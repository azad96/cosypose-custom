from cosypose.utils.tqdm import patch_tqdm; patch_tqdm()  # noqa
import torch.multiprocessing
import time
import json
from collections import OrderedDict
import yaml
import argparse

import torch
import numpy as np
import pandas as pd
import pickle as pkl
import logging

from cosypose.config import EXP_DIR, MEMORY, RESULTS_DIR, LOCAL_DATA_DIR

from cosypose.utils.distributed import init_distributed_mode, get_world_size

from cosypose.lib3d import Transform

from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse, check_update_config
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor

from cosypose.evaluation.meters.pose_meters import PoseErrorMeter
from cosypose.evaluation.pred_runner.multiview_predictions import MultiviewPredictionRunner
from cosypose.evaluation.eval_runner.pose_eval import PoseEvaluation

import cosypose.utils.tensor_collection as tc
from cosypose.evaluation.runner_utils import format_results, gather_predictions
from cosypose.utils.distributed import get_rank


from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset
from cosypose.datasets.bop import remap_bop_targets
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper

from cosypose.datasets.samplers import ListSampler
from cosypose.utils.logging import get_logger
logger = get_logger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@MEMORY.cache
def load_kuatless_detection_results(pickle_file, remove_incorrect_poses=False):
    results_path = LOCAL_DATA_DIR / 'saved_detections' / pickle_file
    pix2pose_results = pkl.loads(results_path.read_bytes())

    infos, poses, bboxes = [], [], []
    for key, result in pix2pose_results.items():
        scene_id, view_id = key.split('/')
        scene_id, view_id = int(scene_id), int(view_id)
        boxes = result['rois']
        scores = result['scores']
        poses_ = result['poses']

        labels = result['labels_txt']
        new_boxes = boxes.copy()
        new_boxes[:,0] = boxes[:,1]
        new_boxes[:,1] = boxes[:,0]
        new_boxes[:,2] = boxes[:,3]
        new_boxes[:,3] = boxes[:,2]
        for o, label in enumerate(labels):
            t = poses_[o][:3, -1]
            if remove_incorrect_poses and (np.sum(t) == 0 or np.max(t) > 100):
                pass
            else:
                infos.append(dict(
                    scene_id=scene_id,
                    view_id=view_id,
                    score=scores[o],
                    label=label,
                ))
                bboxes.append(new_boxes[o])
                poses.append(poses_[o])
    
    data = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        poses=torch.as_tensor(np.stack(poses)),
        bboxes=torch.as_tensor(np.stack(bboxes)).float(),
    ).cpu()
    return data


def get_pose_meters(scene_ds):
    ds_name = scene_ds.name
    compute_add = False
    spheres_overlap_check = True
    large_match_threshold_diameter_ratio = 0.5
    n_top = -1
    visib_gt_min = -1
    dataset_type, test_ds = ds_name.split('.')
    test_type, pbr_type, width, height = test_ds.split('_')
    targets_filename = '{}_{}_{}_{}.json'.format(dataset_type, test_type, width, height) # kuatless_test_1080_810.json

    if targets_filename is not None:
        targets_path = scene_ds.ds_dir / targets_filename
        targets = pd.read_json(targets_path)
        targets = remap_bop_targets(targets)
    else:
        targets = None

    object_ds_name = '{}.eval'.format(dataset_type)
    object_ds = make_object_dataset(object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)

    error_types = ['ADD-S'] + (['ADD(-S)'] if compute_add else [])

    base_kwargs = dict(
        mesh_db=mesh_db,
        exact_meshes=True,
        sample_n_points=None,
        errors_bsz=1,

        # BOP-Like parameters
        n_top=n_top,
        visib_gt_min=visib_gt_min,
        targets=targets,
        spheres_overlap_check=spheres_overlap_check,
    )

    meters = dict()
    for error_type in error_types:
        meters.update({f'{error_type}_ntop=BOP_matching=BOP':  # For ADD-S<0.1d
                        PoseErrorMeter(error_type=error_type, match_threshold=0.1, **base_kwargs),

                        # f'{error_type}_ntop=BOP_matching=OVERLAP': # For measuring ADD-S AUC on T-LESS and average errors on ycbv/tless
                        # PoseErrorMeter(error_type=error_type, consider_all_predictions=False,
                        #                 match_threshold=large_match_threshold_diameter_ratio,
                        #                 report_error_stats=True, report_error_AUC=True, **base_kwargs),

                        # f'{error_type}_ntop=ALL_matching=BOP':  # For mAP
                        # PoseErrorMeter(error_type=error_type, match_threshold=0.1,
                        #                 consider_all_predictions=True,
                        #                 report_AP=True, **base_kwargs),
                    })
    return meters


def load_models(coarse_run_id, refiner_run_id=None, coarse_epoch=None, refiner_epoch=None, n_workers=8, object_set='kuatless'):
    object_ds_name = urdf_ds_name = '{}.cad'.format(object_set)

    object_ds = make_object_dataset(object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=urdf_ds_name, n_workers=n_workers)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id, epoch):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config(cfg)
        checkpoint = f'checkpoint_{epoch}.pth.tar' if epoch else 'checkpoint.pth.tar'
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
            ckpt = torch.load(run_dir / checkpoint) 
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
            ckpt = torch.load(run_dir / checkpoint)
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        return model

    coarse_model = load_model(coarse_run_id, coarse_epoch)
    refiner_model = load_model(refiner_run_id, refiner_epoch)
    model = CoarseRefinePosePredictor(coarse_model=coarse_model,
                                      refiner_model=refiner_model)
    return model, mesh_db


def main():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if 'cosypose' in logger.name:
            logger.setLevel(logging.DEBUG)

    logger.info("Starting ...")
    init_distributed_mode()

    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--config', default='kuatless.test_pbr_1080_810', type=str, required=True)
    parser.add_argument('--comment', type=str, required=True)
    parser.add_argument('--nviews', dest='n_views', default=1, type=int, required=True)
    args = parser.parse_args()

    n_workers = 4
    n_plotters = 4

    n_views = args.n_views
    skip_mv = args.n_views < 2

    # coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-34030' # v1
    # coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-324309' # v2
    # coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-168790' # v3 epoch 60
    # coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-787707' # v4 epoch 30
    # coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-306798' # v5.3 epoch 170
    # coarse_run_id = 'bop-kuatless-coarse-v6' # v6 epoch 140
    # coarse_run_id = 'bop-kuatless-coarse-785678' # v9 epoch 190
    # coarse_run_id = 'bop-kuatless-coarse-n20-981893' # epoch 150
    # coarse_run_id = 'bop-kuatless-coarse-n25-284510' # epoch 130
    # coarse_run_id = 'bop-kuatless-coarse-332k-v6' # epoch 90
    coarse_run_id = 'bop-kuatless-coarse-collision-visib-704232' # epoch 150
    coarse_epoch = 150
    n_coarse_iterations = 1

    # refiner_run_id = None
    # refiner_run_id = 'bop-tless-kuartis-refiner--607469' # v1
    # refiner_run_id = 'bop-tless-kuartis-refiner--434633' # v2
    # refiner_run_id = 'bop-tless-kuartis-refiner--243227' # v3 epoch 90
    # refiner_run_id = 'bop-tless-kuartis-refiner--689761' # v4 epoch 20 but 100 seems better
    # refiner_run_id = 'bop-tless-kuartis-refiner--143806' # v5.1 epoch 200
    refiner_run_id = 'bop-kuatless-refiner-v5.2' # v5.2 epoch 180
    # refiner_run_id = 'bop-kuatless-refiner-332k-v6' # epoch 50
    refiner_epoch = 180
    n_refiner_iterations = 1

    # ds_type, test_type, w, h = args.config.split('-') # kuatless-test-1080-810
    # ds_name = '{}.{}_pbr_{}_{}'.format(ds_type, test_type, w, h) # kuatless.test_pbr_1080_810
    # pickle_file = '{}_{}_{}_{}.pkl'.format(ds_type, test_type, w, h) # kuatless_test_1080_810.pkl
    # object_set = ds_type
    # logger.info(f"DS NAME: {ds_name}")

    ds_name = args.config
    breakpoint()
    ds_type, test_set = ds_name.split('.') # kuatless.test_pbr_1080_810
    test_type, split_type, w, h = test_set.split('_') # test_pbr_1080_810
    pickle_file = '{}_{}_{}_{}.pkl'.format(ds_type, test_type, w, h) # kuatless_test_1080_810.pkl
    object_set = ds_type
    logger.info(f"DS NAME: {ds_name}")

    save_dir = RESULTS_DIR / f'{args.config}-n_views={n_views}-{args.comment}'
    logger.info(f"SAVE DIR: {save_dir}")
    logger.info(f"Coarse: {coarse_run_id}")
    logger.info(f"Refiner: {refiner_run_id}")
    
    # Load dataset
    scene_ds = make_scene_dataset(ds_name)

    # Predictions
    predictor, mesh_db = load_models(coarse_run_id, refiner_run_id, 
                                    coarse_epoch=coarse_epoch, refiner_epoch=refiner_epoch,
                                    n_workers=n_plotters, object_set=object_set)

    mv_predictor = MultiviewScenePredictor(mesh_db)
    base_pred_kwargs = dict(
        n_coarse_iterations=n_coarse_iterations,
        n_refiner_iterations=n_refiner_iterations,
        skip_mv=skip_mv,
        pose_predictor=predictor,
        mv_predictor=mv_predictor,
    )

    pix2pose_detections = load_kuatless_detection_results(pickle_file=pickle_file, remove_incorrect_poses=False).cpu()
    pred_kwargs = {
        'pix2pose_detections': dict(
            detections=pix2pose_detections,
            **base_pred_kwargs
        ),
    }
    
    scene_ds_pred = MultiViewWrapper(scene_ds, n_views=n_views)

    pred_runner = MultiviewPredictionRunner(
        scene_ds_pred, batch_size=1, n_workers=n_workers,
        cache_data=False)

    all_predictions = dict()
    for pred_prefix, pred_kwargs_n in pred_kwargs.items():
        logger.info(f"Prediction: {pred_prefix}")
        preds = pred_runner.get_predictions(**pred_kwargs_n)
        for preds_name, preds_n in preds.items():
            all_predictions[f'{pred_prefix}/{preds_name}'] = preds_n

    logger.info("Done with predictions")
    torch.distributed.barrier()

    results = dict(
        summary=None,
        summary_txt=None,
        predictions=all_predictions,
        metrics=None,
        summary_df=None,
        dfs=None,
    )

    if not save_dir.exists():
        save_dir.mkdir()
    torch.save(results, save_dir / 'results.pth.tar')

    # Evaluation
    predictions_to_evaluate = set()
    det_key = 'pix2pose_detections'
    predictions_to_evaluate.add(f'{det_key}/coarse/iteration={n_coarse_iterations}')
    if refiner_run_id:
        predictions_to_evaluate.add(f'{det_key}/refiner/iteration={n_refiner_iterations}')

    if args.n_views > 1:
        for k in [
                # f'ba_input',
                # f'ba_output',
                f'ba_output+all_cand'
        ]:
            predictions_to_evaluate.add(f'{det_key}/{k}')

    all_predictions = OrderedDict({k: v for k, v in sorted(all_predictions.items(), key=lambda item: item[0])})

    # Evaluation.
    meters = get_pose_meters(scene_ds)
    mv_group_ids = list(iter(pred_runner.sampler))
    scene_ds_ids = np.concatenate(scene_ds_pred.frame_index.loc[mv_group_ids, 'scene_ds_ids'].values)
    sampler = ListSampler(scene_ds_ids)
    eval_runner = PoseEvaluation(scene_ds, meters, n_workers=n_workers,
                                 cache_data=True, batch_size=1, sampler=sampler)

    eval_metrics, eval_dfs = dict(), dict()
    for preds_k, preds in all_predictions.items():
        if preds_k in predictions_to_evaluate:
            logger.info(f"Evaluation : {preds_k} (N={len(preds)})")
            if len(preds) == 0:
                preds = eval_runner.make_empty_predictions()
            eval_metrics[preds_k], eval_dfs[preds_k] = eval_runner.evaluate(preds)
            preds.cpu()
        else:
            logger.info(f"Skipped: {preds_k} (N={len(preds)})")

    all_predictions = gather_predictions(all_predictions)

    metrics_to_print = dict()
    metrics_to_print.update({
        f'{det_key}/coarse/iteration={n_coarse_iterations}/ADD-S_ntop=BOP_matching=OVERLAP/AUC/objects/mean': f'Singleview/AUC of ADD-S',
        f'{det_key}/coarse/iteration={n_coarse_iterations}/ADD-S_ntop=BOP_matching=BOP/0.1d': f'Singleview/ADD-S<0.1d',
        f'{det_key}/coarse/iteration={n_coarse_iterations}/ADD-S_ntop=ALL_matching=BOP/mAP': f'Singleview/mAP@ADD-S<0.1d',

        # f'{det_key}/ba_output+all_cand/ADD-S_ntop=BOP_matching=OVERLAP/AUC/objects/mean': f'Multiview (n={args.n_views})/AUC of ADD-S',
        # f'{det_key}/ba_output+all_cand/ADD-S_ntop=BOP_matching=BOP/0.1d': f'Multiview (n={args.n_views})/ADD-S<0.1d',
        # f'{det_key}/ba_output+all_cand/ADD-S_ntop=ALL_matching=BOP/mAP': f'Multiview (n={args.n_views}/mAP@ADD-S<0.1d)',
    })

    if refiner_run_id:
        metrics_to_print.update({
            f'{det_key}/refiner/iteration={n_refiner_iterations}/ADD-S_ntop=BOP_matching=OVERLAP/AUC/objects/mean': f'Singleview/AUC of ADD-S',
            f'{det_key}/refiner/iteration={n_refiner_iterations}/ADD-S_ntop=BOP_matching=BOP/0.1d': f'Singleview/ADD-S<0.1d',
            f'{det_key}/refiner/iteration={n_refiner_iterations}/ADD-S_ntop=ALL_matching=BOP/mAP': f'Singleview/mAP@ADD-S<0.1d',
        })

    metrics_to_print.update({
        f'{det_key}/ba_input/ADD-S_ntop=BOP_matching=OVERLAP/norm': f'Multiview before BA/ADD-S (m)',
        f'{det_key}/ba_output/ADD-S_ntop=BOP_matching=OVERLAP/norm': f'Multiview after BA/ADD-S (m)',
    })

    if get_rank() == 0:
        if not save_dir.exists():
            save_dir.mkdir()
        results = format_results(all_predictions, eval_metrics, eval_dfs, print_metrics=False)
        (save_dir / 'full_summary.txt').write_text(results.get('summary_txt', ''))

        full_summary = results['summary']
        summary_txt = 'Results:'
        for k, v in metrics_to_print.items():
            if k in full_summary:
                summary_txt += f"\n{v}: {full_summary[k]}"
        logger.info(f"{'-'*80}")
        logger.info(summary_txt)
        logger.info(f"{'-'*80}")

        torch.save(results, save_dir / 'results.pth.tar')
        (save_dir / 'summary.txt').write_text(summary_txt)
        logger.info(f"Saved: {save_dir}")


if __name__ == '__main__':
    patch_tqdm()
    main()
    time.sleep(2)
    if get_world_size() > 1:
        torch.distributed.barrier()

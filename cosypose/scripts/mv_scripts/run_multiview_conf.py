import sys
sys.path.append("/mnt/trains/users/botan/GAG/mmdetection")
from  bm_scripts.bm_inference_azad import BMDetector
from cosypose.utils.tqdm import patch_tqdm; patch_tqdm()  # noqa
import torch.multiprocessing
import time
import json

from collections import OrderedDict
import yaml
import argparse
import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import logging
from pathlib import Path
from bokeh.io import export_png
from PIL import Image

from cosypose.config import EXP_DIR, MEMORY, RESULTS_DIR, LOCAL_DATA_DIR

from cosypose.utils.distributed import init_distributed_mode, get_world_size

from cosypose.lib3d import Transform

from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse, check_update_config
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor

from cosypose.evaluation.meters.pose_meters import PoseErrorMeter
from cosypose.evaluation.pred_runner.single_case_multiview_predictions import SingleCaseMultiviewPredictionRunner
from cosypose.evaluation.eval_runner.pose_eval import PoseEvaluation

import cosypose.utils.tensor_collection as tc
from cosypose.evaluation.runner_utils import format_results, gather_predictions
from cosypose.utils.distributed import get_rank


from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset
from cosypose.datasets.bop import remap_bop_targets
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper

from cosypose.datasets.samplers import ListSampler
from cosypose.utils.logging import get_logger
from cosypose.visualization.multiview import make_colormaps, render_predictions_wrt_camera, nms3d, add_colors_to_predictions, mark_inliers
from cosypose.visualization.plotter import Plotter

logger = get_logger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


def get_twc():

    R1 = [[0.4090992212295532, -0.8999740481376648, 0.150613933801651], 
        [-0.5108031630516052, -0.3626416027545929, -0.7794685363769531], 
        [0.7561203241348267, 0.24194590747356415, -0.6080660223960876]]
    T1 = [-1.5994598865509033, -66.55904388427734, 863.7875366210938]

    R2 = [[0.6589251160621643, -0.5569922924041748, 0.5055466890335083],
        [-0.1656114161014557, -0.7630153894424438, -0.6248047947883606],
        [0.733751118183136, 0.327975332736969, -0.5950143933296204]]
    T2 = [37.58763885498047, -52.60617446899414, 883.1234130859375]

    R3 = [[-0.21939997375011444, -0.8263000249862671, -0.5187408328056335],
        [-0.6794779896736145, 0.5109611749649048, -0.5265247821807861],
        [0.7001238465309143, 0.236953467130661, -0.6735575795173645]]
    T3 = [-58.61317825317383, -41.466285705566406, 780.687744140625]

    R4 = [[0.45168551802635193, -0.745762050151825, -0.4897133409976959],
        [-0.7662615180015564, -0.04312900826334953, -0.6410797238349915],
        [0.45697206258773804, 0.6648148894309998, -0.5909295082092285]]
    T4 = [1.442326307296753, -73.15863037109375, 879.6213989257812]

    def get_extrinsic_matrix(R, T):
        r = np.array(R)
        t = np.array(T).reshape(3, -1) * 0.001
        l = np.array([0, 0, 0, 1]).reshape(1,4)
        m = np.concatenate((r, t), axis=1)
        m = np.concatenate((m, l), axis=0)
        return m

    def my_T(R1, R2, T1, T2):
        R2_inv = np.linalg.inv(R2)
        R = R1 @ R2_inv
        x = R @ T2
        T = T1 - x
        return get_extrinsic_matrix(R, T)

    _twc = np.zeros((4,4,4))
    _twc[0,:,:] = my_T(R1, R1, T1, T1)
    _twc[1,:,:] = my_T(R1, R2, T1, T2)
    _twc[2,:,:] = my_T(R1, R3, T1, T3)
    _twc[3,:,:] = my_T(R1, R4, T1, T4)

    _twc = torch.from_numpy(_twc).cuda().float()
    return _twc


def main():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if 'cosypose' in logger.name:
            logger.setLevel(logging.DEBUG)

    logger.info("Starting ...")
    init_distributed_mode()

    parser = argparse.ArgumentParser('Multiview')
    parser.add_argument('--config', default='tless-bop', type=str)
    parser.add_argument('--comment', default='', type=str)
    args = parser.parse_args()

    n_workers = 4

    object_set = 'kuatless'

    # coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-168790' # v3
    coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-787707' # v4 epoch 30
    coarse_epoch = 30
    n_coarse_iterations = 1

    # refiner_run_id = 'bop-tless-kuartis-refiner--243227' # v3 epoch 90
    refiner_run_id = 'bop-tless-kuartis-refiner--689761' # v4 epoch 20, 100 seems better
    refiner_epoch = 100
    n_refiner_iterations = 1

    # n_rand = np.random.randint(1e10)
    # save_dir = RESULTS_DIR / f'{args.config}-n_views={n_views}-{args.comment}-{n_rand}'
    # logger.info(f"SAVE DIR: {save_dir}")
    logger.info(f"Coarse: {coarse_run_id}")
    logger.info(f"Refiner: {refiner_run_id}")

    # Predictions
    bm_detector = BMDetector()
    predictor, mesh_db = load_models(coarse_run_id, refiner_run_id, coarse_epoch=coarse_epoch, refiner_epoch=refiner_epoch, 
                                    n_workers=n_workers, object_set=object_set)
    mv_predictor = MultiviewScenePredictor(mesh_db)
    
    K = np.array([[1905.52, 0.0, 361.142],
                [0.0, 1902.99, 288.571],
                [0.0, 0.0, 1.0]])
    # K = np.array([[1075.650, 0.0, 360],
    #             [0.0, 1073.903, 270],
    #             [0.0, 0.0, 1.0]])

    folder_name = 'rgb_objects_no_clutter'
    folder_path = '/mnt/trains/users/azad/BM/inputs/cell_4cam/{}'.format(folder_name)
    json_path = '{}/samples.json'.format(folder_path) 
    save_dir = '/mnt/trains/users/azad/BM/results_mv_conf/cell_4cam/{}'.format(folder_name)
    os.makedirs(save_dir, exist_ok=True)
    samples_json = json.loads(Path(json_path).read_text())

    pred_runner = SingleCaseMultiviewPredictionRunner()
    renderer = BulletSceneRenderer(urdf_ds_name)
    plotter = Plotter()
    
    for scene_id, scene in enumerate(samples_json['samples']):
        img_paths = []
        for view in scene['cameras']:
            img_name = view['image_name']
            img_path = os.path.join(folder_path, img_name)
            img_paths.append(img_path)

        n_views = len(img_paths)
        skip_mv = n_views < 2
        view_ids = list(range(1, n_views+1))

        detections = []
        segmentations = []
        for img_path in img_paths:
            det, seg = bm_detector.get_detection(img_path)
            detections.append(det)
            segmentations.append(seg)
        infos = []
        for i, det in enumerate(detections):
            infos.append(
                pd.DataFrame({
                    'label': det.infos['label'],
                    'score': det.infos['score'],
                    'scene_id': 1,
                    'view_id': view_ids[i],
                    'group_id': 1,
                    'batch_im_id': i,
                })
            )

        all_detections = tc.PandasTensorCollection(
            bboxes = torch.cat(list(det.bboxes for det in detections), 0),
            infos = pd.concat(infos, ignore_index=True),
        )

        images = []
        images_dict = dict()
        for i, img_path in enumerate(img_paths):
            img = np.array(Image.open(img_path))
            images_dict[view_ids[i]] = img
            img = np.expand_dims(img, axis=0)
            images.append(img)
        
        images = np.concatenate(images)
        images = torch.from_numpy(images).cuda().float()
        images = images.permute(0, 3, 1, 2) / 255

        cams = np.repeat(K[np.newaxis,:,:], 4, axis=0)
        cams = torch.from_numpy(cams).cuda().float()    
        cam_info = pd.DataFrame({
            'scene_id': 1,
            'view_id': view_ids,
            'group_id': 1,
            'batch_im_id': list(range(n_views)),
            })

        cameras = tc.PandasTensorCollection(
            K=cams,
            infos=cam_info,
        )
        base_pred_kwargs = dict(
            n_coarse_iterations=n_coarse_iterations,
            n_refiner_iterations=n_refiner_iterations,
            skip_mv=skip_mv,
            pose_predictor=predictor,
            mv_predictor=mv_predictor,
        )
        pred_kwargs = dict(
            detections=all_detections,
            images=images,
            cameras=cameras,
            **base_pred_kwargs
        )
        
        all_predictions = dict()
        preds = pred_runner.get_predictions(**pred_kwargs)
        for preds_name, preds_n in preds.items():
            all_predictions[f'{preds_name}'] = preds_n

        logger.info("Done with predictions")
        torch.distributed.barrier()
        all_predictions = OrderedDict({k: v for k, v in sorted(all_predictions.items(), key=lambda item: item[0])})
        results = gather_predictions(all_predictions)

        # os.makedirs(save_dir, exist_ok=False)
        # torch.save(results, save_dir / 'results.pth.tar')

        # renderer = BulletSceneRenderer(urdf_ds_name)
        # plotter = Plotter()
        dict_preds = dict()
        for k in ('cand_inputs', 'cand_matched', 'ba_output'):
            dict_preds[k] = results[f'{k}']

        preds_by_view = dict()
        for view_id in view_ids:
            this_view_dict_preds = dict()
            for k in ('cand_inputs', 'cand_matched', 'ba_output'):
                assert k in dict_preds
                pred_infos = dict_preds[k].infos
                scene_labels = pred_infos['label'].values.tolist()
                keep = np.isin(pred_infos['view_id'], view_id)
                this_view_dict_preds[k] = dict_preds[k][np.where(keep)[0]]
            preds_by_view[view_id] = this_view_dict_preds

        colormap_rgb, colormap_hex = make_colormaps(scene_labels)
        colormap_rgb_3d = colormap_rgb

        for view_id in view_ids:
            this_view_dict_preds = preds_by_view[view_id]
            input_rgb = images_dict[view_id].copy()
            gt_state = dict()
            gt_state['camera'] = dict({'T0C': np.eye(4),
                                        'K': K,
                                        'TWC': np.eye(4),
                                        'resolution': (540, 720),})
            fig_input_im = plotter.plot_image(input_rgb)

            # Detections
            detections = this_view_dict_preds['cand_inputs']
            detections.bboxes = detections.initial_bboxes

            detections = add_colors_to_predictions(detections, colormap_hex)
            fig_detections = plotter.plot_image(input_rgb)
            fig_detections = plotter.plot_maskrcnn_bboxes(fig_detections, detections, colors=detections.infos['color'].tolist())

            # Candidates
            cand_inputs = this_view_dict_preds['cand_inputs']
            cand_matched = this_view_dict_preds['cand_matched']
            cand_inputs = mark_inliers(cand_inputs, cand_matched)
            colors = np.array([(0, 1, 0, 1.0) if is_inlier else (1.0, 0, 0, 0.3) for is_inlier in cand_inputs.infos['is_inlier']])
            cand_inputs.infos['color'] = colors.tolist()

            cand_rgb_rendered = render_predictions_wrt_camera(renderer, cand_inputs, gt_state['camera'])
            cand_binary_mask = cand_rgb_rendered > 0
            cand_binary_mask = np.logical_or(cand_binary_mask[:,:,0], cand_binary_mask[:,:,1], cand_binary_mask[:,:,2])
            cand_rendered_binary = (cand_binary_mask * 255).astype(np.uint8)
            fig_cand = plotter.plot_overlay(input_rgb.copy(), cand_rgb_rendered)
            fig_cand = plotter.plot_confidence_scores(fig_cand, detections, cand_rendered_binary, segmentations[view_id-1])
            
            # Scene reconstruction
            ba_outputs = this_view_dict_preds['ba_output']
            # print("\n\n ba_outputs {} \n\n".format(ba_outputs))

            use_nms3d = True
            if use_nms3d:
                ba_outputs = nms3d(ba_outputs)
            ba_outputs = add_colors_to_predictions(ba_outputs, colormap_rgb_3d)
            outputs_rgb_rendered = render_predictions_wrt_camera(renderer, ba_outputs, gt_state['camera'])
            outputs_binary_mask = outputs_rgb_rendered > 0
            outputs_binary_mask = np.logical_or(outputs_binary_mask[:,:,0], outputs_binary_mask[:,:,1], outputs_binary_mask[:,:,2])
            outputs_rendered_binary = (outputs_binary_mask * 255).astype(np.uint8)
            fig_outputs = plotter.plot_overlay(input_rgb.copy(), outputs_rgb_rendered)
            fig_outputs = plotter.plot_confidence_scores(fig_outputs, detections, outputs_rendered_binary, segmentations[view_id-1])

            save_dir2 = os.path.join(save_dir, str(scene_id))
            os.makedirs(save_dir2, exist_ok=True)

            # cmask = Image.fromarray(cand_rendered_binary)
            # cmask.save(os.path.join(save_dir2, 'mask{}.png'.format(view_id)))
            export_png(fig_input_im, filename = os.path.join(save_dir2, 'input_im_{}.png'.format(view_id)))
            export_png(fig_detections, filename = os.path.join(save_dir2, 'detections_{}.png'.format(view_id)))
            export_png(fig_cand, filename = os.path.join(save_dir2, 'cand_{}.png'.format(view_id)))
            export_png(fig_outputs, filename = os.path.join(save_dir2, 'outputs_{}.png'.format(view_id)))
            del fig_input_im, fig_detections, fig_cand, fig_outputs

if __name__ == '__main__':
    patch_tqdm()
    main()
    time.sleep(2)
    if get_world_size() > 1:
        torch.distributed.barrier()

import sys
sys.path.append("/mnt/trains/users/botan/GAG/mmdetection")
from  bm_scripts.bm_inference import BMDetector
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


def load_models(coarse_run_id, refiner_run_id=None, n_workers=8, object_set='kuatless'):
    object_ds_name = urdf_ds_name = '{}.cad'.format(object_set)

    object_ds = make_object_dataset(object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=urdf_ds_name, n_workers=n_workers)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
            ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
            ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        return model

    coarse_model = load_model(coarse_run_id)
    refiner_model = load_model(refiner_run_id)
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

    parser = argparse.ArgumentParser('Multiview')
    parser.add_argument('--config', default='tless-bop', type=str)
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--nviews', dest='n_views', default=4, type=int)
    args = parser.parse_args()

    n_workers = 2
    n_plotters = 2
    n_views = args.n_views
    skip_mv = args.n_views < 2
    skip_predictions = False

    object_set = 'kuatless'
    coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-34030'
    # coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-34030'
    # refiner_run_id = 'bop-tless-kuartis-refiner--636300'
    # refiner_run_id = 'bop-tless-kuartis-refiner--607469'
    refiner_run_id = None
    n_coarse_iterations = 1
    n_refiner_iterations = 0

    # n_rand = np.random.randint(1e10)
    # save_dir = RESULTS_DIR / f'{args.config}-n_views={n_views}-{args.comment}-{n_rand}'
    # logger.info(f"SAVE DIR: {save_dir}")
    logger.info(f"Coarse: {coarse_run_id}")
    logger.info(f"Refiner: {refiner_run_id}")

    # Predictions
    bm_detector = BMDetector()
    predictor, mesh_db = load_models(coarse_run_id, refiner_run_id, n_workers=n_plotters, object_set=object_set)
    
    mv_predictor = MultiviewScenePredictor(mesh_db)
    base_pred_kwargs = dict(
        n_coarse_iterations=n_coarse_iterations,
        n_refiner_iterations=n_refiner_iterations,
        skip_mv=skip_mv,
        pose_predictor=predictor,
        mv_predictor=mv_predictor,
    )
    view_ids = [1, 2, 3, 4]
    """
    K = np.array([[1905.52, 0.0, 361.142],
                [0.0, 1902.99, 288.571],
                [0.0, 0.0, 1.0]])
    """
    K = np.array([[1075.650, 0.0, 360],
                [0.0, 1073.903, 270],
                [0.0, 0.0, 1.0]])
    # img_path1 = '/mnt/trains/users/botan/BM/results/multiview2/input/000047.jpg'
    # img_path2 = '/mnt/trains/users/botan/BM/results/multiview2/input/000048.jpg'
    # img_path3 = '/mnt/trains/users/botan/BM/results/multiview2/input/000049.jpg'
    # img_path4 = '/mnt/trains/users/botan/BM/results/multiview2/input/000050.jpg'

    img_path1 = '/mnt/trains/users/botan/BM/results/multiview2/input_new/000041.jpg'
    img_path2 = '/mnt/trains/users/botan/BM/results/multiview2/input_new/000044.jpg'
    img_path3 = '/mnt/trains/users/botan/BM/results/multiview2/input_new/000052.jpg'
    img_path4 = '/mnt/trains/users/botan/BM/results/multiview2/input_new/000056.jpg'

    detections1 = bm_detector.get_detection(img_path1)
    detections2 = bm_detector.get_detection(img_path2)
    detections3 = bm_detector.get_detection(img_path3)
    detections4 = bm_detector.get_detection(img_path4)
    infos1 = pd.DataFrame({
        'label': detections1.infos['label'],
        'score': detections1.infos['score'],
        'scene_id': 1,
        'view_id': view_ids[0],
        'group_id': 1,
        'batch_im_id': 0,
        })
    infos2 = pd.DataFrame({
        'label': detections2.infos['label'],
        'score': detections2.infos['score'],
        'scene_id': 1,
        'view_id': view_ids[1],
        'group_id': 1,
        'batch_im_id': 1,
        })
    infos3 = pd.DataFrame({
        'label': detections3.infos['label'],
        'score': detections3.infos['score'],
        'scene_id': 1,
        'view_id': view_ids[2],
        'group_id': 1,
        'batch_im_id': 2,
        })
    infos4 = pd.DataFrame({
        'label': detections4.infos['label'],
        'score': detections4.infos['score'],
        'scene_id': 1,
        'view_id': view_ids[3],
        'group_id': 1,
        'batch_im_id': 3,
        })
    all_detections = tc.PandasTensorCollection(
        bboxes=torch.cat((detections1.bboxes, detections2.bboxes, detections3.bboxes, detections4.bboxes), 0),
        infos=infos1.append(infos2.append(infos3.append(infos4, ignore_index=True), ignore_index=True), ignore_index=True),
    )

    img1 = np.array(Image.open(img_path1))
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.array(Image.open(img_path2))
    img2 = np.expand_dims(img2, axis=0)
    img3 = np.array(Image.open(img_path3))
    img3 = np.expand_dims(img3, axis=0)
    img4 = np.array(Image.open(img_path4))
    img4 = np.expand_dims(img4, axis=0)
    images = np.concatenate((img1, img2, img3, img4))
    images = torch.from_numpy(images).cuda().float()
    images = images.permute(0, 3, 1, 2) / 255

    cams = np.repeat(K[np.newaxis,:,:], 4, axis=0)
    cams = torch.from_numpy(cams).cuda().float()    
    cam_info = pd.DataFrame({
        'scene_id': [1, 1, 1, 1],
        'view_id': [1, 2, 3, 4],
        'group_id': [1, 1, 1, 1],
        'batch_im_id': [0, 1, 2, 3],
        })
    """
    _twc1 = np.array([[ 9.91763014e-01, 5.41808081e-04 , 1.28085246e-01,-163.347],
                        [-1.03748367e-03, 9.99992230e-01,  3.80320196e-03, 0.1302],
                        [-1.28082190e-01, -3.90476139e-03,  9.91755870e-01, 22.4169],
                        [0,0,0,0.9917]])

    _twc2 = np.array([[1,   0,  0, 0],[0, 1, 0, 0],
    [ 0, 0,   1, 0], [0,             0,          0,          1]])
    _twc = np.zeros((2,4,4))
    _twc[0,:,:] = _twc1
    _twc[1,:,:] = _twc2
    _twc = torch.from_numpy(_twc).cuda().float()
    """
    cameras = tc.PandasTensorCollection(
        K=cams,
        infos=cam_info,
    )

    pred_kwargs = dict(detections=all_detections,
                        images=images,
                        cameras=cameras,
                        **base_pred_kwargs)
    pred_runner = SingleCaseMultiviewPredictionRunner()

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

    renderer = BulletSceneRenderer(urdf_ds_name)
    plotter = Plotter()
    dict_preds = dict()
    for k in ('cand_inputs', 'cand_matched', 'ba_output'):
        dict_preds[k] = results[f'{k}']
    #for k in dict_preds.keys():
    #    dict_preds[k] = dict_preds['cand_inputs']

    preds_by_view = dict()
    for view_id in view_ids:
        this_view_dict_preds = dict()
        for k in ('cand_inputs', 'cand_matched', 'ba_output'):
            assert k in dict_preds

            # scene_labels = scene_labels.union(set(dict_preds[k].infos['label'].values.tolist()))
            scene_labels = ['obj_000001' for i in range(len(dict_preds[k].infos['label']))]
            pred_infos = dict_preds[k].infos
            keep = np.logical_and(pred_infos['scene_id'] == 1,
                                  np.isin(pred_infos['view_id'], view_id))
            this_view_dict_preds[k] = dict_preds[k][np.where(keep)[0]]
        preds_by_view[view_id] = this_view_dict_preds

    colormap_rgb, colormap_hex = make_colormaps(scene_labels)
    colormap_rgb_3d = colormap_rgb

    images_dict = dict()
    images_dict[view_ids[0]] = img1[0]
    images_dict[view_ids[1]] = img2[0]
    images_dict[view_ids[2]] = img3[0]
    images_dict[view_ids[3]] = img4[0]
    
    fig_array = []
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
        bboxes = detections.initial_bboxes
        bboxes = bboxes + torch.as_tensor(np.random.randint(30, size=((len(bboxes), 4)))).to(bboxes.dtype).to(bboxes.device)
        detections.bboxes = bboxes

        detections = add_colors_to_predictions(detections, colormap_hex)
        fig_detections = plotter.plot_image(input_rgb)
        fig_detections = plotter.plot_maskrcnn_bboxes(fig_detections, detections, colors=detections.infos['color'].tolist())

        # Candidates
        cand_inputs = this_view_dict_preds['cand_inputs']
        cand_matched = this_view_dict_preds['cand_matched']
        # 3 satiri ekledim
        cand_inputs = mark_inliers(cand_inputs, cand_matched)
        #breakpoint()
        colors = np.array([(0, 1, 0, 1.0) if is_inlier else (1.0, 0, 0, 0.3) for is_inlier in cand_inputs.infos['is_inlier']])
        cand_inputs.infos['color'] = colors.tolist()

        cand_rgb_rendered = render_predictions_wrt_camera(renderer, cand_inputs, gt_state['camera'])
        fig_cand = plotter.plot_overlay(input_rgb.copy(), cand_rgb_rendered)


        # Scene reconstruction
        ba_outputs = this_view_dict_preds['ba_output']
        print("\n\n ba_outputs {} \n\n".format(ba_outputs))

        use_nms3d = True
        if use_nms3d:
            ba_outputs = nms3d(ba_outputs)
        ba_outputs = add_colors_to_predictions(ba_outputs, colormap_rgb_3d)
        outputs_rgb_rendered = render_predictions_wrt_camera(renderer, ba_outputs, gt_state['camera'])
        fig_outputs = plotter.plot_overlay(input_rgb.copy(), outputs_rgb_rendered)

        
        #fig_outputs = plotter.plot_image(outputs_rgb_rendered)
        fig_array.append([fig_input_im, fig_detections, fig_cand, fig_outputs])

        save_dir = '/mnt/trains/users/azad/BM/results/multiview_result/deneme/'
        os.makedirs(save_dir, exist_ok=True)

        export_png(fig_input_im, filename = os.path.join(save_dir, 'input_im_{}.png'.format(view_id)))
        export_png(fig_detections, filename = os.path.join(save_dir, 'detections_{}.png'.format(view_id)))
        export_png(fig_cand, filename = os.path.join(save_dir, 'cand_{}.png'.format(view_id)))
        export_png(fig_outputs, filename = os.path.join(save_dir, 'outputs_{}.png'.format(view_id)))
        del fig_input_im, fig_detections, fig_cand, fig_outputs

if __name__ == '__main__':
    patch_tqdm()
    main()
    time.sleep(2)
    if get_world_size() > 1:
        torch.distributed.barrier()

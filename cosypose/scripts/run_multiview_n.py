import sys
sys.path.append("/mnt/trains/users/botan/GAG/mmdetection")
from  bm_scripts.bm_inference_azad import BMDetector
from cosypose.utils.tqdm import patch_tqdm; patch_tqdm()  # noqa
import torch.multiprocessing
import time
import json

from collections import OrderedDict
import yaml
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image
from pytorch3d.transforms import quaternion_to_matrix

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


# def compute_intrinsic(fx, fy, cx, cy):
#     coeff = 3/7
#     new_fx = fx*coeff
#     new_fy = fy*coeff
#     new_cx = (cx-568/2)*coeff
#     new_cy = (cy-174/2)*coeff
#     K = np.array([[new_fx, 0.0, new_cx],
#                 [0.0, new_fy, new_cy],
#                 [0.0, 0.0, 1.0]])
#     return K

def compute_intrinsic(cam_coeff, input_dim):
    if input_dim == (1080, 810):
        coeff = 3/7
    elif input_dim == (720, 540):
        coeff = 2/7
    else:
        raise ValueError(input_dim)
    
    fx, fy, cx, cy = cam_coeff[:4]
    new_fx = fx*coeff
    new_fy = fy*coeff
    new_cx = (cx-568/2)*coeff
    new_cy = (cy-174/2)*coeff
    K = np.array([[new_fx, 0.0, new_cx],
                [0.0, new_fy, new_cy],
                [0.0, 0.0, 1.0]])
    return K


def get_cam_coeff(img_name, calibration):
    frame_id = img_name.split('_')[2]
    cam_coeff = [comp['properties']['camera_coefficients'] for comp in calibration['components'] if comp['uuid'] == frame_id]
    assert len(cam_coeff) == 1
    return cam_coeff[0]


def main():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if 'cosypose' in logger.name:
            logger.setLevel(logging.DEBUG)

    logger.info("Starting ...")
    init_distributed_mode()

    n_workers = 4
    object_set = 'kuatless'
    input_dim = (1080, 810)

    coarse_run_id = 'bop-kuatless-coarse-v6' # epoch 140
    coarse_epoch = 140
    n_coarse_iterations = 1

    refiner_run_id = 'bop-kuatless-refiner-v5.2' # epoch 180
    refiner_epoch = 180
    n_refiner_iterations = 1

    logger.info(f"Coarse: {coarse_run_id}")
    logger.info(f"Refiner: {refiner_run_id}")

    # Predictions
    bm_detector = BMDetector()
    predictor, mesh_db = load_models(coarse_run_id, refiner_run_id, coarse_epoch=coarse_epoch, refiner_epoch=refiner_epoch, 
                                    n_workers=n_workers, object_set=object_set)
    mv_predictor = MultiviewScenePredictor(mesh_db)

    # K = np.array([[1905.52, 0.0, 361.142],
    #             [0.0, 1902.99, 288.571],
    #             [0.0, 0.0, 1.0]])
    # K = np.array([[1075.650, 0.0, 360],
    #             [0.0, 1073.903, 270],
    #             [0.0, 0.0, 1.0]])

    # img_paths = [
    #     '/mnt/trains/users/azad/BM/inputs/cell_4cam/objects_no_clutter_1080_810/cam_23422354_7b334636-3546-4563-a459-845445645ad5_10-02-2021_12-33-06.png', 
    #     '/mnt/trains/users/azad/BM/inputs/cell_4cam/objects_no_clutter_1080_810/cam_23158292_48aca4b4-b3a5-47bc-8465-c9d5546c47da_10-02-2021_12-33-06.png',
    #     '/mnt/trains/users/azad/BM/inputs/cell_4cam/objects_no_clutter_1080_810/cam_23124735_4b3cc848-5bcc-4765-9883-54d56d8694de_10-02-2021_12-33-06.png', 
    #     '/mnt/trains/users/azad/BM/inputs/cell_4cam/objects_no_clutter_1080_810/cam_23124722_8ac6a535-4639-4a65-9766-9db454b687c8_10-02-2021_12-33-06.png',
    # ]

    org_folder_pth = '/mnt/trains/users/azad/BM/inputs/cell_4cam/objects_no_clutter_1080_810'
    crop_folder_pth = '/mnt/trains/users/azad/BM/inputs/cropped_top_views/cell_4cam/objects_no_clutter_1080_810'
    # img_paths = [
    #     '{}/cam_23422354_7b334636-3546-4563-a459-845445645ad5_10-02-2021_12-33-45.png'.format(org_folder_pth), 
    #     '{}/cam_23158292_48aca4b4-b3a5-47bc-8465-c9d5546c47da_10-02-2021_12-33-45_2.png'.format(crop_folder_pth),
    #     '{}/cam_23158292_48aca4b4-b3a5-47bc-8465-c9d5546c47da_10-02-2021_12-33-45_3.png'.format(crop_folder_pth),
    #     '{}/cam_23124735_4b3cc848-5bcc-4765-9883-54d56d8694de_10-02-2021_12-33-45.png'.format(org_folder_pth), 
    #     '{}/cam_23124722_8ac6a535-4639-4a65-9766-9db454b687c8_10-02-2021_12-33-45.png'.format(org_folder_pth),
    # ]
    img_paths = [
        '{}/cam_23422354_7b334636-3546-4563-a459-845445645ad5_10-02-2021_12-33-45.png'.format(org_folder_pth), 
        '{}/cam_23158292_48aca4b4-b3a5-47bc-8465-c9d5546c47da_10-02-2021_12-33-45_1.png'.format(crop_folder_pth),
        '{}/cam_23158292_48aca4b4-b3a5-47bc-8465-c9d5546c47da_10-02-2021_12-33-45_2.png'.format(crop_folder_pth),
        '{}/cam_23158292_48aca4b4-b3a5-47bc-8465-c9d5546c47da_10-02-2021_12-33-45_3.png'.format(crop_folder_pth),
        '{}/cam_23124735_4b3cc848-5bcc-4765-9883-54d56d8694de_10-02-2021_12-33-45_0.png'.format(crop_folder_pth), 
        '{}/cam_23124735_4b3cc848-5bcc-4765-9883-54d56d8694de_10-02-2021_12-33-45_2.png'.format(crop_folder_pth), 
        '{}/cam_23124735_4b3cc848-5bcc-4765-9883-54d56d8694de_10-02-2021_12-33-45_3.png'.format(crop_folder_pth), 
        '{}/cam_23124722_8ac6a535-4639-4a65-9766-9db454b687c8_10-02-2021_12-33-45.png'.format(org_folder_pth),
    ]

    calib_path = '/'.join(org_folder_pth.split('/')[:-1])
    calibration = json.loads(Path('{}/calibration.json'.format(calib_path)).read_text())

    save_dir = '/mnt/trains/users/azad/BM/results/mv_8view'
    os.makedirs(save_dir, exist_ok=True)

    n_views = len(img_paths)
    skip_mv = n_views < 2
    view_ids = list(range(n_views))

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

    # K1 = compute_intrinsic(6719.372067539465, 6787.764172975213, 1590.7857807217506, 1196.5379228742668)
    # K2 = compute_intrinsic(5047.308848350872, 5044.538622084761, 1482.8328107218515, 1134.1452560470157)
    # K3 = compute_intrinsic(5026.484190265724, 5020.441312027096, 1522.9123039464473, 1098.0725284499242)
    # K4 = compute_intrinsic(6819.976509193038, 6822.473739038355, 1517.7673526582266, 988.6879829910563)
    # K_matrices = [K4, K3, K2, K1]
    K_matrices = []
    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        cam_coeff = get_cam_coeff(img_name, calibration)

        name_splits = img_name.split('_')
        if len(name_splits) == 4: # not cropped images
            K = compute_intrinsic(cam_coeff, input_dim)
        else: #ends with _0.png, _1.png ...
            K = np.array([[cam_coeff[0], 0.0, cam_coeff[2]],
                            [0.0, cam_coeff[1], cam_coeff[3]],
                            [0.0, 0.0, 1.0]])
            if name_splits[-1][0] == '0':
                K[0][2] -= 744
                K[1][2] -= 467
            elif name_splits[-1][0] == '1':
                K[0][2] -= 1244
                K[1][2] -= 467
            elif name_splits[-1][0] == '2':
                K[0][2] -= 744
                K[1][2] -= 787
            elif name_splits[-1][0] == '3':
                K[0][2] -= 1244
                K[1][2] -= 787
            else:
                raise ValueError(name_splits[-1][0])

        K_matrices.append(K)

    cams = np.stack(K_matrices)
    cams = torch.from_numpy(cams).cuda().float()    
    cam_info = pd.DataFrame({
        'scene_id': 1,
        'view_id': view_ids,
        'group_id': 1,
        'batch_im_id': list(range(n_views)),
        })

    TWC = None
    # TWC = get_twc()
    cameras = tc.PandasTensorCollection(
        K=cams,
        infos=cam_info,
    )
    if TWC is not None:
        cameras.TWC=TWC

    pred_kwargs = dict(
        detections=all_detections,
        images=images,
        cameras=cameras,
        use_known_camera_poses=TWC is not None,
        n_coarse_iterations=n_coarse_iterations,
        n_refiner_iterations=n_refiner_iterations,
        skip_mv=skip_mv,
        pose_predictor=predictor,
        mv_predictor=mv_predictor,
    )
    pred_runner = SingleCaseMultiviewPredictionRunner()

    all_predictions = dict()
    preds = pred_runner.get_predictions(**pred_kwargs)
    for preds_name, preds_n in preds.items():
        all_predictions[f'{preds_name}'] = preds_n

    logger.info("Done with predictions")
    torch.distributed.barrier()
    all_predictions = OrderedDict({k: v for k, v in sorted(all_predictions.items(), key=lambda item: item[0])})
    results = gather_predictions(all_predictions)

    renderer = BulletSceneRenderer(urdf_ds_name)
    plotter = Plotter()
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
                                    'K': K_matrices[view_id],
                                    'resolution': input_dim,
                                    'TWC': np.eye(4), # try also putting TWC here
                                    }) 
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
        # colors = np.array([(1.0, 1.0, 1.0, 1.0) for is_inlier in cand_inputs.infos['is_inlier']])
        colors = np.array([(0, 1.0, 0, 1.0) if is_inlier else (0, 1.0, 1.0, 1.0) for is_inlier in cand_inputs.infos['is_inlier']])
        cand_inputs.infos['color'] = colors.tolist()

        cand_rgb_rendered = render_predictions_wrt_camera(renderer, cand_inputs, gt_state['camera'])
        cand_binary_mask = cand_rgb_rendered > 0
        cand_binary_mask = np.logical_or(cand_binary_mask[:,:,0], cand_binary_mask[:,:,1], cand_binary_mask[:,:,2])
        cand_rendered_binary = (cand_binary_mask * 255).astype(np.uint8)
        fig_cand = plotter.plot_overlay(input_rgb.copy(), cand_rgb_rendered)
        fig_cand = plotter.plot_confidence_scores(fig_cand, detections, cand_rendered_binary, segmentations[view_id])

        # Scene reconstruction
        ba_outputs = this_view_dict_preds['ba_output']
        # print("\n\n ba_outputs {} \n\n".format(ba_outputs))

        use_nms3d = True
        if use_nms3d:
            ba_outputs = nms3d(ba_outputs)
        # ba_outputs = add_colors_to_predictions(ba_outputs, colormap_rgb_3d)
        ba_outputs.infos['color'] = [(0, 1.0, 0, 0.0) for i in range(len(ba_outputs))]
        outputs_rgb_rendered = render_predictions_wrt_camera(renderer, ba_outputs, gt_state['camera'])
        outputs_binary_mask = outputs_rgb_rendered > 0
        outputs_binary_mask = np.logical_or(outputs_binary_mask[:,:,0], outputs_binary_mask[:,:,1], outputs_binary_mask[:,:,2])
        outputs_rendered_binary = (outputs_binary_mask * 255).astype(np.uint8)
        fig_outputs = plotter.plot_overlay(input_rgb.copy(), outputs_rgb_rendered)
        fig_outputs = plotter.plot_confidence_scores(fig_outputs, detections, outputs_rendered_binary, segmentations[view_id])

        fig_array = [fig_input_im, fig_detections, fig_cand, fig_outputs]
        res = gridplot(fig_array, ncols=2)
        img_name = img_paths[view_id].split('/')[-1].split('.')[0]
        export_png(res, filename = os.path.join(save_dir, '{}.png'.format(img_name)))
        del fig_input_im, fig_detections, fig_cand, fig_outputs

if __name__ == '__main__':
    patch_tqdm()
    main()
    time.sleep(2)
    if get_world_size() > 1:
        torch.distributed.barrier()

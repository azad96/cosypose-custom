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


def load_models(coarse_run_id, refiner_run_id=None, coarse_epoch=None, refiner_epoch=None, n_workers=8, object_ds_name='kuartis.cad', urdf_ds_name='kuartis.cad'):
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
    #ex1
    # R1 = [[0.4090992212295532, -0.8999740481376648, 0.150613933801651], 
    #     [-0.5108031630516052, -0.3626416027545929, -0.7794685363769531], 
    #     [0.7561203241348267, 0.24194590747356415, -0.6080660223960876]]
    # T1 = [-1.5994598865509033, -66.55904388427734, 863.7875366210938]

    # R2 = [[0.6589251160621643, -0.5569922924041748, 0.5055466890335083],
    #     [-0.1656114161014557, -0.7630153894424438, -0.6248047947883606],
    #     [0.733751118183136, 0.327975332736969, -0.5950143933296204]]
    # T2 = [37.58763885498047, -52.60617446899414, 883.1234130859375]

    # R3 = [[-0.21939997375011444, -0.8263000249862671, -0.5187408328056335],
    #     [-0.6794779896736145, 0.5109611749649048, -0.5265247821807861],
    #     [0.7001238465309143, 0.236953467130661, -0.6735575795173645]]
    # T3 = [-58.61317825317383, -41.466285705566406, 780.687744140625]

    # R4 = [[0.45168551802635193, -0.745762050151825, -0.4897133409976959],
    #     [-0.7662615180015564, -0.04312900826334953, -0.6410797238349915],
    #     [0.45697206258773804, 0.6648148894309998, -0.5909295082092285]]
    # T4 = [1.442326307296753, -73.15863037109375, 879.6213989257812]

    # ex2
    R1 = [[ 0.04824766516685486, 0.7500212788581848, 0.6596516966819763 ],
       [-0.7540968656539917, 0.4604244530200958, -0.4683452248573303],
       [-0.6549886465072632, -0.47484466433525085, 0.587803065776825]]
    T1 = [-0.0001540098455734551, 0.00013891118578612804, 732.7607421875]

    R2 = [[-0.5079371333122253, 0.6439588069915771, -0.5721161961555481 ],
       [-0.5435666441917419, -0.7548550963401794, -0.36705467104911804],
       [-0.6682329177856445, 0.12454259395599365, 0.7334533929824829]]
    T2 = [-0.00030192200210876763, 0.0006318114465102553, 718.2373657226562]

    R3 = [[-0.5166037082672119, -0.500869631767273, 0.6944424510002136],
       [-0.5941299200057983, 0.7937141060829163, 0.1304897516965866],
       [-0.6165471076965332, -0.3451775312423706, -0.7076173424720764]]
    T3 = [3.250774534535594e-05, 7.229084076243453e-06, 778.4479370117188]

    R4 = [[ 0.3140008747577667, -0.9478418231010437, 0.054767049849033356],
       [-0.761124849319458, -0.21682454645633698, 0.6112906336784363],
       [-0.567531943321228, -0.23363037407398224, -0.7895089983940125 ]]
    T4 = [-1.557081486680545e-05, 2.9063976398902014e-05, 845.6792602539062]

    # ex3
    # R1 = [[ 0.9094114303588867, 0.13049247860908508, -0.39489567279815674 ],
    #    [-0.19559234380722046, -0.7037557363510132, -0.6829873323440552],
    #    [-0.36703476309776306, 0.6983550190925598, -0.6144801378250122]]
    # T1 = [0.0021411359775811434, -0.0018981234170496464, 793.3548583984375]

    # R2 = [[-0.7278589606285095, 0.6448631882667542, 0.23318031430244446 ],
    #    [0.2961755394935608, 0.6023331880569458, -0.7412657141685486],
    #    [-0.6184672117233276, -0.4704744815826416, -0.6294063329696655]]
    # T2 = [0.0001024490047711879, -3.291070242994465e-05, 774.54052734375]

    # R3 = [[-0.037765271961688995, -0.8400328159332275, 0.5412197113037109],
    #    [-0.7759043574333191, -0.3166533410549164, -0.5456218719482422],
    #    [0.6297192573547363, -0.44054025411605835, -0.6398266553878784]]
    # T3 = [-1.8385688235866837e-05, -4.060177525389008e-05, 761.9246215820312]

    # R4 = [[ -0.8458238840103149, -0.5225585699081421, -0.10730534791946411],
    #    [-0.22730033099651337, 0.5350028872489929, -0.8136992454528809],
    #    [0.4826141893863678, -0.6638557314872742, -0.5712959170341492]]
    # T4 = [5.853496986674145e-05, 9.393743675900623e-05, 853.3223876953125]

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


def compute_intrinsic(fx, fy, cx, cy):
    new_fx = fx*(2/7)
    new_fy = fy*(2/7)
    new_cx = (cx-568/2)*(2/7)
    new_cy = (cy-174/2)*(2/7)
    K = np.array([[new_fx, 0.0, new_cx],
                [0.0, new_fy, new_cy],
                [0.0, 0.0, 1.0]])
    return K


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

    object_ds_name = 'kuartis.cad'
    urdf_ds_name = 'kuartis.cad'
    
    # coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-168790' # v3
    coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-787707' # v4 epoch 30
    coarse_epoch = 30
    n_coarse_iterations = 1

    # refiner_run_id = 'bop-tless-kuartis-refiner--243227' # v3 epoch 90
    refiner_run_id = 'bop-tless-kuartis-refiner--689761' # v4 epoch 20, 100 seems better
    refiner_epoch = 100
    n_refiner_iterations = 1

    # save_dir = RESULTS_DIR / f'{args.config}-n_views={n_views}-{args.comment}'
    # logger.info(f"SAVE DIR: {save_dir}")
    logger.info(f"Coarse: {coarse_run_id}")
    logger.info(f"Refiner: {refiner_run_id}")

    # Predictions
    bm_detector = BMDetector()
    predictor, mesh_db = load_models(coarse_run_id, refiner_run_id, coarse_epoch=coarse_epoch, refiner_epoch=refiner_epoch, 
                                    n_workers=n_workers, object_ds_name=object_ds_name, urdf_ds_name=urdf_ds_name)
    mv_predictor = MultiviewScenePredictor(mesh_db)

    K = np.array([[1905.52, 0.0, 361.142],
                [0.0, 1902.99, 288.571],
                [0.0, 0.0, 1.0]])
    # K = np.array([[1075.650, 0.0, 360],
    #             [0.0, 1073.903, 270],
    #             [0.0, 0.0, 1.0]])

    img_paths = [
        '/mnt/trains/users/azad/BM/inputs/cell_4cam/rgb_objects_no_clutter/cam_23422354_7b334636-3546-4563-a459-845445645ad5_10-02-2021_12-33-06.png', 
        '/mnt/trains/users/azad/BM/inputs/cell_4cam/rgb_objects_no_clutter/cam_23158292_48aca4b4-b3a5-47bc-8465-c9d5546c47da_10-02-2021_12-33-06.png',
        '/mnt/trains/users/azad/BM/inputs/cell_4cam/rgb_objects_no_clutter/cam_23124735_4b3cc848-5bcc-4765-9883-54d56d8694de_10-02-2021_12-33-06.png', 
        '/mnt/trains/users/azad/BM/inputs/cell_4cam/rgb_objects_no_clutter/cam_23124722_8ac6a535-4639-4a65-9766-9db454b687c8_10-02-2021_12-33-06.png',
        # '/mnt/trains/users/azad/BM/inputs/cell_4cam/rgb_objects_no_clutter/cam_23422354_7b334636-3546-4563-a459-845445645ad5_10-02-2021_12-32-42.png', 
        # '/mnt/trains/users/azad/BM/inputs/cell_4cam/rgb_objects_no_clutter/cam_23158292_48aca4b4-b3a5-47bc-8465-c9d5546c47da_10-02-2021_12-32-42.png', #top1
        # '/mnt/trains/users/azad/BM/inputs/cell_4cam/rgb_objects_no_clutter/cam_23124735_4b3cc848-5bcc-4765-9883-54d56d8694de_10-02-2021_12-32-42.png', #top2
        # '/mnt/trains/users/azad/BM/inputs/cell_4cam/rgb_objects_no_clutter/cam_23124722_8ac6a535-4639-4a65-9766-9db454b687c8_10-02-2021_12-32-42.png',
        # '/mnt/trains/users/azad/BM/inputs/cell_4cam/rgb_objects_no_clutter/cam_23422354_7b334636-3546-4563-a459-845445645ad5_10-02-2021_12-32-48.png', #dark1
        # '/mnt/trains/users/azad/BM/inputs/cell_4cam/rgb_objects_no_clutter/cam_23124722_8ac6a535-4639-4a65-9766-9db454b687c8_10-02-2021_12-32-48.png', #dark2
    ]
    # ex1
    # img_paths = [
    #     '/mnt-ssd/datasets/BM/kuatless_v3/test_pbr/000000/rgb/000004.jpg', 
    #     '/mnt-ssd/datasets/BM/kuatless_v3/test_pbr/000000/rgb/000005.jpg', 
    #     '/mnt-ssd/datasets/BM/kuatless_v3/test_pbr/000000/rgb/000008.jpg', 
    #     '/mnt-ssd/datasets/BM/kuatless_v3/test_pbr/000000/rgb/000010.jpg',
    # ]
    # ex2
    img_paths = [
        '/mnt-ssd/datasets/BM/kuatless_v3/test_pbr/000001/rgb/000974.jpg', 
        '/mnt-ssd/datasets/BM/kuatless_v3/test_pbr/000001/rgb/000975.jpg', 
        '/mnt-ssd/datasets/BM/kuatless_v3/test_pbr/000001/rgb/000976.jpg', 
        '/mnt-ssd/datasets/BM/kuatless_v3/test_pbr/000001/rgb/000977.jpg',
    ]
    # ex3
    # img_paths = [
    #     '/mnt-ssd/datasets/BM/kuatless_v3/test_pbr/000002/rgb/000060.jpg', 
    #     '/mnt-ssd/datasets/BM/kuatless_v3/test_pbr/000002/rgb/000062.jpg', 
    #     '/mnt-ssd/datasets/BM/kuatless_v3/test_pbr/000002/rgb/000063.jpg', 
    #     '/mnt-ssd/datasets/BM/kuatless_v3/test_pbr/000002/rgb/000069.jpg',
    # ]
    save_dir = '/mnt/trains/users/azad/BM/multiview_results/extrinsic/ex2'
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

    cams = np.repeat(K[np.newaxis,:,:], 4, axis=0)
    cams = torch.from_numpy(cams).cuda().float()    
    cam_info = pd.DataFrame({
        'scene_id': 1,
        'view_id': view_ids,
        'group_id': 1,
        'batch_im_id': list(range(n_views)),
        })

    TWC = None
    TWC = get_twc()
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

    # os.makedirs(save_dir, exist_ok=False)
    # torch.save(results, save_dir / 'results.pth.tar')

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
                                    'K': K, 
                                    'TWC': np.eye(4),
                                    'resolution': (540, 720),})
        fig_input_im = plotter.plot_image(input_rgb)

        # Detections
        detections = this_view_dict_preds['cand_inputs']
        bboxes = detections.initial_bboxes
        # bboxes = bboxes + torch.as_tensor(np.random.randint(30, size=((len(bboxes), 4)))).to(bboxes.dtype).to(bboxes.device)
        detections.bboxes = bboxes

        detections = add_colors_to_predictions(detections, colormap_hex)
        fig_detections = plotter.plot_image(input_rgb)
        fig_detections = plotter.plot_maskrcnn_bboxes(fig_detections, detections, colors=detections.infos['color'].tolist())

        # Candidates
        cand_inputs = this_view_dict_preds['cand_inputs']
        cand_matched = this_view_dict_preds['cand_matched']
        cand_inputs = mark_inliers(cand_inputs, cand_matched)
        # colors = np.array([(1.0, 1.0, 1.0, 1.0) for is_inlier in cand_inputs.infos['is_inlier']])
        colors = np.array([(0, 1, 0, 0.0) if is_inlier else (0, 0, 1.0, 0.0) for is_inlier in cand_inputs.infos['is_inlier']])
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
        # export_png(fig_input_im, filename = os.path.join(save_dir, 'input_im_{}.png'.format(view_id)))
        # export_png(fig_detections, filename = os.path.join(save_dir, 'detections_{}.png'.format(view_id)))
        # export_png(fig_cand, filename = os.path.join(save_dir, 'cand_{}.png'.format(view_id)))
        # export_png(fig_outputs, filename = os.path.join(save_dir, 'outputs_{}.png'.format(view_id)))
        del fig_input_im, fig_detections, fig_cand, fig_outputs

if __name__ == '__main__':
    patch_tqdm()
    main()
    time.sleep(2)
    if get_world_size() > 1:
        torch.distributed.barrier()

from PIL import Image
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import argparse
import json
from collections import OrderedDict

from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset

# Pose estimator
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse
from cosypose.training.pose_models_cfg import check_update_config as check_update_config_pose
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.integrated.icp_refiner import ICPRefiner
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor
from cosypose.datasets.wrappers.single_inference_multiview_wrapper import SingleInferenceMultiViewWrapper
from cosypose.datasets.bop import remap_bop_targets
from cosypose.evaluation.meters.pose_meters import PoseErrorMeter
from cosypose.evaluation.eval_runner.single_pose_eval import SinglePoseEvaluation

# Detection
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.training.detector_models_cfg import check_update_config as check_update_config_detector
from cosypose.integrated.detector import Detector

from cosypose.evaluation.pred_runner.bop_predictions import BopPredictionRunner

from cosypose.utils.distributed import get_tmp_dir, get_rank
from cosypose.utils.distributed import init_distributed_mode

from cosypose.config import EXP_DIR, RESULTS_DIR

#From Notebook
import torch
from cosypose.config import LOCAL_DATA_DIR
from cosypose.datasets.datasets_cfg import make_scene_dataset
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.visualization.singleview import make_singleview_prediction_plots, filter_predictions, render_prediction_wrt_camera
from cosypose.visualization.singleview import filter_predictions
from bokeh.plotting import gridplot
from bokeh.io import show, output_notebook; output_notebook()
from bokeh.io import export_png
import os
from cosypose.visualization.plotter import Plotter

#gif
import imageio
from cosypose.visualization.multiview import make_scene_renderings

from cosypose.lib3d import Transform
from bop_toolkit_lib import inout  # noqa
from cosypose.evaluation.data_utils import parse_obs_data
import cosypose.utils.tensor_collection as tc
from cosypose.evaluation.runner_utils import format_results, gather_predictions
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import pose_error

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_pose_meters(ds_name):
    compute_add = False
    spheres_overlap_check = True
    large_match_threshold_diameter_ratio = 0.5
    if ds_name == 'tless.primesense.test.bop19':
        targets_filename = 'test_targets_bop19.json'
        visib_gt_min = -1
        n_top = -1  # Given by targets
    elif ds_name == 'tless.primesense.test':
        targets_filename = 'all_target_tless.json'
        n_top = 1
        visib_gt_min = 0.1
    else:
        raise ValueError

    if 'tless' in ds_name:
        object_ds_name = 'tless.eval'
    else:
        raise ValueError

    if targets_filename is not None:
        targets_path = '/home/kuartis-dgx1/temp/cosypose/local_data/bop_datasets/tless/{}'.format(targets_filename)
        targets = pd.read_json(targets_path)
        targets = remap_bop_targets(targets)
    else:
        targets = None

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
        # For measuring ADD-S AUC on T-LESS and average errors on ycbv/tless.
        meters[f'{error_type}_ntop=BOP_matching=OVERLAP'] = PoseErrorMeter(
            error_type=error_type, consider_all_predictions=False,
            match_threshold=large_match_threshold_diameter_ratio,
            report_error_stats=True, report_error_AUC=True, **base_kwargs)

        if 'tless' in ds_name:
            meters.update({f'{error_type}_ntop=BOP_matching=BOP':  # For ADD-S<0.1d
                           PoseErrorMeter(error_type=error_type, match_threshold=0.1, **base_kwargs),

                           f'{error_type}_ntop=ALL_matching=BOP':  # For mAP
                           PoseErrorMeter(error_type=error_type, match_threshold=0.1,
                                          consider_all_predictions=True,
                                          report_AP=True, **base_kwargs)})
    return meters


def load_detector(run_id):
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model)
    return model


def load_pose_models(coarse_run_id, refiner_run_id=None, n_workers=8):
    run_dir = EXP_DIR / coarse_run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_pose(cfg)
    #object_ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_cad')
    object_ds = make_object_dataset(cfg.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=n_workers)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_pose(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        return model

    coarse_model = load_model(coarse_run_id)
    refiner_model = load_model(refiner_run_id)
    model = CoarseRefinePosePredictor(coarse_model=coarse_model,
                                      refiner_model=refiner_model)
    return model, mesh_db


def getModel(use_icp=False): 
    #load models
    # detector_run_id='detector-bop-tless-pbr--873074'
    # coarse_run_id='coarse-bop-tless-pbr--506801'
    # refiner_run_id='refiner-bop-tless-pbr--233420'
    detector_run_id='detector-bop-tless-synt+real--452847'
    # coarse_run_id='tless-coarse--86926'
    coarse_run_id='coarse-bop-tless-synt+real--160982'    
    refiner_run_id='refiner-bop-tless-synt+real--881314'
    
    detector = load_detector(detector_run_id)
    pose_predictor, mesh_db = load_pose_models(coarse_run_id=coarse_run_id, refiner_run_id=refiner_run_id)

    renderer = pose_predictor.coarse_model.renderer

    icp_refiner = None
    if use_icp:
        icp_refiner = ICPRefiner(mesh_db,
                                renderer=renderer,
                                resolution=pose_predictor.coarse_model.cfg.input_resize)
    return detector, pose_predictor, icp_refiner


def inference(detector, pose_predictor, icp_refiner, image, cameras, n_coarse_iterations, n_refiner_iterations, depth=None):
    #[1,540,720,3]->[1,3,540,720]
    images = torch.from_numpy(image).cuda().float().unsqueeze_(0)
    images = images.permute(0, 3, 1, 2) / 255
    # breakpoint()
    cameras_k = cameras.K.cuda().float()
    use_icp = icp_refiner is not None

    box_detections = detector.get_detections(images=images, one_instance_per_class=False, 
                    detection_th=0.8, output_masks=use_icp, mask_th=0.9)
    if len(box_detections) == 0:
        return None
    final_preds, all_preds=pose_predictor.get_predictions(images, cameras_k, detections=box_detections,
                            n_coarse_iterations=n_coarse_iterations, n_refiner_iterations=n_refiner_iterations)
    
    if use_icp:
        all_preds['icp'] = icp_refiner.refine_poses(final_preds, box_detections.masks, depth, cameras)

    final_preds.register_tensor('initial_bboxes', box_detections.bboxes)
    return final_preds.cpu(), all_preds, box_detections


def get_input_info(dataset_path, scene_id, view_id, load_depth=False):
    rgb_path='{}/{}/rgb/{}.png'.format(dataset_path, scene_id, view_id)
    rgb = np.array(Image.open(rgb_path))
    h, w = rgb.shape[:2]
    # rgb = np.array(Image.open(rgb_path))
    # if rgb.ndim == 2:
    #     rgb = np.repeat(rgb[..., None], 3, axis=-1)
    # rgb = rgb[..., :3]
    # h, w = rgb.shape[:2]
    # rgb = torch.as_tensor(rgb)

    json_files_directory = Path('{}/{}'.format(dataset_path, scene_id))
    scene_camera = json.loads(Path(json_files_directory / 'scene_camera.json').read_text())
    scene_gt = json.loads(Path(json_files_directory / 'scene_gt.json').read_text())
    scene_gt_info = json.loads(Path(json_files_directory / 'scene_gt_info.json').read_text())

    cam_annotation = scene_camera[str(int(view_id))]
    annotation = scene_gt[str(int(view_id))]
    visib = scene_gt_info[str(int(view_id))]

    K = np.array(cam_annotation['cam_K']).reshape(3,3)

    if 'cam_R_w2c' in cam_annotation:
        RC0 = np.array(cam_annotation['cam_R_w2c']).reshape(3, 3)
        tC0 = np.array(cam_annotation['cam_t_w2c']) * 0.001
        TC0 = Transform(RC0, tC0)
    else:
        TC0 = Transform(np.eye(3), np.zeros(3))

    T0C = TC0.inverse()
    T0C = T0C.toHomogeneousMatrix()
    camera = dict(T0C=T0C, K=K, TWC=T0C, resolution=rgb.shape[:2])
    T0C = TC0.inverse()

    objects = []
    mask = np.zeros((h, w), dtype=np.uint8)
    n_objects = len(annotation)  
    for n in range(n_objects):
        RCO = np.array(annotation[n]['cam_R_m2c']).reshape(3, 3)
        tCO = np.array(annotation[n]['cam_t_m2c']) * 0.001
        TCO = Transform(RCO, tCO)
        T0O = T0C * TCO
        T0O = T0O.toHomogeneousMatrix()
        obj_id = annotation[n]['obj_id']
        name = f'obj_{int(obj_id):06d}'
        bbox_visib = np.array(visib[n]['bbox_visib'])
        x, y, w, h = bbox_visib
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        obj = dict(label=name, name=name, TWO=T0O, T0O=T0O,
                    visib_fract=visib[n]['visib_fract'],
                    id_in_segm=n+1, bbox=[x1, y1, x2, y2])
        objects.append(obj)

    mask_path = Path('{}/{}/mask_visib/{}_all.png'.format(dataset_path, scene_id, view_id))
    if mask_path.exists():
        mask = np.array(Image.open(mask_path))
    else:
        for n in range(n_objects):
            mask_n = np.array(Image.open('{}/{}/mask_visib/{}_{:06d}.png'.format(dataset_path, scene_id, view_id, n)))
            mask[mask_n == 255] = n + 1
    mask = torch.as_tensor(mask)            
    
    if load_depth:
        depth_path = Path('{}/{}/depth/{}.png'.format(dataset_path, scene_id, view_id))
        depth = np.array(inout.load_depth(depth_path))
        depth = np.expand_dims(depth, axis=0)
        camera['depth'] = depth * cam_annotation['depth_scale'] / 1000

    obs = dict(
        objects=objects,
        camera=camera,
        frame_info={'scene_id': int(scene_id), 'view_id': int(view_id)},
    )

    cameras = tc.PandasTensorCollection(
        infos=pd.DataFrame({'scene_id': [scene_id], 'view_id': [view_id],
                            'group_id': [0], 'batch_im_id': [0]}),
        K=torch.from_numpy(K.reshape(1,3,3)),
    )
    depth_out = torch.from_numpy(depth).cuda() if 'depth' in camera.keys() else None
    # depth_out = torch.from_numpy(camera['depth']).cuda() if 'depth' in camera.keys() else None

    return rgb, mask, depth_out, obs, cameras


def collate_fn(obs):
    obj_data = []
    obj_data_ = parse_obs_data(obs)
    obj_data.append(obj_data_)
    obj_data = tc.concatenate(obj_data)
    return obj_data


def main():
    n_coarse_iterations = 1
    n_refiner_iterations = 4
    use_icp = True
    evaluation = False
    print("start...........................................")
    detector, pose_predictor, icp_refiner = getModel(use_icp=use_icp)

    dataset_path = '/raid/datasets/Bright_Machines/Bin_Picking/local_data/bop_datasets/tless/test_primesense'
    scene_id = '000001'
    view_id = '000090'
    # image_id = '000000'
    # path_ = '/home/kuartis-dgx1/temp/cosypose/images/tes/images_2/' + image_id + '.jpg'
    
    # K = np.array([[1075.65091572, 0.0, 374.06888344],
    #                 [0.0, 1073.90347929, 255.72159802],
    #                 [0.0, 0.0, 1.0]])
    img, mask, depth, obs, cameras = get_input_info(dataset_path, scene_id, view_id, load_depth=use_icp)
    obs_gt = collate_fn(obs)
    preds, all_predictions, detections = inference(detector, pose_predictor, icp_refiner, img, cameras, 
                                            n_coarse_iterations, n_refiner_iterations, depth=depth)
    all_predictions = OrderedDict({k: v for k, v in sorted(all_predictions.items(), key=lambda item: item[0])})


    models_info = inout.load_json('/raid/datasets/Bright_Machines/Bin_Picking/local_data/bop_datasets/tless/models/models_info.json', keys_to_int=True)
    
    #for vsd
    _, width, height = depth.shape  # w,h mi h,w mi?
    ren = renderer.create_renderer(width, height, 'python', mode='depth')
    ply_paths = '/raid/datasets/Bright_Machines/Bin_Picking/local_data/bop_datasets/tless/models'
    for pred in preds:
        label = pred.infos[1]
        obj_id = int(label.split('_')[1])
        ren.add_object(obj_id, '{}/{}.ply'.format(ply_paths, label))

    for pred in preds:
        R_e = pred.poses[:3, :3]
        t_e = pred.poses[:3, 3]
        label = pred.infos[1]
        obj_id = int(label.split('_')[1])
        gt_index = obs_gt.infos[obs_gt.infos.label == label].frame_obj_id.values[0]
        R_g = obs_gt[gt_index].poses[:3, :3]
        t_g = obs_gt[gt_index].poses[:3, 3]
        depth_mm = obs['camera']['depth'][0] * 1000 # convert to mm
        K = cameras.K[0]
        delta = 15
        vsd_normalized_by_diameter = True
        taus = np.arange(0.05, 0.51, 0.05)

        radius = 0.5 * models_info[obj_id]['diameter']
        sphere_projections_overlap = misc.overlapping_sphere_projections(radius, t_e.squeeze(), t_g.squeeze())

        if not sphere_projections_overlap:
            e = [1.0] * len(taus)
        else:
            e = pose_error.vsd(
                R_e, t_e, R_g, t_g, depth_mm, K, delta,
                taus, vsd_normalized_by_diameter,
                models_info[obj_id]['diameter'], ren, obj_id, 'step')
        print(e)
        breakpoint()


    # if len(preds) == 0:
    #     preds = eval_runner.make_empty_predictions()

    # preds.infos['scene_id'] = int(scene_id)
    # preds.infos['view_id'] = int(view_id)
    # preds.infos['det_id'] = np.arange(len(detections))
    # preds.infos['group_id'] = 0


    if evaluation:
        meters = get_pose_meters('tless.primesense.test')
        eval_runner = SinglePoseEvaluation(obs_gt, dataset_path, scene_id, meters, n_workers=8,
                                            cache_data=True, batch_size=1)
        predictions_to_evaluate = set()
        predictions_to_evaluate.add('icp')
        # predictions_to_evaluate.add(f'refiner/iteration={n_refiner_iterations}')

        eval_metrics, eval_dfs = dict(), dict()
        for preds_k, curr_preds in all_predictions.items():
            # if preds_k in predictions_to_evaluate:
            print(f"Evaluation : {preds_k} (N={len(preds)})")
            if len(curr_preds) == 0:
                curr_preds = eval_runner.make_empty_predictions()

            curr_preds.infos['scene_id'] = int(scene_id)
            curr_preds.infos['view_id'] = int(view_id)
            curr_preds.infos['det_id'] = np.arange(len(detections))
            curr_preds.infos['group_id'] = 0

            eval_metrics[preds_k], eval_dfs[preds_k] = eval_runner.evaluate(curr_preds, obs_gt)
            # else:
                # print(f"Skipped: {preds_k} (N={len(preds)})")
    
        metrics_to_print = dict()
        metrics_to_print.update({
            f'refiner/iteration={n_refiner_iterations}/ADD-S_ntop=BOP_matching=OVERLAP/AUC/objects/mean': f'Singleview/AUC of ADD-S',
            f'refiner/iteration={n_refiner_iterations}/ADD-S_ntop=ALL_matching=BOP/mAP': f'Singleview/mAP@ADD-S<0.1d',

            f'ba_output+all_cand/ADD-S_ntop=BOP_matching=OVERLAP/AUC/objects/mean': f'Multiview (n=1)/AUC of ADD-S',
            f'ba_output+all_cand/ADD-S_ntop=ALL_matching=BOP/mAP': f'Multiview (n=1/mAP@ADD-S<0.1d)',
        })
        metrics_to_print.update({
            f'ba_input/ADD-S_ntop=BOP_matching=OVERLAP/norm': f'Multiview before BA/ADD-S (m)',
            f'ba_output/ADD-S_ntop=BOP_matching=OVERLAP/norm': f'Multiview after BA/ADD-S (m)',
        })
        save_dir = RESULTS_DIR / 'single_image_evaluation'
        results = format_results(all_predictions, eval_metrics, eval_dfs, print_metrics=False)
        (save_dir / 'full_summary.txt').write_text(results.get('summary_txt', ''))

        full_summary = results['summary']
        summary_txt = 'Results:'
        for k, v in metrics_to_print.items():
            if k in full_summary:
                summary_txt += f"\n{v}: {full_summary[k]}"
        print(summary_txt)
        torch.save(results, save_dir / 'results.pth.tar')
        (save_dir / 'summary.txt').write_text(summary_txt)

    # urdf_ds_name = 'tless.cad'
    # # this_preds = all_predictions['icp']
    # this_preds = all_predictions['icp'] if use_icp else preds
    # renderer = BulletSceneRenderer(urdf_ds_name)

    # cam = dict(
    #     resolution=(720, 540),
    #     K=np.array(cameras.K[0]),
    #     TWC=np.eye(4)
    # )
    
    # plotter = Plotter()
    # figures = dict()

    # figures['input_im'] = plotter.plot_image(img)
    
    # fig_dets = plotter.plot_image(img)
    # fig_dets = plotter.plot_maskrcnn_bboxes(fig_dets, detections)
    # figures['detections'] = fig_dets

    # pred_rendered = render_prediction_wrt_camera(renderer, this_preds, cam)

    # figures['pred_rendered'] = plotter.plot_image(pred_rendered)
    # figures['pred_overlay'] = plotter.plot_overlay(img, pred_rendered)    

    # renderer.disconnect()

    # save_dir = 'single_view_inference_results'
    # export_png(figures['input_im'], filename = '{}/input-{}-{}.png'.format(save_dir, scene_id, view_id))
    # export_png(figures['pred_overlay'], filename = '{}/refiner4_icp.png'.format(save_dir))
    # # export_png(figures['pred_overlay'], filename = '{}/pred_overlay-{}-{}.png'.format(save_dir, scene_id, view_id))
    
    # export_png(figures['detections'], filename = '{}/detections-{}-{}.png'.format(save_dir, scene_id, view_id))
    # export_png(figures['pred_rendered'], filename = '{}/pred_rendered-{}-{}.png'.format(save_dir, scene_id, view_id))



    # fps = 25
    # duration = 10
    # n_images = fps * duration
    # n_images = 1
    # images = make_scene_renderings(this_preds, [K],
    #                             urdf_ds_name=urdf_ds_name, 
    #                             distance=1.3, object_scale=2.0,
    #                             show_cameras=False, camera_color=(0, 0, 0, 1),
    #                             theta=np.pi/4, resolution=(720, 540),
    #                             object_id_ref=0, 
    #                             colormap_rgb=None,
    #                             angles=np.linspace(0, 2*np.pi, n_images))
    # imageio.mimsave('deneme.gif', images, fps=fps)


if __name__ == '__main__':
    main()
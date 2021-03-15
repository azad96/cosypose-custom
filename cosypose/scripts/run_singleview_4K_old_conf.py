from pickle import NONE
from cosypose.scripts.run_multiview_n import compute_intrinsic
import sys
sys.path.append("/mnt/trains/users/botan/GAG/mmdetection")
from  bm_scripts.bm_inference_azad import BMDetector
import os
import torch
import numpy as np
from PIL import Image
import yaml
import time
import json
from pathlib import Path
from math import ceil 

from cosypose.config import EXP_DIR
from cosypose.datasets.datasets_cfg import make_object_dataset

# Pose estimator
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg_org import create_model_refiner, create_model_coarse, check_update_config
from cosypose.integrated.pose_predictor_org import CoarseRefinePosePredictor
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer

# From Notebook
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.visualization.singleview import render_prediction_wrt_camera
from cosypose.visualization.plotter import Plotter
from bokeh.io import export_png
from bokeh.plotting import gridplot

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_pose_models(coarse_run_id, refiner_run_id=None, coarse_epoch=None, refiner_epoch=None, n_workers=8):
    run_dir = EXP_DIR / coarse_run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config(cfg)
    object_ds = make_object_dataset(cfg.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=n_workers)
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


def inference(pose_predictor, image, camera_k, detections, coarse_it=1, refiner_it=0):
    images = torch.from_numpy(image).cuda().float().unsqueeze_(0)
    images = images.permute(0, 3, 1, 2) / 255
    cameras_k = torch.from_numpy(camera_k).cuda().float().unsqueeze_(0)

    if len(detections) == 0:
        return None

    final_preds, all_preds = pose_predictor.get_predictions(images, cameras_k, detections=detections,
                                                            n_coarse_iterations=coarse_it, n_refiner_iterations=refiner_it)
    return final_preds.cpu()


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
    urdf_ds_name = 'kuartis.cad'
    input_dim = (1080, 810)
    # coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-168790' # v3 epoch 60
    # coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-306798' # v5.3 epoch 170
    coarse_run_id = 'bop-kuatless-coarse--373078' # v6 epoch 140 cosypose pretrained
    # coarse_run_id = 'bop-kuatless-coarse--865501' # v7 epoch 120 
    coarse_epoch = 140
    n_coarse_iterations = 1

    # refiner_run_id = 'bop-tless-kuartis-refiner--243227' # v3 epoch 90
    refiner_run_id = 'bop-tless-kuartis-refiner--842437' # v5.2 epoch 180
    # refiner_run_id = None
    refiner_epoch = 180
    n_refiner_iterations = 1

    bm_detector = BMDetector()
    pose_predictor, _ = load_pose_models(coarse_run_id=coarse_run_id, refiner_run_id=refiner_run_id,
                                        coarse_epoch=coarse_epoch, refiner_epoch=refiner_epoch)
    renderer = BulletSceneRenderer(urdf_ds_name)
    plotter = Plotter()

    input_folders = [
         'cell_4cam/workingplane_1080_810',
         'cell_4cam/objects_no_clutter_1080_810',
         'cell_4cam/objects_1080_810',
         'bracket_mix/object_mix3_1080_810',
         'bracket_mix/object_mix2_1080_810',
         'bracket_mix/object_mix1_1080_810',
         'bracket/object_1080_810', 
    ]
    total_time = 0.0
    image_count = 0
    start_time = time.time()
    for folder_name in input_folders:
        folder_pth = '/mnt/trains/users/azad/BM/inputs/{}'.format(folder_name)
        save_dir = '/mnt/trains/users/azad/BM/results/v6_with_segmentation/{}'.format(folder_name)
        seg_save_dir = '{}/segmentation_masks'.format(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(seg_save_dir, exist_ok=True)

        file_names = os.listdir(folder_pth)
        img_names = [file_name for file_name in file_names if file_name.endswith('.png') or file_name.endswith('.jpg')]
        img_paths = [os.path.join(folder_pth, img_name) for img_name in img_names]
        image_count += len(img_names)
        
        calib_path = '/'.join(folder_pth.split('/')[:-1])
        calibration = json.loads(Path('{}/calibration.json'.format(calib_path)).read_text())

        for i, (img_name, img_path) in enumerate(zip(img_names, img_paths)):
            img = Image.open(img_path) 
            img = np.array(img)

            cam_coeff = get_cam_coeff(img_name, calibration)
            K = compute_intrinsic(cam_coeff, input_dim)
            cam = dict(
                resolution=input_dim,
                K=K,
                TWC=np.eye(4)
                )

            t0 = time.time()
            detections, segmentation = bm_detector.get_detection(img_path)

            img_seg = Image.fromarray(segmentation)
            img_seg.save('{}/{}'.format(seg_save_dir, img_name))
            # conds = [bbox[1] != 0 and ceil(bbox[2]) != input_dim[0] and 
            #          bbox[0] != 0 and ceil(bbox[3]) != input_dim[1] for bbox in detections.bboxes]
            # conds = [cond.item() if type(cond) == torch.Tensor else cond for cond in conds]
            # conds = [(bbox[1] == 0 and ceil(bbox[2]) == input_dim[0]) or 
            #         (bbox[0] == 0 and ceil(bbox[3]) == input_dim[1]) 
            #         for bbox in detections.bboxes]
            # conds = [not cond.item() if type(cond) == torch.Tensor else not cond for cond in conds ]
            # detections = detections[conds]
            t1 = time.time()
            pred = inference(pose_predictor, img, K, detections, n_coarse_iterations, n_refiner_iterations)
            t2 = time.time()
            pred_rendered = render_prediction_wrt_camera(renderer, pred, cam)
            t3 = time.time()
            total_time += (t2-t1)

            figures = dict()
            figures['input_im'] = plotter.plot_image(img)
            img_seg = plotter.plot_segm_overlay(img, segmentation)
            figures['detections'] = plotter.plot_maskrcnn_bboxes(img_seg, detections)

            binary_mask = pred_rendered[:,:,0] > 0
            pred_rendered_binary = (binary_mask * 255).astype(np.uint8)
            
            figures['pred_rendered'] = plotter.plot_image(pred_rendered)
            figures['pred_overlay'] = plotter.plot_overlay(img, pred_rendered)           
            figures['pred_overlay'] = plotter.plot_confidence_scores(figures['pred_overlay'], detections, pred_rendered_binary, segmentation)
            fig_array = [figures['input_im'], figures['detections'], figures['pred_rendered'], figures['pred_overlay']]
            res = gridplot(fig_array, ncols=2)
            export_png(res, filename = os.path.join(save_dir, img_name))
            print("{}/{} - Detection: {:.3} s Pose: {:.3} s Rendering: {:.3} s".format(i+1, len(img_names), t1-t0, t2-t1, t3-t2))

    end_time = time.time()
    print('Average pose inference time: {} ms'.format(1000*total_time/image_count))
    print('Total inference time {} minutes'.format((end_time-start_time)/60))
    renderer.disconnect()

if __name__ == '__main__':
    main()

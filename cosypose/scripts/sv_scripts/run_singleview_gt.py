import sys
sys.path.append("/mnt/trains/users/botan/GAG/mmdetection")
from  bm_scripts.bm_inference_azad import BMDetector
from PIL import Image
import numpy as np
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import argparse
import json
import pandas as pd
import cosypose.utils.tensor_collection as tc

from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset

# Pose estimator
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse
from cosypose.training.pose_models_cfg import check_update_config as check_update_config_pose
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.integrated.icp_refiner import ICPRefiner
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper

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
import time

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_pose_models(coarse_run_id, refiner_run_id=None, coarse_epoch=None, refiner_epoch=None, n_workers=8):
    run_dir = EXP_DIR / coarse_run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_pose(cfg)
    object_ds = make_object_dataset(cfg.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=n_workers)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id, epoch):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_pose(cfg)
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

def getModel(coarse_run_id, refiner_run_id, coarse_epoch, refiner_epoch): 
    pose_predictor, mesh_db = load_pose_models(coarse_run_id=coarse_run_id, refiner_run_id=refiner_run_id,
                                                coarse_epoch=coarse_epoch, refiner_epoch=refiner_epoch, n_workers=4)

    renderer = pose_predictor.coarse_model.renderer
    icp_refiner = ICPRefiner(mesh_db,
                            renderer=renderer,
                            resolution=pose_predictor.coarse_model.cfg.input_resize)
    return pose_predictor, icp_refiner


def inference(pose_predictor, icp_refiner, image, camera_k, detections = None, coarse_it=1, refiner_it=0):
    #[1,540,720,3]->[1,3,540,720]
    images = torch.from_numpy(image).cuda().float().unsqueeze_(0)
    images = images.permute(0, 3, 1, 2) / 255
    cameras_k = torch.from_numpy(camera_k).cuda().float().unsqueeze_(0)
    box_detections = detections

    if len(box_detections) == 0:
        return None

    final_preds, all_preds = pose_predictor.get_predictions(images, cameras_k, detections=box_detections,
                        n_coarse_iterations=coarse_it, n_refiner_iterations=refiner_it)
    return final_preds.cpu(), box_detections


def main():
    # coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-34030'
    coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-168790'
    # coarse_run_id = 'bop-tless-kuartis-coarse-transnoise-zxyavg-787707'
    coarse_epoch = None
    # coarse_epoch = 30
    n_coarse_iterations = 1

    # refiner_run_id = 'bop-tless-kuartis-refiner--636300'
    # refiner_run_id = 'bop-tless-kuartis-refiner--607469'
    # refiner_run_id = 'bop-tless-kuartis-refiner--243227'
    refiner_run_id = None
    refiner_epoch = None
    n_refiner_iterations = 0

    pose_predictor, icp_refiner = getModel(coarse_run_id, refiner_run_id, coarse_epoch, refiner_epoch)

    urdf_ds_name = 'kuatless.cad'
    renderer = BulletSceneRenderer(urdf_ds_name)
    
    K = np.array([[1905.52, 0.0, 361.142],
                    [0.0, 1902.99, 288.571],
                    [0.0, 0.0, 1.0]])
    cam = dict(
        resolution=(720, 540),
        K=K,
        TWC=np.eye(4)
        )

    folder_name = '000004'
    folder_pth = '/mnt-ssd/datasets/BM/kuatless/test_pbr/{}/rgb'.format(folder_name)
    save_dir = '/mnt/trains/users/azad/BM/results_gt/{}'.format(folder_name)
    model_type = 'coarse-v3'

    file_names = os.listdir(folder_pth)
    img_names = [file_name for file_name in file_names if file_name.endswith('.png') or file_name.endswith('.jpg')]
    img_paths = [os.path.join(folder_pth, img_name) for img_name in img_names]

    gt_folder = '/'.join(folder_pth.split('/')[:-1])
    gt_bbox_path = '{}/scene_gt_info.json'.format(gt_folder)
    bbox_json = json.loads(Path(gt_bbox_path).read_text())
    gt_obj_path = '{}/scene_gt.json'.format(gt_folder)
    obj_json = json.loads(Path(gt_obj_path).read_text())

    for i, (img_name, img_path) in enumerate(zip(img_names, img_paths)):
        img = Image.open(img_path) 
        img = np.array(img)

        t0 = time.time()
        #detections
        label = []
        bboxes = []
        img_id = str(int(img_name.split('.')[0]))
        bbox_infos = bbox_json[img_id]
        obj_infos = obj_json[img_id]

        for obj_it in range(len(bbox_infos)):
            lbl = 'obj_{:06}'.format(obj_json[img_id][obj_it]['obj_id'])
            label.append(lbl)
            bbox = bbox_json[img_id][obj_it]['bbox_obj']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bboxes.append(bbox)

        bboxes=torch.as_tensor(np.stack(bboxes)).float().cuda()
        infos = dict(batch_im_id=0,
                    score=1.0,
                    label=label)
        detections = tc.PandasTensorCollection(infos=pd.DataFrame(infos), bboxes=bboxes)
        t1 = time.time()
        pred, detections = inference(pose_predictor, icp_refiner, img, K, detections, n_coarse_iterations, n_refiner_iterations)
        t2 = time.time()

        this_preds = pred             
        plotter = Plotter()
        figures = dict()

        figures['input_im'] = plotter.plot_image(img)
        
        fig_dets = plotter.plot_image(img)
        fig_dets = plotter.plot_maskrcnn_bboxes(fig_dets, detections)
        figures['detections'] = fig_dets
        t3 = time.time()
        pred_rendered = render_prediction_wrt_camera(renderer, this_preds, cam)
        t4 = time.time()
        print("{}/{} {} - Detection: {:.3f} s Pose: {:.3f} s Rendering: {:.3f} s".format(i+1, len(img_names), img_id, t1-t0, t2-t1, t4-t3))
        
        figures['pred_rendered'] = plotter.plot_image(pred_rendered)
        figures['pred_overlay'] = plotter.plot_overlay(img, pred_rendered)           
        
        save_dir2 = os.path.join(save_dir, img_name.split('.')[0])
        os.makedirs(save_dir2, exist_ok=True)

        input_image_path = Path(os.path.join(save_dir2, 'input_im.png'))
        if not input_image_path.exists():
            export_png(figures['input_im'], filename = input_image_path)

        export_png(figures['pred_rendered'], filename = os.path.join(save_dir2, '{}-rendered.png'.format(model_type)))
        export_png(figures['pred_overlay'], filename = os.path.join(save_dir2, '{}-overlay.png'.format(model_type)))
        export_png(figures['detections'], filename = os.path.join(save_dir2, 'detections.png'.format(model_type)))

    renderer.disconnect()

if __name__ == '__main__':
    main()
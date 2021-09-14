import sys
sys.path.append("/mnt/trains/users/azad/mmdetection")
from bm_scripts.bm_inference_azad import BMDetector
import os
import torch
import numpy as np
from PIL import Image
import time

# From Notebook
from cosypose.visualization.plotter import Plotter
from bokeh.io import export_png
from bokeh.plotting import gridplot

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    bm_detector = BMDetector()
    plotter = Plotter()

    input_folders = [
        '000000',
        # 'double_spur_gear/1080_810'
        # 'cell_4cam/workingplane_1080_810',
        # 'cell_4cam/objects_no_clutter_1080_810',
        # 'cell_4cam/objects_1080_810',
        # 'bracket_mix/object_mix3_1080_810',
        # 'bracket_mix/object_mix2_1080_810',
        # 'bracket_mix/object_mix1_720_540',
        #  'bracket/object_1080_810', 
        # '1080_810/channel_bracket_A',
        # '1080_810/screw_terminal',
    ]

    for folder_name in input_folders:
        # folder_pth = '/mnt/trains/users/azad/BM/inputs/{}'.format(folder_name)
        folder_pth = '/mnt-ssd/datasets/BM/bm3/test_pbr_1080_810/{}/rgb'.format(folder_name)
        save_dir = '/mnt/trains/users/azad/BM/results/dsg_detection/{}'.format(folder_name)
        os.makedirs(save_dir, exist_ok=True)
        print(folder_pth)
        file_names = os.listdir(folder_pth)
        img_names = [file_name for file_name in file_names if file_name.endswith('.png') or file_name.endswith('.jpg')]
        # img_names = ['cam_23422358_4ac75c58-aa59-463c-97b8-9b8d9ba48cac_15-03-2021_14-23-00.png']
        img_names = sorted(img_names)[396:397]
        # img_names = sorted(img_names)[::100]
        img_paths = [os.path.join(folder_pth, img_name) for img_name in img_names]
        
        for i, (img_name, img_path) in enumerate(zip(img_names, img_paths)):
            img = Image.open(img_path) 
            img = np.array(img)
            
            t0 = time.time()
            detections, segmentation = bm_detector.get_detection(img_path)
            # for j,det in enumerate(detections):
            #     print('{} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {}'.format(j, det.bboxes[0], det.bboxes[1], det.bboxes[2], det.bboxes[3], 
            #                                                             det.infos[2],
            #                                                             int(det.infos[1].split('_')[1])))
            t1 = time.time()
            figures = dict()
            figures['input_im'] = plotter.plot_image(img)
            img_seg = plotter.plot_segm_overlay(img, segmentation)
            figures['detections'] = plotter.plot_maskrcnn_bboxes(img_seg, detections)

            fig_array = [figures['detections']]
            res = gridplot(fig_array, ncols=1)
            export_png(res, filename = os.path.join(save_dir, img_name.replace('.jpg', '.png')))
            print("{}/{} - Detection: {:.3} s ".format(i+1, len(img_names), t1-t0))


if __name__ == '__main__':
    main()

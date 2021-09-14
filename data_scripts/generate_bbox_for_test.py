import os
from os.path import join as osp
import json
import numpy as np
import pickle

pth = "/mnt-ssd/datasets/BM/bm3/test_pbr_1080_810"
# pth = "/raid/datasets/Bright_Machines/Bin_Picking/kuatless/test_pbr_1080_810"
# pth = "/mnt/trains/users/azad/BM/docker/local_data/bop_datasets/kuatless/test_pbr_1080_810"
# save_pth = "/mnt/trains/users/azad/BM/docker/local_data/saved_detections/kuatless_test_1080_810.pkl"
save_pth = "/mnt/trains/users/azad/BM/cosypose_azd/local_data/saved_detections/bm3_test_1080_810.pkl"
scene_ids = sorted(os.listdir(pth))
img_id = 0

out_dict = dict()
for scene_id in scene_ids:
    scene_gt_info_pth = osp(pth, scene_id, "scene_gt_info.json")
    scene_gt_pth = osp(pth, scene_id, "scene_gt.json")

    f = open(scene_gt_info_pth ,"r")
    scene_gt_info = json.load(f)
    f.close() 
    f = open(scene_gt_pth ,"r")
    scene_gt = json.load(f)
    f.close() 

    for view_id in scene_gt_info.keys():
        out_dict["{:04}/{:06}".format(int(scene_id), int(view_id))] = dict()
        current_scene_gt_info = scene_gt_info["{}".format(int(view_id))]
        bboxes = []
        scores = []
        poses = []
        labels = []
        dummy_pose = np.array([[1.0, 0., 0., 0.], [0., 1.0, 0., 0.], [0., 0., 1.0, 0.], [0., 0., 0., 1.0]])
        for i ,obj in enumerate(current_scene_gt_info):
            current_gt = scene_gt[view_id][i]
            current_obj_id = current_gt['obj_id']

            # current_bbox = obj["bbox_obj"]
            current_bbox = obj["bbox_visib"]
            x,y,w,h = current_bbox
            if x == -1 and y == -1 and w == -1 and h == -1:
                continue
            # if x < 0 or y < 0 or w < 0 or h < 0:
            #     continue
            #bboxes.append([x, y, x + w, y + h])
            bboxes.append([y, x, y + h, x + w])
            scores.append(1.0)
            poses.append(dummy_pose)
            labels.append("obj_{:06}".format(current_obj_id))
        scores = np.array(scores)
        bboxes = np.array(bboxes)
        out_dict["{:04}/{:06}".format(int(scene_id), int(view_id))]['scores'] = scores
        out_dict["{:04}/{:06}".format(int(scene_id), int(view_id))]['labels_txt'] = labels
        out_dict["{:04}/{:06}".format(int(scene_id), int(view_id))]['poses'] = poses
        out_dict["{:04}/{:06}".format(int(scene_id), int(view_id))]['rois'] = bboxes

f = open(save_pth, "wb")
pickle.dump(out_dict, f)


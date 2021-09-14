import os
from os.path import join as osp
import json



pth = "/mnt-ssd/datasets/BM/bm3/test_pbr_1080_810"
save_pth = "/mnt-ssd/datasets/BM/bm3/bm3_test_1080_810.json"
# pth = "/raid/datasets/Bright_Machines/Bin_Picking/kuatless/test_pbr_1080_810"
# save_pth = "/raid/datasets/Bright_Machines/Bin_Picking/kuatless/kuatless_test_1080_810.json"
# pth = "/mnt/trains/users/azad/BM/docker/local_data/bop_datasets/kuatless/test_pbr_1080_810"
# save_pth = "/mnt/trains/users/azad/BM/docker/local_data/bop_datasets/kuatless/kuatless_test_1080_810.json"
scene_ids = sorted(os.listdir(pth))
img_id = 0

out_dict = dict()
save_list = []
total_entry = 0
for scene_id in scene_ids:
    out_dict["{}".format(int(scene_id))] = dict()

    scene_gt_pth = osp(pth, scene_id, "scene_gt.json")
    f = open(scene_gt_pth ,"r")
    scene_gt = json.load(f)
    f.close() 
    
    for view_id in scene_gt.keys(): #[0, 8, 12, 16]:#scene_gt.keys():
        current_scene_gt = scene_gt["{}".format(int(view_id))]
        out_dict["{}".format(int(scene_id))]["{}".format(int(view_id))] = dict()
        for obj in current_scene_gt:
            obj_id = str(obj["obj_id"])
            if not obj_id in out_dict["{}".format(int(scene_id))]["{}".format(int(view_id))].keys():
                out_dict["{}".format(int(scene_id))]["{}".format(int(view_id))]["{}".format(obj_id)] = 1
                total_entry += 1
            else:
                out_dict["{}".format(int(scene_id))]["{}".format(int(view_id))]["{}".format(obj_id)] += 1
        for k, v in out_dict["{}".format(int(scene_id))]["{}".format(int(view_id))].items():
            save_list.append([int(scene_id), int(view_id), int(k), int(v)])
target_list = [] 
for i, obj in enumerate(save_list):
    dummy = dict()
    dummy["im_id"] = obj[1]
    dummy["inst_count"] = obj[3]
    dummy["obj_id"] = obj[2]
    dummy["scene_id"] = obj[0]
    target_list.append(dummy)

f = open(save_pth, "w")
json.dump(target_list, f)
f.close()

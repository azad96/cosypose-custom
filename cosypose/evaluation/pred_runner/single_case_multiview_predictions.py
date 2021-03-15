import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict

from cosypose.utils.logging import get_logger
from cosypose.datasets.samplers import DistributedSceneSampler
import cosypose.utils.tensor_collection as tc

from torch.utils.data import DataLoader

logger = get_logger(__name__)


class SingleCaseMultiviewPredictionRunner:
    def __init__(self):
        pass

    def get_predictions(self, pose_predictor, mv_predictor,
                        images, cameras, use_known_camera_poses=False,
                        detections=None,
                        n_coarse_iterations=1, n_refiner_iterations=1,
                        sv_score_th=0.0, skip_mv=True,
                        use_detections_TCO=False):
        assert detections is not None
        if detections is not None:
            mask = (detections.infos['score'] >= sv_score_th)
            detections = detections[np.where(mask)[0]]
            detections.infos['det_id'] = np.arange(len(detections))
            det_index = detections.infos.set_index(['scene_id', 'view_id']).sort_index()
        
        predictions = defaultdict(list)
        
        scene_id = np.unique(detections.infos['scene_id'])
        view_ids = np.unique(detections.infos['view_id'])
        group_id = np.unique(detections.infos['group_id'])
        n_gt_dets = len(detections)

        if detections is not None:
            keep_ids, batch_im_ids = [], []
            for group_name, group in cameras.infos.groupby(['scene_id', 'view_id']):
                if group_name in det_index.index:
                    other_group = det_index.loc[group_name]
                    keep_ids_ = other_group['det_id']
                    batch_im_id = np.unique(group['batch_im_id']).item()
                    batch_im_ids.append(np.ones(len(keep_ids_)) * batch_im_id)
                    keep_ids.append(keep_ids_)
            if len(keep_ids) > 0:
                keep_ids = np.concatenate(keep_ids)
                batch_im_ids = np.concatenate(batch_im_ids)
            detections_ = detections[keep_ids]
            detections_.infos['batch_im_id'] = np.array(batch_im_ids).astype(np.int)
        else:
            raise ValueError('No detections')
        detections_ = detections_.cuda().float()
        detections_.infos['group_id'] = group_id.item()
        sv_preds, mv_preds = dict(), dict()
        if len(detections_) > 0:
            data_TCO_init = detections_ if use_detections_TCO else None
            detections__ = detections_ if not use_detections_TCO else None
            candidates, sv_preds = pose_predictor.get_predictions(
                images, cameras.K, detections=detections__,
                n_coarse_iterations=n_coarse_iterations,
                data_TCO_init=data_TCO_init,
                n_refiner_iterations=n_refiner_iterations,
            )
            # for i in range(3):
            #     sv_preds['coarse/iteration=1'].poses[0][i][0] += 0.5
            #     candidates.poses[0][i][0] += 0.5
            # sv_preds['coarse/iteration=1'].poses[1][0][0] += 0.5
            # candidates.poses[1][0][0] += 0.5
            # sv_preds['coarse/iteration=1'].poses[2][0][0] += 0.5
            # candidates.poses[2][0][0] += 0.5

            candidates.register_tensor('initial_bboxes', detections_.bboxes)
            if not skip_mv:
                mv_preds = mv_predictor.predict_scene_state(candidates, cameras, 
                                                        use_known_camera_poses=use_known_camera_poses)
        logger.debug(f"{'-'*80}")

        for k, v in sv_preds.items():
            predictions[k].append(v.cpu())

        for k, v in mv_preds.items():
            predictions[k].append(v.cpu())

        predictions = dict(predictions)
        for k, v in predictions.items():
            predictions[k] = tc.concatenate(v)
        return predictions

from tqdm import tqdm
import numpy as np
import torch
import pandas as pd

from collections import OrderedDict

from torch.utils.data import DataLoader

from cosypose.utils.distributed import get_world_size, get_rank, get_tmp_dir

import cosypose.utils.tensor_collection as tc
from cosypose.evaluation.data_utils import parse_obs_data
from cosypose.datasets.samplers import DistributedSceneSampler


class SinglePoseEvaluation:
    def __init__(self, obs, dataset_path, scene_id, meters, batch_size=1, cache_data=True, n_workers=4, sampler=None):

        self.rank = get_rank()
        self.tmp_dir = get_tmp_dir()

        self.obs = obs
        self.dataset_path = dataset_path
        self.scene_id = scene_id

        self.meters = meters
        self.meters = OrderedDict({k: v for k, v in sorted(self.meters.items(), key=lambda item: item[0])})


    @staticmethod
    def make_empty_predictions():
        infos = dict(view_id=np.empty(0, dtype=np.int),
                     scene_id=np.empty(0, dtype=np.int),
                     label=np.empty(0, dtype=np.object),
                     score=np.empty(0, dtype=np.float))
        poses = torch.empty(0, 4, 4, dtype=torch.float)
        return tc.PandasTensorCollection(infos=pd.DataFrame(infos), poses=poses)

    def evaluate(self, obj_predictions, obj_data_gts, device='cuda'):
        for meter in self.meters.values():
            meter.reset()
        obj_predictions = obj_predictions.to(device)
        for k, meter in self.meters.items():
            meter.add(obj_predictions, obj_data_gts.to(device))
        return self.summary()

    def summary(self):
        summary, dfs = dict(), dict()
        for meter_k, meter in sorted(self.meters.items()):
            meter.gather_distributed(tmp_dir=self.tmp_dir)
            if get_rank() == 0 and len(meter.datas) > 0:
                summary_, df_ = meter.summary()
                dfs[meter_k] = df_
                for k, v in summary_.items():
                    summary[meter_k + '/' + k] = v
        return summary, dfs

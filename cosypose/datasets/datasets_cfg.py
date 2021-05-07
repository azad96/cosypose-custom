import numpy as np
import pandas as pd

from cosypose.config import LOCAL_DATA_DIR, ASSET_DIR, BOP_DS_DIR
from cosypose.utils.logging import get_logger

from .bop_object_datasets import BOPObjectDataset
from .bop import BOPDataset, remap_bop_targets
from .urdf_dataset import BOPUrdfDataset, OneUrdfDataset
from .texture_dataset import TextureDataset


logger = get_logger(__name__)


def keep_bop19(ds):
    targets = pd.read_json(ds.ds_dir / 'test_targets_bop19.json')
    targets = remap_bop_targets(targets)
    targets = targets.loc[:, ['scene_id', 'view_id']].drop_duplicates()
    index = ds.frame_index.merge(targets, on=['scene_id', 'view_id']).reset_index(drop=True)
    assert len(index) == len(targets)
    ds.frame_index = index
    return ds


def make_scene_dataset(ds_name, n_frames=None):
    # BOP challenge
    ds_folder_name, split_name = ds_name.split('.') # kuatless.train_pbr, kuatless.test_pbr_1080_810
    ds_dir = BOP_DS_DIR / ds_folder_name
    ds = BOPDataset(ds_dir, split=split_name)

    if n_frames is not None:
        ds.frame_index = ds.frame_index.iloc[:n_frames].reset_index(drop=True)
    ds.name = ds_name
    return ds


def make_object_dataset(ds_name):
    ds = None
    ds_folder_name, model_type = ds_name.split('.') # kuatless.cad/kuatless.eval
    if model_type == 'cad':
        ds = BOPObjectDataset(BOP_DS_DIR / ds_folder_name / 'models')
    elif model_type == 'eval':
        ds = BOPObjectDataset(BOP_DS_DIR / ds_folder_name / 'models_eval')
    else:
        raise ValueError(ds_name)
    return ds


def make_urdf_dataset(ds_name):
    if isinstance(ds_name, list):
        ds_index = []
        for ds_name_n in ds_name:
            dataset = make_urdf_dataset(ds_name_n)
            ds_index.append(dataset.index)
        dataset.index = pd.concat(ds_index, axis=0)
        return dataset

    ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / ds_name)
    return ds


def make_texture_dataset(ds_name):
    if ds_name == 'shapenet':
        ds = TextureDataset(LOCAL_DATA_DIR / 'texture_datasets' / 'shapenet')
    else:
        raise ValueError(ds_name)
    return ds

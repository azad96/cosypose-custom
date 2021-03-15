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


def keep_kuartis(ds):
    targets = pd.read_json(ds.ds_dir / 'kuartis_pbr_vivo_target.json')
    targets = remap_bop_targets(targets)
    targets = targets.loc[:, ['scene_id', 'view_id']].drop_duplicates()
    index = ds.frame_index.merge(targets, on=['scene_id', 'view_id']).reset_index(drop=True)
    assert len(index) == len(targets)
    ds.frame_index = index
    return ds


def make_scene_dataset(ds_name, n_frames=None):
    # BOP challenge
    folder_name, split_name = ds_name.split('.') # kuatless.train_pbr, kuatless.test_pbr_1080_810
    ds_dir = BOP_DS_DIR / folder_name
    ds = BOPDataset(ds_dir, split=split_name)

    # if ds_name == 'kuatless.train_pbr':
    #     ds_dir = BOP_DS_DIR / 'kuatless'
    #     ds = BOPDataset(ds_dir, split='train_pbr')

    # elif ds_name == 'kuatless.test_pbr_high_res':
    #     ds_dir = BOP_DS_DIR / 'kuatless'
    #     ds = BOPDataset(ds_dir, split='test_pbr_high_res')

    # elif ds_name == 'kuatless.test_pbr_low_res':
    #     ds_dir = BOP_DS_DIR / 'kuatless'
    #     ds = BOPDataset(ds_dir, split='test_pbr_low_res')

    # else:
    #     raise ValueError(ds_name)

    if n_frames is not None:
        ds.frame_index = ds.frame_index.iloc[:n_frames].reset_index(drop=True)
    ds.name = ds_name
    return ds


def make_object_dataset(ds_name):
    ds = None
    if ds_name == 'tless.cad':
        ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models')
    elif ds_name == 'tless.eval' or ds_name == 'tless.bop':
        ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_eval')
    elif ds_name == 'kuartis.cad':
        ds = BOPObjectDataset(BOP_DS_DIR / 'kuatless/models')
    elif ds_name == 'kuartis.eval':
        ds = BOPObjectDataset(BOP_DS_DIR / 'kuatless/models_eval')
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

    # BOP
    if ds_name == 'tless.cad':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'tless.cad')
    elif ds_name == 'kuartis.cad':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'kuartis.cad')
    elif ds_name == 'tless.reconst':
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'urdfs' / 'tless.reconst')

    # Custom scenario
    elif 'custom' in ds_name:
        scenario = ds_name.split('.')[1]
        ds = BOPUrdfDataset(LOCAL_DATA_DIR / 'scenarios' / scenario / 'urdfs')

    elif ds_name == 'camera':
        ds = OneUrdfDataset(ASSET_DIR / 'camera/model.urdf', 'camera')
    else:
        raise ValueError(ds_name)
    return ds


def make_texture_dataset(ds_name):
    if ds_name == 'shapenet':
        ds = TextureDataset(LOCAL_DATA_DIR / 'texture_datasets' / 'shapenet')
    else:
        raise ValueError(ds_name)
    return ds

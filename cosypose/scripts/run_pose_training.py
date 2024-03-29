import argparse
import numpy as np
import os
from colorama import Fore, Style

from cosypose.training.train_pose import train_pose
from cosypose.utils.logging import get_logger
from cosypose.bop_config import BOP_CONFIG
logger = get_logger(__name__)


def make_cfg(args):
    cfg = argparse.ArgumentParser('').parse_args([])
    if args.config:
        logger.info(f"{Fore.GREEN}Training with config: {args.config} {Style.RESET_ALL}")

    cfg.resume_run_id = None
    if len(args.resume) > 0:
        cfg.resume_run_id = args.resume
        logger.info(f"{Fore.RED}Resuming {cfg.resume_run_id} {Style.RESET_ALL}")

    N_CPUS = int(os.environ.get('N_CPUS', 10))
    N_WORKERS = min(N_CPUS - 2, 8)
    N_WORKERS = 8
    N_RAND = np.random.randint(1e6)

    # Data
    cfg.n_symmetries_batch = 64
    cfg.val_epoch_interval = 10
    cfg.test_epoch_interval = 10
    cfg.n_test_frames = None

    cfg.rgb_augmentation = True
    cfg.background_augmentation = False
    cfg.gray_augmentation = False

    # Model
    cfg.backbone_str = 'efficientnet-b3'
    cfg.n_pose_dims = 9
    cfg.n_rendering_workers = 0
    cfg.refiner_run_id_for_test = None
    cfg.refiner_test_epoch = None
    cfg.coarse_run_id_for_test = None
    cfg.coarse_test_epoch = None
    cfg.run_id_pretrain = 'coarse-bop-tless-pbr--506801' 
    # cfg.run_id_pretrain = 'bop-kuatless-coarse-noise-874436' 
    # cfg.run_id_pretrain = 'bop-kuatless-coarse-noise2-824870' 
    # cfg.run_id_pretrain = 'bop-kuatless-coarse-noise3-481715' 
    # cfg.run_id_pretrain = 'bop-kuatless-coarse-noise-132k-582997' 
    # cfg.run_id_pretrain = 'bop-kuatless-coarse-v5.3' # epoch 160
    # cfg.run_id_pretrain = 'bop-kuatless-coarse-v6' # epoch 140
    # cfg.run_id_pretrain = 'bop-kuatless-refiner-v5.2' # v5.2 epoch 180
    cfg.pretrain_epoch = None
    cfg.use_cosypose_model = True

    # Optimizer
    cfg.lr = 3e-5 # adam default: 3e-4, sgd: 3e-2
    cfg.weight_decay = 0.
    cfg.n_epochs_warmup = 10 # 50
    cfg.lr_epoch_decay = 250 # 250
    cfg.clip_grad_norm = 0.5

    # Training
    cfg.batch_size = 48
    cfg.epoch_size = 115200
    cfg.n_epochs = 400          # 700
    cfg.n_dataloader_workers = N_WORKERS

    # Method
    cfg.loss_disentangled = True
    cfg.n_points_loss = 2600
    cfg.TCO_input_generator = 'fixed'
    cfg.n_iterations = 1
    cfg.min_area = None

    if 'bop-' in args.config:
        _, bop_name, model_type = args.config.split('-') # bop-kuatless-coarse
        bop_cfg = BOP_CONFIG[bop_name]

        cfg.train_ds_names = [(bop_cfg['train_pbr_ds_name'][0], 1)]
        cfg.test_ds_names = bop_cfg['test_pbr_ds_name']
        cfg.val_ds_names = cfg.train_ds_names
        cfg.urdf_ds_name = bop_cfg['urdf_ds_name']
        cfg.object_ds_name = bop_cfg['obj_ds_name']
        cfg.input_resize = bop_cfg['input_resize']
        cfg.render_size = bop_cfg['render_size']

        if model_type == 'coarse':
            cfg.init_method = 'z-up+auto-depth'
            cfg.TCO_input_generator = 'fixed+trans_noise'
        elif model_type == 'refiner':
            cfg.TCO_input_generator = 'gt+noise'
        else:
            raise ValueError
    else:
        raise ValueError(args.config)

    if args.no_eval:
        cfg.test_ds_names = []

    cfg.run_id = f'{args.config}-{args.version}-{N_RAND}'

    if args.debug:
        cfg.test_ds_names = []
        cfg.n_epochs = 4
        cfg.val_epoch_interval = 1
        cfg.test_epoch_interval = 1
        cfg.batch_size = 4
        cfg.epoch_size = 4 * cfg.batch_size
        cfg.run_id = 'debug-' + cfg.run_id
        cfg.background_augmentation = False
        cfg.n_dataloader_workers = 8
        cfg.n_rendering_workers = 0
        cfg.n_test_frames = 10

    N_GPUS = int(os.environ.get('N_PROCS', 1))
    cfg.epoch_size = cfg.epoch_size // N_GPUS
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', default='bop-kuatless-coarse', type=str, required=True)
    parser.add_argument('--version', default='v1', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-eval', action='store_true')
    parser.add_argument('--resume', default='', type=str)
    args = parser.parse_args()

    cfg = make_cfg(args)
    train_pose(cfg)
"""5-epoch per-channel QAT of the NuScenes SECOND 3D backbone on v1.0-mini.

Only VoxelResBackBone8x is fake-quantized. The run starts from the official
full-precision SECOND-MultiHead checkpoint and evaluates mini_val with the
same fake-quant path used in training.
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

if not hasattr(np, 'float'):
    np.float = float

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tools'))

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def load_cfg():
    cfg_dir = HERE / 'cfgs'
    prev_cwd = Path.cwd()
    os.chdir(cfg_dir)
    try:
        cfg_from_yaml_file('cbgs_second_multihead_hw_qat.yaml', cfg)
    finally:
        os.chdir(prev_cwd)
    cfg.DATA_CONFIG.DATA_PATH = str(REPO_ROOT / 'data' / 'nuscenes')
    cfg.DATA_CONFIG.VERSION = 'v1.0-mini'
    cfg.DATA_CONFIG.BALANCED_RESAMPLING = False
    cfg.TAG = 'cbgs_second_multihead_hw_qat'
    cfg.EXP_GROUP_PATH = 'nuscenes_models'
    return cfg


def get_backbone(model):
    return model.backbone_3d


def main():
    parser = argparse.ArgumentParser(description='NuScenes SECOND residual backbone QAT on mini')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument(
        '--pretrained_model',
        type=str,
        default=str(HERE / 'cbgs_second_multihead_nds6229_updated.pth'),
    )
    parser.add_argument('--output_dir', type=str, default=str(HERE / 'output' / 'qat_5ep'))
    parser.add_argument('--weight_quant', choices=['per_channel', 'per_tensor'], default='per_channel')
    parser.add_argument('--extra_tag', type=str, default='mini_qat_5ep')
    parser.add_argument('--eval_ckpt', type=str, default=None,
                        help='Skip training and evaluate this QAT checkpoint with fake-quant.')
    args = parser.parse_args()

    model_cfg = load_cfg()
    if args.batch_size is None:
        args.batch_size = model_cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    args.epochs = args.epochs if args.epochs is not None else model_cfg.OPTIMIZATION.NUM_EPOCHS

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)
    logger.info('DATA_PATH: %s' % model_cfg.DATA_CONFIG.DATA_PATH)
    logger.info('VERSION: %s' % model_cfg.DATA_CONFIG.VERSION)
    logger.info('pretrained: %s' % args.pretrained_model)
    logger.info('epochs: %d  batch_size: %d  lr: %s  weight_quant: %s' % (
        args.epochs, args.batch_size, model_cfg.OPTIMIZATION.LR, args.weight_quant
    ))
    log_config_to_file(model_cfg, logger=logger)

    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=model_cfg.DATA_CONFIG,
        class_names=model_cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=True,
        seed=666,
    )
    logger.info('train samples: %d  iters/epoch: %d' % (len(train_set), len(train_loader)))

    model = build_network(
        model_cfg=model_cfg.MODEL,
        num_class=len(model_cfg.CLASS_NAMES),
        dataset=train_set,
    )
    model.cuda()
    backbone = get_backbone(model)
    backbone.enable_hw_qat(enable=True, weight_quant=args.weight_quant, observer='max', fake_quant=True)
    logger.info('quantized sparse convs: %d' % backbone.num_quant_convs())

    if args.eval_ckpt is None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=False, logger=logger)
        model.train()
        optimizer = build_optimizer(model, model_cfg.OPTIMIZATION)
        lr_scheduler, lr_warmup_scheduler = build_scheduler(
            optimizer,
            total_iters_each_epoch=len(train_loader),
            total_epochs=args.epochs,
            last_epoch=-1,
            optim_cfg=model_cfg.OPTIMIZATION,
        )

        train_model(
            model,
            optimizer,
            train_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=lr_scheduler,
            optim_cfg=model_cfg.OPTIMIZATION,
            start_epoch=0,
            total_epochs=args.epochs,
            start_iter=0,
            rank=0,
            tb_log=None,
            ckpt_save_dir=ckpt_dir,
            train_sampler=train_sampler,
            lr_warmup_scheduler=lr_warmup_scheduler,
            ckpt_save_interval=1,
            max_ckpt_save_num=args.epochs,
            merge_all_iters_to_one_epoch=False,
            logger=logger,
            logger_iter_interval=20,
            ckpt_save_time_interval=600,
            use_logger_to_record=True,
            show_gpu_stat=False,
            use_amp=False,
            cfg=model_cfg,
        )
    else:
        model.load_params_from_file(filename=args.eval_ckpt, to_cpu=False, logger=logger)
        logger.info('skip training, evaluate %s' % args.eval_ckpt)

    logger.info('**********************Start fake-quant evaluation**********************')
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=model_cfg.DATA_CONFIG,
        class_names=model_cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    logger.info('val samples: %d' % len(test_set))
    eval_output_dir = output_dir / 'eval' / ('epoch_%d' % args.epochs) / 'val'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    backbone.enable_hw_eval_quant(True)
    eval_args = argparse.Namespace(save_to_file=False, infer_time=False)
    metrics = eval_utils.eval_one_epoch(
        model_cfg, eval_args, model, test_loader, args.epochs, logger,
        dist_test=False, result_dir=eval_output_dir,
    )
    result_path = output_dir / 'qat_mini_metrics.json'
    serializable = {key: float(val) for key, val in metrics.items() if isinstance(val, (int, float))}
    serializable['train_samples'] = len(train_set)
    serializable['val_samples'] = len(test_set)
    serializable['epochs'] = args.epochs
    serializable['weight_quant'] = args.weight_quant
    with open(result_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.info('metrics saved to %s' % result_path)
    logger.info('mAP: %.4f  NDS: %.4f' % (serializable.get('mAP', float('nan')), serializable.get('NDS', float('nan'))))


if __name__ == '__main__':
    main()

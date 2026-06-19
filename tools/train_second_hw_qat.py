import _init_path
import argparse
import csv
import datetime
import glob
import json
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='SECOND VoxelBackBone8x hardware-QAT training')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second_hw_qat.yaml')
    parser.add_argument('--batch_size', type=int, default=None, required=False)
    parser.add_argument('--epochs', type=int, default=None, required=False)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--extra_tag', type=str, default='hw_qat')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888)
    parser.add_argument('--sync_bn', action='store_true', default=False)
    parser.add_argument('--fix_random_seed', action='store_true', default=False)
    parser.add_argument('--ckpt_save_interval', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=None)
    parser.add_argument('--max_ckpt_save_num', type=int, default=30)
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)

    parser.add_argument('--max_waiting_mins', type=int, default=0)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epochs_to_eval', type=int, default=0)
    parser.add_argument('--save_to_file', action='store_true', default=False)
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False)
    parser.add_argument('--logger_iter_interval', type=int, default=50)
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300)
    parser.add_argument('--wo_gpu_stat', action='store_true')
    parser.add_argument('--use_amp', action='store_true')

    parser.add_argument('--hw_qat', action='store_true', default=True,
                        help='Enable hardware-constrained QAT on backbone_3d.')
    parser.add_argument('--no_hw_qat', action='store_false', dest='hw_qat',
                        help='Build the HWQAT backbone without fake quant during training.')
    parser.add_argument('--activation_quant', choices=['signed_symmetric'], default='signed_symmetric')
    parser.add_argument('--weight_quant', choices=['per_channel', 'per_tensor'], default='per_channel')
    parser.add_argument('--observer', choices=['max', 'momentum'], default='max')
    parser.add_argument('--observer_momentum', type=float, default=0.95)
    parser.add_argument('--freeze_observer_epoch', type=int, default=-1,
                        help='Reserved metadata field; shared OpenPCDet train loop has no epoch hook here.')
    parser.add_argument('--export_qparams', action='store_true', default=False)

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def get_backbone(model):
    return model.module.backbone_3d if hasattr(model, 'module') else model.backbone_3d


def enable_hw_qat(model, args, logger):
    backbone = get_backbone(model)
    if not hasattr(backbone, 'enable_hw_qat'):
        raise RuntimeError('Configured backbone does not support HW-QAT: %s' % type(backbone).__name__)

    backbone.enable_hw_qat(
        enable=args.hw_qat,
        weight_quant=args.weight_quant,
        observer=args.observer,
        observer_momentum=args.observer_momentum,
        fake_quant=args.hw_qat,
    )
    logger.info('HW-QAT backbone setup:')
    logger.info('  enabled: %s' % args.hw_qat)
    logger.info('  activation_quant: %s' % args.activation_quant)
    logger.info('  weight_quant: %s' % args.weight_quant)
    logger.info('  observer: %s' % args.observer)
    logger.info('  observer_momentum: %.6f' % args.observer_momentum)
    logger.info('  freeze_observer_epoch metadata: %s' % args.freeze_observer_epoch)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_train_qparams(model, output_dir, args, logger):
    backbone = get_backbone(model)
    if not hasattr(backbone, 'get_activation_qparams'):
        logger.warning('Backbone has no qparam export helpers, skipping.')
        return

    export_dir = output_dir / 'hw_qparams'
    export_dir.mkdir(parents=True, exist_ok=True)

    write_csv(export_dir / 'layer_inventory.csv', backbone.get_layer_inventory())
    write_csv(export_dir / 'activation_scales.csv', backbone.get_activation_qparams())
    write_csv(export_dir / 'weight_scales.csv', backbone.get_weight_qparams())

    metadata = {
        'cfg_file': args.cfg_file,
        'extra_tag': args.extra_tag,
        'activation_quant': args.activation_quant,
        'weight_quant': args.weight_quant,
        'observer': args.observer,
        'observer_momentum': args.observer_momentum,
        'freeze_observer_epoch': args.freeze_observer_epoch,
        'scope': 'MODEL.BACKBONE_3D only',
        'zero_point': 0,
        'qmin': -127,
        'qmax': 127,
    }
    with open(export_dir / 'qparam_metadata.json', 'w') as json_file:
        json.dump(metadata, json_file, indent=2)
    logger.info('Exported HW-QAT training qparams to %s' % export_dir)


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start HW-QAT logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    logger.info('Training in %s mode' % ('distributed' if dist_train else 'single-process'))
    for key, val in vars(args).items():
        logger.info('{:24} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info('----------- Create dataloader & network & optimizer -----------')
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except Exception:
                    ckpt_list = ckpt_list[:-1]

    model.train()
    enable_hw_qat(model, args, logger)

    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info('----------- Model %s created, param count: %d -----------'
                % (cfg.MODEL.NAME, sum([m.numel() for m in model.parameters()])))
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    logger.info('**********************Start HW-QAT training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record,
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg
    )

    if args.export_qparams and cfg.LOCAL_RANK == 0:
        export_train_qparams(model, output_dir, args, logger)

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End HW-QAT training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()

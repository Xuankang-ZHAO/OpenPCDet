import _init_path
import argparse
import csv
import datetime
import math
import os
import re
import struct
from pathlib import Path

import numpy as np
import torch

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


RTL_SCHEDULE = [
    (4, 16), (16, 16), (16, 32), (32, 32), (32, 32), (32, 64),
    (64, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 128),
]


def parse_config():
    parser = argparse.ArgumentParser(description='SECOND HW-QAT check/export/eval')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second_hw_qat.yaml')
    parser.add_argument('--batch_size', type=int, default=None, required=False)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--extra_tag', type=str, default='hw_qat')
    parser.add_argument('--ckpt', type=str, default=None, required=True)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888)
    parser.add_argument('--local_rank', type=int, default=None)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--save_to_file', action='store_true', default=False)
    parser.add_argument('--infer_time', action='store_true', default=False)
    parser.add_argument('--eval_tag', type=str, default='hw_qat')

    parser.add_argument('--calibrate', action='store_true', default=False)
    parser.add_argument('--calib_batches', type=int, default=8)
    parser.add_argument('--check_export', action='store_true', default=False)
    parser.add_argument('--eval_hw_ref', action='store_true', default=False)
    parser.add_argument('--export_dir', type=str, default=None)
    parser.add_argument('--weight_quant', choices=['per_channel', 'per_tensor'], default='per_channel')
    parser.add_argument('--observer', choices=['max', 'momentum'], default='max')
    parser.add_argument('--observer_momentum', type=float, default=0.95)
    parser.add_argument('--bias_bits', type=int, default=32)
    parser.add_argument('--shift_bits', type=int, default=5)
    parser.add_argument('--max_shift_rel_error', type=float, default=1.0)
    parser.add_argument('--emit_binary', action='store_true', default=False)

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def get_backbone(model):
    return model.module.backbone_3d if hasattr(model, 'module') else model.backbone_3d


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_report(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as report_file:
        report_file.write('\n'.join(lines))
        report_file.write('\n')


def signed_bits_needed(values):
    values = [int(v) for v in values]
    if not values:
        return 1
    min_val, max_val = min(values), max(values)
    for bits in range(1, 65):
        if min_val >= -(1 << (bits - 1)) and max_val <= (1 << (bits - 1)) - 1:
            return bits
    return 65


def signed_range(bits):
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def output_channel_axis(conv, weight):
    if weight.shape[0] == conv.out_channels:
        return 0
    if weight.shape[-1] == conv.out_channels:
        return weight.dim() - 1
    return 0


def view_channel_vector(conv, weight, vector):
    if vector.dim() == 0:
        return vector
    axis = output_channel_axis(conv, weight)
    shape = [1] * weight.dim()
    shape[axis] = vector.numel()
    return vector.view(*shape)


def per_channel_or_tensor_scale(conv, weight, mode):
    qmax = 127.0
    if mode == 'per_tensor':
        return (weight.detach().abs().amax() / qmax).clamp_min(1e-8)
    axis = output_channel_axis(conv, weight)
    reduce_dims = [idx for idx in range(weight.dim()) if idx != axis]
    return (weight.detach().abs().amax(dim=reduce_dims) / qmax).clamp_min(1e-8)


def fold_bn(conv, bn):
    weight = conv.weight.detach().float().cpu()
    gamma = bn.weight.detach().float().cpu()
    beta = bn.bias.detach().float().cpu()
    mean = bn.running_mean.detach().float().cpu()
    var = bn.running_var.detach().float().cpu()
    bn_scale = gamma / torch.sqrt(var + bn.eps)
    folded_weight = weight * view_channel_vector(conv, weight, bn_scale)
    folded_bias = beta - mean * bn_scale
    return folded_weight, folded_bias


def quantize_weight(conv, folded_weight, mode):
    scale = per_channel_or_tensor_scale(conv, folded_weight, mode).cpu()
    q = (folded_weight / view_channel_vector(conv, folded_weight, scale)).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def normalize_scale(scale, cout):
    if isinstance(scale, torch.Tensor):
        if scale.dim() == 0:
            return [float(scale)] * cout
        return [float(v) for v in scale.view(-1)]
    return [float(scale)] * cout


def kernel_tuple(conv):
    return tuple(int(v) for v in conv.kernel_size)


def stride_tuple(conv):
    return tuple(int(v) for v in conv.stride)


def run_calibration(model, dataloader, args, logger):
    backbone = get_backbone(model)
    if not hasattr(backbone, 'enable_calibration'):
        raise RuntimeError('Backbone does not support HW-QAT calibration: %s' % type(backbone).__name__)

    logger.info('Start HW-QAT activation calibration for %d batches' % args.calib_batches)
    backbone.reset_activation_observers()
    backbone.enable_calibration(True, observer=args.observer, observer_momentum=args.observer_momentum)
    backbone.enable_hw_qat(False, weight_quant=args.weight_quant, observer=args.observer,
                           observer_momentum=args.observer_momentum, fake_quant=False)
    model.eval()

    for batch_idx, batch_dict in enumerate(dataloader):
        if batch_idx >= args.calib_batches:
            break
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            model(batch_dict)
        logger.info('  calibrated batch %d/%d' % (batch_idx + 1, args.calib_batches))

    backbone.enable_calibration(False)
    logger.info('Finished HW-QAT activation calibration')


def export_hw_payload(backbone, export_dir, args, logger):
    if not hasattr(backbone, 'get_layer_modules'):
        raise RuntimeError('Backbone has no HW-QAT layer export helpers.')

    export_dir.mkdir(parents=True, exist_ok=True)
    backbone.enable_hw_qat(False, weight_quant=args.weight_quant, observer=args.observer,
                           observer_momentum=args.observer_momentum, fake_quant=False)

    act_rows = backbone.get_activation_qparams()
    act_scale = {row['name']: float(row['scale']) for row in act_rows}
    act_seen = {row['name']: bool(row['seen']) for row in act_rows}

    inventory_rows = []
    folded_bias_rows = []
    weight_scale_rows = []
    bias_stat_rows = []
    requant_rows = []
    layer_qparams = []
    hard_failures = []
    warnings = []
    bias_min_allowed, bias_max_allowed = signed_range(args.bias_bits)
    max_shift = (1 << args.shift_bits) - 1

    layer_modules = backbone.get_layer_modules()
    for layer_id, (name, conv, bn, relu, act_key) in enumerate(layer_modules):
        rtl_cin, rtl_cout = RTL_SCHEDULE[layer_id]
        layer_ok = conv.in_channels == rtl_cin and conv.out_channels == rtl_cout
        if not layer_ok:
            hard_failures.append('Layer %d channel mismatch: got %d->%d expected %d->%d'
                                 % (layer_id, conv.in_channels, conv.out_channels, rtl_cin, rtl_cout))
        if bn is None:
            hard_failures.append('Layer %d missing BN for folding' % layer_id)

        input_key = 'input' if layer_id == 0 else layer_modules[layer_id - 1][4]
        sx = act_scale.get(input_key, 1.0)
        sy = act_scale.get(act_key, 1.0)
        if not act_seen.get(input_key, False):
            warnings.append('Activation scale %s was not calibrated; using %.6e' % (input_key, sx))
        if not act_seen.get(act_key, False):
            warnings.append('Activation scale %s was not calibrated; using %.6e' % (act_key, sy))

        folded_weight, folded_bias = fold_bn(conv, bn)
        weight_int8, weight_scale = quantize_weight(conv, folded_weight, args.weight_quant)
        sw_values = normalize_scale(weight_scale, conv.out_channels)

        inventory_rows.append({
            'layer_id': layer_id,
            'module_name': name,
            'conv_type': type(conv).__name__,
            'kernel': kernel_tuple(conv),
            'stride': stride_tuple(conv),
            'cin': int(conv.in_channels),
            'cout': int(conv.out_channels),
            'has_bn': True,
            'has_relu': relu is not None,
            'has_residual': False,
            'rtl_match': layer_ok,
        })

        for channel, bias_fp32 in enumerate(folded_bias.tolist()):
            folded_bias_rows.append({
                'layer_id': layer_id,
                'module_name': name,
                'channel': channel,
                'folded_bias_fp32': bias_fp32,
            })

        for channel, sw in enumerate(sw_values):
            weight_scale_rows.append({
                'layer_id': layer_id,
                'module_name': name,
                'channel': channel if args.weight_quant == 'per_channel' else -1,
                'weight_scale': sw,
                'zero_point': 0,
                'qmin': int(weight_int8.min()),
                'qmax': int(weight_int8.max()),
            })
            if args.weight_quant == 'per_tensor':
                break

        layer_bias_int = []
        layer_shifts = []
        params_rows = []
        for channel, bias_fp32 in enumerate(folded_bias.tolist()):
            sw = sw_values[channel]
            denom = max(sx * sw, 1e-12)
            bias_int = int(round(bias_fp32 / denom))
            real_multiplier = max(sx * sw / max(sy, 1e-12), 1e-12)
            nearest_shift = int(round(-math.log(real_multiplier, 2)))
            nearest_shift = max(0, nearest_shift)
            pow2_multiplier = 2.0 ** (-nearest_shift)
            rel_error = abs(real_multiplier - pow2_multiplier) / max(real_multiplier, 1e-12)

            layer_bias_int.append(bias_int)
            layer_shifts.append(nearest_shift)
            if bias_int < bias_min_allowed or bias_int > bias_max_allowed:
                hard_failures.append('Layer %d channel %d bias_int=%d exceeds signed %d-bit'
                                     % (layer_id, channel, bias_int, args.bias_bits))
            if nearest_shift > max_shift:
                hard_failures.append('Layer %d channel %d shift=%d exceeds %d-bit max %d'
                                     % (layer_id, channel, nearest_shift, args.shift_bits, max_shift))
            if rel_error > args.max_shift_rel_error:
                warnings.append('Layer %d channel %d shift-only rel_error %.6f exceeds %.6f'
                                % (layer_id, channel, rel_error, args.max_shift_rel_error))

            row = {
                'layer_id': layer_id,
                'module_name': name,
                'cout_channel': channel,
                'bias_int': bias_int,
                'bias_bits_needed': signed_bits_needed([bias_int]),
                'shift': nearest_shift,
                'sx': sx,
                'sw': sw,
                'sy': sy,
                'real_multiplier': real_multiplier,
                'pow2_multiplier': pow2_multiplier,
                'relative_error': rel_error,
                'relu_en': relu is not None,
                'rounding_en': True,
            }
            requant_rows.append(row)
            params_rows.append(row)

        layer_qparams.append({
            'layer_id': layer_id,
            'module_name': name,
            'weight_int8': weight_int8,
            'bias_int': torch.tensor(layer_bias_int, dtype=torch.int32),
            'shift': torch.tensor(layer_shifts, dtype=torch.int32),
            'sx': sx,
            'sy': sy,
            'relu_en': relu is not None,
        })

        bias_stat_rows.append({
            'layer_id': layer_id,
            'module_name': name,
            'cout': int(conv.out_channels),
            'min': min(layer_bias_int),
            'max': max(layer_bias_int),
            'max_abs': max(abs(v) for v in layer_bias_int),
            'signed_bits_needed': signed_bits_needed(layer_bias_int),
            'p99_abs': float(np.percentile(np.abs(layer_bias_int), 99)),
            'p999_abs': float(np.percentile(np.abs(layer_bias_int), 99.9)),
            'outlier_channels': ';'.join(str(idx) for idx, val in enumerate(layer_bias_int)
                                         if val < bias_min_allowed or val > bias_max_allowed),
            'shift_min': min(layer_shifts),
            'shift_max': max(layer_shifts),
        })

        write_csv(export_dir / ('params_layer%02d.csv' % layer_id), params_rows)
        write_csv(export_dir / ('weights_layer%02d_meta.csv' % layer_id), [{
            'layer_id': layer_id,
            'module_name': name,
            'shape': tuple(int(v) for v in weight_int8.shape),
            'dtype': 'int8',
            'weight_quant': args.weight_quant,
            'qmin': int(weight_int8.min()),
            'qmax': int(weight_int8.max()),
        }])

        if args.emit_binary:
            weight_int8.numpy().tofile(export_dir / ('weights_layer%02d.bin' % layer_id))
            with open(export_dir / ('params_layer%02d.bin' % layer_id), 'wb') as bin_file:
                for row in params_rows:
                    bin_file.write(struct.pack('<ib', int(row['bias_int']), int(row['shift'])))

    write_csv(export_dir / 'layer_inventory.csv', inventory_rows)
    write_csv(export_dir / 'activation_scales.csv', act_rows)
    write_csv(export_dir / 'weight_scales.csv', weight_scale_rows)
    write_csv(export_dir / 'folded_bias_fp32.csv', folded_bias_rows)
    write_csv(export_dir / 'bias_int_stats.csv', bias_stat_rows)
    write_csv(export_dir / 'requant_shift_stats.csv', requant_rows)

    report_lines = [
        '# SECOND HW-QAT Quant Scheme Report',
        '',
        '- Scope: VoxelBackBone8x only',
        '- Activation: signed symmetric INT8, zero_point=0',
        '- Weight: signed symmetric INT8, zero_point=0, %s' % args.weight_quant,
        '- BN: folded offline into sparse conv weight/bias for export',
        '- Accumulator: INT32',
        '- Requant: shift-only nearest power-of-two',
        '- Bias bits checked: %d' % args.bias_bits,
        '- Shift bits checked: %d' % args.shift_bits,
        '- Hard failures: %d' % len(hard_failures),
        '- Warnings: %d' % len(warnings),
        '',
        '## Hard Failures',
    ]
    report_lines.extend(['- ' + item for item in hard_failures] or ['- None'])
    report_lines.append('')
    report_lines.append('## Warnings')
    report_lines.extend(['- ' + item for item in warnings] or ['- None'])
    append_report(export_dir / 'quant_scheme_report.md', report_lines)

    logger.info('Exported HW-QAT payload/check reports to %s' % export_dir)
    logger.info('Hard failures: %d, warnings: %d' % (len(hard_failures), len(warnings)))
    for item in hard_failures:
        logger.error(item)
    for item in warnings[:20]:
        logger.warning(item)
    if len(warnings) > 20:
        logger.warning('... %d additional warnings omitted from log; see quant_scheme_report.md' % (len(warnings) - 20))

    if args.check_export and hard_failures:
        raise RuntimeError('HW-QAT export checks failed; see %s' % (export_dir / 'quant_scheme_report.md'))

    return {
        'hard_failures': hard_failures,
        'warnings': warnings,
        'export_dir': export_dir,
        'qparams': layer_qparams,
    }


def eval_hw_reference(model, test_loader, args, eval_output_dir, export_dir, qparams, logger):
    backbone = get_backbone(model)
    if not hasattr(backbone, 'enable_hw_reference'):
        raise RuntimeError('Backbone does not support HW reference mode.')
    backbone.enable_hw_qat(False, weight_quant=args.weight_quant, observer=args.observer,
                           observer_momentum=args.observer_momentum, fake_quant=False)
    backbone.enable_hw_reference(True, qparams=qparams)
    logger.info('Enabled HW-equivalent integer bias/shift reference path for evaluation')
    tb_dict = eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, 'hw_ref', logger, dist_test=False,
        result_dir=eval_output_dir / 'hw_ref'
    )
    report_lines = [
        '# SECOND HW-QAT HW Reference Evaluation',
        '',
        'This report records OpenPCDet metrics from the HW-equivalent integer reference path.',
        'The 3D backbone reference uses int8-valued sparse features/weights, integer bias,',
        'per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.',
        'Exact bias/shift checks are in requant_shift_stats.csv.',
        '',
        '```text',
        str(tb_dict),
        '```',
    ]
    append_report(eval_output_dir / 'shift_only_accuracy_report.md', report_lines)
    append_report(export_dir / 'shift_only_accuracy_report.md', report_lines)
    backbone.enable_hw_reference(False)


def main():
    args, cfg = parse_config()
    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher != 'none':
        raise NotImplementedError('test_second_hw_qat.py currently supports --launcher none only')

    dist_test = False
    total_gpus = 1
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = output_dir / 'eval' / ('epoch_%s' % (re.findall(r'\d+', args.ckpt)[-1] if re.findall(r'\d+', args.ckpt) else 'no_number')) / cfg.DATA_CONFIG.DATA_SPLIT['test'] / args.eval_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    export_dir = Path(args.export_dir) if args.export_dir is not None else output_dir / 'hw_export'

    log_file = eval_output_dir / ('log_hw_qat_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    logger.info('**********************Start SECOND HW-QAT check/export**********************')
    for key, val in vars(args).items():
        logger.info('{:24} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test,
                                pre_trained_path=args.pretrained_model)
    model.cuda()

    if args.calibrate:
        run_calibration(model, test_loader, args, logger)

    backbone = get_backbone(model)
    export_result = export_hw_payload(backbone, export_dir, args, logger)

    if args.eval_hw_ref:
        eval_hw_reference(model, test_loader, args, eval_output_dir, export_dir, export_result['qparams'], logger)

    logger.info('**********************SECOND HW-QAT check/export done**********************')
    logger.info('Export directory: %s' % export_result['export_dir'])


if __name__ == '__main__':
    main()

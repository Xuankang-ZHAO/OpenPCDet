#!/usr/bin/python3
"""Profile per-layer GPU latency of SECOND VoxelBackBone8x HW-QAT INT8 inference.

Uses CUDA events around each HW-reference backbone layer. Important caveats:

1. Values are INT8, but spconv CUDA kernels still execute in FP32 on this path.
2. Latency is GPU-kernel wall time for that layer (including quant/requant bookkeeping).
3. Warmup + repeated timed runs are used to reduce first-call / cache effects.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mycode.kitti_frame_loader import (
    add_data_mode_args,
    build_kitti_dataset,
    build_template_dataset,
    choose_kitti_frame_ids,
    choose_raw_frame_paths,
    load_kitti_sample,
    load_raw_sample,
    resolve_data_mode,
    resolve_project_root,
)


LAYER_KEEP_KEYS = {
    1: 'x_conv1',
    4: 'x_conv2',
    7: 'x_conv3',
    10: 'x_conv4',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Per-layer GPU latency for SECOND VoxelBackBone8x INT8 HW-reference inference'
    )
    parser.add_argument('--cfg', type=str, default='tools/cfgs/kitti_models/second_hw_qat.yaml')
    parser.add_argument(
        '--ckpt',
        type=str,
        default='output/kitti_models/second_hw_qat/hw_qat_10ep/ckpt/checkpoint_epoch_10.pth',
    )
    parser.add_argument('--velodyne_dir', type=str, default='data/kitti/training/velodyne')
    add_data_mode_args(parser)
    parser.add_argument('--frame_id', type=str, default='000216')
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--out_dir', type=str, default='mycode/output/second_backbone_int8_layer_latency')
    parser.add_argument('--weight_quant', choices=['per_channel', 'per_tensor'], default='per_channel')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=50)
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == 'auto':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for latency profiling')
        return torch.device('cuda:0')
    device = torch.device(device_arg)
    if device.type != 'cuda':
        raise RuntimeError('This profiler must run on CUDA, got %s' % device)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    return device


def move_batch_to_device(batch_dict, device):
    skip_keys = {'frame_id', 'metadata', 'calib', 'image_paths', 'ori_shape', 'img_process_infos'}
    output = {}
    for key, value in batch_dict.items():
        if key in skip_keys:
            output[key] = value
            continue
        if isinstance(value, np.ndarray):
            if key in {'image_shape', 'voxel_coords'}:
                output[key] = torch.from_numpy(value).int().to(device)
            else:
                output[key] = torch.from_numpy(value).float().to(device)
        else:
            output[key] = value
    output['batch_size'] = batch_dict.get('batch_size', 1)
    return output


def describe_gpu(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        'index': int(index),
        'name': props.name,
        'total_memory_bytes': int(props.total_memory),
        'major': int(props.major),
        'minor': int(props.minor),
        'multi_processor_count': int(props.multi_processor_count),
    }


def enable_hw_reference(model, project_root, args, logger):
    tools_dir = project_root / 'tools'
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import test_second_hw_qat as hw_qat_tools

    backbone = model.backbone_3d
    export_dir = project_root / args.out_dir / 'hw_export_for_latency'
    export_args = SimpleNamespace(
        weight_quant=args.weight_quant,
        observer='max',
        observer_momentum=0.95,
        bias_bits=32,
        shift_bits=5,
        max_shift_rel_error=1.0,
        emit_binary=False,
        check_export=False,
    )
    export_result = hw_qat_tools.export_hw_payload(backbone, export_dir, export_args, logger)
    backbone.enable_hw_qat(
        False, weight_quant=args.weight_quant, observer='max',
        observer_momentum=0.95, fake_quant=False,
    )
    backbone.enable_hw_reference(True, qparams=export_result['qparams'])
    return export_result


def layer_static_info(backbone):
    rows = []
    for layer_id, (name, conv, _bn, _relu, act_key) in enumerate(backbone.get_layer_modules()):
        rows.append({
            'layer_id': layer_id,
            'module_name': name,
            'conv_type': type(conv).__name__,
            'activation_key': act_key,
            'indice_key': getattr(conv, 'indice_key', ''),
            'kernel': [int(v) for v in conv.kernel_size],
            'stride': [int(v) for v in conv.stride],
            'cin': int(conv.in_channels),
            'cout': int(conv.out_channels),
            'kept_as': LAYER_KEEP_KEYS.get(layer_id),
        })
    return rows


def summarize_ms(samples):
    samples = [float(x) for x in samples]
    if not samples:
        return {
            'mean_ms': 0.0,
            'std_ms': 0.0,
            'min_ms': 0.0,
            'max_ms': 0.0,
            'p50_ms': 0.0,
            'p90_ms': 0.0,
            'p99_ms': 0.0,
        }
    ordered = sorted(samples)
    n = len(ordered)

    def percentile(p):
        if n == 1:
            return ordered[0]
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return ordered[idx]

    return {
        'mean_ms': float(statistics.fmean(samples)),
        'std_ms': float(statistics.stdev(samples)) if n > 1 else 0.0,
        'min_ms': float(ordered[0]),
        'max_ms': float(ordered[-1]),
        'p50_ms': float(percentile(50)),
        'p90_ms': float(percentile(90)),
        'p99_ms': float(percentile(99)),
    }


def profile_backbone_layers_once(backbone, batch_torch, collect_shapes=False):
    static_rows = {row['layer_id']: row for row in layer_static_info(backbone)}
    orig_layer = backbone._forward_hw_reference_layer
    records = []

    def wrapped(sparse_tensor, layer_id, layer_info):
        static = static_rows[layer_id]
        nin = int(sparse_tensor.features.shape[0])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = orig_layer(sparse_tensor, layer_id, layer_info)
        end.record()
        end.synchronize()
        nout = int(output.features.shape[0])
        record = {
            **static,
            'latency_ms': float(start.elapsed_time(end)),
            'num_input_voxels': nin,
            'num_output_voxels': nout,
        }
        if collect_shapes:
            record['input_spatial_shape_zyx'] = [int(v) for v in list(sparse_tensor.spatial_shape)]
            record['output_spatial_shape_zyx'] = [int(v) for v in list(output.spatial_shape)]
        records.append(record)
        return output

    backbone._forward_hw_reference_layer = wrapped
    try:
        with torch.no_grad():
            _ = backbone(batch_torch)
    finally:
        backbone._forward_hw_reference_layer = orig_layer
    return records


def run_vfe(model, batch_torch):
    with torch.no_grad():
        return model.vfe(batch_torch)


def analyze_frame(model, batch_torch, frame_info, warmup, repeat):
    backbone = model.backbone_3d
    vfe_batch = run_vfe(model, dict(batch_torch))

    for _ in range(max(int(warmup), 0)):
        with torch.no_grad():
            _ = backbone(dict(vfe_batch))
    torch.cuda.synchronize()

    timed_runs = []
    for idx in range(max(int(repeat), 1)):
        records = profile_backbone_layers_once(
            backbone, dict(vfe_batch), collect_shapes=(idx == 0),
        )
        timed_runs.append(records)

    layer_samples = {}
    for run in timed_runs:
        for record in run:
            layer_samples.setdefault(record['layer_id'], []).append(record['latency_ms'])

    summary_records = []
    total_samples = []
    for layer_id in sorted(layer_samples.keys()):
        base = dict(timed_runs[0][layer_id])
        stats = summarize_ms(layer_samples[layer_id])
        base.update(stats)
        base['samples_ms'] = [float(x) for x in layer_samples[layer_id]]
        summary_records.append(base)

    for run in timed_runs:
        total_samples.append(sum(rec['latency_ms'] for rec in run))
    backbone_stats = summarize_ms(total_samples)

    return {
        'frame_id': frame_info['frame_id'],
        'point_cloud_path': frame_info['point_cloud_path'],
        'data_loader': frame_info['data_loader'],
        'fov_points_only': frame_info['fov_points_only'],
        'num_voxels_after_vfe': int(vfe_batch['voxel_features'].shape[0]),
        'warmup': int(warmup),
        'repeat': int(repeat),
        'backbone_total_ms': backbone_stats,
        'records': summary_records,
    }


def format_records(frame_id, gpu_name, payload):
    total = payload['backbone_total_ms']
    lines = [
        'Frame: %s' % frame_id,
        'GPU: %s' % gpu_name,
        'Mode: HW-reference INT8 values in FP32 spconv kernels',
        'Warmup/Repeat: %d / %d' % (payload['warmup'], payload['repeat']),
        'Backbone total mean: %.3f ms  (p50=%.3f, p90=%.3f, std=%.3f)' % (
            total['mean_ms'], total['p50_ms'], total['p90_ms'], total['std_ms'],
        ),
        '',
        '{:<4} {:<16} {:<13} {:>8} {:>4} {:>8} {:>4} {:>9} {:>9} {:>9} {:>9} {:>8}'.format(
            'Id', 'Name', 'Type', 'Nin', 'Cin', 'Nout', 'Cout',
            'MeanMs', 'P50Ms', 'P90Ms', 'StdMs', 'Share%',
        ),
        '-' * 118,
    ]
    total_mean = max(total['mean_ms'], 1e-12)
    for record in payload['records']:
        share = 100.0 * record['mean_ms'] / total_mean
        lines.append(
            '{:<4} {:<16} {:<13} {:>8d} {:>4d} {:>8d} {:>4d} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>7.1f}%'.format(
                record['layer_id'],
                record['module_name'],
                record['conv_type'],
                record['num_input_voxels'],
                record['cin'],
                record['num_output_voxels'],
                record['cout'],
                record['mean_ms'],
                record['p50_ms'],
                record['p90_ms'],
                record['std_ms'],
                share,
            )
        )
    return '\n'.join(lines)


def save_outputs(out_dir, run_tag, payload):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / ('%s.json' % run_tag)
    csv_path = out_path / ('%s.csv' % run_tag)

    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)

    rows = []
    for frame in payload['frames']:
        for record in frame['records']:
            rows.append({
                'frame_id': frame['frame_id'],
                'layer_id': record['layer_id'],
                'module_name': record['module_name'],
                'conv_type': record['conv_type'],
                'indice_key': record['indice_key'],
                'cin': record['cin'],
                'cout': record['cout'],
                'num_input_voxels': record['num_input_voxels'],
                'num_output_voxels': record['num_output_voxels'],
                'mean_ms': round(record['mean_ms'], 4),
                'p50_ms': round(record['p50_ms'], 4),
                'p90_ms': round(record['p90_ms'], 4),
                'p99_ms': round(record['p99_ms'], 4),
                'std_ms': round(record['std_ms'], 4),
                'min_ms': round(record['min_ms'], 4),
                'max_ms': round(record['max_ms'], 4),
                'share_pct': round(100.0 * record['mean_ms'] / max(frame['backbone_total_ms']['mean_ms'], 1e-12), 2),
                'kept_as': record.get('kept_as'),
            })

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main():
    args = parse_args()
    project_root = resolve_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.utils import common_utils

    cfg_path = project_root / args.cfg
    ckpt_path = project_root / args.ckpt
    if not cfg_path.exists():
        raise FileNotFoundError('Config file not found: %s' % cfg_path)
    if not ckpt_path.exists():
        raise FileNotFoundError('Checkpoint not found: %s' % ckpt_path)

    original_cwd = Path.cwd()
    try:
        os.chdir(project_root / 'tools')
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(original_cwd)

    device = resolve_device(args.device)
    torch.cuda.set_device(device)
    gpu_info = describe_gpu(device)

    logger = common_utils.create_logger()
    data_mode = resolve_data_mode(cfg, args.data_mode)
    if data_mode == 'kitti':
        dataset = build_kitti_dataset(cfg, project_root, args.kitti_root, logger)
        frame_id = '' if args.frame_id in {'', 'random'} else args.frame_id
        frame_refs = choose_kitti_frame_ids(dataset, frame_id, args.seed, args.num_frames)
    else:
        dataset = build_template_dataset(cfg)
        frame_id = '' if args.frame_id in {'', 'random'} else args.frame_id
        frame_refs = choose_raw_frame_paths(
            project_root / args.velodyne_dir, frame_id, args.seed, args.num_frames,
        )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=False)
    model.to(device)
    model.eval()
    enable_hw_reference(model, project_root, args, logger)

    run_tag = 'second_backbone_int8_layer_latency_%s_%s' % (
        gpu_info['name'].replace(' ', ''),
        datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    frames = []
    printed = []
    for frame_ref in frame_refs:
        if data_mode == 'kitti':
            sample, frame_info = load_kitti_sample(dataset, frame_ref)
        else:
            sample, frame_info, _raw_points = load_raw_sample(dataset, frame_ref)
        batch = dataset.collate_batch([sample])
        batch_torch = move_batch_to_device(batch, device)
        frame_payload = analyze_frame(
            model=model,
            batch_torch=batch_torch,
            frame_info=frame_info,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        frames.append(frame_payload)
        printed.append(format_records(frame_info['frame_id'], gpu_info['name'], frame_payload))
        print(printed[-1])
        print('')

    payload = {
        'cfg': str(cfg_path),
        'ckpt': str(ckpt_path),
        'device': str(device),
        'gpu': gpu_info,
        'weight_quant': args.weight_quant,
        'mode': 'hw_reference_int8_values_in_fp32_tensors',
        'data_mode': data_mode,
        'warmup': args.warmup,
        'repeat': args.repeat,
        'notes': [
            'CUDA event timing around each _forward_hw_reference_layer call.',
            'INT8 is value-level; spconv CUDA kernels still run FP32 on this GPU path.',
            'Share% is mean_layer / mean_backbone_sum.',
            'First SparseConv with a new indice_key pays pair-generation cost; later SubM reuse is cheaper.',
        ],
        'frames': frames,
    }
    json_path, csv_path = save_outputs(project_root / args.out_dir, run_tag, payload)
    log_path = project_root / args.out_dir / ('%s.log' % run_tag)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as handle:
        handle.write('\n\n'.join(printed))
        handle.write('\n\n')
        handle.write(json.dumps({
            'gpu': gpu_info,
            'ckpt': str(ckpt_path),
            'frames': [frame['frame_id'] for frame in frames],
            'json': str(json_path),
            'csv': str(csv_path),
        }, indent=2))
        handle.write('\n')

    print('Saved CSV:  %s' % csv_path)
    print('Saved JSON: %s' % json_path)
    print('Saved log:  %s' % log_path)


if __name__ == '__main__':
    main()

#!/usr/bin/python3
"""Profile per-layer GPU memory of SECOND VoxelBackBone8x HW-QAT INT8 inference.

The HW reference path stores INT8-valued features/weights inside FP32 CUDA
tensors, because spconv's CUDA kernels here still run in floating point. This
script therefore reports three complementary views:

1. Live CUDA allocator stats around each backbone layer (actual GPU occupancy)
2. Tensor breakdown of each SparseConvTensor (features / indices / indice pairs)
3. Logical INT8 payload sizes (N * C bytes) that an INT8 accelerator would keep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
        description='Per-layer GPU memory profile for SECOND VoxelBackBone8x INT8 HW-reference inference'
    )
    parser.add_argument('--cfg', type=str, default='tools/cfgs/kitti_models/second_hw_qat.yaml')
    parser.add_argument(
        '--ckpt',
        type=str,
        default='output/kitti_models/second_hw_qat/hw_qat_10ep/ckpt/checkpoint_epoch_10.pth',
    )
    parser.add_argument('--velodyne_dir', type=str, default='data/kitti/training/velodyne')
    add_data_mode_args(parser)
    parser.add_argument('--frame_id', type=str, default='000216', help='KITTI frame id, comma-separated, or empty for random')
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--out_dir', type=str, default='mycode/output/second_backbone_int8_layer_vram')
    parser.add_argument('--weight_quant', choices=['per_channel', 'per_tensor'], default='per_channel')
    parser.add_argument('--warmup', type=int, default=1, help='Unprofiled forwards used to settle CUDA/spconv caches')
    parser.add_argument('--repeat', type=int, default=1, help='Profiled forwards per frame after warmup')
    return parser.parse_args()


def bytes_to_mb(num_bytes):
    return float(num_bytes) / (1024.0 * 1024.0)


def round_mb(num_bytes, digits=4):
    return round(bytes_to_mb(num_bytes), digits)


def resolve_device(device_arg):
    if device_arg == 'auto':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for GPU VRAM profiling')
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


def cuda_sync():
    torch.cuda.synchronize()


def snapshot_allocator():
    cuda_sync()
    return {
        'allocated_bytes': int(torch.cuda.memory_allocated()),
        'reserved_bytes': int(torch.cuda.memory_reserved()),
        'max_allocated_bytes': int(torch.cuda.max_memory_allocated()),
        'max_reserved_bytes': int(torch.cuda.max_memory_reserved()),
    }


def reset_peak_from_current():
    cuda_sync()
    torch.cuda.reset_peak_memory_stats()


def tensor_cuda_bytes(tensor):
    if not torch.is_tensor(tensor):
        return 0
    if tensor.device.type != 'cuda':
        return 0
    return int(tensor.numel() * tensor.element_size())


def walk_cuda_bytes(obj, seen=None, depth=0, max_depth=6):
    if obj is None or depth > max_depth:
        return 0
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if torch.is_tensor(obj):
        return tensor_cuda_bytes(obj)
    if isinstance(obj, (bytes, bytearray, str, int, float, bool, np.ndarray)):
        return 0
    if isinstance(obj, dict):
        return sum(walk_cuda_bytes(value, seen, depth + 1, max_depth) for value in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(walk_cuda_bytes(value, seen, depth + 1, max_depth) for value in obj)
    if hasattr(obj, '__dict__'):
        return walk_cuda_bytes(vars(obj), seen, depth + 1, max_depth)
    if hasattr(obj, '_asdict'):
        return walk_cuda_bytes(obj._asdict(), seen, depth + 1, max_depth)
    return 0


def sparse_tensor_breakdown(sparse_tensor):
    if sparse_tensor is None:
        return {
            'active_voxels': 0,
            'channels': 0,
            'spatial_shape_zyx': [],
            'feature_dtype': '',
            'feature_bytes': 0,
            'indices_bytes': 0,
            'indice_dict_bytes': 0,
            'other_sparse_bytes': 0,
            'total_sparse_bytes': 0,
        }

    features = sparse_tensor.features
    indices = sparse_tensor.indices
    feature_bytes = tensor_cuda_bytes(features)
    indices_bytes = tensor_cuda_bytes(indices)
    indice_dict_bytes = walk_cuda_bytes(getattr(sparse_tensor, 'indice_dict', None))
    total_bytes = walk_cuda_bytes(sparse_tensor)
    other_bytes = max(total_bytes - feature_bytes - indices_bytes - indice_dict_bytes, 0)
    return {
        'active_voxels': int(features.shape[0]) if features is not None else 0,
        'channels': int(features.shape[1]) if features is not None and features.ndim > 1 else 0,
        'spatial_shape_zyx': [int(v) for v in list(sparse_tensor.spatial_shape)],
        'feature_dtype': str(features.dtype).replace('torch.', '') if features is not None else '',
        'feature_bytes': feature_bytes,
        'indices_bytes': indices_bytes,
        'indice_dict_bytes': indice_dict_bytes,
        'other_sparse_bytes': other_bytes,
        'total_sparse_bytes': total_bytes,
    }


def logical_int8_feature_bytes(num_voxels, channels):
    return int(num_voxels) * int(channels)


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


def format_mb(num_bytes):
    return '%8.3f' % bytes_to_mb(num_bytes)


def format_records(frame_id, gpu_name, records):
    lines = [
        'Frame: %s' % frame_id,
        'GPU: %s' % gpu_name,
        '',
        'Note: HW-reference INT8 values still live in FP32 CUDA tensors. '
        'delta/peak are allocator stats; int8_feat is the logical payload.',
        '',
        '{:<4} {:<16} {:<13} {:>8} {:>4} {:>8} {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
            'Id', 'Name', 'Type', 'Nin', 'Cin', 'Nout', 'Cout',
            'OutFP32MB', 'OutINT8MB', 'IndiceMB', 'Wint8MB', 'DeltaMB', 'Peak+MB', 'LiveMB',
        ),
        '-' * 148,
    ]
    for record in records:
        lines.append(
            '{:<4} {:<16} {:<13} {:>8d} {:>4d} {:>8d} {:>4d} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
                record['layer_id'],
                record['module_name'],
                record['conv_type'],
                record['num_input_voxels'],
                record['cin'],
                record['num_output_voxels'],
                record['cout'],
                format_mb(record['output_feature_bytes']),
                format_mb(record['output_feature_int8_bytes']),
                format_mb(record['output_indice_dict_bytes']),
                format_mb(record['weight_int8_bytes']),
                format_mb(record['delta_allocated_bytes']),
                format_mb(record['peak_extra_allocated_bytes']),
                format_mb(record['live_allocated_bytes']),
            )
        )
    return '\n'.join(lines)


def enable_hw_reference(model, project_root, args, logger):
    tools_dir = project_root / 'tools'
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import test_second_hw_qat as hw_qat_tools

    backbone = model.backbone_3d
    export_dir = project_root / args.out_dir / 'hw_export_for_vram'
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
    backbone.enable_hw_qat(False, weight_quant=args.weight_quant, observer='max',
                           observer_momentum=0.95, fake_quant=False)
    backbone.enable_hw_reference(True, qparams=export_result['qparams'])
    return export_result


def layer_static_info(backbone):
    rows = []
    for layer_id, (name, conv, _bn, _relu, act_key) in enumerate(backbone.get_layer_modules()):
        qparam = backbone._hw_qparams[layer_id]
        weight_int8 = qparam['weight_int8']
        bias_int = qparam['bias_int']
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
            'weight_numel': int(weight_int8.numel()),
            'weight_int8_bytes': int(weight_int8.numel()),
            'weight_fp32_runtime_bytes': int(conv.weight.numel() * conv.weight.element_size()),
            'bias_int32_bytes': int(bias_int.numel() * 4),
            'kept_as': LAYER_KEEP_KEYS.get(layer_id),
        })
    return rows


def profile_backbone_layers(backbone, batch_torch):
    records = []
    static_rows = {row['layer_id']: row for row in layer_static_info(backbone)}
    orig_layer = backbone._forward_hw_reference_layer

    def wrapped(sparse_tensor, layer_id, layer_info):
        static = static_rows[layer_id]
        input_break = sparse_tensor_breakdown(sparse_tensor)
        reset_peak_from_current()
        before = snapshot_allocator()
        output = orig_layer(sparse_tensor, layer_id, layer_info)
        after = snapshot_allocator()
        output_break = sparse_tensor_breakdown(output)
        peak_extra = max(int(after['max_allocated_bytes']) - int(before['allocated_bytes']), 0)
        record = {
            **static,
            'num_input_voxels': input_break['active_voxels'],
            'num_output_voxels': output_break['active_voxels'],
            'input_spatial_shape_zyx': input_break['spatial_shape_zyx'],
            'output_spatial_shape_zyx': output_break['spatial_shape_zyx'],
            'input_feature_dtype': input_break['feature_dtype'],
            'output_feature_dtype': output_break['feature_dtype'],
            'input_feature_bytes': input_break['feature_bytes'],
            'output_feature_bytes': output_break['feature_bytes'],
            'input_indices_bytes': input_break['indices_bytes'],
            'output_indices_bytes': output_break['indices_bytes'],
            'input_indice_dict_bytes': input_break['indice_dict_bytes'],
            'output_indice_dict_bytes': output_break['indice_dict_bytes'],
            'input_sparse_total_bytes': input_break['total_sparse_bytes'],
            'output_sparse_total_bytes': output_break['total_sparse_bytes'],
            'input_feature_int8_bytes': logical_int8_feature_bytes(input_break['active_voxels'], static['cin']),
            'output_feature_int8_bytes': logical_int8_feature_bytes(output_break['active_voxels'], static['cout']),
            'before_allocated_bytes': before['allocated_bytes'],
            'after_allocated_bytes': after['allocated_bytes'],
            'live_allocated_bytes': after['allocated_bytes'],
            'live_reserved_bytes': after['reserved_bytes'],
            'delta_allocated_bytes': int(after['allocated_bytes']) - int(before['allocated_bytes']),
            'delta_reserved_bytes': int(after['reserved_bytes']) - int(before['reserved_bytes']),
            'peak_allocated_bytes': after['max_allocated_bytes'],
            'peak_extra_allocated_bytes': peak_extra,
        }
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
    for _ in range(max(int(warmup), 0)):
        with torch.no_grad():
            _ = backbone(run_vfe(model, dict(batch_torch)))

    cuda_sync()
    torch.cuda.empty_cache()
    reset_peak_from_current()

    repeats = []
    for _ in range(max(int(repeat), 1)):
        before_vfe = snapshot_allocator()
        vfe_batch = run_vfe(model, dict(batch_torch))
        after_vfe = snapshot_allocator()
        records = profile_backbone_layers(backbone, vfe_batch)
        after_backbone = snapshot_allocator()
        repeats.append({
            'vfe': {
                'num_voxels': int(vfe_batch['voxel_features'].shape[0]),
                'channels': int(vfe_batch['voxel_features'].shape[1]),
                'feature_bytes': tensor_cuda_bytes(vfe_batch['voxel_features']),
                'coords_bytes': tensor_cuda_bytes(vfe_batch['voxel_coords']),
                'before_allocated_bytes': before_vfe['allocated_bytes'],
                'after_allocated_bytes': after_vfe['allocated_bytes'],
                'delta_allocated_bytes': int(after_vfe['allocated_bytes']) - int(before_vfe['allocated_bytes']),
            },
            'backbone_peak_allocated_bytes': after_backbone['max_allocated_bytes'],
            'backbone_live_allocated_bytes': after_backbone['allocated_bytes'],
            'backbone_live_reserved_bytes': after_backbone['reserved_bytes'],
            'records': records,
        })

    chosen = repeats[-1]
    return {
        'frame_id': frame_info['frame_id'],
        'point_cloud_path': frame_info['point_cloud_path'],
        'data_loader': frame_info['data_loader'],
        'fov_points_only': frame_info['fov_points_only'],
        'vfe': chosen['vfe'],
        'backbone_peak_allocated_bytes': chosen['backbone_peak_allocated_bytes'],
        'backbone_live_allocated_bytes': chosen['backbone_live_allocated_bytes'],
        'backbone_live_reserved_bytes': chosen['backbone_live_reserved_bytes'],
        'records': chosen['records'],
        'repeats': repeats,
    }


def mb_fields(record):
    keys = [key for key in record.keys() if key.endswith('_bytes')]
    extra = {}
    for key in keys:
        extra[key.replace('_bytes', '_mb')] = round_mb(record[key])
    return extra


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
            row = {
                'frame_id': frame['frame_id'],
                **record,
                **mb_fields(record),
            }
            rows.append(row)

    fieldnames = [
        'frame_id', 'layer_id', 'module_name', 'conv_type', 'indice_key',
        'cin', 'cout', 'num_input_voxels', 'num_output_voxels',
        'input_feature_mb', 'output_feature_mb', 'output_feature_int8_mb',
        'output_indices_mb', 'output_indice_dict_mb', 'weight_int8_mb',
        'weight_fp32_runtime_mb', 'delta_allocated_mb', 'peak_extra_allocated_mb',
        'live_allocated_mb', 'live_reserved_mb', 'kept_as',
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
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
        frame_refs = choose_raw_frame_paths(project_root / args.velodyne_dir, frame_id, args.seed, args.num_frames)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=False)
    model.to(device)
    model.eval()
    enable_hw_reference(model, project_root, args, logger)

    run_tag = 'second_backbone_int8_layer_vram_%s_%s' % (
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
        printed.append(format_records(frame_info['frame_id'], gpu_info['name'], frame_payload['records']))
        print(printed[-1])
        print('')
        print('VFE voxels: %d, VFE delta: %.3f MB, backbone live: %.3f MB, reserved: %.3f MB, peak: %.3f MB' % (
            frame_payload['vfe']['num_voxels'],
            bytes_to_mb(frame_payload['vfe']['delta_allocated_bytes']),
            bytes_to_mb(frame_payload['backbone_live_allocated_bytes']),
            bytes_to_mb(frame_payload['backbone_live_reserved_bytes']),
            bytes_to_mb(frame_payload['backbone_peak_allocated_bytes']),
        ))
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
            'PyTorch/spconv CUDA kernels still execute in FP32; INT8 is value-level, not Tensor-Core INT8.',
            'delta_allocated_bytes is the net live-memory change after the layer returns.',
            'peak_extra_allocated_bytes includes transient GEMM/indice workspace during that layer.',
            'output_feature_int8_bytes is N_out * C_out, i.e. the logical INT8 activation payload.',
            'x_conv1/2/3/4 stay alive because OpenPCDet keeps multi-scale 3D features.',
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

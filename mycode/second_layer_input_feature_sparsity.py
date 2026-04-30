#!/usr/bin/env python3
"""Measure per-layer input feature sparsity for SECOND sparse convolution layers.

The script samples one or more KITTI point clouds, runs voxelization + MeanVFE,
then records the input SparseConvTensor features for every sparse convolution in
the 3D backbone. It reports how many feature channels are zero inside active
voxels, which captures feature sparsity created by preceding ReLU operations.
"""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Per-layer input feature sparsity analysis for SECOND backbone')
    parser.add_argument('--cfg', type=str, default='tools/cfgs/kitti_models/second.yaml')
    parser.add_argument('--ckpt', type=str, default='mycode/second_7862.pth')
    parser.add_argument('--velodyne_dir', type=str, default='data/kitti/training/velodyne')
    parser.add_argument('--frame_id', type=str, default='', help='KITTI frame id, e.g. 000123. Leave empty for random pick.')
    parser.add_argument('--num_frames', type=int, default=1, help='Number of frames to analyze.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto', help='auto|cpu|cuda|cuda:0')
    parser.add_argument('--out_dir', type=str, default='mycode/output')
    parser.add_argument('--log_file', type=str, default='', help='Optional combined log file path.')
    parser.add_argument('--zero_threshold', type=float, default=0.0, help='Treat abs(x) <= threshold as zero.')
    parser.add_argument('--int8_inference', action='store_true', default=False,
                        help='Enable INT8 inference mode for VoxelBackBone8x_INT8 if supported.')
    return parser.parse_args()


def resolve_project_root():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def choose_frames(velodyne_dir, frame_id, seed, num_frames):
    velodyne_path = Path(velodyne_dir)
    if not velodyne_path.exists():
        raise FileNotFoundError(f'Velodyne directory not found: {velodyne_path}')

    if frame_id:
        tokens = [token.strip() for token in frame_id.split(',') if token.strip()]
        if num_frames != len(tokens):
            raise ValueError('When --frame_id is provided, --num_frames must match the number of frame ids.')

        frames = []
        for token in tokens:
            frame_name = token if token.endswith('.bin') else f'{token}.bin'
            frame_path = velodyne_path / frame_name
            if not frame_path.exists():
                raise FileNotFoundError(f'Frame not found: {frame_path}')
            frames.append(frame_path)
        return frames

    candidates = sorted(velodyne_path.glob('*.bin'))
    if not candidates:
        raise RuntimeError(f'No .bin files found in {velodyne_path}')
    if num_frames > len(candidates):
        raise ValueError(f'Requested {num_frames} frames, but only found {len(candidates)} point clouds.')

    rng = random.Random(seed)
    return rng.sample(candidates, num_frames)


def build_inference_dataset(cfg_obj):
    from pcdet.datasets.dataset import DatasetTemplate

    return DatasetTemplate(dataset_cfg=cfg_obj.DATA_CONFIG, class_names=cfg_obj.CLASS_NAMES, training=False)


def load_single_batch(dataset, frame_path):
    points = np.fromfile(str(frame_path), dtype=np.float32).reshape(-1, 4)
    sample = dataset.prepare_data({'points': points, 'frame_id': frame_path.stem})
    batch = dataset.collate_batch([sample])
    return batch, int(points.shape[0])


def move_batch_to_device(batch_dict, device):
    skip_keys = {'frame_id', 'metadata', 'calib', 'image_paths', 'ori_shape', 'img_process_infos'}
    output = {}
    for key, value in batch_dict.items():
        if key in skip_keys:
            output[key] = value
            continue
        if isinstance(value, np.ndarray):
            if key == 'image_shape':
                output[key] = torch.from_numpy(value).int().to(device)
            else:
                output[key] = torch.from_numpy(value).float().to(device)
        else:
            output[key] = value
    output['batch_size'] = batch_dict.get('batch_size', 1)
    return output


def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def to_int_list(values):
    return [int(value) for value in list(values)]


def humanize_layer_name(name):
    if name.endswith('.0'):
        return name[:-2]
    return name


def compute_zero_mask(features, zero_threshold):
    if zero_threshold > 0:
        return features.abs() <= zero_threshold
    return features == 0


def compute_feature_sparsity_stats(features, zero_threshold):
    active_voxels = int(features.shape[0])
    channels = int(features.shape[1]) if features.ndim > 1 else 0
    total_elements = int(features.numel())

    if total_elements == 0 or channels == 0:
        return {
            'active_voxels': active_voxels,
            'channels': channels,
            'total_elements': total_elements,
            'zero_elements': 0,
            'nonzero_elements': 0,
            'zero_ratio': 0.0,
            'nonzero_ratio': 0.0,
            'avg_zero_channels_per_voxel': 0.0,
            'avg_nonzero_channels_per_voxel': 0.0,
            'all_zero_voxel_ratio': 0.0,
            'zero_channel_ratio_p50': 0.0,
            'zero_channel_ratio_p90': 0.0,
            'zero_channel_ratio_p95': 0.0,
            'zero_channel_ratio_p99': 0.0,
            'zero_channel_ratio_max': 0.0,
        }

    zero_mask = compute_zero_mask(features, zero_threshold)
    zero_elements = int(zero_mask.sum().item())
    nonzero_elements = total_elements - zero_elements

    zero_channels_per_voxel = zero_mask.sum(dim=1).to(torch.float32)
    zero_channel_ratio_per_voxel = zero_channels_per_voxel / float(channels)
    quantiles = torch.quantile(
        zero_channel_ratio_per_voxel,
        torch.tensor([0.5, 0.9, 0.95, 0.99], device=zero_channel_ratio_per_voxel.device, dtype=torch.float32)
    )

    return {
        'active_voxels': active_voxels,
        'channels': channels,
        'total_elements': total_elements,
        'zero_elements': zero_elements,
        'nonzero_elements': nonzero_elements,
        'zero_ratio': float(zero_elements / total_elements),
        'nonzero_ratio': float(nonzero_elements / total_elements),
        'avg_zero_channels_per_voxel': float(zero_channels_per_voxel.mean().item()),
        'avg_nonzero_channels_per_voxel': float(channels - zero_channels_per_voxel.mean().item()),
        'all_zero_voxel_ratio': float((zero_channels_per_voxel == channels).to(torch.float32).mean().item()),
        'zero_channel_ratio_p50': float(quantiles[0].item()),
        'zero_channel_ratio_p90': float(quantiles[1].item()),
        'zero_channel_ratio_p95': float(quantiles[2].item()),
        'zero_channel_ratio_p99': float(quantiles[3].item()),
        'zero_channel_ratio_max': float(zero_channel_ratio_per_voxel.max().item()),
    }


def build_record(record_name, layer_type, indice_key, sparse_tensor, zero_threshold, module_name='', source='layer_input'):
    stats = compute_feature_sparsity_stats(sparse_tensor.features, zero_threshold)
    return {
        'record_name': record_name,
        'module_name': module_name or record_name,
        'source': source,
        'layer_type': layer_type,
        'indice_key': indice_key,
        'spatial_shape_zyx': to_int_list(sparse_tensor.spatial_shape),
        'feature_shape_nc': [int(v) for v in sparse_tensor.features.shape],
        **stats,
    }


def make_input_record(backbone, batch_dict, zero_threshold):
    class BatchFeatureView:
        def __init__(self, features, coords, spatial_shape):
            self.features = features
            self.indices = coords
            self.spatial_shape = spatial_shape

    return build_record(
        record_name='input_voxel_features',
        layer_type='InputFeatures',
        indice_key='',
        sparse_tensor=BatchFeatureView(batch_dict['voxel_features'], batch_dict['voxel_coords'], backbone.sparse_shape),
        zero_threshold=zero_threshold,
        module_name='conv_input',
        source='backbone_input',
    )


class SparseInputRecorder:
    def __init__(self, spconv_module, zero_threshold):
        self.spconv_module = spconv_module
        self.zero_threshold = zero_threshold
        self.records = []
        self.handles = []

    def register(self, backbone):
        allowed = (self.spconv_module.SubMConv3d, self.spconv_module.SparseConv3d)
        for name, module in backbone.named_modules():
            if isinstance(module, allowed):
                handle = module.register_forward_pre_hook(self._make_pre_hook(name, module))
                self.handles.append(handle)

    def _make_pre_hook(self, name, module):
        def hook(_module, inputs):
            if not inputs:
                return
            sparse_tensor = inputs[0]
            display_name = humanize_layer_name(name)
            self.records.append(build_record(
                record_name=display_name,
                module_name=name,
                layer_type=module.__class__.__name__,
                indice_key=getattr(module, 'indice_key', ''),
                sparse_tensor=sparse_tensor,
                zero_threshold=self.zero_threshold,
            ))

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def format_records(frame_id, device, records):
    lines = [
        f'Frame: {frame_id}',
        f'Device: {device}',
        '',
        f"{'Record':<20} {'Type':<16} {'Active':>9} {'C':>5} {'ZeroRatio':>11} {'AvgZeroC':>11} {'AllZeroV':>10} {'P50':>8} {'P90':>8} {'P99':>8} {'SpatialShape(Z,Y,X)':>24}",
        '-' * 140,
    ]
    for record in records:
        shape_str = str(tuple(record['spatial_shape_zyx']))
        lines.append(
            f"{record['record_name']:<20} {record['layer_type']:<16} {record['active_voxels']:>9d} "
            f"{record['channels']:>5d} {record['zero_ratio']:>11.6f} {record['avg_zero_channels_per_voxel']:>11.3f} "
            f"{record['all_zero_voxel_ratio']:>10.6f} {record['zero_channel_ratio_p50']:>8.4f} "
            f"{record['zero_channel_ratio_p90']:>8.4f} {record['zero_channel_ratio_p99']:>8.4f} {shape_str:>24}"
        )
    return '\n'.join(lines)


def save_records(out_dir, frame_id, payload):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / f'{frame_id}_second_input_feature_sparsity.json'
    csv_path = out_path / f'{frame_id}_second_input_feature_sparsity.csv'

    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)

    fieldnames = [
        'record_name', 'module_name', 'source', 'layer_type', 'indice_key', 'spatial_shape_zyx', 'feature_shape_nc',
        'active_voxels', 'channels', 'total_elements', 'zero_elements', 'nonzero_elements', 'zero_ratio',
        'nonzero_ratio', 'avg_zero_channels_per_voxel', 'avg_nonzero_channels_per_voxel', 'all_zero_voxel_ratio',
        'zero_channel_ratio_p50', 'zero_channel_ratio_p90', 'zero_channel_ratio_p95', 'zero_channel_ratio_p99',
        'zero_channel_ratio_max'
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in payload['records']:
            row = dict(record)
            row['spatial_shape_zyx'] = 'x'.join(str(v) for v in row['spatial_shape_zyx'])
            row['feature_shape_nc'] = 'x'.join(str(v) for v in row['feature_shape_nc'])
            writer.writerow(row)

    return json_path, csv_path


def save_combined_summary(out_dir, run_tag, payloads):
    summary_path = Path(out_dir) / f'{run_tag}_summary.json'
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump({'frames': payloads}, handle, indent=2)
    return summary_path


def resolve_log_path(project_root, out_dir, requested_log_file, run_tag):
    if requested_log_file:
        return project_root / requested_log_file
    return project_root / out_dir / f'{run_tag}.log'


def setup_int8_inference(model, enable_int8):
    if not enable_int8:
        return False

    backbone = model.backbone_3d
    if not (hasattr(backbone, 'convert_to_int8') and hasattr(backbone, 'enable_int8_inference')):
        return False

    backbone.convert_to_int8()
    backbone.enable_int8_inference(True)
    return True


def analyze_frame(model, backbone, spconv, batch_torch, frame_path, raw_points, zero_threshold):
    with torch.no_grad():
        batch_torch = model.vfe(batch_torch)
        input_record = make_input_record(backbone, batch_torch, zero_threshold)

        recorder = SparseInputRecorder(spconv, zero_threshold=zero_threshold)
        recorder.register(backbone)
        try:
            _ = backbone(batch_torch)
        finally:
            recorder.remove()

    return {
        'frame_id': frame_path.stem,
        'point_cloud_path': str(frame_path),
        'num_raw_points': int(raw_points),
        'num_voxels_after_vfe': int(input_record['active_voxels']),
        'records': [input_record] + recorder.records,
    }


def main():
    args = parse_args()
    project_root = resolve_project_root()

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.utils import common_utils
    from pcdet.utils.spconv_utils import spconv

    cfg_path = project_root / args.cfg
    ckpt_path = project_root / args.ckpt
    velodyne_dir = project_root / args.velodyne_dir

    if not cfg_path.exists():
        raise FileNotFoundError(f'Config file not found: {cfg_path}')
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    original_cwd = Path.cwd()
    try:
        os.chdir(project_root / 'tools')
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(original_cwd)

    frame_paths = choose_frames(velodyne_dir, args.frame_id, args.seed, args.num_frames)
    dataset = build_inference_dataset(cfg)
    device = resolve_device(args.device)

    logger = common_utils.create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=(device.type == 'cpu'))
    model.to(device)

    int8_enabled = setup_int8_inference(model, args.int8_inference)
    model.eval()
    backbone = model.backbone_3d

    run_tag = (
        f"second_input_feature_sparsity_{args.num_frames}frames_seed{args.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    log_path = resolve_log_path(project_root, args.out_dir, args.log_file, run_tag)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    payloads = []
    log_sections = []
    for frame_path in frame_paths:
        batch, raw_points = load_single_batch(dataset, frame_path)
        batch_torch = move_batch_to_device(batch, device)
        frame_payload = analyze_frame(
            model=model,
            backbone=backbone,
            spconv=spconv,
            batch_torch=batch_torch,
            frame_path=frame_path,
            raw_points=raw_points,
            zero_threshold=args.zero_threshold,
        )
        frame_payload.update({
            'cfg': str(cfg_path),
            'ckpt': str(ckpt_path),
            'device': str(device),
            'seed': args.seed,
            'zero_threshold': args.zero_threshold,
            'int8_inference': bool(int8_enabled),
        })
        payloads.append(frame_payload)

        formatted = format_records(frame_path.stem, device, frame_payload['records'])
        print(formatted)
        json_path, csv_path = save_records(project_root / args.out_dir, frame_path.stem, frame_payload)
        section_lines = [
            formatted,
            '',
            f'Raw points: {raw_points}',
            f'Checkpoint: {ckpt_path}',
            f'Saved JSON: {json_path}',
            f'Saved CSV:  {csv_path}',
        ]
        log_sections.append('\n'.join(section_lines))

        print('')
        print(f'Raw points: {raw_points}')
        print(f'Saved JSON: {json_path}')
        print(f'Saved CSV:  {csv_path}')
        print('')

    run_payload = {
        'cfg': str(cfg_path),
        'ckpt': str(ckpt_path),
        'device': str(device),
        'seed': args.seed,
        'num_frames': args.num_frames,
        'zero_threshold': args.zero_threshold,
        'int8_inference': bool(int8_enabled),
        'frames': [frame_payload['frame_id'] for frame_payload in payloads],
    }

    summary_path = save_combined_summary(project_root / args.out_dir, run_tag, payloads)
    with log_path.open('w', encoding='utf-8') as handle:
        handle.write('\n\n'.join(log_sections))
        handle.write('\n\n')
        handle.write(json.dumps(run_payload, indent=2))
        handle.write(f'\nSummary JSON: {summary_path}\n')

    print(f'Combined log: {log_path}')
    print(f'Combined summary JSON: {summary_path}')


if __name__ == '__main__':
    main()
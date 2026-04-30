#!/usr/bin/env python3
"""Measure per-sparse-conv-layer voxel sparsity for SECOND 3D backbone.

This script starts from voxelization output, runs MeanVFE and the 3D sparse
backbone, and reports sparsity after every sparse convolution layer in
`spconv_backbone_qat.py`.
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
    parser = argparse.ArgumentParser(description='Per-layer sparse conv sparsity analysis for SECOND backbone')
    parser.add_argument('--cfg', type=str, default='tools/cfgs/kitti_models/second.yaml')
    parser.add_argument('--ckpt', type=str, default='mycode/second_7862.pth')
    parser.add_argument('--velodyne_dir', type=str, default='data/kitti/training/velodyne')
    parser.add_argument('--frame_id', type=str, default='', help='KITTI frame id, e.g. 000123. Leave empty for random pick.')
    parser.add_argument('--device', type=str, default='auto', help='auto|cpu|cuda|cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=1, help='Number of different point clouds to analyze.')
    parser.add_argument('--out_dir', type=str, default='mycode/output')
    parser.add_argument('--log_file', type=str, default='', help='Optional combined log file path.')
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
        frame_tokens = [token.strip() for token in frame_id.split(',') if token.strip()]
        frames = []
        for token in frame_tokens:
            frame_name = token if token.endswith('.bin') else f'{token}.bin'
            frame_path = velodyne_path / frame_name
            if not frame_path.exists():
                raise FileNotFoundError(f'Frame not found: {frame_path}')
            frames.append(frame_path)
        if num_frames != len(frames):
            raise ValueError('When --frame_id is provided, --num_frames must match the number of frame ids.')
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
    return batch


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


def to_int_list(spatial_shape):
    return [int(value) for value in list(spatial_shape)]


def total_cells(spatial_shape):
    total = 1
    for value in spatial_shape:
        total *= int(value)
    return int(total)


def feature_shape_of(features):
    return [int(value) for value in list(features.shape)]


def make_record(layer_name, layer_type, indice_key, sparse_tensor, previous_active, input_active):
    shape = to_int_list(sparse_tensor.spatial_shape)
    active = int(sparse_tensor.indices.shape[0])
    feature_shape = feature_shape_of(sparse_tensor.features)
    cells = total_cells(shape)
    sparsity = float(active / cells) if cells > 0 else 0.0
    ratio_prev = float(active / previous_active) if previous_active else None
    ratio_input = float(active / input_active) if input_active else None
    return {
        'layer_name': layer_name,
        'layer_type': layer_type,
        'indice_key': indice_key,
        'spatial_shape_zyx': shape,
        'feature_shape_nc': feature_shape,
        'feature_channels': int(feature_shape[1]) if len(feature_shape) > 1 else 0,
        'total_cells': cells,
        'active_voxels': active,
        'sparsity': sparsity,
        'active_ratio_vs_prev': ratio_prev,
        'active_ratio_vs_input': ratio_input,
    }


def make_input_record(backbone, batch_dict):
    coords = batch_dict['voxel_coords']
    features = batch_dict['voxel_features']
    active = int(coords.shape[0])
    shape = to_int_list(backbone.sparse_shape)
    feature_shape = feature_shape_of(features)
    cells = total_cells(shape)
    return {
        'layer_name': 'input_voxelized',
        'layer_type': 'InputSparseTensor',
        'indice_key': '',
        'spatial_shape_zyx': shape,
        'feature_shape_nc': feature_shape,
        'feature_channels': int(feature_shape[1]) if len(feature_shape) > 1 else 0,
        'total_cells': cells,
        'active_voxels': active,
        'sparsity': float(active / cells) if cells > 0 else 0.0,
        'active_ratio_vs_prev': 1.0,
        'active_ratio_vs_input': 1.0,
    }


class SparseLayerRecorder:
    def __init__(self, spconv_module, input_active):
        self.spconv_module = spconv_module
        self.input_active = input_active
        self.previous_active = input_active
        self.records = []
        self.handles = []

    def register(self, backbone):
        allowed = (self.spconv_module.SubMConv3d, self.spconv_module.SparseConv3d)
        for name, module in backbone.named_modules():
            if isinstance(module, allowed):
                handle = module.register_forward_hook(self._make_hook(name, module))
                self.handles.append(handle)

    def _make_hook(self, name, module):
        def hook(_module, _inputs, output):
            record = make_record(
                layer_name=name,
                layer_type=module.__class__.__name__,
                indice_key=getattr(module, 'indice_key', ''),
                sparse_tensor=output,
                previous_active=self.previous_active,
                input_active=self.input_active,
            )
            self.records.append(record)
            self.previous_active = record['active_voxels']

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
        f"{'Layer':<18} {'Type':<14} {'FeatShape(N,C)':>18} {'Active':>10} {'TotalCells':>12} {'Sparsity':>12} {'VsPrev':>10} {'VsInput':>10} {'SpatialShape(Z,Y,X)':>24}",
        '-' * 142,
    ]
    for record in records:
        prev_ratio = '-' if record['active_ratio_vs_prev'] is None else f"{record['active_ratio_vs_prev']:.6f}"
        input_ratio = '-' if record['active_ratio_vs_input'] is None else f"{record['active_ratio_vs_input']:.6f}"
        shape_str = str(tuple(record['spatial_shape_zyx']))
        feature_shape_str = str(tuple(record['feature_shape_nc']))
        lines.append(
            f"{record['layer_name']:<18} {record['layer_type']:<14} {feature_shape_str:>18} "
            f"{record['active_voxels']:>10d} {record['total_cells']:>12d} "
            f"{record['sparsity']:>12.8f} {prev_ratio:>10} {input_ratio:>10} {shape_str:>24}"
        )
    return '\n'.join(lines)


def save_records(out_dir, frame_id, payload):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / f'{frame_id}_second_layer_sparsity.json'
    csv_path = out_path / f'{frame_id}_second_layer_sparsity.csv'

    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)

    fieldnames = [
        'layer_name', 'layer_type', 'indice_key', 'spatial_shape_zyx', 'feature_shape_nc', 'feature_channels',
        'total_cells', 'active_voxels', 'sparsity', 'active_ratio_vs_prev', 'active_ratio_vs_input'
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in payload['records']:
            row = dict(record)
            row['spatial_shape_zyx'] = 'x'.join(str(value) for value in row['spatial_shape_zyx'])
            row['feature_shape_nc'] = 'x'.join(str(value) for value in row['feature_shape_nc'])
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


def analyze_frame(model, backbone, spconv, batch_torch, frame_path):
    with torch.no_grad():
        batch_torch = model.vfe(batch_torch)
        input_record = make_input_record(backbone, batch_torch)

        recorder = SparseLayerRecorder(spconv, input_active=input_record['active_voxels'])
        recorder.register(backbone)
        try:
            _ = backbone(batch_torch)
        finally:
            recorder.remove()

    records = [input_record] + recorder.records
    return {
        'frame_id': frame_path.stem,
        'point_cloud_path': str(frame_path),
        'records': records,
    }


def main():
    args = parse_args()
    project_root = resolve_project_root()

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.utils import common_utils
    from pcdet.utils.spconv_utils import spconv

    original_cwd = Path.cwd()
    try:
        os.chdir(project_root / 'tools')
        cfg_from_yaml_file(str(project_root / args.cfg), cfg)
    finally:
        os.chdir(original_cwd)

    frame_paths = choose_frames(project_root / args.velodyne_dir, args.frame_id, args.seed, args.num_frames)
    dataset = build_inference_dataset(cfg)

    device = resolve_device(args.device)

    logger = common_utils.create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)

    ckpt_path = project_root / args.ckpt
    if ckpt_path.exists():
        model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=(device.type == 'cpu'))
    else:
        logger.info(f'Checkpoint not found, continue without loading weights: {ckpt_path}')

    model.to(device)
    model.eval()

    backbone = model.backbone_3d

    run_tag = f"second_layer_sparsity_{args.num_frames}frames_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = resolve_log_path(project_root, args.out_dir, args.log_file, run_tag)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    payloads = []
    log_sections = []
    for frame_path in frame_paths:
        batch = load_single_batch(dataset, frame_path)
        batch_torch = move_batch_to_device(batch, device)
        frame_payload = analyze_frame(model, backbone, spconv, batch_torch, frame_path)
        frame_payload.update({
            'cfg': str(project_root / args.cfg),
            'ckpt': str(ckpt_path),
            'device': str(device),
            'seed': args.seed,
        })
        payloads.append(frame_payload)

        formatted = format_records(frame_path.stem, device, frame_payload['records'])
        print(formatted)
        json_path, csv_path = save_records(project_root / args.out_dir, frame_path.stem, frame_payload)
        tail_lines = [formatted, '', f'Saved JSON: {json_path}', f'Saved CSV:  {csv_path}']
        log_sections.append('\n'.join(tail_lines))
        print('')
        print(f'Saved JSON: {json_path}')
        print(f'Saved CSV:  {csv_path}')
        print('')

    payload = {
        'cfg': str(project_root / args.cfg),
        'ckpt': str(ckpt_path),
        'device': str(device),
        'seed': args.seed,
        'num_frames': args.num_frames,
        'frames': [frame_payload['frame_id'] for frame_payload in payloads],
    }

    summary_path = save_combined_summary(project_root / args.out_dir, run_tag, payloads)
    with log_path.open('w', encoding='utf-8') as handle:
        handle.write('\n\n'.join(log_sections))
        handle.write('\n\n')
        handle.write(json.dumps(payload, indent=2))
        handle.write(f'\nSummary JSON: {summary_path}\n')

    print(f'Combined log: {log_path}')
    print(f'Combined summary JSON: {summary_path}')


if __name__ == '__main__':
    main()
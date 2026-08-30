#!/usr/bin/env python3
"""Compute Chebyshev-distance voxel statistics at SECOND 3D-backbone stages.

For each frame this script voxelizes the point cloud, runs MeanVFE and
VoxelBackBone8x, then writes one CSV per analyzed stage:

- conv1: voxelization / VFE / conv1 (SubM, stride 1, same occupied voxels)
- conv2: after the first XY stride-2 SparseConv block
- conv3: after the second XY stride-2 SparseConv block
- conv4: after the third XY stride-2 SparseConv block

`conv_out` (Z-only stride 2) is not analyzed.

Chebyshev distance is still 2D in voxel XY. The LiDAR center is scaled with
the backbone stride: (cx, cy) at stride s becomes (cx // s, cy // s), so
KITTI default (0, 800) maps to (0, 400), (0, 200), (0, 100).

This script is locked to KITTI inference input: `KittiDataset.__getitem__()`
with `FOV_POINTS_ONLY=True`, matching `tools/test.py` and RTL golden packages.
Train and val infos are merged so consecutive `training/velodyne` IDs can be
loaded without changing the eval preprocessor.

`voxel_occupancy` is non_empty / total. `voxel_sparsity` is 1 - occupancy.

Each CSV ends with a `MEAN_<N>_frames` row: per-bin Chebyshev counts averaged
over the N processed frames.
"""
import os
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mycode.kitti_frame_loader import (
    add_data_mode_args,
    build_kitti_dataset,
    choose_frame_ids,
    get_class_names,
    get_dataset_cfg,
    load_kitti_sample,
    normalize_frame_token,
    normalize_voxel_coords,
    resolve_data_mode,
    resolve_project_root,
)
from pcdet.config import cfg, cfg_from_yaml_file


STAGE_SPECS = (
    {'name': 'conv1', 'feature_key': 'x_conv1', 'stride': 1},
    {'name': 'conv2', 'feature_key': 'x_conv2', 'stride': 2},
    {'name': 'conv3', 'feature_key': 'x_conv3', 'stride': 4},
    {'name': 'conv4', 'feature_key': 'x_conv4', 'stride': 8},
)

BASE_CSV_KEYS = [
    'file',
    'stage',
    'stride',
    'spatial_shape_zyx',
    'lidar_center_x',
    'lidar_center_y',
    'data_loader',
    'fov_points_only',
    'data_mode',
    'max_voxels',
    'total_voxels',
    'non_empty_voxels',
    'voxel_occupancy',
    'voxel_sparsity',
]


def attach_trainval_infos(dataset):
    """Keep eval preprocessing, but index every training/velodyne sample ID."""
    infos = []
    seen = set()
    for name in ('kitti_infos_train.pkl', 'kitti_infos_val.pkl'):
        path = Path(dataset.root_path) / name
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open('rb') as handle:
            for info in pickle.load(handle):
                lidar_idx = str(info['point_cloud']['lidar_idx'])
                if lidar_idx in seen:
                    continue
                seen.add(lidar_idx)
                infos.append(info)
    dataset.kitti_infos = infos
    return {str(info['point_cloud']['lidar_idx']): idx for idx, info in enumerate(infos)}


def load_model_cfg(cfg_file):
    project_root = Path(cfg.ROOT_DIR)
    cfg_path = Path(cfg_file)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    original_cwd = Path.cwd()
    try:
        os.chdir(project_root / 'tools')
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(original_cwd)
    return cfg


def parse_lidar_center(text):
    parts = [part.strip() for part in str(text).split(',')]
    if len(parts) != 2:
        raise ValueError('lidar_center must contain exactly two integers: x,y')
    return int(parts[0]), int(parts[1])


def max_voxels_from_cfg(cfg_obj, training=False):
    mode = 'train' if training else 'test'
    dataset_cfg = get_dataset_cfg(cfg_obj)
    for processor in dataset_cfg.DATA_PROCESSOR:
        if processor.get('NAME', '') == 'transform_points_to_voxels':
            return int(processor.MAX_NUMBER_OF_VOXELS[mode])
    return None


def occupancy_and_sparsity(non_empty_voxels, total_voxels):
    occupancy = float(non_empty_voxels) / float(total_voxels) if total_voxels > 0 else 0.0
    return occupancy, 1.0 - occupancy


def max_chebyshev_distance(nx, ny, cx, cy):
    max_dx = max(abs(0 - cx), abs((int(nx) - 1) - cx))
    max_dy = max(abs(0 - cy), abs((int(ny) - 1) - cy))
    return int(max(max_dx, max_dy))


def lidar_center_for_stride(center_xy, stride):
    stride = int(stride)
    if stride < 1:
        raise ValueError(f'Invalid stride: {stride}')
    return int(center_xy[0]) // stride, int(center_xy[1]) // stride


def grid_xyz_from_spatial_shape(spatial_shape):
    nz, ny, nx = (int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2]))
    return nx, ny, nz


def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


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


def coords_from_sparse_tensor(sparse_tensor):
    indices = sparse_tensor.indices
    if hasattr(indices, 'detach'):
        indices = indices.detach().cpu().numpy()
    coords = normalize_voxel_coords(indices)
    if coords is None:
        return np.zeros((0, 3), dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError(f'Unexpected sparse indices shape: {getattr(coords, "shape", None)}')
    return np.unique(coords, axis=0)


def analyze_coords(
    coords,
    file_label,
    metadata,
    grid_size_xyz,
    lidar_center_xy,
    max_voxels=None,
    stage='',
    stride=1,
    spatial_shape_zyx=None,
):
    coords = normalize_voxel_coords(coords)
    if coords is None:
        raise RuntimeError(f'No voxel coordinates for {file_label} stage={stage}')

    nx, ny, nz = (int(grid_size_xyz[0]), int(grid_size_xyz[1]), int(grid_size_xyz[2]))
    total_voxels = int(nx) * int(ny) * int(nz)
    cx, cy = int(lidar_center_xy[0]), int(lidar_center_xy[1])
    bins_max = max_chebyshev_distance(nx, ny, cx, cy)
    if spatial_shape_zyx is None:
        spatial_shape_zyx = (nz, ny, nx)

    common = {
        'file': file_label,
        'stage': stage,
        'stride': int(stride),
        'spatial_shape_zyx': 'x'.join(str(int(v)) for v in spatial_shape_zyx),
        'lidar_center_x': cx,
        'lidar_center_y': cy,
        'data_loader': metadata.get('data_loader', ''),
        'fov_points_only': metadata.get('fov_points_only', ''),
        'data_mode': metadata.get('data_mode', ''),
        'max_voxels': '' if max_voxels is None else int(max_voxels),
        'total_voxels': int(total_voxels),
    }

    if coords.size == 0:
        occupancy, sparsity = occupancy_and_sparsity(0, total_voxels)
        common.update({
            'non_empty_voxels': 0,
            'voxel_occupancy': float(occupancy),
            'voxel_sparsity': float(sparsity),
            'chebyshev_counts': np.zeros(bins_max + 1, dtype=np.int64),
            'dropped_out_of_range': 0,
        })
        return common

    z_idx = coords[:, 0]
    y_idx = coords[:, 1]
    x_idx = coords[:, 2]
    non_empty_voxels = int(coords.shape[0])

    valid_mask = (z_idx >= 0) & (z_idx < nz) & (y_idx >= 0) & (y_idx < ny) & (x_idx >= 0) & (x_idx < nx)
    dropped_out_of_range = int((~valid_mask).sum())
    if dropped_out_of_range:
        z_idx = z_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        x_idx = x_idx[valid_mask]
        non_empty_voxels = int(z_idx.shape[0])

    occupancy, sparsity = occupancy_and_sparsity(non_empty_voxels, total_voxels)
    cheb = np.maximum(np.abs(x_idx - cx), np.abs(y_idx - cy)).astype(np.int64)
    cheb_counts = np.bincount(cheb, minlength=(bins_max + 1))
    if cheb_counts.size > bins_max + 1:
        extra = int(cheb_counts[bins_max + 1:].sum())
        raise RuntimeError(
            f'{file_label} stage={stage}: Chebyshev distance exceeded grid max {bins_max} ({extra} voxels)'
        )
    cheb_counts = cheb_counts[: bins_max + 1]

    common.update({
        'non_empty_voxels': int(non_empty_voxels),
        'voxel_occupancy': float(occupancy),
        'voxel_sparsity': float(sparsity),
        'chebyshev_counts': cheb_counts,
        'dropped_out_of_range': dropped_out_of_range,
    })
    return common


def _result_to_row(result, max_bins_seen):
    row = {key: result.get(key, '') for key in BASE_CSV_KEYS}
    counts = result.get('chebyshev_counts', np.zeros(0, dtype=np.int64))
    for i in range(max_bins_seen):
        row[f'dist_{i}'] = int(counts[i]) if i < counts.size else 0
    return row


def mean_summary_row(results, max_bins_seen):
    n_frames = len(results)
    if n_frames == 0:
        raise ValueError('Cannot write a mean row without per-frame results')

    sample = results[0]
    dist_sums = np.zeros(max_bins_seen, dtype=np.float64)
    for result in results:
        counts = result.get('chebyshev_counts', np.zeros(0, dtype=np.int64))
        n = min(max_bins_seen, int(counts.size))
        if n:
            dist_sums[:n] += counts[:n]

    occupancy = float(np.mean([r['voxel_occupancy'] for r in results]))
    sparsity = float(np.mean([r['voxel_sparsity'] for r in results]))
    non_empty_mean = float(np.mean([r['non_empty_voxels'] for r in results]))
    row = {
        'file': f'MEAN_{n_frames}_frames',
        'stage': sample.get('stage', ''),
        'stride': sample.get('stride', ''),
        'spatial_shape_zyx': sample.get('spatial_shape_zyx', ''),
        'lidar_center_x': sample.get('lidar_center_x', ''),
        'lidar_center_y': sample.get('lidar_center_y', ''),
        'data_loader': sample.get('data_loader', ''),
        'fov_points_only': sample.get('fov_points_only', ''),
        'data_mode': sample.get('data_mode', ''),
        'max_voxels': sample.get('max_voxels', ''),
        'total_voxels': sample.get('total_voxels', ''),
        'non_empty_voxels': f'{non_empty_mean:.4f}',
        'voxel_occupancy': occupancy,
        'voxel_sparsity': sparsity,
    }
    for i in range(max_bins_seen):
        row[f'dist_{i}'] = f'{dist_sums[i] / n_frames:.4f}'
    return row


def write_stage_csv(out_path, results):
    max_bins_seen = max((r['chebyshev_counts'].size for r in results), default=0)
    dist_cols = [f'dist_{i}' for i in range(max_bins_seen)]
    fieldnames = BASE_CSV_KEYS + dist_cols
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(_result_to_row(result, max_bins_seen))
        if results:
            writer.writerow(mean_summary_row(results, max_bins_seen))
    return out_path


def summarize_stage(stage_name, results, max_voxels):
    occupancies = [r['voxel_occupancy'] for r in results]
    sparsities = [r['voxel_sparsity'] for r in results]
    dropped = sum(int(r.get('dropped_out_of_range', 0)) for r in results)
    if not results:
        print(f'{stage_name}: no frames')
        return
    sample = results[0]
    extra = ''
    if stage_name == 'conv1' and max_voxels is not None:
        capped = sum(1 for r in results if int(r['non_empty_voxels']) == int(max_voxels))
        extra = f' Hit voxelizer max_voxels={max_voxels} on {capped}/{len(results)} frames.'
    print(
        f'{stage_name}: stride={sample["stride"]} spatial_shape={sample["spatial_shape_zyx"]} '
        f'lidar_center=({sample["lidar_center_x"]},{sample["lidar_center_y"]}) '
        f'mean occupancy={np.mean(occupancies):.6f} mean sparsity={np.mean(sparsities):.6f} '
        f'mean non_empty={np.mean([r["non_empty_voxels"] for r in results]):.1f}.{extra}'
    )
    if dropped:
        print(f'  dropped {dropped} out-of-range voxels')


def load_frame_batch(dataset, frame_id):
    sample, metadata = load_kitti_sample(dataset, frame_id)
    batch = dataset.collate_batch([sample])
    return batch, metadata


def extract_stage_coords(batch_dict, stage_spec, file_label, metadata, lidar_center_xy, max_voxels):
    features = batch_dict['multi_scale_3d_features']
    sparse_tensor = features[stage_spec['feature_key']]
    coords = coords_from_sparse_tensor(sparse_tensor)
    spatial_shape = [int(v) for v in sparse_tensor.spatial_shape]
    nx, ny, nz = grid_xyz_from_spatial_shape(spatial_shape)
    center = lidar_center_for_stride(lidar_center_xy, stage_spec['stride'])
    return analyze_coords(
        coords,
        file_label,
        metadata,
        grid_size_xyz=(nx, ny, nz),
        lidar_center_xy=center,
        max_voxels=max_voxels,
        stage=stage_spec['name'],
        stride=stage_spec['stride'],
        spatial_shape_zyx=spatial_shape,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Chebyshev-distance histograms at SECOND conv1/conv2/conv3/conv4. '
        'Locked to KITTI FOV inference input (FOV_POINTS_ONLY=True).'
    )
    parser.add_argument('--cfg', type=str, default='tools/cfgs/kitti_models/second.yaml')
    parser.add_argument('--ckpt', type=str, default='', help='Optional checkpoint; occupancy indices do not need weights')
    parser.add_argument('--velodyne_dir', type=str, default='data/kitti/training/velodyne')
    parser.add_argument('--list_file', type=str, default='draw/chebyshev_analyze/frame_list_200.txt')
    add_data_mode_args(parser, default_mode='kitti')
    parser.add_argument('--out_dir', type=str, default='draw/chebyshev_analyze', help='Directory for per-stage CSVs')
    parser.add_argument('--lidar_center', type=str, default='0,800', help='LiDAR voxel center at stride 1 as "x,y"')
    parser.add_argument('--device', type=str, default='auto', help='auto|cpu|cuda|cuda:0')
    args = parser.parse_args()

    project_root = resolve_project_root()
    cfg_local = load_model_cfg(args.cfg)
    data_mode = resolve_data_mode(cfg_local, args.data_mode)
    if data_mode != 'kitti':
        raise RuntimeError(
            'Chebyshev analysis is locked to --data_mode kitti (FOV_POINTS_ONLY=True) '
            'so results match tools/test.py and RTL golden. Do not use raw.'
        )
    fov = bool(get_dataset_cfg(cfg_local).get('FOV_POINTS_ONLY', False))
    if not fov:
        raise RuntimeError('FOV_POINTS_ONLY must be True in the dataset config')
    max_voxels = max_voxels_from_cfg(cfg_local, training=False)
    lc = parse_lidar_center(args.lidar_center)
    device = resolve_device(args.device)

    velodyne_dir = args.velodyne_dir if args.velodyne_dir is not None else os.path.join(cfg.ROOT_DIR, 'data', 'kitti', 'training', 'velodyne')
    if not os.path.exists(velodyne_dir):
        raise FileNotFoundError(f'velodyne dir not found: {velodyne_dir}')

    frame_ids = [normalize_frame_token(frame_id) for frame_id in choose_frame_ids(velodyne_dir, args.list_file)]
    if len(frame_ids) == 0:
        raise RuntimeError('No .bin files found to process')

    from pcdet.models import build_network
    from pcdet.utils import common_utils

    logger = common_utils.create_logger()
    dataset = build_kitti_dataset(cfg_local, Path(cfg.ROOT_DIR), args.kitti_root, logger)
    info_by_id = attach_trainval_infos(dataset)
    missing = [frame_id for frame_id in frame_ids if frame_id not in info_by_id]
    if missing:
        raise RuntimeError(f'Frames not found in KITTI train/val infos: {missing[:8]}')
    print(
        f'data_mode=kitti; FOV_POINTS_ONLY={fov}; infos=train+val; '
        f'max_voxels={max_voxels}; lidar_center={lc[0]},{lc[1]}; device={device}; '
        f'n_frames={len(frame_ids)}. Matches tools/test.py and RTL golden input.'
    )

    for spec in STAGE_SPECS:
        cx, cy = lidar_center_for_stride(lc, spec['stride'])
        print(f'  {spec["name"]}: stride={spec["stride"]} lidar_center=({cx},{cy})')

    model = build_network(model_cfg=cfg_local.MODEL, num_class=len(get_class_names(cfg_local)), dataset=dataset)
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if ckpt_path:
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
        if ckpt_path.exists():
            model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=(device.type == 'cpu'))
        else:
            print(f'Checkpoint not found, continue without weights: {ckpt_path}')
    model.to(device)
    model.eval()

    stage_results = {spec['name']: [] for spec in STAGE_SPECS}
    for frame_id in tqdm(frame_ids, desc='Processing'):
        batch, metadata = load_frame_batch(dataset, frame_id)
        batch_torch = move_batch_to_device(batch, device)
        with torch.no_grad():
            batch_torch = model.vfe(batch_torch)
            batch_torch = model.backbone_3d(batch_torch)
        file_label = f'{frame_id}.bin'
        for spec in STAGE_SPECS:
            stage_results[spec['name']].append(
                extract_stage_coords(batch_torch, spec, file_label, metadata, lc, max_voxels)
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in STAGE_SPECS:
        results = stage_results[spec['name']]
        out_path = out_dir / f'chebyshev_stats_{spec["name"]}.csv'
        write_stage_csv(out_path, results)
        summarize_stage(spec['name'], results, max_voxels)
        print(f'  wrote {out_path}')


if __name__ == '__main__':
    main()

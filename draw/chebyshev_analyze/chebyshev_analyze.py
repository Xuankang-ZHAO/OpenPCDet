#!/usr/bin/env python3
"""Compute Chebyshev-distance voxel statistics for many LiDAR frames.

This script follows the same pipeline used elsewhere in `mycode/`:
- by default it uses `KittiDataset.__getitem__()` so FOV filtering matches inference
- for each non-empty voxel we compute the Chebyshev distance in the XY
  plane to a provided `lidar_center` (voxel-space) and tally how many
  voxels fall into each Chebyshev distance bin.

Output CSV: one row per input .bin file. Columns include
`file,total_voxels,non_empty_voxels,voxel_sparsity` followed by
one column per Chebyshev distance: `dist_0,dist_1,...` up to the
maximum distance derived from the voxel coordinates (see code).

This file intentionally does NOT include block-partition statistics.
"""
import os
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mycode.kitti_frame_loader import (
    add_data_mode_args,
    build_kitti_dataset,
    choose_frame_ids,
    load_kitti_voxels,
    load_raw_voxels_via_data_processor,
    normalize_voxel_coords,
    resolve_data_mode,
)
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.processor.data_processor import DataProcessor


# load KITTI dataset config (same helper used across mycode)
def load_cfg_for_kitti():
    cfg_file = os.path.join(cfg.ROOT_DIR, 'tools', 'cfgs', 'dataset_configs', 'kitti_dataset.yaml')
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg


def analyze_coords(coords, file_label, metadata, data_proc, lidar_center_xy=(0, 0)):
    coords = normalize_voxel_coords(coords)
    if coords is None:
        return None

    if coords.size == 0:
        nx, ny, nz = int(data_proc.grid_size[0]), int(data_proc.grid_size[1]), int(data_proc.grid_size[2])
        total_voxels = nx * ny * nz
        return {
            'file': file_label,
            'data_loader': metadata.get('data_loader', ''),
            'fov_points_only': metadata.get('fov_points_only', ''),
            'data_mode': metadata.get('data_mode', ''),
            'total_voxels': int(total_voxels),
            'non_empty_voxels': 0,
            'voxel_sparsity': 0.0,
            'chebyshev_counts': np.zeros(1, dtype=np.int64),
        }

    z_idx = coords[:, 0]
    y_idx = coords[:, 1]
    x_idx = coords[:, 2]

    nx, ny, nz = int(data_proc.grid_size[0]), int(data_proc.grid_size[1]), int(data_proc.grid_size[2])
    total_voxels = int(nx) * int(ny) * int(nz)
    non_empty_voxels = coords.shape[0]
    voxel_sparsity = float(non_empty_voxels) / float(total_voxels) if total_voxels > 0 else 0.0

    valid_mask = (z_idx >= 0) & (z_idx < nz) & (y_idx >= 0) & (y_idx < ny) & (x_idx >= 0) & (x_idx < nx)
    if not np.all(valid_mask):
        z_idx = z_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        x_idx = x_idx[valid_mask]
        non_empty_voxels = z_idx.shape[0]
        voxel_sparsity = float(non_empty_voxels) / float(total_voxels) if total_voxels > 0 else 0.0

    cx, cy = int(lidar_center_xy[0]), int(lidar_center_xy[1])
    cheb = np.maximum(np.abs(x_idx - cx), np.abs(y_idx - cy)).astype(np.int64)
    bins_max = int(max(abs(int(nx) - cx), abs(int(ny) - cy)))
    cheb_counts = np.bincount(cheb, minlength=(bins_max + 1))

    return {
        'file': file_label,
        'data_loader': metadata.get('data_loader', ''),
        'fov_points_only': metadata.get('fov_points_only', ''),
        'data_mode': metadata.get('data_mode', ''),
        'total_voxels': int(total_voxels),
        'non_empty_voxels': int(non_empty_voxels),
        'voxel_sparsity': float(voxel_sparsity),
        'chebyshev_counts': cheb_counts,
    }


def analyze_file(bin_path, data_proc, lidar_center_xy=(0, 0)):
    coords, metadata = load_raw_voxels_via_data_processor(bin_path, data_proc)
    return analyze_coords(coords, os.path.basename(bin_path), metadata, data_proc, lidar_center_xy)


def main():
    parser = argparse.ArgumentParser(description='Compute Chebyshev-distance voxel histograms')
    parser.add_argument('--velodyne_dir', type=str, default='data/kitti/training/velodyne', help='Path to KITTI velodyne folder (bin files)')
    parser.add_argument('--list_file', type=str, default='data/kitti/ImageSets/analyze.txt', help='Optional frame id list (one id per line)')
    add_data_mode_args(parser)
    parser.add_argument('--out', type=str, default='mycode/output/chebyshev_stats.csv', help='CSV output file')
    parser.add_argument('--lidar_center', type=str, default='0,800', help='LiDAR voxel center as "x,y" (e.g. 0,800)')
    args = parser.parse_args()

    cfg_local = load_cfg_for_kitti()
    data_mode = resolve_data_mode(cfg_local, args.data_mode)

    velodyne_dir = args.velodyne_dir if args.velodyne_dir is not None else os.path.join(cfg.ROOT_DIR, 'data', 'kitti', 'training', 'velodyne')
    if not os.path.exists(velodyne_dir):
        raise FileNotFoundError(f'velodyne dir not found: {velodyne_dir}')

    frame_ids = choose_frame_ids(velodyne_dir, args.list_file)
    if len(frame_ids) == 0:
        raise RuntimeError('No .bin files found to process')

    num_point_features = len(cfg_local.POINT_FEATURE_ENCODING.used_feature_list) if 'POINT_FEATURE_ENCODING' in cfg_local else 4
    data_proc = DataProcessor(processor_configs=cfg_local.DATA_PROCESSOR,
                              point_cloud_range=np.array(cfg_local.POINT_CLOUD_RANGE),
                              training=False,
                              num_point_features=num_point_features)

    kitti_dataset = None
    if data_mode == 'kitti':
        from pcdet.utils import common_utils

        logger = common_utils.create_logger()
        kitti_dataset = build_kitti_dataset(cfg_local, Path(cfg.ROOT_DIR), args.kitti_root, logger)

    lc = tuple(int(x) for x in args.lidar_center.split(',')) if args.lidar_center is not None else (0, 0)

    results = []
    max_bins_seen = 0
    for frame_id in tqdm(frame_ids, desc='Processing'):
        if data_mode == 'kitti':
            coords, metadata = load_kitti_voxels(kitti_dataset, frame_id)
            res = analyze_coords(coords, f'{frame_id}.bin', metadata, data_proc, lidar_center_xy=lc)
        else:
            bin_path = os.path.join(velodyne_dir, frame_id + '.bin')
            res = analyze_file(bin_path, data_proc, lidar_center_xy=lc)
        if res is None:
            continue
        results.append(res)
        max_bins_seen = max(max_bins_seen, res['chebyshev_counts'].size)

    base_keys = ['file', 'data_loader', 'fov_points_only', 'data_mode', 'total_voxels', 'non_empty_voxels', 'voxel_sparsity']
    dist_cols = [f'dist_{i}' for i in range(max_bins_seen)]
    fieldnames = base_keys + dist_cols

    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) != '' else None
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in base_keys}
            counts = r.get('chebyshev_counts', np.zeros(0, dtype=np.int64))
            for i in range(max_bins_seen):
                row[f'dist_{i}'] = int(counts[i]) if i < counts.size else 0
            writer.writerow(row)

    sparsities = [r['voxel_sparsity'] for r in results]
    if len(sparsities) > 0:
        print(f"Processed {len(results)} files. Mean voxel_sparsity: {np.mean(sparsities):.6f}, median: {np.median(sparsities):.6f}")
    else:
        print('No valid frames processed.')


if __name__ == '__main__':
    main()

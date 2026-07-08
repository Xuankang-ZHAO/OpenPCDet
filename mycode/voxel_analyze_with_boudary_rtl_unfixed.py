#!/usr/bin/env python3
"""Analyze voxel sparsity with RTL-style unfixed zone block partitioning.

This script keeps the same voxelization pipeline, parser style, default KITTI
data paths, and CSV statistics as `voxel_analyze_with_boudary_v2.py`, but uses
the RTL-aligned unfixed partition semantics described in
`online_block_partitioning_algorithm_summary.md`.

Run directly in the OpenPCDet folder: (openpcd) vipuser@ubuntu22:~/桌面/OpenPCDet$ python mycode/voxel_analyze_with_boudary_rtl_unfixed.py
"""

import argparse
import csv
import os
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
from mycode.rtl_unfixed_block_partition import (
    compute_rtl_unfixed_partition_counts,
    load_zone_specs,
)
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.processor.data_processor import DataProcessor


def load_cfg_for_kitti():
    cfg_file = os.path.join(cfg.ROOT_DIR, 'tools', 'cfgs', 'dataset_configs', 'kitti_dataset.yaml')
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg


def analyze_coords(coords, file_label, metadata, data_proc, zone_specs, lidar_center_xy):
    coords = normalize_voxel_coords(coords)
    if coords is None:
        return None

    nx, ny, nz = (int(data_proc.grid_size[0]), int(data_proc.grid_size[1]), int(data_proc.grid_size[2]))
    total_voxels = nx * ny * nz
    non_empty_voxels = int(coords.shape[0])
    voxel_sparsity = non_empty_voxels / total_voxels

    counts, total_blocks, block_voxel_limit = compute_rtl_unfixed_partition_counts(
        coords,
        (nx, ny, nz),
        zone_specs,
        lidar_center_xy,
    )

    max_voxels_in_block = int(counts.max()) if counts.size > 0 else 0
    mean_voxels_per_valid_block = float(counts[counts > 0].mean()) if counts.size > 0 else 0.0
    empty_blocks = int(np.sum(counts == 0)) if counts.size > 0 else 0
    nonempty_blocks = int(np.sum(counts > 0)) if counts.size > 0 else 0
    empty_fraction = float(empty_blocks / total_blocks) if total_blocks > 0 else 0.0

    result = {
        'file': file_label,
        'data_loader': metadata.get('data_loader', ''),
        'fov_points_only': metadata.get('fov_points_only', ''),
        'data_mode': metadata.get('data_mode', ''),
        'total_voxels': int(total_voxels),
        'non_empty_voxels': int(non_empty_voxels),
        'voxel_sparsity': float(voxel_sparsity),
        'block_voxel_limit': int(block_voxel_limit),
        'blocks_total': int(total_blocks),
        'blocks_empty': empty_blocks,
        'blocks_nonempty': nonempty_blocks,
        'blocks_fraction_empty': empty_fraction,
        'blocks_max_voxels': int(max_voxels_in_block),
        'blocks_mean_voxels_per_block': float(mean_voxels_per_valid_block),
    }

    hist = np.bincount(counts) if counts.size > 0 else np.zeros(0, dtype=np.int64)
    result['block_count_hist'] = ';'.join(f'{index}:{int(value)}' for index, value in enumerate(hist) if value > 0)

    nonempty_block_counts = counts[counts > 0]
    result['blocks_nonempty_voxel_counts_list'] = [int(value) for value in nonempty_block_counts.tolist()]
    return result


def analyze_file(bin_path, data_proc, zone_specs, lidar_center_xy):
    coords, metadata = load_raw_voxels_via_data_processor(bin_path, data_proc)
    return analyze_coords(coords, os.path.basename(bin_path), metadata, data_proc, zone_specs, lidar_center_xy)


def write_results_csv(results, output_path):
    base_keys = [
        'file',
        'data_loader',
        'fov_points_only',
        'data_mode',
        'total_voxels',
        'non_empty_voxels',
        'voxel_sparsity',
        'block_voxel_limit',
        'blocks_total',
        'blocks_empty',
        'blocks_nonempty',
        'blocks_fraction_empty',
        'blocks_max_voxels',
        'blocks_mean_voxels_per_block',
        'block_count_hist',
    ]

    max_nonempty = 0
    for result in results:
        max_nonempty = max(max_nonempty, len(result.get('blocks_nonempty_voxel_counts_list', [])))

    block_cols = [f'Nonempty_block{index}' for index in range(max_nonempty)]
    fieldnames = base_keys + block_cols

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key, '') for key in base_keys}
            block_values = result.get('blocks_nonempty_voxel_counts_list', [])
            for index, column_name in enumerate(block_cols):
                row[column_name] = block_values[index] if index < len(block_values) else ''
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--velodyne_dir', type=str, default='data/kitti/training/velodyne', help='Path to KITTI velodyne folder (bin files)')
    parser.add_argument('--list_file', type=str, default='data/kitti/ImageSets/trainval.txt', help='Optional frame id list (one id per line); pass an empty string to scan the directory directly')
    add_data_mode_args(parser)
    parser.add_argument('--out', type=str, default='mycode/output/block_v2_rtl_unfixed_zone4.csv', help='CSV output file')
    parser.add_argument('--zone_lut', type=str, default='mycode/block_size_lut_rtl_unfixed.txt', help='Path to zone block-size LUT for RTL unfixed partition')
    parser.add_argument('--lidar_center', type=str, default='0,800', help='LiDAR voxel center as "x,y" for zone lookup (e.g. 0,800 for KITTI)')
    parser.add_argument('--max_files', type=int, default=200, help='Optional limit for the number of frames to process')
    args = parser.parse_args()

    if args.list_file == '':
        args.list_file = None

    cfg_local = load_cfg_for_kitti()
    data_mode = resolve_data_mode(cfg_local, args.data_mode)

    if not os.path.exists(args.velodyne_dir):
        raise FileNotFoundError(f'velodyne dir not found: {args.velodyne_dir}')
    if args.list_file is not None and not os.path.exists(args.list_file):
        raise FileNotFoundError(f'list file not found: {args.list_file}')

    frame_ids = choose_frame_ids(args.velodyne_dir, args.list_file)
    if args.max_files is not None:
        frame_ids = frame_ids[:args.max_files]
    if not frame_ids:
        raise RuntimeError('No .bin files found to process')

    lidar_center_xy = tuple(int(value) for value in args.lidar_center.split(','))
    if len(lidar_center_xy) != 2:
        raise ValueError('lidar_center must contain exactly two integers: x,y')

    num_point_features = len(cfg_local.POINT_FEATURE_ENCODING.used_feature_list) if 'POINT_FEATURE_ENCODING' in cfg_local else 4
    data_proc = DataProcessor(
        processor_configs=cfg_local.DATA_PROCESSOR,
        point_cloud_range=np.array(cfg_local.POINT_CLOUD_RANGE),
        training=False,
        num_point_features=num_point_features,
    )

    kitti_dataset = None
    if data_mode == 'kitti':
        from pcdet.utils import common_utils

        logger = common_utils.create_logger()
        kitti_dataset = build_kitti_dataset(cfg_local, Path(cfg.ROOT_DIR), args.kitti_root, logger)

    grid_size = (int(data_proc.grid_size[0]), int(data_proc.grid_size[1]), int(data_proc.grid_size[2]))
    zone_specs = load_zone_specs(args.zone_lut, grid_size, lidar_center_xy)

    results = []
    for frame_id in tqdm(frame_ids, desc='Processing'):
        if data_mode == 'kitti':
            coords, metadata = load_kitti_voxels(kitti_dataset, frame_id)
            result = analyze_coords(coords, f'{frame_id}.bin', metadata, data_proc, zone_specs, lidar_center_xy)
        else:
            frame_path = os.path.join(args.velodyne_dir, frame_id + '.bin')
            result = analyze_file(frame_path, data_proc, zone_specs, lidar_center_xy)
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError('No valid voxel analysis results were produced')

    write_results_csv(results, args.out)

    sparsities = [result['voxel_sparsity'] for result in results]
    print(
        f'Processed {len(results)} files. '
        f'Mean voxel_sparsity: {np.mean(sparsities):.6f}, median: {np.median(sparsities):.6f}'
    )


if __name__ == '__main__':
    main()
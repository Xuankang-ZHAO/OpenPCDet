#!/usr/bin/env python3
"""Analyze voxel sparsity with a fixed block size and no zone partition.

The voxel grid is tiled from origin (0, 0, 0) with one XYZ block size
(default 10x10x6). There is no Chebyshev distance and no zone LUT.
Boundary voxels still emit RTL-style halo copies into neighbor blocks.

Run directly in the OpenPCDet folder:
python mycode/rtl_fixed/voxel_analyze.py
"""

import argparse
import csv
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

from mycode.kitti_frame_loader import (
    add_data_mode_args,
    build_kitti_dataset,
    choose_frame_ids,
    get_dataset_cfg,
    load_kitti_voxels,
    load_raw_voxels_via_data_processor,
    normalize_voxel_coords,
    resolve_data_mode,
)
from mycode.rtl_fixed.partition import compute_rtl_fixed_partition_counts
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.processor.data_processor import DataProcessor


def load_cfg_for_kitti():
    cfg_file = os.path.join(cfg.ROOT_DIR, 'tools', 'cfgs', 'dataset_configs', 'kitti_dataset.yaml')
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg


def analyze_coords(coords, file_label, metadata, data_proc, block_size_xyz):
    coords = normalize_voxel_coords(coords)
    if coords is None:
        return None

    nx, ny, nz = (int(data_proc.grid_size[0]), int(data_proc.grid_size[1]), int(data_proc.grid_size[2]))
    total_voxels = nx * ny * nz
    non_empty_voxels = int(coords.shape[0])
    voxel_sparsity = non_empty_voxels / total_voxels

    counts, total_blocks, block_voxel_limit = compute_rtl_fixed_partition_counts(
        coords,
        (nx, ny, nz),
        block_size_xyz,
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
        'block_size_x': int(block_size_xyz[0]),
        'block_size_y': int(block_size_xyz[1]),
        'block_size_z': int(block_size_xyz[2]),
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


def analyze_file(bin_path, data_proc, block_size_xyz):
    coords, metadata = load_raw_voxels_via_data_processor(bin_path, data_proc)
    return analyze_coords(coords, os.path.basename(bin_path), metadata, data_proc, block_size_xyz)


def write_results_csv(results, output_path):
    base_keys = [
        'file',
        'data_loader',
        'fov_points_only',
        'data_mode',
        'block_size_x',
        'block_size_y',
        'block_size_z',
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
    parser.add_argument('--out', type=str, default='mycode/rtl_fixed/block_rtl_fixed_10x10x6.csv', help='CSV output file')
    parser.add_argument('--block_size_x', type=int, default=10, help='Fixed block size along X in voxels')
    parser.add_argument('--block_size_y', type=int, default=10, help='Fixed block size along Y in voxels')
    parser.add_argument('--block_size_z', type=int, default=6, help='Fixed block size along Z in voxels')
    parser.add_argument('--max_files', type=int, default=200, help='Optional limit for the number of frames to process')
    args = parser.parse_args()

    if args.list_file == '':
        args.list_file = None

    if min(args.block_size_x, args.block_size_y, args.block_size_z) <= 0:
        raise ValueError('block sizes must be positive')

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

        dataset_cfg = get_dataset_cfg(cfg_local)
        if dataset_cfg.get('INFO_PATH', None) is not None:
            dataset_cfg.INFO_PATH['test'] = ['kitti_infos_trainval.pkl']

        logger = common_utils.create_logger()
        kitti_dataset = build_kitti_dataset(cfg_local, Path(cfg.ROOT_DIR), args.kitti_root, logger)

    block_size_xyz = (args.block_size_x, args.block_size_y, args.block_size_z)

    results = []
    for frame_id in tqdm(frame_ids, desc='Processing'):
        if data_mode == 'kitti':
            coords, metadata = load_kitti_voxels(kitti_dataset, frame_id)
            result = analyze_coords(coords, f'{frame_id}.bin', metadata, data_proc, block_size_xyz)
        else:
            frame_path = os.path.join(args.velodyne_dir, frame_id + '.bin')
            result = analyze_file(frame_path, data_proc, block_size_xyz)
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError('No valid voxel analysis results were produced')

    write_results_csv(results, args.out)

    sparsities = [result['voxel_sparsity'] for result in results]
    print(
        f'Processed {len(results)} files. '
        f'block={block_size_xyz[0]}x{block_size_xyz[1]}x{block_size_xyz[2]}. '
        f'Mean voxel_sparsity: {np.mean(sparsities):.6f}, median: {np.median(sparsities):.6f}'
    )


if __name__ == '__main__':
    main()

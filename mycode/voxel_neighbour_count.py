#!/usr/bin/env python3
"""Compute 3x3x3 neighbor-count distribution for voxels after voxelization.

For each non-empty voxel, count how many non-empty voxels are present in the
3x3x3 neighborhood centered on it (including itself). Print overall total
non-empty voxels and the histogram of neighbor counts (1..27).
"""
import os
import argparse
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


# load KITTI dataset config
def load_cfg_for_kitti():
    cfg_file = os.path.join(cfg.ROOT_DIR, 'tools', 'cfgs', 'dataset_configs', 'kitti_dataset.yaml')
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg


def analyze_coords(coords, file_label, metadata, data_proc):
    coords = normalize_voxel_coords(coords)
    if coords is None:
        return None

    # coords expected as [z, y, x]
    z_idx = coords[:, 0]
    y_idx = coords[:, 1]
    x_idx = coords[:, 2]

    # grid_size in data_proc is [nx, ny, nz] (x,y,z). We build occupancy as (nz, ny, nx)
    nx, ny, nz = int(data_proc.grid_size[0]), int(data_proc.grid_size[1]), int(data_proc.grid_size[2])

    non_empty_voxels = coords.shape[0]

    # occupancy indexed as occupancy[z, y, x]
    occupancy = np.zeros((nz, ny, nx), dtype=np.uint8)
    # Clip indices just in case
    valid_mask = (z_idx >= 0) & (z_idx < nz) & (y_idx >= 0) & (y_idx < ny) & (x_idx >= 0) & (x_idx < nx)
    if not np.all(valid_mask):
        z_idx = z_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        x_idx = x_idx[valid_mask]
        non_empty_voxels = z_idx.shape[0]

    occupancy[z_idx, y_idx, x_idx] = 1

    # pad by 1 to allow 3x3x3 centered slices without boundary checks
    occ_p = np.pad(occupancy, pad_width=1, mode='constant', constant_values=0)

    # For each voxel compute sum over 3x3x3 slice in padded array
    neigh_counts = np.empty(non_empty_voxels, dtype=np.int32)
    for i in range(non_empty_voxels):
        z = int(z_idx[i])
        y = int(y_idx[i])
        x = int(x_idx[i])
        # in padded array the voxel is at (z+1, y+1, x+1); slice [z:z+3) covers -1..+1
        s = occ_p[z:(z + 3), y:(y + 3), x:(x + 3)]
        neigh_counts[i] = int(s.sum())

    # histogram for counts 0..27 (we will ignore 0)
    hist = np.bincount(neigh_counts, minlength=28)

    return {
        'file': file_label,
        'data_loader': metadata.get('data_loader', ''),
        'fov_points_only': metadata.get('fov_points_only', ''),
        'data_mode': metadata.get('data_mode', ''),
        'non_empty_voxels': int(non_empty_voxels),
        'neigh_hist': hist,
    }


def analyze_file(bin_path, data_proc):
    coords, metadata = load_raw_voxels_via_data_processor(bin_path, data_proc)
    return analyze_coords(coords, os.path.basename(bin_path), metadata, data_proc)


def main():
    parser = argparse.ArgumentParser(description='Compute 3x3x3 neighbor-count distribution')
    parser.add_argument('--velodyne_dir', type=str, default='data/kitti/training/velodyne', help='Path to KITTI velodyne folder (bin files)')
    parser.add_argument('--list_file', type=str, default='data/kitti/ImageSets/analyze.txt', help='Optional frame id list (one id per line)')
    add_data_mode_args(parser)
    args = parser.parse_args()

    cfg_local = load_cfg_for_kitti()
    data_mode = resolve_data_mode(cfg_local, args.data_mode)

    if args.velodyne_dir is None:
        velodyne_dir = os.path.join(cfg.ROOT_DIR, 'data', 'kitti', 'training', 'velodyne')
    else:
        velodyne_dir = args.velodyne_dir

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

    # Print a neat table: header once, then one row per file
    fname_w = 24
    count_w = 12
    col_w = 6
    header = f"{'file':{fname_w}} {'有效体素数':>{count_w}}"
    for k in range(1, 28):
        header += f" {k:>{col_w}}"
    print(f'data_mode={data_mode} data_loader={"KittiDataset.__getitem__" if data_mode == "kitti" else "DataProcessor(raw_points)"}')
    print(header)

    for frame_id in tqdm(frame_ids, desc='Processing'):
        if data_mode == 'kitti':
            coords, metadata = load_kitti_voxels(kitti_dataset, frame_id)
            res = analyze_coords(coords, f'{frame_id}.bin', metadata, data_proc)
        else:
            bin_path = os.path.join(velodyne_dir, frame_id + '.bin')
            res = analyze_file(bin_path, data_proc)
        if res is None:
            continue
        row = f"{res['file'][:fname_w]:{fname_w}} {res['non_empty_voxels']:>{count_w}d}"
        for k in range(1, 28):
            row += f" {int(res['neigh_hist'][k]):>{col_w}d}"
        print(row)


if __name__ == '__main__':
    main()

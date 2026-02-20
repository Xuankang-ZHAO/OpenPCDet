#!/usr/bin/env python3
"""Compute 3x3x3 neighbor-count distribution for voxels after voxelization.

For each non-empty voxel, count how many non-empty voxels are present in the
3x3x3 neighborhood centered on it (including itself). Print overall total
non-empty voxels and the histogram of neighbor counts (1..27).
"""
import os
import argparse
import numpy as np
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.processor.data_processor import DataProcessor


# load KITTI dataset config
def load_cfg_for_kitti():
    cfg_file = os.path.join(cfg.ROOT_DIR, 'tools', 'cfgs', 'dataset_configs', 'kitti_dataset.yaml')
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg


def find_bin_files(velodyne_dir, list_file=None):
    if list_file is not None and os.path.exists(list_file):
        with open(list_file, 'r') as f:
            ids = [l.strip() for l in f.readlines() if l.strip()]
        paths = [os.path.join(velodyne_dir, x + '.bin') for x in ids]
        paths = [p for p in paths if os.path.exists(p)]
    else:
        paths = sorted([os.path.join(velodyne_dir, f) for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
    return paths


def analyze_file(bin_path, data_proc):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    data_dict = {'points': points, 'use_lead_xyz': True}

    # Apply data processing pipeline to generate voxels
    for proc in data_proc.data_processor_queue:
        data_dict = proc(data_dict)

    coords = data_dict.get('voxel_coords', None)
    if coords is None:
        return None

    if isinstance(coords, list):
        coords = coords[0]

    if hasattr(coords, 'cpu'):
        coords = coords.cpu().numpy()

    coords = coords.astype(np.int64)
    if coords.ndim == 2 and coords.shape[1] == 4:
        # format: [batch_idx, z, y, x]
        coords = coords[:, 1:4]

    # coords expected as [z, y, x]
    z_idx = coords[:, 0]
    y_idx = coords[:, 1]
    x_idx = coords[:, 2]

    # grid_size in data_proc is [nx, ny, nz] (x,y,z). We build occupancy as (nz, ny, nx)
    nx, ny, nz = int(data_proc.grid_size[0]), int(data_proc.grid_size[1]), int(data_proc.grid_size[2])
    total_voxels = nx * ny * nz

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

    return {'file': os.path.basename(bin_path), 'non_empty_voxels': int(non_empty_voxels), 'neigh_hist': hist}


def main():
    parser = argparse.ArgumentParser(description='Compute 3x3x3 neighbor-count distribution')
    parser.add_argument('--velodyne_dir', type=str, default='data/kitti/training/velodyne', help='Path to KITTI velodyne folder (bin files)')
    parser.add_argument('--list_file', type=str, default='data/kitti/ImageSets/train.txt', help='Optional frame id list (one id per line)')
    args = parser.parse_args()

    cfg_local = load_cfg_for_kitti()

    if args.velodyne_dir is None:
        velodyne_dir = os.path.join(cfg.ROOT_DIR, 'data', 'kitti', 'training', 'velodyne')
    else:
        velodyne_dir = args.velodyne_dir

    if not os.path.exists(velodyne_dir):
        raise FileNotFoundError(f'velodyne dir not found: {velodyne_dir}')

    paths = find_bin_files(velodyne_dir, args.list_file)
    if len(paths) == 0:
        raise RuntimeError('No .bin files found to process')

    num_point_features = len(cfg_local.POINT_FEATURE_ENCODING.used_feature_list) if 'POINT_FEATURE_ENCODING' in cfg_local else 4
    data_proc = DataProcessor(processor_configs=cfg_local.DATA_PROCESSOR,
                              point_cloud_range=np.array(cfg_local.POINT_CLOUD_RANGE),
                              training=False,
                              num_point_features=num_point_features)

    # Print a neat table: header once, then one row per file
    fname_w = 24
    count_w = 12
    col_w = 6
    header = f"{'file':{fname_w}} {'有效体素数':>{count_w}}"
    for k in range(1, 28):
        header += f" {k:>{col_w}}"
    print(header)

    for p in tqdm(paths, desc='Processing'):
        res = analyze_file(p, data_proc)
        if res is None:
            continue
        row = f"{res['file'][:fname_w]:{fname_w}} {res['non_empty_voxels']:>{count_w}d}"
        for k in range(1, 28):
            row += f" {int(res['neigh_hist'][k]):>{col_w}d}"
        print(row)


if __name__ == '__main__':
    main()

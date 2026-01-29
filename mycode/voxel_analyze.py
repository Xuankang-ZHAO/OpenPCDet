#!/usr/bin/env python3
"""Analyze voxel sparsity and block partition sparsity for KITTI point clouds.

Usage examples:
  python mycode/voxel_analyze.py --velodyne_dir /path/to/kitti/velodyne --out stats.csv

The script hooks into OpenPCDet's DataProcessor transform pipeline to reproduce
the same voxelization used during inference (spconv VoxelGenerator wrapper).
"""
import os
import math
import argparse
import numpy as np
import csv

# progress bar
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.processor.data_processor import DataProcessor


def load_cfg_for_kitti():
    # 拼接得到完整的kitti数据集配置文件路径
    cfg_file = os.path.join(cfg.ROOT_DIR, 'tools', 'cfgs', 'dataset_configs', 'kitti_dataset.yaml')
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg


def find_bin_files(velodyne_dir, list_file=None):
    if list_file is not None:
        with open(list_file, 'r') as f:
            ids = [l.strip() for l in f.readlines() if l.strip()]
        paths = [os.path.join(velodyne_dir, x + '.bin') for x in ids]
        paths = [p for p in paths if os.path.exists(p)]
    else:
        paths = sorted([os.path.join(velodyne_dir, f) for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
    return paths


def analyze_file(bin_path, data_proc, block_size_xyz):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    data_dict = {'points': points, 'use_lead_xyz': True}

    # run the configured processors (this mirrors dataset pipeline)
    for proc in data_proc.data_processor_queue:
        data_dict = proc(data_dict)

    coords = data_dict.get('voxel_coords', None)
    if coords is None:
        return None

    # coords could be list (double flip) or numpy/torch array
    if isinstance(coords, list):
        coords = coords[0]

    if hasattr(coords, 'cpu'):
        coords = coords.cpu().numpy()

    coords = coords.astype(np.int64)
    if coords.ndim == 2 and coords.shape[1] == 4:
        # format: [batch_idx, z, y, x]
        coords = coords[:, 1:4]

    # coords expected [z, y, x]
    z_idx = coords[:, 0]
    y_idx = coords[:, 1]
    x_idx = coords[:, 2]

    nx, ny, nz = data_proc.grid_size[0], data_proc.grid_size[1], data_proc.grid_size[2]
    total_voxels = int(nx) * int(ny) * int(nz)
    non_empty = coords.shape[0]
    sparsity = non_empty / total_voxels

    # Block partition in x/y/z (tile over all three dims)
    bx, by, bz = int(block_size_xyz[0]), int(block_size_xyz[1]), int(block_size_xyz[2])
    num_blocks_x = math.ceil(nx / bx)
    num_blocks_y = math.ceil(ny / by)
    num_blocks_z = math.ceil(nz / bz)
    total_blocks = num_blocks_x * num_blocks_y * num_blocks_z

    block_x = (x_idx // bx).astype(np.int64)
    block_y = (y_idx // by).astype(np.int64)
    block_z = (z_idx // bz).astype(np.int64)
    block_ids = block_x + block_y * num_blocks_x + block_z * (num_blocks_x * num_blocks_y)

    counts = np.bincount(block_ids, minlength=total_blocks)
    voxels_per_block = int(bx) * int(by) * int(bz)
    occupancy = counts / voxels_per_block

    result = {
        'file': os.path.basename(bin_path),
        'non_empty_voxels': int(non_empty),
        'total_voxels': int(total_voxels),
        'sparsity': float(sparsity),
        'blocks_total': int(total_blocks),
        'blocks_empty': int(np.sum(counts == 0)),
        'blocks_nonempty': int(np.sum(counts > 0)),
        'blocks_mean_occupied_voxels': float(counts.mean()),
        'blocks_median_occupied_voxels': float(np.median(counts)),
        'blocks_mean_occupancy': float(occupancy.mean()),
        'blocks_fraction_empty': float(np.sum(counts == 0) / total_blocks),
    }

    # optional distribution summary
    hist = np.bincount(counts)
    # return also hist as a compact string
    result['block_count_hist'] = ';'.join([f"{i}:{int(v)}" for i, v in enumerate(hist) if v > 0])
    return result


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--velodyne_dir', type=str, default=None, help='Path to KITTI velodyne folder (bin files)')
    parser.add_argument('--list_file', type=str, default=None, help='Optional frame id list (one id per line)')
    parser.add_argument('--out', type=str, default='voxel_stats.csv', help='CSV output file')
    parser.add_argument('--block_size', type=int, default=16, help='Fallback single block size for all dims')
    parser.add_argument('--block_size_x', type=int, default=None, help='Block size in voxels along X (optional)')
    parser.add_argument('--block_size_y', type=int, default=None, help='Block size in voxels along Y (optional)')
    parser.add_argument('--block_size_z', type=int, default=None, help='Block size in voxels along Z (optional)')
    args = parser.parse_args()

    cfg_local = load_cfg_for_kitti()

    if args.velodyne_dir is None:
        default_dir = os.path.join(cfg.ROOT_DIR, 'data', 'kitti', 'velodyne')
        velodyne_dir = default_dir
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

    results = []
    # determine block sizes per-dimension (x,y,z)
    bx = args.block_size_x if args.block_size_x is not None else args.block_size
    by = args.block_size_y if args.block_size_y is not None else args.block_size
    bz = args.block_size_z if args.block_size_z is not None else args.block_size

    for p in tqdm(paths, desc='Processing'):
        res = analyze_file(p, data_proc, (bx, by, bz))
        if res is not None:
            results.append(res)

    # write CSV
    keys = ['file','non_empty_voxels','total_voxels','sparsity','blocks_total','blocks_empty','blocks_nonempty',
            'blocks_mean_occupied_voxels','blocks_median_occupied_voxels','blocks_mean_occupancy','blocks_fraction_empty','block_count_hist']
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in keys})

    # simple aggregate print
    sparsities = [r['sparsity'] for r in results]
    print(f"Processed {len(results)} files. Mean sparsity: {np.mean(sparsities):.6f}, median: {np.median(sparsities):.6f}")


if __name__ == '__main__':
    main()

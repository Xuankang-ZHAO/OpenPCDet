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

# load KITTI dataset config
def load_cfg_for_kitti():
    # 拼接得到完整的kitti数据集配置文件路径
    cfg_file = os.path.join(cfg.ROOT_DIR, 'tools', 'cfgs', 'dataset_configs', 'kitti_dataset.yaml')
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg

# Find all .bin files in velodyne_dir, optionally filtering by list_file
def find_bin_files(velodyne_dir, list_file=None):
    if list_file is not None:
        with open(list_file, 'r') as f:
            ids = [l.strip() for l in f.readlines() if l.strip()]
        paths = [os.path.join(velodyne_dir, x + '.bin') for x in ids]
        paths = [p for p in paths if os.path.exists(p)]
    else:
        paths = sorted([os.path.join(velodyne_dir, f) for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
    return paths

# Analyze a single .bin file and return voxel/block statistics
#   bin_path: path to .bin file
#   data_proc: configured DataProcessor instance
#   block_size_xyz: tuple of (block_size_x, block_size_y, block_size_z
def analyze_file(bin_path, data_proc, block_size_xyz):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # print(points.shape)
    data_dict = {'points': points, 'use_lead_xyz': True}

    # Apply data processing pipeline to generate voxels
    for proc in data_proc.data_processor_queue:
        data_dict = proc(data_dict)

    # Extract voxel coordinates
    coords = data_dict.get('voxel_coords', None)
    if coords is None:
        return None

    # Double-check the format of coords, actually coords is not a list in this case
    if isinstance(coords, list):
        coords = coords[0]
        print('Warning: voxel_coords is a list, using first element.')
        
    # Convert to numpy if needed, actually coords has no 'cpu' attribute and it is already numpy array in this case 
    if hasattr(coords, 'cpu'):
        coords = coords.cpu().numpy()
        print('Message: voxel_coords is a tensor, converting to numpy.')

    # Ensure integer type
    coords = coords.astype(np.int64)
    if coords.ndim == 2 and coords.shape[1] == 4:
        # format: [batch_idx, z, y, x]
        coords = coords[:, 1:4]
        print('Message: voxel_coords has batch dimension, removing it.')

    # Coords expected [z, y, x]
    z_idx = coords[:, 0]
    y_idx = coords[:, 1]
    x_idx = coords[:, 2]
    # Debug print, the result is 1406, 1358, 39 for kitti 000001.bin, which is correct for x, y, z sequence
    # print(max(x_idx), max(y_idx), max(z_idx))

    # Compute voxel sparsity
    nx, ny, nz = data_proc.grid_size[0], data_proc.grid_size[1], data_proc.grid_size[2]
    total_voxels = int(nx) * int(ny) * int(nz)
    non_empty_voxels = coords.shape[0]
    voxel_sparsity = non_empty_voxels / total_voxels

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
    block_voxel_limit = int(bx) * int(by) * int(bz)


    max_voxels_in_block = int(counts.max()) if counts.size > 0 else 0
    mean_voxels_per_valid_block = float(counts[counts>0].mean()) if counts.size > 0 else 0.0

    result = {
        'file': os.path.basename(bin_path),
        'total_voxels': int(total_voxels),
        'non_empty_voxels': int(non_empty_voxels),
        'voxel_sparsity': float(voxel_sparsity),
        'block_voxel_limit': int(block_voxel_limit),
        'blocks_total': int(total_blocks),
        'blocks_empty': int(np.sum(counts == 0)),
        'blocks_nonempty': int(np.sum(counts > 0)),
        'blocks_fraction_empty': float(np.sum(counts == 0) / total_blocks),
        'blocks_max_voxels': int(max_voxels_in_block),
        'blocks_mean_voxels_per_block': float(mean_voxels_per_valid_block),
    }

    # Histogram of block voxel counts, e.g. "0:100;1:50;2:20" means 100 blocks with 0 voxels, 50 blocks with 1 voxel, 20 blocks with 2 voxels
    hist = np.bincount(counts)
    # Return hist as a compact string
    result['block_count_hist'] = ';'.join([f"{i}:{int(v)}" for i, v in enumerate(hist) if v > 0])

    # List non-empty block voxel counts (one value per non-empty block)
    # print('counts shape:', counts.shape)
    nonempty_block_counts = counts[counts > 0]
    # print('nonempty_block_counts shape:', nonempty_block_counts.shape)
    if nonempty_block_counts.size > 0:
        # keep original order (by block id); keep as list of ints
        nonempty_block_counts_list = [int(x) for x in nonempty_block_counts.tolist()]
    else:
        nonempty_block_counts_list = []
    result['blocks_nonempty_voxel_counts_list'] = nonempty_block_counts_list
    return result

# Main function to parse arguments and run analysis
def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--velodyne_dir', type=str, default='data/kitti/training/velodyne', help='Path to KITTI velodyne folder (bin files)')
    parser.add_argument('--list_file', type=str, default='data/kitti/ImageSets/train.txt', help='Optional frame id list (one id per line)')
    parser.add_argument('--out', type=str, default='mycode/output/voxel_stats.csv', help='CSV output file')
    parser.add_argument('--block_size', type=int, default=16, help='Fallback single block size for all dims')
    parser.add_argument('--block_size_x', type=int, default=16, help='Block size in voxels along X (optional)')
    parser.add_argument('--block_size_y', type=int, default=16, help='Block size in voxels along Y (optional)')
    parser.add_argument('--block_size_z', type=int, default=16, help='Block size in voxels along Z (optional)')
    args = parser.parse_args()

    cfg_local = load_cfg_for_kitti()

    if args.velodyne_dir is None:
        default_dir = os.path.join(cfg.ROOT_DIR, 'data', 'kitti', 'training', 'velodyne')
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
        base_keys = ['file','total_voxels','non_empty_voxels','voxel_sparsity','block_voxel_limit','blocks_total','blocks_empty','blocks_nonempty','blocks_fraction_empty','blocks_max_voxels','blocks_mean_voxels_per_block','block_count_hist']

        # Determine max number of non-empty blocks across all frames to create columns
        max_nonempty = 0
        for r in results:
            lst = r.get('blocks_nonempty_voxel_counts_list', [])
            if len(lst) > max_nonempty:
                max_nonempty = len(lst)

        block_cols = [f'Nonempty_block{i}' for i in range(max_nonempty)]
        keys = base_keys + block_cols

        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                row = {k: r.get(k, '') for k in base_keys}
                lst = r.get('blocks_nonempty_voxel_counts_list', [])
                for i in range(max_nonempty):
                    row[block_cols[i]] = lst[i] if i < len(lst) else ''
                writer.writerow(row)

    # simple aggregate print
    sparsities = [r['voxel_sparsity'] for r in results]
    print(f"Processed {len(results)} files. Mean voxel_sparsity: {np.mean(sparsities):.6f}, median: {np.median(sparsities):.6f}")


if __name__ == '__main__':
    main()

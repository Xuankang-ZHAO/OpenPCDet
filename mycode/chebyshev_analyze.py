#!/usr/bin/env python3
"""Compute Chebyshev-distance voxel statistics for many LiDAR frames.

This script follows the same pipeline used elsewhere in `mycode/`:
- it uses OpenPCDet's `DataProcessor` to produce voxel coordinates from
  raw `.bin` LiDAR files.
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
import numpy as np
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.processor.data_processor import DataProcessor


# load KITTI dataset config (same helper used across mycode)
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


def analyze_file(bin_path, data_proc, lidar_center_xy=(0, 0)):
    """Return a dict of stats including chebyshev-distance histogram.

    Args:
        bin_path: path to .bin file
        data_proc: configured DataProcessor instance
        lidar_center_xy: tuple (cx, cy) in voxel indices (x,y)

    Returns:
        dict with keys 'file','total_voxels','non_empty_voxels','voxel_sparsity'
        and 'chebyshev_counts' (numpy array of counts indexed by distance).
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    data_dict = {'points': points, 'use_lead_xyz': True}

    # Apply processing pipeline to generate voxels
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

    # Expect coords in order [z, y, x]
    if coords.size == 0:
        # Build empty result using grid_size
        nx, ny, nz = int(data_proc.grid_size[0]), int(data_proc.grid_size[1]), int(data_proc.grid_size[2])
        total_voxels = nx * ny * nz
        return {
            'file': os.path.basename(bin_path),
            'total_voxels': int(total_voxels),
            'non_empty_voxels': 0,
            'voxel_sparsity': 0.0,
            'chebyshev_counts': np.zeros(1, dtype=np.int64)
        }

    z_idx = coords[:, 0]
    y_idx = coords[:, 1]
    x_idx = coords[:, 2]

    nx, ny, nz = int(data_proc.grid_size[0]), int(data_proc.grid_size[1]), int(data_proc.grid_size[2])
    total_voxels = int(nx) * int(ny) * int(nz)
    non_empty_voxels = coords.shape[0]
    voxel_sparsity = float(non_empty_voxels) / float(total_voxels) if total_voxels > 0 else 0.0
    
    #print(nx, ny, nz)
    # Clip indices to grid bounds just in case
    valid_mask = (z_idx >= 0) & (z_idx < nz) & (y_idx >= 0) & (y_idx < ny) & (x_idx >= 0) & (x_idx < nx)
    if not np.all(valid_mask):
        z_idx = z_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        x_idx = x_idx[valid_mask]
        non_empty_voxels = z_idx.shape[0]
        voxel_sparsity = float(non_empty_voxels) / float(total_voxels) if total_voxels > 0 else 0.0

    # Parse lidar center
    cx, cy = int(lidar_center_xy[0]), int(lidar_center_xy[1])

    # Chebyshev distance per voxel in XY plane: max(|x-cx|, |y-cy|)
    cheb = np.maximum(np.abs(x_idx - cx), np.abs(y_idx - cy)).astype(np.int64)

    # bins_max should be the maximum of |nx - cx| and |ny - cy|
    # (distance from the lidar center to the far edge of the voxel grid)
    bins_max = int(max(abs(int(nx) - cx), abs(int(ny) - cy)))

    # Build bincount for distances 0..bins_max inclusive
    cheb_counts = np.bincount(cheb, minlength=(bins_max + 1))

    return {
        'file': os.path.basename(bin_path),
        'total_voxels': int(total_voxels),
        'non_empty_voxels': int(non_empty_voxels),
        'voxel_sparsity': float(voxel_sparsity),
        'chebyshev_counts': cheb_counts
    }


def main():
    parser = argparse.ArgumentParser(description='Compute Chebyshev-distance voxel histograms')
    parser.add_argument('--velodyne_dir', type=str, default='data/kitti/training/velodyne', help='Path to KITTI velodyne folder (bin files)')
    parser.add_argument('--list_file', type=str, default='data/kitti/ImageSets/analyze.txt', help='Optional frame id list (one id per line)')
    parser.add_argument('--out', type=str, default='mycode/output/chebyshev_stats.csv', help='CSV output file')
    parser.add_argument('--lidar_center', type=str, default='0,800', help='LiDAR voxel center as "x,y" (e.g. 0,800)')
    args = parser.parse_args()

    cfg_local = load_cfg_for_kitti()

    velodyne_dir = args.velodyne_dir if args.velodyne_dir is not None else os.path.join(cfg.ROOT_DIR, 'data', 'kitti', 'training', 'velodyne')
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

    # parse lidar_center
    lc = tuple(int(x) for x in args.lidar_center.split(',')) if args.lidar_center is not None else (0, 0)

    results = []
    max_bins_seen = 0
    for p in tqdm(paths, desc='Processing'):
        res = analyze_file(p, data_proc, lidar_center_xy=lc)
        if res is None:
            continue
        results.append(res)
        max_bins_seen = max(max_bins_seen, res['chebyshev_counts'].size)

    # Prepare CSV header: base keys then dist_0..dist_N
    base_keys = ['file', 'total_voxels', 'non_empty_voxels', 'voxel_sparsity']
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

    # Simple summary print
    sparsities = [r['voxel_sparsity'] for r in results]
    if len(sparsities) > 0:
        print(f"Processed {len(results)} files. Mean voxel_sparsity: {np.mean(sparsities):.6f}, median: {np.median(sparsities):.6f}")
    else:
        print('No valid frames processed.')


if __name__ == '__main__':
    main()

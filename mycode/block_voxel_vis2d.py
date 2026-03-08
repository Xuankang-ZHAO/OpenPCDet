#!/usr/bin/env python3
"""Generate 2D per-file visualizations of block voxel counts.

For each input `.bin` file this script produces a PNG where the image
grid resolution follows the voxel-space `x,y` grid size. Each block is
drawn as a rectangle whose size corresponds to the block's x,y sizes
in voxels; pixel values are the maximum voxel-count among blocks that
cover that pixel (to handle replication/overlap). The LiDAR center is
drawn as a red `x` marker in the voxel coordinate space.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.processor.data_processor import DataProcessor
from mycode.block_partition import compute_block_partition_map


def load_cfg_for_kitti():
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


def make_image_from_blocks(counts, blocks, grid_size):
    nx, ny, nz = grid_size
    nx = int(nx); ny = int(ny)
    img = np.zeros((ny, nx), dtype=np.float32)

    # For each block, paint its xy footprint with the block's count (max if overlap)
    for block_id, info in blocks.items():
        # counts array may be shorter than some block ids in certain modes
        if block_id >= counts.size:
            continue
        cnt = int(counts[block_id])
        if cnt <= 0:
            continue
        bx0 = int(info['bx0']); by0 = int(info['by0'])
        bx = int(info['bx']); by = int(info['by'])
        x0 = bx0 * bx
        x1 = min(nx, x0 + bx)
        y0 = by0 * by
        y1 = min(ny, y0 + by)
        if x0 >= nx or y0 >= ny:
            continue
        # Note image index order is [y,x]
        img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], float(cnt))

    return img


def vis_and_save(img, lidar_center, out_path, cmap_name='magma'):
    # create figure/axes so colorbar can steal space from the axes (no warning)
    fig, ax = plt.subplots(figsize=(8, 8))

    # prepare normalized image and colormap
    img_f = img.astype(np.float32)
    maxv = float(img_f.max()) if img_f.size > 0 else 0.0
    norm = colors.Normalize(vmin=0.0, vmax=maxv if maxv > 0 else 1.0)

    # use the recommended API and reverse the colormap so low->light, high->dark
    base_cmap = matplotlib.colormaps.get_cmap(cmap_name)
    cmap = base_cmap.reversed()

    # map to RGBA; then make zero-valued pixels white & fully transparent
    normed = norm(img_f)
    rgba = cmap(normed)

    # make low-count pixels more transparent: alpha = 0 for zeros, otherwise scaled
    alpha = 0.2 + 0.8 * normed
    alpha[img_f == 0] = 0.0
    rgba[..., 3] = alpha

    # force pure zeros to white-transparent to emphasize emptiness
    zero_mask = (img_f == 0)
    rgba[zero_mask, :3] = 1.0

    ax.imshow(rgba, interpolation='nearest', origin='lower')

    # add x/y ticks with fixed interval of 200 (drawn on bottom and left)
    ny, nx = img.shape
    tick_interval = 200
    # generate x ticks: start at 0, step 200, stop before exceeding nx
    xticks = np.arange(0, nx+1, tick_interval)
    # ensure ticks don't exceed the last pixel index
    xticks = xticks[xticks < nx+1]

    # generate y ticks: start at 0, step 200, stop before exceeding ny
    yticks = np.arange(0, ny+1, tick_interval)
    yticks = yticks[yticks < ny+1]

    # ensure image covers integer voxel centers; set explicit limits
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)

    # draw axis lines (bottom and left) in data coords
    # ax.plot([ -0.5, nx - 0.5 ], [ -0.5, -0.5 ], color='black', lw=1)
    # ax.plot([ -0.5, -0.5 ], [ -0.5, ny - 0.5 ], color='black', lw=1)
    # ax.plot([ -0.5, nx - 0.5 ], [ ny - 0.5, ny - 0.5 ], color='black', lw=1)
    # ax.plot([ nx - 0.5, nx - 0.5 ], [ -0.5, ny - 0.5 ], color='black', lw=1)
    for spine in ax.spines.values():
        spine.set_visible(True)

    # tick length and label padding in data coordinates (small fraction)
    tick_len = max(nx, ny) * 0.005

    # draw x ticks and labels below the image
    for x in xticks:
        ax.plot([x, x], [-0.5, 2 * tick_len], color='black', lw=1)
        ax.text(x, -0.5 - tick_len, str(int(x)), ha='center', va='top', fontsize=8)

    # draw y ticks and labels to the left of the image
    for y in yticks:
        ax.plot([-0.5, 2 * tick_len], [y, y], color='black', lw=1)
        ax.text(-0.5 - tick_len, y, str(int(y)), ha='right', va='center', fontsize=8)

    # axis labels
    ax.text((nx - 1) / 2.0, -0.5 - 5 * tick_len, 'x (voxels)', ha='center', va='top', fontsize=10)
    ax.text(-0.5 - 15 * tick_len, (ny - 1) / 2.0, 'y (voxels)', ha='center', va='bottom', rotation='vertical', fontsize=10)

    # hide default axis ticks and spines so only the custom ticks are shown
    ax.set_xticks([])
    ax.set_yticks([])

    # ensure there is room for left/bottom labels and the colorbar
    fig.subplots_adjust(left=0.18, bottom=0.18)

    # add a colorbar showing the color mapping (ignores alpha)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(img_f)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label('voxels per block (max overlap)')

    if lidar_center is not None:
        cx, cy = int(lidar_center[0]), int(lidar_center[1])
        ny, nx = img.shape
        if 0 <= cx < nx and 0 <= cy < ny:
            ax.scatter([cx], [cy], c='red', marker='x', s=200)

    # keep custom axis lines/labels visible; finalize layout
    # use tight bbox so externally-drawn labels/ticks are not clipped
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--velodyne_dir', type=str, default='data/kitti/training/velodyne')
    parser.add_argument('--list_file', type=str, default='data/kitti/ImageSets/analyze.txt', help='Optional frame id list (one id per line)')
    parser.add_argument('--out_dir', type=str, default='mycode/output/block_vis2d_fixed')
    parser.add_argument('--block_size_x', type=int, default=10)
    parser.add_argument('--block_size_y', type=int, default=10)
    parser.add_argument('--block_size_z', type=int, default=6)
    parser.add_argument('--fixed_block_partition', type=str, default='True')
    parser.add_argument('--block_lut', type=str, default='mycode/block_size_lut.txt')
    parser.add_argument('--lidar_center', type=str, default='0,800')
    args = parser.parse_args()

    cfg_local = load_cfg_for_kitti()

    velodyne_dir = args.velodyne_dir
    if not os.path.exists(velodyne_dir):
        raise FileNotFoundError(f'velodyne dir not found: {velodyne_dir}')

    paths = find_bin_files(velodyne_dir, args.list_file)
    if len(paths) == 0:
        raise RuntimeError('No .bin files found to process')

    os.makedirs(args.out_dir, exist_ok=True)

    num_point_features = len(cfg_local.POINT_FEATURE_ENCODING.used_feature_list) if 'POINT_FEATURE_ENCODING' in cfg_local else 4
    data_proc = DataProcessor(processor_configs=cfg_local.DATA_PROCESSOR,
                              point_cloud_range=np.array(cfg_local.POINT_CLOUD_RANGE),
                              training=False,
                              num_point_features=num_point_features)

    fixed_flag = str(args.fixed_block_partition).lower() in ('1', 'true', 'yes', 'y', 't')
    bx = args.block_size_x; by = args.block_size_y; bz = args.block_size_z
    lidar_center = tuple(int(x) for x in args.lidar_center.split(',')) if args.lidar_center is not None else None

    for p in tqdm(paths, desc='Visualizing'):
        points = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
        data_dict = {'points': points, 'use_lead_xyz': True}
        for proc in data_proc.data_processor_queue:
            data_dict = proc(data_dict)

        coords = data_dict.get('voxel_coords', None)
        if coords is None:
            continue
        if isinstance(coords, list):
            coords = coords[0]
        if hasattr(coords, 'cpu'):
            coords = coords.cpu().numpy()
        coords = coords.astype(np.int64)
        if coords.ndim == 2 and coords.shape[1] == 4:
            coords = coords[:, 1:4]

        mode = 'fixed' if fixed_flag else 'unfixed'
        # prefer grid_size from data_proc
        grid_sz = getattr(data_proc, 'grid_size', None)
        if grid_sz is None:
            # fallback: try cfg
            grid_sz = (cfg_local.POINT_CLOUD_RANGE[0], cfg_local.POINT_CLOUD_RANGE[1], cfg_local.POINT_CLOUD_RANGE[2]) if hasattr(cfg_local, 'POINT_CLOUD_RANGE') else (400, 400, 40)

        if mode == 'fixed':
            counts, blocks, total_blocks, _ = compute_block_partition_map(coords, grid_sz, (bx, by, bz), mode='fixed')
        else:
            counts, blocks, total_blocks, _ = compute_block_partition_map(coords, grid_sz, (bx, by, bz), mode='unfixed', lut_path=args.block_lut, lidar_center_xy=lidar_center)

        gsize = tuple(int(x) for x in grid_sz)

        img = make_image_from_blocks(counts, blocks, gsize)

        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(args.out_dir, base + '.png')
        vis_and_save(img, lidar_center, out_path)


if __name__ == '__main__':
    main()

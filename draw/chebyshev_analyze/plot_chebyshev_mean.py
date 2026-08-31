#!/usr/bin/env python3
"""Plot MEAN_200_frames Chebyshev histograms as four stacked stage panels.

CSV bins are raw occupied-voxel counts per Chebyshev ring. The plotted y
values are those counts divided by the actual XY ring length on the finite
grid, so the curves show local density rather than circumference-biased totals.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


STAGES = (
    ('conv1', 'chebyshev_stats_conv1.csv'),
    ('conv2', 'chebyshev_stats_conv2.csv'),
    ('conv3', 'chebyshev_stats_conv3.csv'),
    ('conv4', 'chebyshev_stats_conv4.csv'),
)

VECTOR_SUFFIXES = ('.pdf', '.svg')

# IEEE TCAS-I / IEEEtran journal: one column is about 3.5 in.
# The stacked figure is drawn ~1/3 wider than one column (~4.67 in).
IEEE_COLUMN_INCHES = 3.5
IEEE_FIGURE_WIDTH_INCHES = IEEE_COLUMN_INCHES * (4.0 / 3.0)


def load_mean_histogram(csv_path):
    with Path(csv_path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    mean_rows = [row for row in rows if str(row.get('file', '')).startswith('MEAN_')]
    if not mean_rows:
        raise RuntimeError(f'No MEAN_* row found in {csv_path}')
    row = mean_rows[-1]
    fov = str(row.get('fov_points_only', '')).strip().lower()
    mode = str(row.get('data_mode', '')).strip().lower()
    if mode == 'raw' or fov in ('false', '0', ''):
        raise RuntimeError(
            f'{csv_path} is not FOV=true KITTI inference input '
            f'(data_mode={row.get("data_mode")}, fov_points_only={row.get("fov_points_only")}). '
            'Regenerate with draw/chebyshev_analyze/chebyshev_analyze.py --data_mode kitti.'
        )
    dist_cols = sorted(
        (col for col in row if col.startswith('dist_')),
        key=lambda name: int(name.split('_', 1)[1]),
    )
    distances = [int(col.split('_', 1)[1]) for col in dist_cols]
    counts = [float(row[col]) for col in dist_cols]
    return distances, counts, row


def parse_spatial_shape_zyx(spatial_shape_zyx):
    nz, ny, nx = (int(part) for part in str(spatial_shape_zyx).split('x'))
    return nx, ny, nz


def grid_label_xyz(spatial_shape_zyx):
    nx, ny, nz = parse_spatial_shape_zyx(spatial_shape_zyx)
    return f'{nx} × {ny} × {nz}'


def _clipped_inclusive_len(lo, hi, bound_lo, bound_hi):
    start = max(int(lo), int(bound_lo))
    stop = min(int(hi), int(bound_hi) - 1)
    return max(0, stop - start + 1)


def chebyshev_ring_circumference(nx, ny, cx, cy, dist):
    """XY cell count at Chebyshev distance `dist`, clipped to the finite grid.

    A full square ring has length 8d (d>=1), but KITTI's LiDAR center sits on
    the x=0 edge and distant rings are cut by the Y/X bounds, so the plotted
    density uses this actual ring length rather than 8d.
    """
    nx, ny, cx, cy, dist = int(nx), int(ny), int(cx), int(cy), int(dist)
    if dist < 0:
        return 0
    if dist == 0:
        return int(0 <= cx < nx and 0 <= cy < ny)

    length = 0
    right_x = cx + dist
    if 0 <= right_x < nx:
        length += _clipped_inclusive_len(cy - dist, cy + dist, 0, ny)
    left_x = cx - dist
    if 0 <= left_x < nx:
        length += _clipped_inclusive_len(cy - dist, cy + dist, 0, ny)
    top_y = cy + dist
    if 0 <= top_y < ny:
        length += _clipped_inclusive_len(cx - dist + 1, cx + dist - 1, 0, nx)
    bottom_y = cy - dist
    if 0 <= bottom_y < ny:
        length += _clipped_inclusive_len(cx - dist + 1, cx + dist - 1, 0, nx)
    return length


def counts_to_ring_density(distances, counts, row):
    nx, ny, _nz = parse_spatial_shape_zyx(row['spatial_shape_zyx'])
    cx = int(row['lidar_center_x'])
    cy = int(row['lidar_center_y'])
    density = []
    for dist, count in zip(distances, counts):
        ring_len = chebyshev_ring_circumference(nx, ny, cx, cy, dist)
        density.append((count / ring_len) if ring_len else 0.0)
    return density


def plot_mean_curves(csv_dir, out_path):
    csv_dir = Path(csv_dir)
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(IEEE_FIGURE_WIDTH_INCHES, 4.6),
        sharex=False,
    )
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for ax, color, (_stage, filename) in zip(axes, colors, STAGES):
        distances, counts, row = load_mean_histogram(csv_dir / filename)
        density = counts_to_ring_density(distances, counts, row)
        ax.plot(distances, density, color=color, linewidth=1.0)
        ax.set_xlim(0, distances[-1] if distances else 1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, min_n_ticks=3))
        ax.text(
            0.985,
            0.90,
            grid_label_xyz(row["spatial_shape_zyx"]),
            transform=ax.transAxes,
            va='top',
            ha='right',
            fontsize=12,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 1.0},
        )

    axes[-1].set_xlabel('Chebyshev Distance')
    fig.supylabel('Voxels / Ring Length', fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.38)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = [out_path]
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    for suffix in VECTOR_SUFFIXES:
        vector_path = out_path.with_suffix(suffix)
        fig.savefig(vector_path, bbox_inches='tight')
        saved.append(vector_path)
    plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser(description='Plot mean Chebyshev histograms for conv1–conv4')
    parser.add_argument(
        '--csv_dir',
        type=str,
        default=str(Path(__file__).resolve().parent),
        help='Directory containing chebyshev_stats_conv{1,2,3,4}.csv',
    )
    parser.add_argument(
        '--out',
        type=str,
        default='',
        help='Raster output path. PDF and SVG are written beside it. '
        'Default: <csv_dir>/chebyshev_mean_200frames.png',
    )
    args = parser.parse_args()
    csv_dir = Path(args.csv_dir)
    out_path = Path(args.out) if args.out else csv_dir / 'chebyshev_mean_200frames.png'
    saved = plot_mean_curves(csv_dir, out_path)
    for path in saved:
        print(f'Saved {path}')


if __name__ == '__main__':
    main()

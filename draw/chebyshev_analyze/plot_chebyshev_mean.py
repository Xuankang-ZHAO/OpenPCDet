#!/usr/bin/env python3
"""Plot MEAN_200_frames Chebyshev histograms for conv1–conv4 on one figure."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


STAGES = (
    ('conv1', 'chebyshev_stats_conv1.csv', 'conv1 (stride 1)'),
    ('conv2', 'chebyshev_stats_conv2.csv', 'conv2 (stride 2)'),
    ('conv3', 'chebyshev_stats_conv3.csv', 'conv3 (stride 4)'),
    ('conv4', 'chebyshev_stats_conv4.csv', 'conv4 (stride 8)'),
)

VECTOR_SUFFIXES = ('.pdf', '.svg')


def load_mean_histogram(csv_path):
    with Path(csv_path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    mean_rows = [row for row in rows if str(row.get('file', '')).startswith('MEAN_')]
    if not mean_rows:
        raise RuntimeError(f'No MEAN_* row found in {csv_path}')
    row = mean_rows[-1]
    dist_cols = sorted(
        (col for col in row if col.startswith('dist_')),
        key=lambda name: int(name.split('_', 1)[1]),
    )
    distances = [int(col.split('_', 1)[1]) for col in dist_cols]
    counts = [float(row[col]) for col in dist_cols]
    return distances, counts, row


def plot_mean_curves(csv_dir, out_path):
    csv_dir = Path(csv_dir)
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })
    fig, ax = plt.subplots(figsize=(11, 6))
    for _stage, filename, label in STAGES:
        distances, counts, _row = load_mean_histogram(csv_dir / filename)
        ax.plot(distances, counts, linewidth=1.6, label=label)

    ax.set_xlabel('Chebyshev Distance')
    ax.set_ylabel('Voxel Count')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

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

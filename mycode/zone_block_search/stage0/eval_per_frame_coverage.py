#!/usr/bin/env python3
"""Per-frame whole-scene two-page coverage under a fixed zone LUT.

Uses rtl_unfixed multi-zone partition (variable Bx×By×Bz + 3D halo) on the
cached stage-0 voxels, then R_frame = #{Nb<=128} / #{materialized blocks}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mycode.rtl_unfixed.partition import (
    compute_rtl_unfixed_partition_counts,
    load_zone_specs,
    summarize_zone_specs,
)
from mycode.zone_block_search.stage0.config import (
    GRID_SIZE_XYZ,
    LIDAR_CENTER_XY,
    RESULTS_DIR,
    TWO_PAGE_LIMIT,
    VOXEL_CACHE_PATH,
)
from mycode.zone_block_search.stage0.voxels import load_stage0_voxel_frames


def raw_zones_to_lut_lines(raw_zones_csv: Path) -> list:
    lines = []
    with raw_zones_csv.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: int(r['zone_id']))
    for row in rows:
        outer = row['outer_T'].strip()
        size = row['size'].replace('x', ',')
        lines.append(f"{row['zone_id']}:{outer}:{size}")
    return lines


def write_lut(path: Path, lines: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        handle.write('# Auto-generated from Step4 raw_zones.csv\n')
        handle.write('# zone:outer_half_open_T:bx,by,bz\n')
        for line in lines:
            handle.write(line + '\n')


def frame_coverage(coords, grid_size, zone_specs, lidar_center, limit=TWO_PAGE_LIMIT):
    counts, n_blocks, _ = compute_rtl_unfixed_partition_counts(
        coords, grid_size, zone_specs, lidar_center
    )
    if n_blocks == 0:
        return {
            'n_blocks': 0,
            'n_le_limit': 0,
            'coverage': float('nan'),
            'reshape_frac': float('nan'),
            'mean_nb': float('nan'),
            'p95_nb': float('nan'),
            'max_nb': 0,
        }
    le = int(np.sum(counts <= limit))
    return {
        'n_blocks': int(n_blocks),
        'n_le_limit': le,
        'coverage': float(le / n_blocks),
        'reshape_frac': float(np.mean(counts > limit)),
        'mean_nb': float(np.mean(counts)),
        'p95_nb': float(np.percentile(counts, 95)),
        'max_nb': int(counts.max()),
    }


def main():
    parser = argparse.ArgumentParser(description='Per-frame R under zone LUT')
    parser.add_argument('--in_dir', type=str, default=str(RESULTS_DIR))
    parser.add_argument('--raw_zones', type=str, default='')
    parser.add_argument('--cache', type=str, default=str(VOXEL_CACHE_PATH))
    parser.add_argument('--list_file', type=str, default='')
    parser.add_argument('--lut', type=str, default='')
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    raw_zones = Path(args.raw_zones) if args.raw_zones else in_dir / 'raw_zones.csv'
    lut_path = Path(args.lut) if args.lut else in_dir / 'raw_zones_lut.txt'

    lines = raw_zones_to_lut_lines(raw_zones)
    write_lut(lut_path, lines)
    zone_specs = load_zone_specs(str(lut_path), GRID_SIZE_XYZ, LIDAR_CENTER_XY)
    print(f'LUT {lut_path}: {summarize_zone_specs(zone_specs)}')

    list_file = Path(args.list_file) if args.list_file else None
    kwargs = {'cache_path': Path(args.cache)}
    if list_file is not None:
        kwargs['list_file'] = list_file
    frame_ids, coords_list, meta = load_stage0_voxel_frames(**kwargs)
    print(f'Frames={len(frame_ids)} cache={meta["from_cache"]}')

    rows = []
    for frame_id, coords in tqdm(list(zip(frame_ids, coords_list)), desc='Per-frame R'):
        stats = frame_coverage(coords, GRID_SIZE_XYZ, zone_specs, LIDAR_CENTER_XY)
        rows.append({'frame_id': frame_id, **stats})

    out_csv = in_dir / 'per_frame_coverage.csv'
    with out_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    coverages = np.array([r['coverage'] for r in rows if r['n_blocks'] > 0], dtype=np.float64)
    n_blocks = np.array([r['n_blocks'] for r in rows], dtype=np.int64)
    n_le = np.array([r['n_le_limit'] for r in rows], dtype=np.int64)
    pooled = float(n_le.sum() / n_blocks.sum()) if n_blocks.sum() else float('nan')

    summary = {
        'lut': summarize_zone_specs(zone_specs),
        'lut_path': str(lut_path),
        'n_frames': len(rows),
        'two_page_limit': TWO_PAGE_LIMIT,
        'per_frame_R_mean': float(np.mean(coverages)),
        'per_frame_R_median': float(np.median(coverages)),
        'per_frame_R_std': float(np.std(coverages)),
        'per_frame_R_p10': float(np.percentile(coverages, 10)),
        'per_frame_R_p90': float(np.percentile(coverages, 90)),
        'per_frame_R_min': float(np.min(coverages)),
        'per_frame_R_max': float(np.max(coverages)),
        'pooled_R': pooled,
        'total_materialized_blocks': int(n_blocks.sum()),
        'mean_blocks_per_frame': float(np.mean(n_blocks)),
        'frames_below_0_95': int(np.sum(coverages < 0.95)),
        'frames_below_0_90': int(np.sum(coverages < 0.90)),
    }
    out_json = in_dir / 'per_frame_coverage_summary.json'
    with out_json.open('w') as handle:
        json.dump(summary, handle, indent=2)

    # Optional histogram plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(coverages, bins=20, color='steelblue', edgecolor='white')
        ax.axvline(0.95, color='C3', linestyle='--', label='q=0.95')
        ax.axvline(summary['per_frame_R_mean'], color='C1', linestyle='-', label='mean')
        ax.set_xlabel('Per-frame two-page coverage R')
        ax.set_ylabel('Frame count')
        ax.set_title('Whole-scene R under Step4 raw zones')
        ax.legend()
        fig.tight_layout()
        fig.savefig(in_dir / 'per_frame_coverage_hist.png', dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f'plot skipped: {exc}')

    print(f'Wrote {out_csv}')
    print(f'Wrote {out_json}')
    print('\n=== Per-frame whole-scene R (Step4 raw zones) ===')
    print(f"pooled_R (all blocks) = {pooled:.4f}")
    print(
        f"per-frame R: mean={summary['per_frame_R_mean']:.4f}  "
        f"median={summary['per_frame_R_median']:.4f}  "
        f"std={summary['per_frame_R_std']:.4f}"
    )
    print(
        f"per-frame R: p10={summary['per_frame_R_p10']:.4f}  "
        f"p90={summary['per_frame_R_p90']:.4f}  "
        f"min={summary['per_frame_R_min']:.4f}  max={summary['per_frame_R_max']:.4f}"
    )
    print(
        f"frames with R<0.95: {summary['frames_below_0_95']}/{len(rows)}; "
        f"R<0.90: {summary['frames_below_0_90']}/{len(rows)}"
    )
    print(f"mean materialized blocks / frame: {summary['mean_blocks_per_frame']:.1f}")


if __name__ == '__main__':
    main()

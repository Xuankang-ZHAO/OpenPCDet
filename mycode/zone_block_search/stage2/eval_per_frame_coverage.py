#!/usr/bin/env python3
"""Per-frame whole-scene two-page coverage under a Stage-2 zone LUT.

Uses rtl_unfixed multi-zone partition (variable Bx×By×Bz + 3D halo) on conv3
occupancy. Bz=8 is allowed here (rtl_unfixed LUT parser still requires 16).
R_frame = #{Nb<=128} / #{materialized blocks}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mycode.rtl_unfixed.partition import (
    ZoneSpec,
    _ilog2,
    _validate_nested_squares_and_block_alignment,
    compute_rtl_unfixed_partition_counts,
    summarize_zone_specs,
)
from mycode.zone_block_search.stage2.config import (
    FRAME_LIST_PATH,
    GRID_SIZE_XYZ,
    LIDAR_CENTER_XY,
    RESULTS_DIR,
    TWO_PAGE_LIMIT,
    VOXEL_CACHE_PATH,
)
from mycode.zone_block_search.stage2.voxels import load_stage2_voxel_frames


def _parse_outer(text: str) -> Optional[int]:
    token = str(text).strip()
    if token in ('', '*', 'inf', '+inf', 'None'):
        return None
    return int(float(token)) if token else None


def raw_zones_to_lut_lines(raw_zones_csv: Path) -> list:
    lines = []
    with raw_zones_csv.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: int(r['zone_id']))
    for row in rows:
        outer = _parse_outer(row['outer_T'])
        outer_s = '*' if outer is None else str(outer)
        size = row['size'].replace('x', ',')
        lines.append(f"{row['zone_id']}:{outer_s}:{size}")
    return lines


def write_lut(path: Path, lines: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        handle.write('# Auto-generated Stage2 zone LUT (Bz=8)\n')
        handle.write('# zone:outer_half_open_T:bx,by,bz\n')
        for line in lines:
            handle.write(line + '\n')


def zone_specs_from_lut_lines(lines: List[str]) -> List[ZoneSpec]:
    """Build nested zone specs without rtl_unfixed's Bz==16 LUT constraint."""
    parsed: List[Tuple[int, Optional[int], Tuple[int, int, int]]] = []
    for line in lines:
        zone_s, bound_s, size_s = line.split(':')
        zone_id = int(zone_s)
        outer = _parse_outer(bound_s)
        bx, by, bz = (int(p) for p in size_s.split(','))
        parsed.append((zone_id, outer, (bx, by, bz)))

    finite = [item for item in parsed if item[1] is not None]
    unbounded = [item for item in parsed if item[1] is None]
    finite.sort(key=lambda item: item[1])
    if len(unbounded) != 1 or unbounded[0][0] != parsed[-1][0]:
        raise ValueError('Zone LUT must end with exactly one unbounded zone')

    ordered = finite + unbounded
    specs: List[ZoneSpec] = []
    for index, (zone_id, outer_half_open, block_size_xyz) in enumerate(ordered):
        inner = 0 if index == 0 else ordered[index - 1][1]
        specs.append(
            ZoneSpec(
                zone_id=zone_id,
                inner_half_open=inner,
                outer_half_open=outer_half_open,
                block_size_xyz=block_size_xyz,
                log2_block_size_xyz=tuple(_ilog2(size) for size in block_size_xyz),
            )
        )
    _validate_nested_squares_and_block_alignment(specs)
    return specs


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
    parser = argparse.ArgumentParser(description='Stage2 per-frame R under zone LUT')
    parser.add_argument('--in_dir', type=str, default=str(RESULTS_DIR))
    parser.add_argument('--raw_zones', type=str, default='')
    parser.add_argument('--cache', type=str, default=str(VOXEL_CACHE_PATH))
    parser.add_argument('--list_file', type=str, default=str(FRAME_LIST_PATH))
    parser.add_argument('--lut', type=str, default='')
    parser.add_argument('--max_frames', type=int, default=0)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    in_dir.mkdir(parents=True, exist_ok=True)
    raw_zones = Path(args.raw_zones) if args.raw_zones else in_dir / 'raw_zones_rle.csv'
    lut_path = Path(args.lut) if args.lut else in_dir / 'raw_zones_lut.txt'

    lines = raw_zones_to_lut_lines(raw_zones)
    write_lut(lut_path, lines)
    zone_specs = zone_specs_from_lut_lines(lines)
    print(f'LUT {lut_path}: {summarize_zone_specs(zone_specs)}')
    print(f'grid={GRID_SIZE_XYZ} lidar={LIDAR_CENTER_XY}')

    frame_ids, coords_list, meta = load_stage2_voxel_frames(
        list_file=Path(args.list_file),
        cache_path=Path(args.cache),
    )
    if args.max_frames and args.max_frames > 0:
        frame_ids = frame_ids[:args.max_frames]
        coords_list = coords_list[:args.max_frames]
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
        'grid_size_xyz': list(GRID_SIZE_XYZ),
        'lidar_center_xy': list(LIDAR_CENTER_XY),
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
        'frames_below_0_97': int(np.sum(coverages < 0.97)),
        'frames_below_0_95': int(np.sum(coverages < 0.95)),
        'frames_below_0_90': int(np.sum(coverages < 0.90)),
    }
    out_json = in_dir / 'per_frame_coverage_summary.json'
    with out_json.open('w') as handle:
        json.dump(summary, handle, indent=2)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(coverages, bins=20, color='steelblue', edgecolor='white')
        ax.axvline(0.95, color='C3', linestyle='--', label='0.95')
        ax.axvline(0.97, color='C4', linestyle=':', label='q=0.97')
        ax.axvline(summary['per_frame_R_mean'], color='C1', linestyle='-', label='mean')
        ax.set_xlabel('Per-frame two-page coverage R')
        ax.set_ylabel('Frame count')
        ax.set_title('Stage2 whole-scene R')
        ax.legend()
        fig.tight_layout()
        fig.savefig(in_dir / 'per_frame_coverage_hist.png', dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f'plot skipped: {exc}')

    print(f'Wrote {out_csv}')
    print(f'Wrote {out_json}')
    print('\n=== Per-frame whole-scene R ===')
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
        f"frames with R<0.97: {summary['frames_below_0_97']}/{len(rows)}; "
        f"R<0.95: {summary['frames_below_0_95']}/{len(rows)}; "
        f"R<0.90: {summary['frames_below_0_90']}/{len(rows)}"
    )
    print(f"mean materialized blocks / frame: {summary['mean_blocks_per_frame']:.1f}")


if __name__ == '__main__':
    main()

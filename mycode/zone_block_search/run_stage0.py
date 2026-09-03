#!/usr/bin/env python3
"""Stage-0 zone / block-size search: Steps 0–3 then pause.

Profiles the Bx<By menu over 200 KITTI FOV frames, writes per-ring two-page
coverage and B*(j), then stops for human review before Step 4.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mycode.zone_block_search.config import (
    COVERAGE_Q,
    DELTA_D,
    FRAME_LIST_PATH,
    GRID_SIZE_XYZ,
    LIDAR_CENTER_XY,
    LOW_SAMPLE_THRESHOLD,
    MENU_BX_LT_BY,
    RESULTS_DIR,
    TWO_PAGE_LIMIT,
    VOXEL_CACHE_PATH,
    menu_volume_rank,
    size_label,
)
from mycode.zone_block_search.partition import (
    accumulate_ring_nb_samples,
    summarize_ring_coverage,
    validate_against_rtl_unfixed,
)
from mycode.zone_block_search.voxels import load_stage0_voxel_frames


def select_bstar(coverage_by_size: dict, menu=MENU_BX_LT_BY, q: float = COVERAGE_Q):
    """Pick largest menu size with coverage >= q; all-fail → smallest; all-pass → largest.

    coverage_by_size: {size_label: coverage float or nan}
    Returns (label, bx, by, bz) or (None, ...) if no usable coverage.
    """
    vol = menu_volume_rank(menu)
    ranked = sorted(menu, key=lambda s: vol[size_label(*s)])
    passing = []
    any_finite = False
    for shape in ranked:
        label = size_label(*shape)
        cov = coverage_by_size.get(label, float('nan'))
        if cov != cov:  # NaN
            continue
        any_finite = True
        if cov >= q:
            passing.append(shape)
    if not any_finite:
        return None, None, None, None
    chosen = passing[-1] if passing else ranked[0]
    return size_label(*chosen), chosen[0], chosen[1], chosen[2]


def write_coverage_csv(path: Path, rings: list, menu, coverage_table: dict, q: float = COVERAGE_Q):
    fieldnames = [
        'ring_j', 'inner_T', 'outer_T',
        'size', 'bx', 'by', 'bz',
        'n_samples', 'coverage_Rjk', 'p95_nb', 'reshape_frac', 'mean_nb', 'max_nb',
        'passes_q',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for j in rings:
            for shape in menu:
                label = size_label(*shape)
                stats = coverage_table[label].get(j, {
                    'n_samples': 0,
                    'coverage': float('nan'),
                    'p95_nb': float('nan'),
                    'reshape_frac': float('nan'),
                    'mean_nb': float('nan'),
                    'max_nb': 0,
                })
                cov = stats['coverage']
                passes = (cov == cov) and (cov >= q)
                writer.writerow({
                    'ring_j': j,
                    'inner_T': j * DELTA_D,
                    'outer_T': (j + 1) * DELTA_D,
                    'size': label,
                    'bx': shape[0],
                    'by': shape[1],
                    'bz': shape[2],
                    'n_samples': stats['n_samples'],
                    'coverage_Rjk': '' if cov != cov else f'{cov:.6f}',
                    'p95_nb': '' if stats['p95_nb'] != stats['p95_nb'] else f"{stats['p95_nb']:.2f}",
                    'reshape_frac': '' if stats['reshape_frac'] != stats['reshape_frac'] else f"{stats['reshape_frac']:.6f}",
                    'mean_nb': '' if stats['mean_nb'] != stats['mean_nb'] else f"{stats['mean_nb']:.2f}",
                    'max_nb': stats['max_nb'],
                    'passes_q': int(passes),
                })


def write_bstar_csv(path: Path, rows: list):
    fieldnames = [
        'ring_j', 'inner_T', 'outer_T', 'Bstar', 'bx', 'by', 'bz',
        'n_samples_at_Bstar', 'coverage_at_Bstar', 'low_sample', 'note',
    ]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(out_dir: Path, rings: list, menu, coverage_table: dict, bstar_rows: list,
                 q: float = COVERAGE_Q):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available; skipping plots')
        return []

    labels = [size_label(*s) for s in menu]
    # Coverage heatmap: rings x sizes
    mat = np.full((len(rings), len(labels)), np.nan, dtype=np.float64)
    samples = np.zeros((len(rings), len(labels)), dtype=np.int64)
    for i, j in enumerate(rings):
        for k, label in enumerate(labels):
            stats = coverage_table[label].get(j)
            if stats is None:
                continue
            samples[i, k] = stats['n_samples']
            if stats['n_samples'] > 0:
                mat[i, k] = stats['coverage']

    saved = []
    fig, ax = plt.subplots(figsize=(10, max(4, 0.22 * len(rings) + 2)))
    im = ax.imshow(mat, aspect='auto', vmin=0.0, vmax=1.0, cmap='RdYlGn', origin='lower')
    for i in range(len(rings)):
        for k in range(len(labels)):
            if samples[i, k] <= 0:
                continue
            val = mat[i, k]
            color = 'black' if 0.35 < val < 0.85 else 'white'
            ax.text(k, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticks(range(len(rings)))
    ax.set_yticklabels([f'Z{j} [{j*DELTA_D},{(j+1)*DELTA_D})' for j in rings])
    ax.set_xlabel('Block size')
    ax.set_ylabel('Fine square ring')
    ax.set_title(f'Stage0 two-page coverage R (q={q}, limit={TWO_PAGE_LIMIT})')
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    heat_path = out_dir / 'coverage_heatmap.png'
    fig.savefig(heat_path, dpi=160)
    fig.savefig(out_dir / 'coverage_heatmap.svg')
    plt.close(fig)
    saved.extend([heat_path, out_dir / 'coverage_heatmap.svg'])

    # B*(j) step plot
    fig, ax = plt.subplots(figsize=(10, 4))
    vol = menu_volume_rank(menu)
    rank_of = {lab: i for i, lab in enumerate(labels)}
    xs, ys, low = [], [], []
    for row in bstar_rows:
        if not row['Bstar']:
            continue
        xs.append(int(row['ring_j']))
        ys.append(rank_of[row['Bstar']])
        low.append(int(row['low_sample']))
    ax.step(xs, ys, where='mid', color='C0', linewidth=1.5)
    ax.scatter(xs, ys, c=['C3' if flag else 'C0' for flag in low], s=28, zorder=3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Ring index j')
    ax.set_ylabel('B*(j)')
    ax.set_title('Stage0 preferred size per fine ring (red = low_sample)')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    bstar_path = out_dir / 'bstar_by_ring.png'
    fig.savefig(bstar_path, dpi=160)
    fig.savefig(out_dir / 'bstar_by_ring.svg')
    plt.close(fig)
    saved.extend([bstar_path, out_dir / 'bstar_by_ring.svg'])

    # Sample counts at selected size
    fig, ax = plt.subplots(figsize=(10, 3.2))
    sj = [int(r['ring_j']) for r in bstar_rows]
    sn = [int(r['n_samples_at_Bstar'] or 0) for r in bstar_rows]
    ax.bar(sj, sn, color='steelblue', width=0.8)
    ax.axhline(LOW_SAMPLE_THRESHOLD, color='C3', linestyle='--', linewidth=1, label=f'low_sample<{LOW_SAMPLE_THRESHOLD}')
    ax.set_xlabel('Ring index j')
    ax.set_ylabel('Pooled block-frame samples')
    ax.set_title('Samples at selected B*(j)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    samp_path = out_dir / 'samples_by_ring.png'
    fig.savefig(samp_path, dpi=160)
    plt.close(fig)
    saved.append(samp_path)
    return saved


def print_bstar_table(bstar_rows: list, q: float = COVERAGE_Q):
    print(f'\n=== Step 3: per-ring B*(j) (Bx<By menu, q={q}) — PAUSE for review ===')
    print(f'{"j":>4}  {"[Tin,Tout)":>14}  {"B*":>12}  {"n":>8}  {"R":>7}  {"flag":>10}')
    print('-' * 68)
    for row in bstar_rows:
        j = row['ring_j']
        interval = f"[{row['inner_T']},{row['outer_T']})"
        bstar = row['Bstar'] or '-'
        n = row['n_samples_at_Bstar']
        cov = row['coverage_at_Bstar']
        flag = 'low_sample' if int(row['low_sample']) else ''
        if row['note']:
            flag = (flag + ' ' + row['note']).strip()
        cov_s = f'{float(cov):.3f}' if cov not in ('', None) else '-'
        n_s = str(n) if n not in ('', None) else '0'
        print(f'{j:4d}  {interval:>14}  {bstar:>12}  {n_s:>8}  {cov_s:>7}  {flag:>10}')
    print('-' * 68)
    print('Step 4 (debounce / merge) not run. Inspect results/ then instruct next steps.')


def run_validation(coords_list, n_frames: int = 1):
    print('Validating vectorized partition vs rtl_unfixed single-zone...')
    coords = coords_list[0]
    all_ok = True
    for shape in MENU_BX_LT_BY:
        report = validate_against_rtl_unfixed(coords, shape)
        status = 'OK' if report['match'] else 'MISMATCH'
        print(
            f'  {report["block_size"]}: {status} '
            f'rtl_n={report["rtl_n_blocks"]} ours_n={report["ours_n_blocks"]} '
            f'sum rtl/ours={report["rtl_sum_nb"]}/{report["ours_sum_nb"]}'
        )
        all_ok = all_ok and report['match']
    if not all_ok:
        raise RuntimeError('Partition validation failed against rtl_unfixed')
    print('Validation passed.')


def profile_menu(coords_list, menu=MENU_BX_LT_BY):
    """Return coverage_table[size_label][ring_j] = stats dict."""
    coverage_table = {}
    for shape in menu:
        label = size_label(*shape)
        ring_samples = defaultdict(list)
        for coords in tqdm(coords_list, desc=f'Profile {label}', leave=False):
            accumulate_ring_nb_samples(coords, shape, ring_samples)
        coverage_table[label] = summarize_ring_coverage(ring_samples)
        n_blocks = sum(s['n_samples'] for s in coverage_table[label].values())
        print(f'  {label}: {n_blocks} materialized block-frame instances, '
              f'{len(coverage_table[label])} rings with samples')
    return coverage_table


def build_bstar_rows(coverage_table, menu=MENU_BX_LT_BY, q: float = COVERAGE_Q):
    all_rings = sorted({j for stats in coverage_table.values() for j in stats})
    # Also include empty leading rings up to max observed so the table is contiguous
    if all_rings:
        rings = list(range(0, max(all_rings) + 1))
    else:
        rings = []

    rows = []
    for j in rings:
        cov_by_size = {
            label: coverage_table[label].get(j, {}).get('coverage', float('nan'))
            for label in (size_label(*s) for s in menu)
        }
        n_by_size = {
            label: coverage_table[label].get(j, {}).get('n_samples', 0)
            for label in (size_label(*s) for s in menu)
        }
        total_n = max(n_by_size.values()) if n_by_size else 0
        # Prefer sample count from the selected size; rings with zero everywhere get no label
        label, bx, by, bz = select_bstar(cov_by_size, menu=menu, q=q)
        note = ''
        if total_n == 0:
            label, bx, by, bz = None, None, None, None
            note = 'no_samples'
            n_star = 0
            cov_star = ''
            low = 1
        else:
            n_star = n_by_size.get(label, 0)
            cov_star = cov_by_size.get(label, float('nan'))
            low = int(n_star < LOW_SAMPLE_THRESHOLD)
            # If all sizes failed q, note that we fell back to smallest
            passing = [
                lab for lab, cov in cov_by_size.items()
                if cov == cov and cov >= q and n_by_size.get(lab, 0) > 0
            ]
            if not passing:
                note = 'all_fail_use_min'

        rows.append({
            'ring_j': j,
            'inner_T': j * DELTA_D,
            'outer_T': (j + 1) * DELTA_D,
            'Bstar': label or '',
            'bx': bx if bx is not None else '',
            'by': by if by is not None else '',
            'bz': bz if bz is not None else '',
            'n_samples_at_Bstar': n_star,
            'coverage_at_Bstar': '' if cov_star == '' or cov_star != cov_star else f'{float(cov_star):.6f}',
            'low_sample': low,
            'note': note,
        })
    return rings, rows


def main():
    parser = argparse.ArgumentParser(description='Stage0 zone/block search Steps 0–3')
    parser.add_argument('--list_file', type=str, default=str(FRAME_LIST_PATH))
    parser.add_argument('--out_dir', type=str, default=str(RESULTS_DIR))
    parser.add_argument('--cache', type=str, default=str(VOXEL_CACHE_PATH))
    parser.add_argument('--q', type=float, default=COVERAGE_Q, help='Two-page coverage threshold')
    parser.add_argument('--force_reload', action='store_true')
    parser.add_argument('--skip_validate', action='store_true')
    parser.add_argument('--max_frames', type=int, default=0, help='Debug: limit frames (0=all)')
    args = parser.parse_args()
    q = float(args.q)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=== Step 0–1: load voxels / fixed inputs ===')
    print(f'grid={GRID_SIZE_XYZ} lidar={LIDAR_CENTER_XY} delta_d={DELTA_D} q={q}')
    print(f'menu Bx<By: {[size_label(*s) for s in MENU_BX_LT_BY]}')

    frame_ids, coords_list, meta = load_stage0_voxel_frames(
        list_file=Path(args.list_file),
        cache_path=Path(args.cache),
        force_reload=args.force_reload,
    )
    if args.max_frames and args.max_frames > 0:
        frame_ids = frame_ids[:args.max_frames]
        coords_list = coords_list[:args.max_frames]
        print(f'Debug truncate to {len(frame_ids)} frames')
    print(f'Loaded {len(frame_ids)} frames (cache={meta["from_cache"]})')

    if not args.skip_validate:
        run_validation(coords_list)

    print('=== Step 2: per-size global profiling ===')
    coverage_table = profile_menu(coords_list, MENU_BX_LT_BY)

    print(f'=== Step 3: per-ring max passing size (q={q}) ===')
    rings, bstar_rows = build_bstar_rows(coverage_table, MENU_BX_LT_BY, q=q)

    coverage_csv = out_dir / 'coverage_Rjk.csv'
    bstar_csv = out_dir / 'selected_Bstar.csv'
    write_coverage_csv(coverage_csv, rings, MENU_BX_LT_BY, coverage_table, q=q)
    write_bstar_csv(bstar_csv, bstar_rows)

    summary = {
        'stage': 0,
        'grid_size_xyz': list(GRID_SIZE_XYZ),
        'lidar_center_xy': list(LIDAR_CENTER_XY),
        'delta_d': DELTA_D,
        'bz': 16,
        'q': q,
        'two_page_limit': TWO_PAGE_LIMIT,
        'menu': 'Bx<By',
        'menu_sizes': [size_label(*s) for s in MENU_BX_LT_BY],
        'n_frames': len(frame_ids),
        'frame_list': str(args.list_file),
        'n_rings': len(rings),
        'low_sample_threshold': LOW_SAMPLE_THRESHOLD,
        'bstar': [
            {
                'ring_j': r['ring_j'],
                'interval': [r['inner_T'], r['outer_T']],
                'Bstar': r['Bstar'],
                'n_samples': r['n_samples_at_Bstar'],
                'coverage': r['coverage_at_Bstar'],
                'low_sample': bool(int(r['low_sample'])),
                'note': r['note'],
            }
            for r in bstar_rows
        ],
        'outputs': {
            'coverage_Rjk': str(coverage_csv),
            'selected_Bstar': str(bstar_csv),
        },
        'stopped_after': 'step3',
        'next': 'Await review before Step 4 (debounce + run-length merge)',
    }
    summary_path = out_dir / 'summary.json'
    with summary_path.open('w') as handle:
        json.dump(summary, handle, indent=2)
    print(f'Wrote {coverage_csv}')
    print(f'Wrote {bstar_csv}')
    print(f'Wrote {summary_path}')

    saved_plots = plot_results(out_dir, rings, MENU_BX_LT_BY, coverage_table, bstar_rows, q=q)
    for path in saved_plots:
        print(f'Wrote {path}')

    print_bstar_table(bstar_rows, q=q)


if __name__ == '__main__':
    main()

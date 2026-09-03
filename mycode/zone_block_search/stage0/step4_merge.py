#!/usr/bin/env python3
"""Step 4: Rule-A debounce (2-ring persistence) + run-length merge.

Reads selected_Bstar.csv from Step 3, suppresses single-ring label flips,
merges contiguous identical labels into raw zones, and writes evaluation
artifacts. Does not compress to hardware zone count (Step 5).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mycode.zone_block_search.stage0.config import (
    DELTA_D,
    LIDAR_CENTER_XY,
    MENU_BX_LT_BY,
    RESULTS_DIR,
    menu_volume_rank,
    size_label,
)

# Table I stage-0 reference (not a search target; for evaluation only).
TABLE_I_REF = [
    {'zone_id': 0, 'outer_T': 64, 'size': '16x32x16'},
    {'zone_id': 1, 'outer_T': 512, 'size': '16x16x16'},
    {'zone_id': 2, 'outer_T': 768, 'size': '16x32x16'},
    {'zone_id': 3, 'outer_T': None, 'size': '64x64x16'},
]


def parse_size(label: str) -> Tuple[int, int, int]:
    parts = label.split('x')
    if len(parts) != 3:
        raise ValueError(f'Bad size label: {label!r}')
    return int(parts[0]), int(parts[1]), int(parts[2])


def load_bstar_rows(path: Path) -> List[dict]:
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: int(r['ring_j']))
    for row in rows:
        row['ring_j'] = int(row['ring_j'])
        row['inner_T'] = int(row['inner_T'])
        row['outer_T'] = int(row['outer_T'])
        row['n_samples_at_Bstar'] = int(row['n_samples_at_Bstar'] or 0)
        cov = row.get('coverage_at_Bstar', '')
        row['coverage_at_Bstar'] = float(cov) if cov not in ('', None) else float('nan')
    # Drop trailing no-label rings if any
    return [r for r in rows if r.get('Bstar')]


def debounce_rule_a(labels: Sequence[str], min_run: int = 2) -> Tuple[List[str], List[dict]]:
    """Confirm a label change only if the new label spans >= min_run consecutive rings.

    Rejected short runs keep the previous stable label.
    Returns (debounced_labels, event_log).
    """
    if not labels:
        return [], []
    n = len(labels)
    out = [None] * n
    events = []
    stable = labels[0]
    i = 0
    while i < n:
        if labels[i] == stable:
            out[i] = stable
            i += 1
            continue
        new = labels[i]
        j = i
        while j < n and labels[j] == new:
            j += 1
        run_len = j - i
        if run_len >= min_run:
            events.append({
                'ring_start': i,
                'ring_end_exclusive': j,
                'from': stable,
                'to': new,
                'run_len': run_len,
                'action': 'confirm',
            })
            stable = new
            for k in range(i, j):
                out[k] = stable
        else:
            events.append({
                'ring_start': i,
                'ring_end_exclusive': j,
                'from': stable,
                'to': new,
                'run_len': run_len,
                'action': 'reject',
            })
            for k in range(i, j):
                out[k] = stable
        i = j
    return out, events


def run_length_merge(
    labels: Sequence[str],
    delta_d: int = DELTA_D,
) -> List[dict]:
    """Merge consecutive identical labels into raw zones with half-open T bounds."""
    if not labels:
        return []
    zones = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i < len(labels) and labels[i] == labels[start]:
            continue
        bx, by, bz = parse_size(labels[start])
        inner_T = start * delta_d
        # Last zone is unbounded in XY beyond the profiled rings / FOV remainder.
        is_last = i == len(labels)
        outer_T = None if is_last else i * delta_d
        zones.append({
            'zone_id': len(zones),
            'ring_start': start,
            'ring_end_exclusive': i,
            'n_rings': i - start,
            'inner_T': inner_T,
            'outer_T': outer_T,
            'size': labels[start],
            'bx': bx,
            'by': by,
            'bz': bz,
        })
        start = i
    return zones


def load_coverage_lookup(path: Path) -> dict:
    """Map (ring_j, size_label) -> stats from coverage_Rjk.csv."""
    lookup = {}
    if not path.exists():
        return lookup
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            j = int(row['ring_j'])
            size = row['size']
            n = int(row['n_samples'] or 0)
            cov = row.get('coverage_Rjk', '')
            lookup[(j, size)] = {
                'n_samples': n,
                'coverage': float(cov) if cov not in ('', None) else float('nan'),
            }
    return lookup


def evaluate_zones(zones: List[dict], labels_raw: Sequence[str], labels_deb: Sequence[str],
                   coverage_lookup: dict, events: List[dict]) -> dict:
    flipped = [i for i, (a, b) in enumerate(zip(labels_raw, labels_deb)) if a != b]
    zone_stats = []
    for z in zones:
        n_sum = 0
        cov_weighted = 0.0
        cov_weight = 0
        ring_covs = []
        for j in range(z['ring_start'], z['ring_end_exclusive']):
            stats = coverage_lookup.get((j, z['size']))
            if stats is None or stats['n_samples'] <= 0:
                continue
            n_sum += stats['n_samples']
            if stats['coverage'] == stats['coverage']:
                cov_weighted += stats['coverage'] * stats['n_samples']
                cov_weight += stats['n_samples']
                ring_covs.append(stats['coverage'])
        pooled_cov = (cov_weighted / cov_weight) if cov_weight else float('nan')
        zone_stats.append({
            **z,
            'outer_T_display': '*' if z['outer_T'] is None else z['outer_T'],
            'n_samples_pooled': n_sum,
            'coverage_pooled': None if pooled_cov != pooled_cov else round(pooled_cov, 6),
            'coverage_min_ring': None if not ring_covs else round(min(ring_covs), 6),
        })

    return {
        'n_rings': len(labels_raw),
        'n_raw_unique_before': len(set(labels_raw)),
        'n_label_changes_before': sum(
            1 for i in range(1, len(labels_raw)) if labels_raw[i] != labels_raw[i - 1]
        ),
        'n_raw_zones_after': len(zones),
        'n_rings_relabeled': len(flipped),
        'relabeled_rings': flipped,
        'reject_events': [e for e in events if e['action'] == 'reject'],
        'confirm_events': [e for e in events if e['action'] == 'confirm'],
        'zones': zone_stats,
        'table_i_reference': TABLE_I_REF,
        'lidar_center_xy': list(LIDAR_CENTER_XY),
        'delta_d': DELTA_D,
        'rule': 'A_two_ring_persistence',
        'min_run': 2,
    }


def write_debounced_csv(path: Path, rows: List[dict], labels_deb: Sequence[str]):
    fieldnames = [
        'ring_j', 'inner_T', 'outer_T',
        'Bstar_raw', 'Bstar_debounced', 'changed',
        'n_samples_at_raw', 'coverage_at_raw',
    ]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, deb in zip(rows, labels_deb):
            writer.writerow({
                'ring_j': row['ring_j'],
                'inner_T': row['inner_T'],
                'outer_T': row['outer_T'],
                'Bstar_raw': row['Bstar'],
                'Bstar_debounced': deb,
                'changed': int(row['Bstar'] != deb),
                'n_samples_at_raw': row['n_samples_at_Bstar'],
                'coverage_at_raw': (
                    '' if row['coverage_at_Bstar'] != row['coverage_at_Bstar']
                    else f"{row['coverage_at_Bstar']:.6f}"
                ),
            })


def write_raw_zones_csv(path: Path, zones: List[dict]):
    fieldnames = [
        'zone_id', 'ring_start', 'ring_end_exclusive', 'n_rings',
        'inner_T', 'outer_T', 'size', 'bx', 'by', 'bz',
        'n_samples_pooled', 'coverage_pooled', 'coverage_min_ring',
    ]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for z in zones:
            writer.writerow({
                'zone_id': z['zone_id'],
                'ring_start': z['ring_start'],
                'ring_end_exclusive': z['ring_end_exclusive'],
                'n_rings': z['n_rings'],
                'inner_T': z['inner_T'],
                'outer_T': '*' if z['outer_T'] is None else z['outer_T'],
                'size': z['size'],
                'bx': z['bx'],
                'by': z['by'],
                'bz': z['bz'],
                'n_samples_pooled': z.get('n_samples_pooled', ''),
                'coverage_pooled': (
                    '' if z.get('coverage_pooled') is None else f"{z['coverage_pooled']:.6f}"
                ),
                'coverage_min_ring': (
                    '' if z.get('coverage_min_ring') is None else f"{z['coverage_min_ring']:.6f}"
                ),
            })


def plot_labels(out_dir: Path, rows: List[dict], labels_deb: Sequence[str], zones: List[dict]):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available; skip plot')
        return None

    menu_labels = [size_label(*s) for s in MENU_BX_LT_BY]
    rank = {lab: i for i, lab in enumerate(menu_labels)}
    xs = [r['ring_j'] for r in rows]
    y_raw = [rank[r['Bstar']] for r in rows]
    y_deb = [rank[lab] for lab in labels_deb]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(xs, y_raw, where='mid', color='C1', linewidth=1.2, alpha=0.85, label='B*(j) raw')
    ax.step(xs, y_deb, where='mid', color='C0', linewidth=1.8, label='Rule A debounced')
    ax.scatter(xs, y_raw, s=18, color='C1', zorder=3)
    ax.scatter(xs, y_deb, s=18, color='C0', zorder=4)
    for z in zones:
        lo = z['ring_start'] - 0.5
        hi = z['ring_end_exclusive'] - 0.5
        ax.axvspan(lo, hi, alpha=0.08, color='C0')
        mid = 0.5 * (z['ring_start'] + z['ring_end_exclusive'] - 1)
        ax.text(
            mid, rank[z['size']] + 0.35, f"Z{z['zone_id']}\n{z['size']}",
            ha='center', va='bottom', fontsize=8, color='0.25',
        )
    ax.set_yticks(range(len(menu_labels)))
    ax.set_yticklabels(menu_labels)
    ax.set_xlabel('Ring index j')
    ax.set_ylabel('Block size')
    ax.set_title('Step 4 Rule A debounce + RLE raw zones')
    ax.legend(loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    path = out_dir / 'step4_labels.png'
    fig.savefig(path, dpi=160)
    fig.savefig(out_dir / 'step4_labels.svg')
    plt.close(fig)
    return path


def print_report(eval_doc: dict):
    print('\n=== Step 4: Rule A (2-ring) + RLE ===')
    print(f"rings={eval_doc['n_rings']}  label_changes_before={eval_doc['n_label_changes_before']}  "
          f"raw_zones_after={eval_doc['n_raw_zones_after']}  "
          f"rings_relabeled={eval_doc['n_rings_relabeled']}")
    if eval_doc['reject_events']:
        print('Rejected short runs:')
        for e in eval_doc['reject_events']:
            print(
                f"  rings [{e['ring_start']},{e['ring_end_exclusive']}) "
                f"{e['to']} (len={e['run_len']}) → keep {e['from']}"
            )
    if eval_doc['confirm_events']:
        print('Confirmed transitions:')
        for e in eval_doc['confirm_events']:
            print(
                f"  rings [{e['ring_start']},{e['ring_end_exclusive']}) "
                f"{e['from']} → {e['to']} (len={e['run_len']})"
            )
    print('\nRaw zones (left-closed / right-open T about LiDAR center):')
    print(f"{'id':>3}  {'[Tin,Tout)':>14}  {'size':>12}  {'rings':>7}  {'n_pool':>8}  {'R_pool':>7}  {'R_min':>7}")
    print('-' * 72)
    for z in eval_doc['zones']:
        tout = '*' if z['outer_T'] is None else str(z['outer_T'])
        interval = f"[{z['inner_T']},{tout})"
        rp = '-' if z['coverage_pooled'] is None else f"{z['coverage_pooled']:.3f}"
        rm = '-' if z['coverage_min_ring'] is None else f"{z['coverage_min_ring']:.3f}"
        print(
            f"{z['zone_id']:3d}  {interval:>14}  {z['size']:>12}  "
            f"{z['n_rings']:7d}  {z['n_samples_pooled']:8d}  {rp:>7}  {rm:>7}"
        )
    print('-' * 72)
    print('Table I reference (stage0): T=64/512/768 → 16x32, 16x16, 16x32, 64x64')
    print('Step 5 (hardware 4-zone merge / Golden model) not run.')


def main():
    parser = argparse.ArgumentParser(description='Step 4 Rule A debounce + RLE')
    parser.add_argument('--in_dir', type=str, default=str(RESULTS_DIR))
    parser.add_argument('--bstar_csv', type=str, default='')
    parser.add_argument('--min_run', type=int, default=2)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    bstar_path = Path(args.bstar_csv) if args.bstar_csv else in_dir / 'selected_Bstar.csv'
    coverage_path = in_dir / 'coverage_Rjk.csv'
    if not bstar_path.exists():
        raise FileNotFoundError(bstar_path)

    rows = load_bstar_rows(bstar_path)
    labels_raw = [r['Bstar'] for r in rows]
    labels_deb, events = debounce_rule_a(labels_raw, min_run=args.min_run)
    zones = run_length_merge(labels_deb, delta_d=DELTA_D)
    coverage_lookup = load_coverage_lookup(coverage_path)
    eval_doc = evaluate_zones(zones, labels_raw, labels_deb, coverage_lookup, events)

    write_debounced_csv(in_dir / 'debounced_labels.csv', rows, labels_deb)
    write_raw_zones_csv(in_dir / 'raw_zones.csv', eval_doc['zones'])
    with (in_dir / 'raw_zones.json').open('w') as handle:
        json.dump(eval_doc, handle, indent=2)

    plot_path = plot_labels(in_dir, rows, labels_deb, eval_doc['zones'])
    print(f'Wrote {in_dir / "debounced_labels.csv"}')
    print(f'Wrote {in_dir / "raw_zones.csv"}')
    print(f'Wrote {in_dir / "raw_zones.json"}')
    if plot_path:
        print(f'Wrote {plot_path}')
    print_report(eval_doc)


if __name__ == '__main__':
    main()

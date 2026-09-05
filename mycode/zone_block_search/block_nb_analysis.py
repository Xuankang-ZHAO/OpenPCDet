"""Shared halo-aware N_b histogram for the closed final zone/block LUTs.

Collects materialized-block voxel counts (active + halo copies, including
halo-only blocks) over the profiling frame list, then bins them in closed
intervals of width 16: 1-16, 17-32, ...
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from mycode.rtl_unfixed.partition import (
    ZoneSpec,
    _ilog2,
    _validate_nested_squares_and_block_alignment,
    compute_rtl_unfixed_partition_counts,
    summarize_zone_specs,
)

BIN_WIDTH = 16
TWO_PAGE_LIMIT = 128

# Closed final configs from Zone_Block_Methodology.md.
# Each entry is (zone_id, outer_half_open_T or None, (Bx, By, Bz)).
FINAL_LUTS: Dict[int, List[Tuple[int, Optional[int], Tuple[int, int, int]]]] = {
    0: [
        (0, 192, (8, 8, 16)),
        (1, 384, (16, 16, 16)),
        (2, 576, (32, 32, 16)),
        (3, None, (64, 64, 16)),
    ],
    1: [
        (0, 224, (8, 8, 8)),
        (1, 352, (16, 8, 8)),
        (2, 448, (16, 16, 8)),
        (3, None, (32, 16, 8)),
    ],
    2: [
        (0, 96, (4, 4, 8)),
        (1, 160, (8, 4, 8)),
        (2, 272, (8, 8, 8)),
        (3, None, (16, 16, 8)),
    ],
    3: [
        (0, 16, (16, 8, 8)),
        (1, 64, (4, 4, 8)),
        (2, 96, (8, 4, 8)),
        (3, None, (8, 8, 8)),
    ],
}


def _parse_outer(text: str) -> Optional[int]:
    token = str(text).strip()
    if token in ('', '*', 'inf', '+inf', 'None'):
        return None
    return int(float(token)) if token else None


def lut_lines_from_final(stage: int) -> List[str]:
    if stage not in FINAL_LUTS:
        raise KeyError(f'No final LUT for stage {stage}')
    lines = []
    for zone_id, outer, size in FINAL_LUTS[stage]:
        outer_s = '*' if outer is None else str(outer)
        bx, by, bz = size
        lines.append(f'{zone_id}:{outer_s}:{bx},{by},{bz}')
    return lines


def zone_specs_from_lut_lines(lines: Sequence[str]) -> List[ZoneSpec]:
    """Build nested zone specs without rtl_unfixed's Bz==16 LUT constraint."""
    parsed: List[Tuple[int, Optional[int], Tuple[int, int, int]]] = []
    for line in lines:
        token = line.strip()
        if not token or token.startswith('#'):
            continue
        zone_s, bound_s, size_s = token.split(':')
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


def write_lut(path: Path, lines: Sequence[str], stage: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        handle.write(f'# Closed final Stage-{stage} zone LUT (halo N_b analysis)\n')
        handle.write('# zone:outer_half_open_T:bx,by,bz\n')
        for line in lines:
            handle.write(line + '\n')


def make_bin_edges(max_nb: int, width: int = BIN_WIDTH) -> List[Tuple[int, int]]:
    if max_nb < 1:
        return [(1, width)]
    n_bins = (int(max_nb) - 1) // width + 1
    return [(i * width + 1, (i + 1) * width) for i in range(n_bins)]


def histogram_rows(nb: np.ndarray, scope: str, bin_edges: Sequence[Tuple[int, int]]) -> List[dict]:
    total = int(nb.size)
    rows = []
    for lo, hi in bin_edges:
        if total == 0:
            count = 0
        else:
            count = int(np.sum((nb >= lo) & (nb <= hi)))
        pct = float(count / total) if total else 0.0
        rows.append({
            'scope': scope,
            'bin_lo': lo,
            'bin_hi': hi,
            'bin_label': f'{lo}-{hi}',
            'n_blocks': count,
            'pct': pct,
        })
    return rows


def nb_summary(nb: np.ndarray, two_page_limit: int = TWO_PAGE_LIMIT) -> dict:
    if nb.size == 0:
        return {
            'n_blocks': 0,
            'mean_nb': float('nan'),
            'median_nb': float('nan'),
            'max_nb': 0,
            'n_le_two_page': 0,
            'two_page_coverage': float('nan'),
        }
    le = int(np.sum(nb <= two_page_limit))
    return {
        'n_blocks': int(nb.size),
        'mean_nb': float(np.mean(nb)),
        'median_nb': float(np.median(nb)),
        'max_nb': int(nb.max()),
        'n_le_two_page': le,
        'two_page_coverage': float(le / nb.size),
    }


def collect_materialized_nb(
    coords_list: Sequence[np.ndarray],
    grid_size: Tuple[int, int, int],
    zone_specs: Sequence[ZoneSpec],
    lidar_center: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return concatenated N_b and zone_id arrays over all frames."""
    all_nb: List[np.ndarray] = []
    all_zones: List[np.ndarray] = []
    for coords in tqdm(coords_list, desc='Materialized N_b'):
        counts, n_blocks, _, keys = compute_rtl_unfixed_partition_counts(
            coords,
            grid_size,
            zone_specs,
            lidar_center,
            return_keys=True,
        )
        if n_blocks == 0:
            continue
        all_nb.append(counts)
        all_zones.append(keys[:, 0])
    if not all_nb:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(all_nb), np.concatenate(all_zones)


def _try_plot(rows: List[dict], out_path: Path, title: str) -> None:
    all_rows = [r for r in rows if r['scope'] == 'all']
    if not all_rows:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'plot skipped: {exc}')
        return

    labels = [r['bin_label'] for r in all_rows]
    counts = [r['n_blocks'] for r in all_rows]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(labels)), 3.8))
    ax.bar(range(len(labels)), counts, color='steelblue', edgecolor='white')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('N_b (stored voxels including halo)')
    ax.set_ylabel('Materialized blocks')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def print_histogram(rows: List[dict], scope: str = 'all') -> None:
    scoped = [r for r in rows if r['scope'] == scope]
    total = sum(r['n_blocks'] for r in scoped)
    print(f'\n=== {scope}: N_b bins (width={BIN_WIDTH}, n={total}) ===')
    print(f'{"bin":>12}  {"n_blocks":>12}  {"pct":>8}')
    for row in scoped:
        print(f'{row["bin_label"]:>12}  {row["n_blocks"]:12d}  {100.0 * row["pct"]:7.2f}%')


def run_analysis(
    stage: int,
    frame_ids: Sequence[str],
    coords_list: Sequence[np.ndarray],
    grid_size: Tuple[int, int, int],
    lidar_center: Tuple[int, int],
    out_dir: Path,
    two_page_limit: int = TWO_PAGE_LIMIT,
    cache_meta: Optional[dict] = None,
) -> dict:
    lines = lut_lines_from_final(stage)
    zone_specs = zone_specs_from_lut_lines(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    lut_path = out_dir / 'final_zones_lut.txt'
    write_lut(lut_path, lines, stage)

    print(f'Stage {stage} final LUT: {summarize_zone_specs(zone_specs)}')
    print(f'grid={tuple(grid_size)} lidar={tuple(lidar_center)} frames={len(frame_ids)}')

    nb, zone_ids = collect_materialized_nb(coords_list, grid_size, zone_specs, lidar_center)
    bin_edges = make_bin_edges(int(nb.max()) if nb.size else 0)

    rows = histogram_rows(nb, 'all', bin_edges)
    zone_summaries = {'all': nb_summary(nb, two_page_limit)}
    for spec in zone_specs:
        mask = zone_ids == spec.zone_id
        scope = f'zone{spec.zone_id}'
        rows.extend(histogram_rows(nb[mask], scope, bin_edges))
        zone_summaries[scope] = nb_summary(nb[mask], two_page_limit)

    csv_path = out_dir / 'nb_bin_histogram.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['scope', 'bin_lo', 'bin_hi', 'bin_label', 'n_blocks', 'pct'],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'stage': stage,
        'lut': summarize_zone_specs(zone_specs),
        'lut_path': str(lut_path),
        'n_frames': len(frame_ids),
        'grid_size_xyz': list(grid_size),
        'lidar_center_xy': list(lidar_center),
        'halo': True,
        'bin_width': BIN_WIDTH,
        'two_page_limit': two_page_limit,
        'cache': cache_meta or {},
        'summaries': zone_summaries,
        'bins': rows,
    }
    json_path = out_dir / 'nb_bin_summary.json'
    with json_path.open('w') as handle:
        json.dump(summary, handle, indent=2)

    plot_path = out_dir / 'nb_bin_histogram.png'
    _try_plot(rows, plot_path, f'Stage {stage} materialized-block N_b (halo on)')

    print(f'Wrote {csv_path}')
    print(f'Wrote {json_path}')
    print_histogram(rows, 'all')
    for spec in zone_specs:
        print_histogram(rows, f'zone{spec.zone_id}')
    print(
        f"\nall: mean={zone_summaries['all']['mean_nb']:.2f}  "
        f"median={zone_summaries['all']['median_nb']:.1f}  "
        f"max={zone_summaries['all']['max_nb']}  "
        f"R(N_b<={two_page_limit})={zone_summaries['all']['two_page_coverage']:.4f}"
    )
    return summary


def run_stage_cli(
    stage: int,
    load_frames: Callable,
    grid_size: Tuple[int, int, int],
    lidar_center: Tuple[int, int],
    default_out_dir: Path,
    default_cache: Path,
    default_list_file: Path,
    two_page_limit: int = TWO_PAGE_LIMIT,
) -> dict:
    parser = argparse.ArgumentParser(
        description=f'Stage {stage} final-config block N_b histogram (halo on)'
    )
    parser.add_argument('--cache', type=str, default=str(default_cache))
    parser.add_argument('--list_file', type=str, default=str(default_list_file))
    parser.add_argument('--out_dir', type=str, default=str(default_out_dir))
    parser.add_argument('--max_frames', type=int, default=0)
    args = parser.parse_args()

    frame_ids, coords_list, meta = load_frames(
        list_file=Path(args.list_file),
        cache_path=Path(args.cache),
    )
    if args.max_frames and args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
        coords_list = coords_list[: args.max_frames]
    print(f'Loaded {len(frame_ids)} frames (cache={meta.get("from_cache")})')

    return run_analysis(
        stage=stage,
        frame_ids=frame_ids,
        coords_list=coords_list,
        grid_size=grid_size,
        lidar_center=lidar_center,
        out_dir=Path(args.out_dir),
        two_page_limit=two_page_limit,
        cache_meta={
            'from_cache': bool(meta.get('from_cache')),
            'cache_path': str(meta.get('cache_path', args.cache)),
        },
    )

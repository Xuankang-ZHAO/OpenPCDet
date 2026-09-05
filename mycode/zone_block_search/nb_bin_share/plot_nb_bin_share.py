#!/usr/bin/env python3
"""Plot per-stage N_b bin shares (width=16) from final_nb_hist CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BIN_WIDTH = 16
ONE_PAGE = 64
TWO_PAGE = 128
STAGES = (0, 1, 2, 3)

PACKAGE_DIR = Path(__file__).resolve().parent
SEARCH_DIR = PACKAGE_DIR.parent

COLOR_LE64 = '#4C78A8'
COLOR_LE128 = '#F2A900'
COLOR_RESHAPE = '#E45756'


def default_csv(stage: int) -> Path:
    return SEARCH_DIR / f'stage{stage}' / 'results' / 'final_nb_hist' / 'nb_bin_histogram.csv'


def load_scope_rows(csv_path: Path, scope: str = 'all') -> List[dict]:
    with csv_path.open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['scope'] == scope]
    if not rows:
        raise ValueError(f'No rows with scope={scope!r} in {csv_path}')
    parsed = []
    for row in rows:
        parsed.append({
            'bin_lo': int(row['bin_lo']),
            'bin_hi': int(row['bin_hi']),
            'bin_label': row['bin_label'],
            'n_blocks': int(row['n_blocks']),
            'pct': float(row['pct']),
        })
    return parsed


def collapse_tail(rows: Sequence[dict], display_hi: int) -> List[dict]:
    """Keep bins with bin_hi <= display_hi; merge the rest into one overflow bin."""
    kept = [dict(row) for row in rows if row['bin_hi'] <= display_hi]
    tail = [row for row in rows if row['bin_hi'] > display_hi]
    if not tail:
        return kept
    overflow_lo = display_hi + 1
    overflow_hi = max(row['bin_hi'] for row in tail)
    kept.append({
        'bin_lo': overflow_lo,
        'bin_hi': overflow_hi,
        'bin_label': f'{overflow_lo}+',
        'n_blocks': sum(row['n_blocks'] for row in tail),
        'pct': sum(row['pct'] for row in tail),
    })
    return kept


def cumulative_share(rows: Sequence[dict], limit: int) -> float:
    return sum(row['pct'] for row in rows if row['bin_hi'] <= limit)


def bar_color(bin_hi: int, label: str) -> str:
    if label.endswith('+') and bin_hi > TWO_PAGE:
        return COLOR_RESHAPE
    if bin_hi <= ONE_PAGE:
        return COLOR_LE64
    if bin_hi <= TWO_PAGE:
        return COLOR_LE128
    return COLOR_RESHAPE


def apply_axes_style(ax) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def draw_stage_bars(
    ax,
    rows: Sequence[dict],
    title: str,
    ylabel: bool = True,
    annotate: bool = True,
    source_rows: Optional[Sequence[dict]] = None,
) -> None:
    labels = [row['bin_label'] for row in rows]
    shares = [100.0 * row['pct'] for row in rows]
    colors = [bar_color(row['bin_hi'], row['bin_label']) for row in rows]
    xs = range(len(rows))
    ax.bar(xs, shares, color=colors, edgecolor='white', linewidth=0.4, width=0.86)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r'$N_b$ interval (bin width = 16)')
    if ylabel:
        ax.set_ylabel('Share of materialized blocks (%)')
    ymax = max(shares) if shares else 1.0
    ax.set_ylim(0.0, max(ymax * 1.18, 5.0))
    apply_axes_style(ax)

    if annotate:
        ref_rows = source_rows if source_rows is not None else rows
        le64 = 100.0 * cumulative_share(ref_rows, ONE_PAGE)
        le128 = 100.0 * cumulative_share(ref_rows, TWO_PAGE)
        ax.text(
            0.98,
            0.96,
            f'$N_b\\leq 64$: {le64:.2f}%\n$N_b\\leq 128$: {le128:.2f}%',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=8,
            bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': '#cccccc', 'alpha': 0.92},
        )


def legend_handles() -> List[Patch]:
    return [
        Patch(facecolor=COLOR_LE64, edgecolor='white', label=r'$N_b\leq 64$ (1 page)'),
        Patch(facecolor=COLOR_LE128, edgecolor='white', label=r'$65\leq N_b\leq 128$ (2 pages)'),
        Patch(facecolor=COLOR_RESHAPE, edgecolor='white', label=r'$N_b>128$ (reshape)'),
    ]


def save_figure(fig, out_dir: Path, stem: str) -> Tuple[Path, Path]:
    png_path = out_dir / f'{stem}.png'
    svg_path = out_dir / f'{stem}.svg'
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    return png_path, svg_path


def plot_stage(rows: Sequence[dict], stage: int, out_dir: Path, display_hi: int) -> Tuple[Path, Path]:
    plot_rows = collapse_tail(rows, display_hi)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    draw_stage_bars(ax, plot_rows, f'Stage {stage}', source_rows=rows)
    ax.legend(handles=legend_handles(), loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, out_dir, f'stage{stage}_nb_bin_share')


def plot_all_stages(
    stage_rows: Dict[int, List[dict]],
    out_dir: Path,
    display_hi: int,
) -> Tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.6), sharey=False)
    for ax, stage in zip(axes.ravel(), STAGES):
        rows = stage_rows[stage]
        plot_rows = collapse_tail(rows, display_hi)
        draw_stage_bars(ax, plot_rows, f'Stage {stage}', source_rows=rows)
    fig.legend(handles=legend_handles(), loc='upper center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.supxlabel(r'$N_b$ interval (bin width = 16)', y=0.01, fontsize=10)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    return save_figure(fig, out_dir, 'all_stages_nb_bin_share')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot final-config N_b bin shares (width=16)')
    parser.add_argument('--out_dir', type=str, default=str(PACKAGE_DIR))
    parser.add_argument('--display_hi', type=int, default=256, help='Keep explicit bins up to this N_b; merge the tail')
    parser.add_argument('--scope', type=str, default='all')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_rows: Dict[int, List[dict]] = {}
    for stage in STAGES:
        csv_path = default_csv(stage)
        rows = load_scope_rows(csv_path, scope=args.scope)
        stage_rows[stage] = rows
        n_blocks = sum(row['n_blocks'] for row in rows)
        le64 = 100.0 * cumulative_share(rows, ONE_PAGE)
        le128 = 100.0 * cumulative_share(rows, TWO_PAGE)
        print(
            f'Stage {stage}: {csv_path}  n={n_blocks}  '
            f'N_b<=64={le64:.3f}%  N_b<=128={le128:.3f}%'
        )
        png_path, svg_path = plot_stage(rows, stage, out_dir, args.display_hi)
        print(f'  wrote {png_path.name} / {svg_path.name}')

    png_path, svg_path = plot_all_stages(stage_rows, out_dir, args.display_hi)
    print(f'wrote {png_path}')
    print(f'wrote {svg_path}')


if __name__ == '__main__':
    main()

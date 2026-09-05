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
        'bin_label': f'>{display_hi}',
        'n_blocks': sum(row['n_blocks'] for row in tail),
        'pct': sum(row['pct'] for row in tail),
    })
    return kept


def cumulative_share(rows: Sequence[dict], limit: int) -> float:
    return sum(row['pct'] for row in rows if row['bin_hi'] <= limit)


def bar_color(bin_hi: int, label: str) -> str:
    if label.startswith('>') or (label.endswith('+') and bin_hi > TWO_PAGE):
        return COLOR_RESHAPE
    if bin_hi <= ONE_PAGE:
        return COLOR_LE64
    if bin_hi <= TWO_PAGE:
        return COLOR_LE128
    return COLOR_RESHAPE


def apply_axes_style(ax) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', pad=3.0)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def add_axis_arrows(ax) -> None:
    arrowprops = {
        'arrowstyle': '-|>',
        'mutation_scale': 9,
        'color': 'black',
        'lw': 0.8,
        'shrinkA': 0,
        'shrinkB': 0,
        'clip_on': False,
    }
    ax.annotate(
        '',
        xy=(1.04, 0.0),
        xytext=(1.0, 0.0),
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=arrowprops,
        clip_on=False,
        annotation_clip=False,
    )
    ax.annotate(
        '',
        xy=(0.0, 1.04),
        xytext=(0.0, 1.0),
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=arrowprops,
        clip_on=False,
        annotation_clip=False,
    )


def draw_stage_bars(
    ax,
    rows: Sequence[dict],
    title: str,
    ylabel: bool = True,
    xlabel: Optional[str] = None,
    title_below: bool = False,
    annotate: bool = True,
    source_rows: Optional[Sequence[dict]] = None,
) -> None:
    shares = [100.0 * row['pct'] for row in rows]
    colors = [bar_color(row['bin_hi'], row['bin_label']) for row in rows]
    n_bins = len(rows)
    xs = range(n_bins)
    ax.bar(xs, shares, color=colors, edgecolor='none', width=1.0, align='center')
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.margins(x=0)
    tick_pos = []
    tick_labels = []
    for i, row in enumerate(rows):
        if row['bin_label'].startswith('>'):
            continue
        tick_pos.append(i + 0.5)
        tick_labels.append(str(row['bin_hi']))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    if title_below:
        ax.set_xlabel(title, fontsize=11, labelpad=2)
    else:
        ax.set_title(title, fontsize=11)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel('Fraction of nonempty blocks (%)', fontsize=12, labelpad=6)
    ymax = max(shares) if shares else 1.0
    ax.set_ylim(0.0, max(ymax * 1.18, 5.0))
    apply_axes_style(ax)
    add_axis_arrows(ax)

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
            fontsize=12,
            bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': '#cccccc', 'alpha': 0.92},
        )


def legend_handles() -> List[Patch]:
    return [
        Patch(facecolor=COLOR_LE64, edgecolor='white', label='$N_b\\leq 64$\n(1 page)'),
        Patch(facecolor=COLOR_LE128, edgecolor='white', label='$65\\leq N_b\\leq 128$\n(2 pages)'),
        Patch(facecolor=COLOR_RESHAPE, edgecolor='white', label='$N_b>128$\n(reshape)'),
    ]


def save_figure(fig, out_dir: Path, stem: str) -> Tuple[Path, Path]:
    png_path = out_dir / f'{stem}.png'
    svg_path = out_dir / f'{stem}.svg'
    fig.savefig(png_path, dpi=200, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(svg_path, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    return png_path, svg_path


XLABEL_NB = rf'$N_b$ (bin width = {BIN_WIDTH})'


def plot_stage(rows: Sequence[dict], stage: int, out_dir: Path, display_hi: int) -> Tuple[Path, Path]:
    plot_rows = collapse_tail(rows, display_hi)
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    draw_stage_bars(ax, plot_rows, f'Stage {stage}', xlabel=XLABEL_NB, source_rows=rows)
    ax.legend(handles=legend_handles(), loc='upper center', bbox_to_anchor=(0.5, 1.28), ncol=3, frameon=False, fontsize=10)
    fig.tight_layout()
    return save_figure(fig, out_dir, f'stage{stage}_nb_bin_share')


def plot_all_stages(
    stage_rows: Dict[int, List[dict]],
    out_dir: Path,
    display_hi: int,
) -> Tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.7), sharey=False)
    for index, (ax, stage) in enumerate(zip(axes.ravel(), STAGES)):
        rows = stage_rows[stage]
        plot_rows = collapse_tail(rows, display_hi)
        panel = chr(ord('a') + index)
        draw_stage_bars(
            ax,
            plot_rows,
            f'({panel}) Stage {stage}',
            ylabel=False,
            title_below=True,
            source_rows=rows,
        )
    fig.supylabel('Fraction of nonempty blocks (%)', fontsize=12, x=0.02)
    fig.tight_layout(rect=(0.02, 0.018, 1.0, 0.97), h_pad=1.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y0 = min(
        inv.transform((0.0, ax.get_tightbbox(renderer).y0))[1]
        for ax in axes[1, :]
    )
    y1 = max(
        inv.transform((0.0, ax.get_tightbbox(renderer).y1))[1]
        for ax in axes[0, :]
    )
    fig.legend(
        handles=legend_handles(),
        loc='lower center',
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, min(y1 + 0.008, 0.99)),
    )
    fig.supxlabel(XLABEL_NB, y=max(y0 - 0.024, 0.0), fontsize=12)
    return save_figure(fig, out_dir, 'all_stages_nb_bin_share')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot final-config N_b bin shares (width=16)')
    parser.add_argument('--out_dir', type=str, default=str(PACKAGE_DIR))
    parser.add_argument('--display_hi', type=int, default=128, help='Keep explicit bins up to this N_b; merge the tail')
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

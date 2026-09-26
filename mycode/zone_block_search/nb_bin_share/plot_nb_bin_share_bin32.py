#!/usr/bin/env python3
"""Plot per-stage N_b bin shares (width=32) by rebinning final_nb_hist CSVs.

Four stage panels are placed in one row at IEEE TCAS-I single-column width.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from plot_nb_bin_share import (  # noqa: E402
    BAR_WIDTH,
    COLOR_LE128,
    COLOR_LE64,
    COLOR_RESHAPE,
    ONE_PAGE,
    STAGES,
    TWO_PAGE,
    apply_axes_style,
    bar_color,
    collapse_tail,
    cumulative_share,
    default_csv,
    load_scope_rows,
)

BIN_WIDTH = 32
IEEE_COLUMN_INCHES = 3.5
XLABEL_NB = rf'$N_v$ (bin width: {BIN_WIDTH})'


def rebin_rows(rows: Sequence[dict], width: int = BIN_WIDTH) -> List[dict]:
    """Merge consecutive source bins so each displayed bin has the given width."""
    grouped: Dict[Tuple[int, int], dict] = {}
    order: List[Tuple[int, int]] = []
    for row in rows:
        new_hi = ((int(row['bin_hi']) + width - 1) // width) * width
        new_lo = new_hi - width + 1
        key = (new_lo, new_hi)
        if key not in grouped:
            grouped[key] = {
                'bin_lo': new_lo,
                'bin_hi': new_hi,
                'bin_label': f'{new_lo}-{new_hi}',
                'n_blocks': 0,
                'pct': 0.0,
            }
            order.append(key)
        grouped[key]['n_blocks'] += int(row['n_blocks'])
        grouped[key]['pct'] += float(row['pct'])
    return [grouped[key] for key in order]


def configure_style() -> None:
    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 7,
            'axes.labelsize': 7,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.fontsize': 7,
            'axes.linewidth': 0.6,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'xtick.major.size': 2.2,
            'ytick.major.size': 2.2,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'svg.fonttype': 'none',
        }
    )


def draw_xaxis_break_compact(ax, x: float) -> None:
    yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
    dy = 0.035 * yspan
    dx = 0.14
    gap = 0.09
    ax.plot(
        [x - 0.16, x + 0.22],
        [0.0, 0.0],
        color='white',
        lw=1.4,
        solid_capstyle='butt',
        clip_on=False,
        zorder=5,
    )
    for shift in (-gap / 2.0, gap / 2.0):
        ax.plot(
            [x + shift - dx, x + shift + dx],
            [-dy, dy],
            color='black',
            lw=0.55,
            solid_capstyle='butt',
            clip_on=False,
            zorder=6,
        )


def add_axis_arrows_compact(ax) -> None:
    arrowprops = {
        'arrowstyle': '-|>',
        'mutation_scale': 5,
        'color': 'black',
        'lw': 0.5,
        'shrinkA': 0,
        'shrinkB': 0,
        'clip_on': False,
    }
    ax.annotate(
        '',
        xy=(1.02, 0.0),
        xytext=(1.0, 0.0),
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=arrowprops,
        clip_on=False,
        annotation_clip=False,
    )
    ax.annotate(
        '',
        xy=(0.0, 1.02),
        xytext=(0.0, 1.0),
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=arrowprops,
        clip_on=False,
        annotation_clip=False,
    )


def draw_stage_bars_compact(ax, rows: Sequence[dict], stage: int) -> None:
    regular = [row for row in rows if not row['bin_label'].startswith('>')]
    overflow = [row for row in rows if row['bin_label'].startswith('>')]
    shares = [100.0 * row['pct'] for row in regular]
    colors = [bar_color(row['bin_hi'], row['bin_label']) for row in regular]
    n_reg = len(regular)
    xs = list(range(n_reg))
    ax.bar(xs, shares, color=colors, edgecolor='none', width=BAR_WIDTH, align='center')
    x_right = n_reg - 0.5
    overflow_x = None
    overflow_pct = 0.0
    if overflow:
        overflow_x = n_reg + 0.38
        overflow_pct = 100.0 * overflow[0]['pct']
        ax.bar(
            [overflow_x],
            [overflow_pct],
            color=bar_color(overflow[0]['bin_hi'], overflow[0]['bin_label']),
            edgecolor='none',
            width=BAR_WIDTH,
            align='center',
        )
        x_right = overflow_x + 0.28
    ax.set_xlim(-0.48, x_right)
    ax.margins(x=0)
    # Labels sit on bin edges, midway between adjacent bars.
    ax.set_xticks([i + 0.5 for i in xs])
    ax.set_xticklabels([str(row['bin_hi']) for row in regular], fontsize=6)
    all_shares = shares + [100.0 * row['pct'] for row in overflow]
    ymax = max(all_shares) if all_shares else 1.0
    ax.set_ylim(0.0, max(ymax * 1.16, 5.0))
    apply_axes_style(ax)
    ax.tick_params(axis='x', pad=0.6, labelsize=6)
    ax.tick_params(axis='y', pad=1.0, labelsize=6)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.45, alpha=0.7)
    if overflow and overflow_x is not None:
        draw_xaxis_break_compact(ax, n_reg - 0.5 + 0.32)
        ax.text(
            overflow_x,
            overflow_pct + 0.04 * ax.get_ylim()[1],
            r'$>\!128$',
            ha='center',
            va='bottom',
            fontsize=6,
            color='#333333',
            clip_on=False,
        )
    add_axis_arrows_compact(ax)
    ax.text(
        0.98,
        0.97,
        f'Stage {stage}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=7,
        color='#222222',
        clip_on=False,
    )


def legend_handles() -> List[Patch]:
    return [
        Patch(facecolor=COLOR_LE64, edgecolor='white', label=r'$N_v\leq 64$'),
        Patch(facecolor=COLOR_LE128, edgecolor='white', label=r'$65\leq N_v\leq 128$'),
        Patch(facecolor=COLOR_RESHAPE, edgecolor='white', label=r'$N_v>128$'),
    ]


def save_column_figure(fig, out_dir: Path, stem: str) -> Tuple[Path, Path]:
    """Save PNG/SVG at column width, trimming empty bands above and below the ink."""
    from io import BytesIO

    import numpy as np
    from PIL import Image

    dpi = 600
    width = fig.get_figwidth()
    height = fig.get_figheight()
    canvas = Bbox.from_bounds(0.0, 0.0, width, height)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches=canvas, pad_inches=0)
    buf.seek(0)
    pixels = np.asarray(Image.open(buf).convert('RGB'))
    ink_rows = np.where(~np.all(pixels == 255, axis=2).all(axis=1))[0]
    pad_px = int(round(0.015 * dpi))
    top = max(int(ink_rows[0]) - pad_px, 0)
    bottom = min(int(ink_rows[-1]) + pad_px, pixels.shape[0] - 1)
    export_bbox = Bbox.from_extents(
        0.0,
        height - (bottom + 1) / dpi,
        width,
        height - top / dpi,
    )
    png_path = out_dir / f'{stem}.png'
    svg_path = out_dir / f'{stem}.svg'
    fig.savefig(png_path, dpi=dpi, bbox_inches=export_bbox, pad_inches=0)
    fig.savefig(svg_path, bbox_inches=export_bbox, pad_inches=0)
    plt.close(fig)
    return png_path, svg_path


def plot_all_stages(
    stage_rows: Dict[int, List[dict]],
    out_dir: Path,
    display_hi: int,
) -> Tuple[Path, Path]:
    configure_style()
    fig, axes = plt.subplots(1, 4, figsize=(IEEE_COLUMN_INCHES, 1.40), sharey=False)
    fig.subplots_adjust(left=0.093, right=0.972, bottom=0.155, top=0.90, wspace=0.42)
    for ax, stage in zip(axes, STAGES):
        plot_rows = collapse_tail(stage_rows[stage], display_hi)
        draw_stage_bars_compact(ax, plot_rows, stage)
    fig.legend(
        handles=legend_handles(),
        loc='upper center',
        bbox_to_anchor=(0.535, 1.0),
        ncol=3,
        frameon=False,
        fontsize=7,
        handlelength=0.9,
        handleheight=0.7,
        handletextpad=0.3,
        columnspacing=1.15,
        borderpad=0.15,
        borderaxespad=0.0,
    )
    fig.supylabel('Fraction of blocks (%)', fontsize=7, x=0.002)
    fig.text(0.535, 0.0, XLABEL_NB, ha='center', va='bottom', fontsize=7)
    return save_column_figure(fig, out_dir, 'all_stages_nb_bin_share_bin32')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot final-config N_b bin shares (width=32)')
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
        rows = rebin_rows(load_scope_rows(csv_path, scope=args.scope), width=BIN_WIDTH)
        stage_rows[stage] = rows
        n_blocks = sum(row['n_blocks'] for row in rows)
        le64 = 100.0 * cumulative_share(rows, ONE_PAGE)
        le128 = 100.0 * cumulative_share(rows, TWO_PAGE)
        print(
            f'Stage {stage}: {csv_path}  n={n_blocks}  '
            f'N_b<=64={le64:.3f}%  N_b<=128={le128:.3f}%'
        )

    png_path, svg_path = plot_all_stages(stage_rows, out_dir, args.display_hi)
    print(f'wrote {png_path}')
    print(f'wrote {svg_path}')


if __name__ == '__main__':
    main()

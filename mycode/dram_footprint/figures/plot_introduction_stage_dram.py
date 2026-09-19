#!/usr/bin/env python3
"""Plot representative input/stage-output storage costs for TCAS-I."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext, LogLocator


OUT_DIR = Path(__file__).resolve().parent

# KITTI frame 000216. The five points map to Input and Stage 1--4,
# respectively. Labels show the storage-relevant active voxels x channels.
# Source: ../feature_map_dram_hash_entry_comparison.md
FEATURE_MAP_LABELS = (
    "15.0K$\\times$4",
    "15.0K$\\times$16",
    "25.8K$\\times$32",
    "19.1K$\\times$64",
    "8.5K$\\times$64",
)
FIXED_CAPACITY_MIB = np.array((29.7318, 59.4635, 45.9824, 22.7829, 6.1798))
PROPOSED_MIB = np.array((1.1230, 1.6846, 4.3481, 5.8008, 1.6040))
# Theoretical compact lower bound: N_voxel * (8-byte coordinate + C-byte INT8 feature).
MINIMUM_MIB = np.array((0.171661, 0.343323, 0.982742, 1.310738, 0.583305))
FIXED_CAPACITY_HASH = np.array((4330, 4330, 2009, 553, 150))
PROPOSED_HASH = np.array((1150, 1150, 1781, 1320, 365))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    configure_style()

    x = np.arange(len(FEATURE_MAP_LABELS))
    bar_width = 0.24
    bar_offset = 0.16

    # IEEE/TCAS-I single-column width is approximately 3.5 in (88.9 mm).
    fig, ax = plt.subplots(figsize=(3.5, 1.72))

    ax.bar(
        x - bar_offset,
        FIXED_CAPACITY_MIB,
        bar_width,
        label="Base",
        color="#D0D0D0",
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    ax.bar(
        x + bar_offset,
        PROPOSED_MIB,
        bar_width,
        label="Our",
        color="#9DB9DD",
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    # Overlay the compact-storage lower bound within both scheme bars.
    ax.bar(
        x - bar_offset,
        MINIMUM_MIB,
        bar_width,
        color="#A8D5AE",
        edgecolor="black",
        linewidth=0.45,
        zorder=4,
    )
    ax.bar(
        x + bar_offset,
        MINIMUM_MIB,
        bar_width,
        color="#A8D5AE",
        edgecolor="black",
        linewidth=0.45,
        zorder=4,
    )

    ax_hash = ax.twinx()
    ax_hash.plot(
        x - bar_offset,
        FIXED_CAPACITY_HASH,
        color="#111111",
        marker="o",
        markersize=2.2,
        linewidth=0.75,
        label="Base",
        zorder=5,
    )
    ax_hash.plot(
        x + bar_offset,
        PROPOSED_HASH,
        color="#0072BD",
        marker="D",
        markersize=2.0,
        linewidth=0.75,
        label="Our",
        zorder=5,
    )

    ax.set_ylabel("DRAM (MiB)")
    ax.set_xticks(x, FEATURE_MAP_LABELS)
    ax.tick_params(axis="x", labelsize=5.6, pad=2)
    ax.set_xlabel("Active voxels $\\times$ channels", labelpad=2)
    ax.set_yscale("log", base=10)
    ax.set_ylim(0.1, 100)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.45, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_hash.set_ylabel("Hash entries")
    ax_hash.set_yscale("log", base=10)
    ax_hash.set_ylim(100, 10000)
    ax_hash.yaxis.set_major_locator(LogLocator(base=10, numticks=3))
    ax_hash.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax_hash.spines["top"].set_visible(False)
    legend_handles = [
        Line2D([], [], linestyle="none", label="DRAM"),
        Line2D([], [], linestyle="none", label="Hash"),
        Patch(facecolor="#D0D0D0", edgecolor="black", linewidth=0.6, label="Base"),
        Line2D([], [], color="#111111", marker="o", markersize=2.2, linewidth=0.75, label="Base"),
        Patch(facecolor="#9DB9DD", edgecolor="black", linewidth=0.6, label="Our"),
        Line2D([], [], color="#0072BD", marker="D", markersize=2.0, linewidth=0.75, label="Our"),
        Patch(facecolor="#A8D5AE", edgecolor="black", linewidth=0.45, label="Min."),
        Line2D([], [], linestyle="none", label=""),
    ]
    ax_hash.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.995),
        frameon=False,
        ncols=4,
        fontsize=6.2,
        handlelength=1.0,
        handletextpad=0.25,
        columnspacing=0.45,
        labelspacing=0.15,
        borderpad=0.1,
        borderaxespad=0.2,
    )

    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.25, top=0.79)

    stem = OUT_DIR / "introduction_stage_dram_grouped_bar"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()

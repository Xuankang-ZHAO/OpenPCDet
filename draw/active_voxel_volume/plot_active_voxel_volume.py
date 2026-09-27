#!/usr/bin/env python3
"""Scatter of train active-voxel caps against sensing-box volume.

Each point is one OpenPCDet 3D sparse-convolution config. The x value is
MAX_NUMBER_OF_VOXELS['train']. The y value is the axis-aligned sensing box
(xmax-xmin)*(ymax-ymin)*(zmax-zmin) from POINT_CLOUD_RANGE. Volume spans
about 2.5 decades, so the y-axis is logarithmic; the figure width stays the
IEEE TCAS-I single-column width of 3.5 in.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator


OUT_DIR = Path(__file__).resolve().parent
# IEEE TCAS-I / IEEEtran journal single-column width.
FIG_WIDTH_IN = 3.5
# 10^4–10^7 span, with the top and bottom white margins cropped off.
FIG_HEIGHT_IN = 1.228
# White margin outside the y-axis label, matched on the right.
SIDE_PAD_IN = 0.16


def box_volume(xmin: float, ymin: float, zmin: float, xmax: float, ymax: float, zmax: float) -> float:
    return (xmax - xmin) * (ymax - ymin) * (zmax - zmin)


# Same algorithm shares one circle color. Caps and ranges follow the merged yaml.
SECOND = "#E8E8E8"
CENTERPOINT = "#0072B2"
VOXELNEXT = "#C4845A"

# label, train cap, volume (m^3), color, text offset (pt), ha, va.
# Labels name the dataset only; the algorithm is the legend color.
POINTS = (
    (
        "KITTI",
        16_000,
        box_volume(0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
        SECOND,
        (0, 3),
        "center",
        "bottom",
    ),
    (
        "ONCE",
        60_000,
        box_volume(-75.2, -75.2, -5.0, 75.2, 75.2, 3.0),
        SECOND,
        (-3, 0),
        "right",
        "center",
    ),
    (
        "nuScenes",
        60_000,
        box_volume(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        SECOND,
        (4, -6),
        "center",
        "top",
    ),
    (
        "Lyft",
        80_000,
        box_volume(-80.0, -80.0, -5.0, 80.0, 80.0, 3.0),
        SECOND,
        (4, 0),
        "left",
        "center",
    ),
    (
        "nuScenes",
        120_000,
        box_volume(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        VOXELNEXT,
        (0, -3),
        "center",
        "top",
    ),
    (
        "Argoverse 2",
        120_000,
        box_volume(-200.0, -200.0, -20.0, 200.0, 200.0, 20.0),
        VOXELNEXT,
        (-4, -3),
        "right",
        "center",
    ),
    (
        "Waymo",
        150_000,
        box_volume(-75.2, -75.2, -2.0, 75.2, 75.2, 4.0),
        CENTERPOINT,
        (0, 3),
        "center",
        "bottom",
    ),
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_increasing_scale_arrow(fig, ax) -> None:
    """Dashed diagonal across the plot, in the style of a workload-scale arrow.

    Endpoints stay on the data diagonal used before the axis was cut at 10^7.
    """
    fig.canvas.draw()
    log_lo, log_hi = 4.0, math.log10(2.2e7)
    start = (
        (69 / 281.75) * 196_000,
        10 ** (log_lo + (32 / 154) * (log_hi - log_lo)),
    )
    end = (
        (205 / 281.75) * 196_000,
        10 ** (log_lo + (124 / 154) * (log_hi - log_lo)),
    )
    color = "#7E9BB5"
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transData,
            arrowstyle="-|>",
            mutation_scale=8,
            linestyle=(0, (3.0, 1.6)),
            linewidth=0.85,
            color=color,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
            zorder=2,
        )
    )
    fig.canvas.draw()
    p1 = ax.transData.transform(start)
    p2 = ax.transData.transform(end)
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    norm = math.hypot(dx, dy)
    offset = 6.0
    midpoint = ((p1[0] + p2[0]) / 2 - dy / norm * offset, (p1[1] + p2[1]) / 2 + dx / norm * offset)
    text_xy = ax.transData.inverted().transform(midpoint)
    ax.text(
        text_xy[0],
        text_xy[1],
        "Increasing scale",
        rotation=angle,
        rotation_mode="anchor",
        ha="center",
        va="center",
        fontsize=6,
        color=color,
        zorder=2,
        clip_on=False,
    )


def add_algorithm_legend(ax) -> None:
    """Algorithm key in the empty upper-right of the axes."""
    marker_size = math.sqrt(22)
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=marker_size,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=name,
        )
        for name, color in (
            ("SECOND", SECOND),
            ("VoxelNeXt", VOXELNEXT),
            ("CenterPoint", CENTERPOINT),
        )
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.955),
        frameon=True,
        fontsize=6,
        handlelength=0.9,
        handletextpad=0.35,
        borderpad=0.3,
        borderaxespad=0,
        labelspacing=0.25,
    )
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_facecolor("white")


def main() -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    for label, cap, volume, color, offset, ha, va in POINTS:
        ax.scatter(
            [cap],
            [volume],
            s=22,
            marker="o",
            facecolor=color,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax.annotate(
            label,
            (cap, volume),
            textcoords="offset points",
            xytext=offset,
            ha=ha,
            va=va,
            fontsize=6,
            linespacing=0.9,
            color="black",
            zorder=4,
        )

    ax.set_xlabel("Active voxel numbers", fontsize=7)
    ax.set_ylabel(r"3-D ROI volume (m$^3$)")
    ax.set_xlim(0, 196_000)
    ax.set_xticks([0, 50_000, 100_000, 150_000])
    ax.set_xticklabels(["0", "50k", "100k", "150k"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yscale("log")
    ax.set_ylim(1e4, 1e7)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.tick_params(which="both", top=False, right=False)
    ax.tick_params(which="minor", left=False, bottom=False)
    ax.tick_params(axis="x", pad=1)
    ax.xaxis.labelpad = -0.5
    ax.grid(False)
    ax.yaxis.grid(which="major", color="#D0D0D0", linewidth=0.4, linestyle="--", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
    # Left fraction keeps the y-label ink about SIDE_PAD_IN from the figure edge.
    left = 0.164
    right = 1.0 - (SIDE_PAD_IN + 0.005) / FIG_WIDTH_IN
    fig.subplots_adjust(left=left, right=right, bottom=0.167, top=0.953)
    add_algorithm_legend(ax)
    add_increasing_scale_arrow(fig, ax)

    stem = OUT_DIR / "active_voxel_volume"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)
    for suffix in (".pdf", ".svg", ".png"):
        print(f"Saved {stem.with_suffix(suffix)}")


if __name__ == "__main__":
    main()

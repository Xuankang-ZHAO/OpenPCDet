#!/usr/bin/env python3
"""Plot per-frame peak DRAM and hash demand across 200 KITTI val frames."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import LogFormatterMathtext, NullLocator
from matplotlib.transforms import Bbox


OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR.parent
SOURCES = (
    DATA_DIR / "val_sampled_200_feature_map_dram_hash_entry_comparison.json",
)
EXPECTED_FRAMES = 200
COLORS = {"Base": "#D0D0D0", "Our": "#9DB9DD", "Min": "#A8D5AE"}


class WhiskerLegendHandler(HandlerBase):
    """Draw a vertical min--max whisker with horizontal end caps."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        center = -xdescent + width / 2
        bottom = -ydescent
        top = bottom + height
        cap_half = width * 0.25
        style = dict(color=orig_handle.get_color(),
                     linewidth=orig_handle.get_linewidth(), transform=trans)
        return [
            Line2D([center, center], [bottom, top], **style),
            Line2D([center - cap_half, center + cap_half], [bottom, bottom], **style),
            Line2D([center - cap_half, center + cap_half], [top, top], **style),
        ]


def read_per_frame_peaks() -> dict[str, np.ndarray]:
    """Read measured peaks and recompute compact IFM+OFM peaks per frame."""
    values: dict[str, list[float]] = {
        "Base DRAM": [],
        "Our DRAM": [],
        "Min DRAM": [],
        "Base hash": [],
        "Our hash": [],
    }
    seen_frames: set[str] = set()

    for source in SOURCES:
        with source.open(encoding="utf-8") as handle:
            frames = json.load(handle)["frames"]
        for frame in frames:
            frame_id = str(frame["frame_id"])
            if frame_id in seen_frames:
                raise ValueError(f"Duplicate frame ID: {frame_id}")
            seen_frames.add(frame_id)

            base_maps = frame["feature_maps"]["fixed_capacity"]
            our_maps = frame["feature_maps"]["proposed"]
            if len(base_maps) != len(our_maps) or len(base_maps) < 2:
                raise ValueError(f"Unmatched feature maps for frame {frame_id}")
            if any(
                (base["voxels"], base["channels"]) != (our["voxels"], our["channels"])
                for base, our in zip(base_maps, our_maps)
            ):
                raise ValueError(f"Inconsistent voxel counts or channels for frame {frame_id}")

            # Check that the stored peaks use simultaneously resident IFM+OFM.
            for scheme, maps in (("fixed_capacity", base_maps), ("proposed", our_maps)):
                adjacent_bytes = max(
                    first["dram_bytes"] + second["dram_bytes"]
                    for first, second in zip(maps, maps[1:])
                )
                adjacent_hash = max(
                    first["hash_entries"] + second["hash_entries"]
                    for first, second in zip(maps, maps[1:])
                )
                if (adjacent_bytes != frame["peaks"][scheme]["dram_bytes"]
                        or adjacent_hash != frame["peaks"][scheme]["hash_entries"]):
                    raise ValueError(f"Inconsistent stored peaks for {frame_id}, {scheme}")

            compact_bytes = [
                item["voxels"] * (8 + item["channels"]) for item in base_maps
            ]
            compact_peak_bytes = max(
                first + second
                for first, second in zip(compact_bytes, compact_bytes[1:])
            )

            values["Base DRAM"].append(frame["peaks"]["fixed_capacity"]["dram_bytes"] / 2**20)
            values["Our DRAM"].append(frame["peaks"]["proposed"]["dram_bytes"] / 2**20)
            values["Min DRAM"].append(compact_peak_bytes / 2**20)
            values["Base hash"].append(frame["peaks"]["fixed_capacity"]["hash_entries"])
            values["Our hash"].append(frame["peaks"]["proposed"]["hash_entries"])

    if len(seen_frames) != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} distinct frames, got {len(seen_frames)}")
    return {key: np.asarray(sample, dtype=float) for key, sample in values.items()}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
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


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.grid(axis="y", color="#D5D5D5", linewidth=0.4, linestyle="--")
    ax.set_axisbelow(True)


def add_box_summaries(
    ax: plt.Axes,
    groups: tuple[tuple[str, np.ndarray], ...],
    number_formats: tuple[str, ...],
    value_scale: float = 1,
    suffix: str = "",
) -> None:
    for x, (name, sample) in enumerate(groups):
        color = COLORS[name]
        minimum = sample.min()
        q1, q3 = np.percentile(sample, (25, 75))
        mean = sample.mean()
        maximum = sample.max()
        ax.vlines(x, minimum, maximum, color="#222222", linewidth=0.75, zorder=3)
        ax.hlines((minimum, maximum), x - 0.07, x + 0.07,
                  color="#222222", linewidth=0.75, zorder=3)
        ax.add_patch(Rectangle(
            (x - 0.17, q1), 0.34, q3 - q1,
            facecolor=color, edgecolor="#222222", linewidth=0.7, zorder=4,
        ))
        ax.hlines(mean, x - 0.19, x + 0.19, color="#111111", linewidth=1.0, zorder=5)
        ax.annotate(
            format(mean / value_scale, number_formats[x]) + suffix,
            (x + 0.19, mean),
            xytext=(2.5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.15},
            zorder=7,
        )


def main() -> None:
    configure_style()
    data = read_per_frame_peaks()

    # 3.5 inches is the standard IEEE/TCAS-I single-column width.
    fig, (ax_dram, ax_hash) = plt.subplots(
        1, 2, figsize=(3.5, 1.40), gridspec_kw={"width_ratios": (1.35, 1)}
    )
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.15, top=0.86, wspace=0.58)

    dram_groups = tuple((name, data[f"{name} DRAM"]) for name in ("Base", "Our", "Min"))
    add_box_summaries(ax_dram, dram_groups, (".1f", ".1f", ".2f"))
    ax_dram.set_xticks(range(3), ("Base", "Ours", "Fully Packed"))
    ax_dram.set_xlim(-0.42, 2.62)
    ax_dram.set_yscale("log", base=10)
    ax_dram.set_ylim(1, 250)
    ax_dram.set_yticks((1, 10, 100))
    ax_dram.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax_dram.yaxis.set_minor_locator(NullLocator())
    ax_dram.set_ylabel("Peak DRAM Footprint (MiB)", labelpad=2)
    style_axis(ax_dram)

    hash_groups = tuple((name, data[f"{name} hash"]) for name in ("Base", "Our"))
    add_box_summaries(ax_hash, hash_groups, (".2f", ".2f"), value_scale=1000, suffix="K")
    ax_hash.set_xticks(range(2), ("Base", "Ours"))
    ax_hash.set_xlim(-0.38, 1.85)
    ax_hash.set_ylim(0, 13500)
    ax_hash.set_yticks((0, 4000, 8000, 12000), ("0", "4K", "8K", "12K"))
    ax_hash.set_ylabel("Peak Hash Entries", labelpad=2)
    style_axis(ax_hash)

    whisker_handle = Line2D([], [], color="#222222", linewidth=0.75, label="Min–max")
    handles = (
        Patch(facecolor="#D0D0D0", edgecolor="#222222", linewidth=0.7, label="25%-75%"),
        whisker_handle,
        Line2D([], [], color="#111111", linewidth=1.0, label="Mean"),
    )
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.99),
        ncol=3, frameon=False, fontsize=6.2, handlelength=1.0,
        handletextpad=0.3, columnspacing=0.7,
        handler_map={whisker_handle: WhiskerLegendHandler()},
    )

    stem = OUT_DIR / "introduction_peak_resource_comparison_val200"
    # Trim 40 px at 600 dpi from each end while retaining the 3.5-in column width.
    vertical_trim = 40 / 600
    export_bbox = Bbox.from_bounds(
        0, vertical_trim, fig.get_figwidth(), fig.get_figheight() - 2 * vertical_trim
    )
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=export_bbox)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches=export_bbox)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches=export_bbox)
    plt.close(fig)


if __name__ == "__main__":
    main()

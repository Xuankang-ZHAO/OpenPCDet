#!/usr/bin/env python3
"""Bar chart of fixed-capacity conv1.0.0 DRAM for KITTI frames 000000-000019.

Allocated DRAM is the reserved IFM+OFM block storage. Effective DRAM counts
every voxel stored in those blocks, including halo copies:
Sum_N_b * (8-byte coordinate + 1 byte per feature channel).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator
from matplotlib.transforms import Bbox


OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = OUT_DIR.parent / "train_000000_000019_feature_map_dram_hash_entry_comparison.json"
USEFUL_PATH = OUT_DIR / "conv1_fixed_capacity_useful_dram.json"
FRAME_IDS = [f"{index:06d}" for index in range(20)]
IFM_NAME = "conv_input.0"
OFM_NAME = "conv1.0.0"
SCHEME = "fixed_capacity"
COORD_BYTES = 8
BYTES_PER_CHANNEL = 1


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def coresident_mib(frame: dict, scheme: str) -> float:
    rows = {row["name"]: row for row in frame["feature_maps"][scheme]}
    ifm = rows[IFM_NAME]
    ofm = rows[OFM_NAME]
    if int(ifm["channels"]) != 16 or int(ofm["channels"]) != 16:
        raise ValueError(f"{frame['frame_id']} {scheme} channels are not 16/16")
    return (int(ifm["dram_bytes"]) + int(ofm["dram_bytes"])) / (1024.0 * 1024.0)


def load_dram() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with DATA_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    with USEFUL_PATH.open(encoding="utf-8") as handle:
        useful_payload = json.load(handle)
    by_id = {str(frame["frame_id"]): frame for frame in payload["frames"]}
    useful_by_id = {str(frame["frame_id"]): frame for frame in useful_payload["frames"]}
    missing = [frame_id for frame_id in FRAME_IDS if frame_id not in by_id or frame_id not in useful_by_id]
    if missing or len(by_id) != len(FRAME_IDS):
        raise ValueError(f"Expected frames 000000-000019, missing {missing}")
    allocated = []
    effective = []
    blocks = []
    for frame_id in FRAME_IDS:
        frame = by_id[frame_id]
        measured = useful_by_id[frame_id]
        rows = {row["name"]: row for row in frame["feature_maps"][SCHEME]}
        allocated_mib = coresident_mib(frame, SCHEME)
        if frame["peaks"][SCHEME]["dram_layer_name"] != OFM_NAME:
            raise ValueError(f"{frame_id} fixed-capacity peak is not {OFM_NAME}")
        stored = float(frame["peaks"][SCHEME]["dram_mib"])
        if abs(allocated_mib - stored) > 5e-4:
            raise ValueError(f"{frame_id} coresident {allocated_mib:.4f} != peak {stored}")
        if int(measured["allocated_dram_bytes"]) != int(round(allocated_mib * 1024 * 1024)):
            stored_bytes = int(frame["peaks"][SCHEME]["dram_bytes"])
            if int(measured["allocated_dram_bytes"]) != stored_bytes:
                raise ValueError(f"{frame_id} useful-file allocation does not match the comparison")
        channels = int(measured["ifm_channels"])
        if channels != 16 or int(measured["ofm_channels"]) != 16:
            raise ValueError(f"{frame_id} conv1.0.0 channels are not 16/16")
        voxel_bytes = COORD_BYTES + channels * BYTES_PER_CHANNEL
        useful_bytes = (int(measured["ifm_sum_nb"]) + int(measured["ofm_sum_nb"])) * voxel_bytes
        if useful_bytes != int(measured["useful_bytes"]):
            raise ValueError(f"{frame_id} useful bytes {useful_bytes} != stored {measured['useful_bytes']}")
        allocated.append(allocated_mib)
        effective.append(useful_bytes / (1024.0 * 1024.0))
        blocks.append(int(rows[IFM_NAME]["blocks"]) + int(rows[OFM_NAME]["blocks"]))
    return (
        np.asarray(allocated, dtype=float),
        np.asarray(effective, dtype=float),
        np.asarray(blocks, dtype=float),
    )


def main() -> None:
    configure_style()
    allocated, effective, blocks = load_dram()
    x = np.arange(len(FRAME_IDS))
    bar_width = 0.72

    # IEEE/TCAS-I single-column width is 3.5 in.
    # Log scale keeps the ~1 MiB effective portion visible inside the allocated bar.
    fig, ax = plt.subplots(figsize=(3.5, 1.2))
    axis_floor = 10 ** 0
    base_bars = ax.bar(
        x,
        allocated - axis_floor,
        bar_width,
        bottom=axis_floor,
        label="Base",
        color="#A9D18E",
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
    )
    effective_bars = ax.bar(
        x,
        effective - axis_floor,
        bar_width,
        bottom=axis_floor,
        label="Effective",
        color="#F4B183",
        edgecolor="black",
        linewidth=0.45,
        zorder=4,
    )

    ax_blocks = ax.twinx()
    block_line = ax_blocks.scatter(
        x,
        blocks,
        s=14,
        facecolors="white",
        edgecolors="#111111",
        linewidths=0.6,
        marker="o",
        label="Blocks",
        zorder=5,
    )

    ax.set_xlabel("Frame index", fontsize=7, labelpad=0)
    ax.set_ylabel("DRAM footprint (MiB)", fontsize=7)
    ax.set_xticks(x[::2], [str(index) for index in range(0, 20, 2)])
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_xlim(-0.6, 19.6)
    ax.set_yscale("log", base=10)
    ax.set_ylim(axis_floor, 250)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", which="major", color="#D0D0D0", linewidth=0.45, linestyle="--", zorder=0)
    ax.grid(axis="y", which="minor", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_blocks.set_ylabel("Block numbers", fontsize=7, labelpad=2)
    ax_blocks.set_ylim(0, 12000)
    ax_blocks.set_yticks([0, 4000, 8000, 12000])
    ax_blocks.set_yticklabels(["0", "4k", "8k", "12k"])
    ax_blocks.yaxis.set_minor_locator(NullLocator())
    ax_blocks.tick_params(axis="y", labelsize=6.5, pad=1)
    ax_blocks.spines["top"].set_visible(False)
    ax.legend(
        handles=[base_bars, effective_bars, block_line],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.96),
        frameon=False,
        ncols=3,
        handlelength=1.0,
        handletextpad=0.2,
        columnspacing=0.5,
        borderaxespad=0.1,
    )
    fig.subplots_adjust(left=0.155, right=0.845, bottom=0.20, top=0.84)

    # Crop only the vertical whitespace, leaving 3 px above and below at 600 DPI.
    output_dpi = 600
    vertical_pad_px = 3
    fig.set_dpi(output_dpi)
    fig.canvas.draw()
    rendered = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    content_rows = np.any(rendered != 255, axis=(1, 2))
    occupied_rows = np.flatnonzero(content_rows)
    if occupied_rows.size == 0:
        raise RuntimeError("Rendered figure is empty")
    first_row = int(occupied_rows[0])
    last_row = int(occupied_rows[-1])
    canvas_height_px = rendered.shape[0]
    vertical_bbox = Bbox.from_extents(
        0,
        max(0, canvas_height_px - (last_row + 1 + vertical_pad_px)) / output_dpi,
        fig.get_figwidth(),
        min(canvas_height_px, canvas_height_px - first_row + vertical_pad_px) / output_dpi,
    )

    stem = OUT_DIR / "conv1_dram_frames_000000_000019"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=vertical_bbox)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches=vertical_bbox)
    fig.savefig(stem.with_suffix(".png"), dpi=output_dpi, bbox_inches=vertical_bbox)
    plt.close(fig)
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.svg')}")
    print(f"Saved {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()

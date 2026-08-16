"""2D schematic of SECOND BEV features: full mesh with occupied cells colored."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BIN_PATH = Path(__file__).with_name("000008.bin")
OUT_PATH = Path(__file__).with_name("bev_feature_000008.png")

PC_RANGE = np.array([0.0, -40.0, -3.0, 70.4, 40.0, 1.0], dtype=np.float64)
CELL = np.array([4.0, 4.0], dtype=np.float64)
X_LIM = (0.0, 32.0)
Y_LIM = (-16.0, 16.0)


def jet_from_z(z):
    z = np.asarray(z, dtype=np.float64)
    lo, hi = np.percentile(z, [2, 98])
    t = np.clip((z - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=1)


def load_kitti_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def bev_grid(pts):
    nx = int(np.round((X_LIM[1] - X_LIM[0]) / CELL[0]))
    ny = int(np.round((Y_LIM[1] - Y_LIM[0]) / CELL[1]))
    ix0 = int(np.floor((X_LIM[0] - PC_RANGE[0]) / CELL[0]))
    iy0 = int(np.floor((Y_LIM[0] - PC_RANGE[1]) / CELL[1]))

    xyz = pts[:, :3]
    ix = np.floor((xyz[:, 0] - PC_RANGE[0]) / CELL[0]).astype(np.int32)
    iy = np.floor((xyz[:, 1] - PC_RANGE[1]) / CELL[1]).astype(np.int32)
    lx = ix - ix0
    ly = iy - iy0
    valid = (lx >= 0) & (lx < nx) & (ly >= 0) & (ly < ny)

    max_z = np.full((nx, ny), -np.inf, dtype=np.float64)
    np.maximum.at(max_z, (lx[valid], ly[valid]), xyz[valid, 2])
    max_z[np.isneginf(max_z)] = np.nan
    return max_z, ix0, iy0


def save_bev_figure(max_z, ix0, iy0, path):
    nx, ny = max_z.shape
    occ = np.isfinite(max_z)
    colors = np.ones((nx, ny, 3), dtype=np.float64)
    if occ.any():
        colors[occ] = jet_from_z(max_z[occ])

    fig, ax = plt.subplots(figsize=(8.0, 8.0), facecolor="white")
    ax.set_facecolor("white")
    for i in range(nx):
        for j in range(ny):
            x = PC_RANGE[0] + (i + ix0) * CELL[0]
            y = PC_RANGE[1] + (j + iy0) * CELL[1]
            ax.add_patch(
                Rectangle(
                    (y, x),
                    CELL[1],
                    CELL[0],
                    facecolor=colors[i, j],
                    edgecolor=(0.35, 0.35, 0.35),
                    linewidth=0.8,
                )
            )

    ax.set_xlim(Y_LIM[1], Y_LIM[0])
    ax.set_ylim(X_LIM[0], X_LIM[1])
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def main():
    pts = load_kitti_bin(BIN_PATH)
    max_z, ix0, iy0 = bev_grid(pts)
    print(f"cell: {CELL.tolist()} m")
    print(f"mesh: {max_z.shape[0]} x {max_z.shape[1]}")
    print(f"occupied: {int(np.isfinite(max_z).sum())}")
    save_bev_figure(max_z, ix0, iy0, OUT_PATH)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

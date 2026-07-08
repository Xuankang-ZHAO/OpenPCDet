#!/usr/bin/env python3
"""Export one frame point cloud to 80-bit packed voxel stream.

This script reuses OpenPCDet's official data pipeline (point feature encoder +
data processor voxelization) and exports one voxel per line in the format:

    voxel_data_in[79:0] = {feat[31:0], z[15:0], y[15:0], x[15:0]}

Each output line is a 20-hex-digit token (MSB-first):

    <feat:8 hex><z:4 hex><y:4 hex><x:4 hex>

Example:
python mycode/export_single_frame_voxel_stream80.py \
  --cfg tools/cfgs/kitti_models/second.yaml \
  --pcd data/kitti/training/velodyne/000014.bin \
  --out mycode/output/voxel_stream80/000014_voxel_stream80.txt \
  --feat-mode num_points
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from mycode.kitti_frame_loader import (
    add_data_mode_args,
    build_kitti_dataset,
    build_template_dataset,
    load_kitti_sample,
    normalize_frame_token,
    resolve_data_mode,
    resolve_project_root,
)


UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1


def parse_args():
    parser = argparse.ArgumentParser(description="Export single-frame voxel stream in 80-bit packed format")
    parser.add_argument("--cfg", type=str, default="tools/cfgs/kitti_models/second.yaml", help="Model config yaml")
    parser.add_argument("--pcd", type=str, required=True, help="Input point cloud file (.bin or .npy); frame id is taken from the stem in kitti mode")
    add_data_mode_args(parser)
    parser.add_argument("--out", type=str, required=True, help="Output stream file path")

    parser.add_argument(
        "--feat-mode",
        type=str,
        default="num_points",
        choices=["num_points", "occupancy", "const", "mean_channel", "max_channel"],
        help="How to generate 32-bit feat field",
    )
    parser.add_argument("--feat-const", type=int, default=1, help="feat value when feat-mode=const")
    parser.add_argument("--feat-channel", type=int, default=3, help="Feature channel index in voxel tensor")
    parser.add_argument(
        "--feat-scale",
        type=float,
        default=1000.0,
        help="Scale factor for float feature in mean_channel/max_channel mode",
    )
    parser.add_argument(
        "--feat-offset",
        type=float,
        default=0.0,
        help="Offset added before scaling in mean_channel/max_channel mode",
    )

    parser.add_argument(
        "--sort",
        type=str,
        default="none",
        choices=["none", "xyz", "zyx"],
        help="Optional voxel order in output stream",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional path to write export summary json",
    )
    return parser.parse_args()


def load_points(pcd_path: Path) -> np.ndarray:
    if not pcd_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {pcd_path}")

    if pcd_path.suffix.lower() == ".bin":
        pts = np.fromfile(str(pcd_path), dtype=np.float32)
        if pts.size % 4 != 0:
            raise ValueError(f".bin point cloud size is not divisible by 4: {pcd_path}")
        pts = pts.reshape(-1, 4)
    elif pcd_path.suffix.lower() == ".npy":
        pts = np.load(str(pcd_path))
        if pts.ndim != 2 or pts.shape[1] < 4:
            raise ValueError(f".npy point cloud must have shape (N, >=4), got {pts.shape}")
        pts = pts[:, :4].astype(np.float32, copy=False)
    else:
        raise ValueError("Unsupported point cloud extension. Use .bin or .npy")

    return pts


def _to_uint32_from_float(values: np.ndarray, scale: float, offset: float) -> np.ndarray:
    mapped = np.round((values + offset) * scale).astype(np.int64)
    mapped = np.clip(mapped, 0, UINT32_MAX)
    return mapped.astype(np.uint32)


def build_feat(
    feat_mode: str,
    voxels: np.ndarray,
    voxel_num_points: np.ndarray,
    feat_const: int,
    feat_channel: int,
    feat_scale: float,
    feat_offset: float,
) -> np.ndarray:
    num_voxels = voxel_num_points.shape[0]

    if feat_mode == "num_points":
        feat = np.clip(voxel_num_points.astype(np.int64), 0, UINT32_MAX).astype(np.uint32)
        return feat

    if feat_mode == "occupancy":
        return np.ones((num_voxels,), dtype=np.uint32)

    if feat_mode == "const":
        if feat_const < 0 or feat_const > UINT32_MAX:
            raise ValueError(f"feat-const out of uint32 range: {feat_const}")
        return np.full((num_voxels,), feat_const, dtype=np.uint32)

    if feat_channel < 0 or feat_channel >= voxels.shape[2]:
        raise ValueError(
            f"feat-channel={feat_channel} out of range for voxel tensor channel size={voxels.shape[2]}"
        )

    out = np.zeros((num_voxels,), dtype=np.uint32)
    for i in range(num_voxels):
        n = int(voxel_num_points[i])
        if n <= 0:
            out[i] = np.uint32(0)
            continue
        v = voxels[i, :n, feat_channel].astype(np.float64)
        if feat_mode == "mean_channel":
            value = float(v.mean())
        elif feat_mode == "max_channel":
            value = float(v.max())
        else:
            raise ValueError(f"Unsupported feat mode: {feat_mode}")
        out[i] = _to_uint32_from_float(np.array([value], dtype=np.float64), feat_scale, feat_offset)[0]

    return out


def pack_word_hex80(x: int, y: int, z: int, feat: int) -> str:
    if not (0 <= x <= UINT16_MAX):
        raise ValueError(f"x out of uint16 range: {x}")
    if not (0 <= y <= UINT16_MAX):
        raise ValueError(f"y out of uint16 range: {y}")
    if not (0 <= z <= UINT16_MAX):
        raise ValueError(f"z out of uint16 range: {z}")
    if not (0 <= feat <= UINT32_MAX):
        raise ValueError(f"feat out of uint32 range: {feat}")

    word = (int(feat) << 48) | (int(z) << 32) | (int(y) << 16) | int(x)
    return f"{word:020X}"


def reorder_indices(x: np.ndarray, y: np.ndarray, z: np.ndarray, order: str) -> np.ndarray:
    if order == "none":
        return np.arange(x.shape[0], dtype=np.int64)
    if order == "xyz":
        return np.lexsort((z, y, x))
    if order == "zyx":
        return np.lexsort((x, y, z))
    raise ValueError(f"Unsupported sort order: {order}")


def extract_voxel_arrays(sample):
    voxels = sample.get("voxels", None)
    voxel_coords = sample.get("voxel_coords", None)
    voxel_num_points = sample.get("voxel_num_points", None)

    if voxels is None or voxel_coords is None or voxel_num_points is None:
        raise RuntimeError("Voxelization output missing. Check DATA_PROCESSOR in config")

    if isinstance(voxel_coords, list):
        voxel_coords = voxel_coords[0]
    if isinstance(voxels, list):
        voxels = voxels[0]
    if isinstance(voxel_num_points, list):
        voxel_num_points = voxel_num_points[0]

    voxel_coords = np.asarray(voxel_coords).astype(np.int64)
    voxels = np.asarray(voxels)
    voxel_num_points = np.asarray(voxel_num_points).astype(np.int64)

    if voxel_coords.ndim != 2 or voxel_coords.shape[1] not in (3, 4):
        raise ValueError(f"Unexpected voxel_coords shape: {voxel_coords.shape}")

    if voxel_coords.shape[1] == 4:
        voxel_coords = voxel_coords[:, 1:4]

    z = voxel_coords[:, 0]
    y = voxel_coords[:, 1]
    x = voxel_coords[:, 2]
    return voxels, x, y, z, voxel_num_points


def main():
    args = parse_args()
    project_root = resolve_project_root()
    pcd_path = Path(args.pcd)
    if not pcd_path.is_absolute():
        pcd_path = project_root / pcd_path

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.utils import common_utils

    original_cwd = os.getcwd()
    try:
        os.chdir(project_root / 'tools')
        cfg_from_yaml_file(str(project_root / args.cfg), cfg)
    finally:
        os.chdir(original_cwd)

    data_mode = resolve_data_mode(cfg, args.data_mode)
    frame_id = normalize_frame_token(pcd_path.name)

    if data_mode == "kitti":
        logger = common_utils.create_logger()
        dataset = build_kitti_dataset(cfg, project_root, args.kitti_root, logger)
        sample, frame_info = load_kitti_sample(dataset, frame_id)
        voxels, x, y, z, voxel_num_points = extract_voxel_arrays(sample)
        num_input_points = int(frame_info.get("num_raw_points", sample["points"].shape[0]))
        data_loader = frame_info["data_loader"]
        fov_points_only = frame_info["fov_points_only"]
        grid_size = dataset.grid_size
        voxel_size = dataset.voxel_size
    else:
        points = load_points(pcd_path)
        dataset = build_template_dataset(cfg)
        sample = dataset.prepare_data({"points": points, "frame_id": frame_id})
        voxels, x, y, z, voxel_num_points = extract_voxel_arrays(sample)
        num_input_points = int(points.shape[0])
        data_loader = "DatasetTemplate.prepare_data(raw_points)"
        fov_points_only = False
        grid_size = dataset.grid_size
        voxel_size = dataset.voxel_size

    feat = build_feat(
        feat_mode=args.feat_mode,
        voxels=voxels,
        voxel_num_points=voxel_num_points,
        feat_const=args.feat_const,
        feat_channel=args.feat_channel,
        feat_scale=args.feat_scale,
        feat_offset=args.feat_offset,
    )

    if x.shape[0] != feat.shape[0]:
        raise ValueError(f"Length mismatch: coords={x.shape[0]} feat={feat.shape[0]}")

    order_idx = reorder_indices(x, y, z, args.sort)
    x_ord = x[order_idx]
    y_ord = y[order_idx]
    z_ord = z[order_idx]
    feat_ord = feat[order_idx]

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="ascii", newline="\n") as f:
        for xi, yi, zi, fi in zip(x_ord, y_ord, z_ord, feat_ord):
            token = pack_word_hex80(int(xi), int(yi), int(zi), int(fi))
            f.write(token + "\n")

    summary = {
        "cfg": str(args.cfg),
        "pcd": str(pcd_path),
        "out": str(out_path),
        "frame_id": frame_id,
        "data_loader": data_loader,
        "fov_points_only": fov_points_only,
        "data_mode": data_mode,
        "num_input_points": num_input_points,
        "num_voxels": int(x_ord.shape[0]),
        "grid_size_xyz": [int(v) for v in grid_size.tolist()] if grid_size is not None else None,
        "voxel_size_xyz": [float(v) for v in voxel_size] if voxel_size is not None else None,
        "coord_min_xyz": [int(x_ord.min()) if x_ord.size else 0, int(y_ord.min()) if y_ord.size else 0, int(z_ord.min()) if z_ord.size else 0],
        "coord_max_xyz": [int(x_ord.max()) if x_ord.size else 0, int(y_ord.max()) if y_ord.size else 0, int(z_ord.max()) if z_ord.size else 0],
        "feat_mode": args.feat_mode,
        "sort": args.sort,
    }

    if args.summary_json:
        summary_path = Path(args.summary_json)
        if not summary_path.is_absolute():
            summary_path = project_root / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2)

    print("Export finished")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

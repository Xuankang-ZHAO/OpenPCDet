"""Run INT8 SECOND HW-QAT on local 000008.bin and save boxes from the same view."""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from open3d.visualization import rendering

from vis_topdown import forward_camera, has_display, load_kitti_bin

LOCAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOCAL_DIR.parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
CFG_FILE = TOOLS_DIR / "cfgs/kitti_models/second_hw_qat.yaml"
CKPT_PATH = REPO_ROOT / "output/kitti_models/second_hw_qat/hw_qat_10ep/ckpt/checkpoint_epoch_10.pth"
BIN_PATH = LOCAL_DIR / "000008.bin"
OUT_BIN = LOCAL_DIR / "det_000008.bin"
OUT_PATH = LOCAL_DIR / "det_000008.png"
BOX_EDGE_SAMPLES = 80
BOX_EDGES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
BOX_COLORS = {
    1: [0.05, 0.72, 0.18],
    2: [0.05, 0.62, 0.92],
    3: [0.95, 0.72, 0.08],
}


def ensure_project_imports():
    for path in (str(REPO_ROOT), str(TOOLS_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(TOOLS_DIR)


def box_corners(box7):
    x, y, z, length, width, height, yaw = [float(v) for v in box7[:7]]
    dx, dy, dz = length / 2.0, width / 2.0, height / 2.0
    local = np.array(
        [
            [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz],
            [dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],
        ],
        dtype=np.float64,
    )
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]])
    return local @ rot.T + np.array([x, y, z], dtype=np.float64)


def sample_box_points(box7, label, n_per_edge=BOX_EDGE_SAMPLES):
    corners = box_corners(box7)
    t = np.linspace(0.0, 1.0, n_per_edge, dtype=np.float64)[:, None]
    intensity = np.full((n_per_edge, 1), float(label), dtype=np.float64)
    parts = []
    for i, j in BOX_EDGES:
        xyz = (1.0 - t) * corners[i] + t * corners[j]
        parts.append(np.concatenate([xyz, intensity], axis=1))
    return np.concatenate(parts, axis=0)


def save_det_bin(src_pts, boxes, labels, path):
    src_pts = np.asarray(src_pts, dtype=np.float32).reshape(-1, 4)
    box_parts = [sample_box_points(box, label) for box, label in zip(boxes, labels)]
    if box_parts:
        merged = np.concatenate([src_pts.astype(np.float64)] + box_parts, axis=0)
    else:
        merged = src_pts.astype(np.float64)
    merged.astype(np.float32).tofile(path)
    return path, int(src_pts.shape[0]), int(merged.shape[0] - src_pts.shape[0])


def box_line_set(box7):
    center = box7[0:3]
    lwh = box7[3:6]
    axis_angles = np.array([0.0, 0.0, box7[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def save_offscreen(pcd, pts, boxes, labels, path):
    eye, lookat, up = forward_camera(pts)
    renderer = rendering.OffscreenRenderer(1280, 800)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    pcd_mat = rendering.MaterialRecord()
    pcd_mat.shader = "defaultUnlit"
    pcd_mat.point_size = 3.0
    renderer.scene.add_geometry("pcd", pcd, pcd_mat)

    for idx, (box, label) in enumerate(zip(boxes, labels)):
        line_set = box_line_set(box)
        color = BOX_COLORS.get(int(label), [0.1, 0.1, 0.1])
        line_set.paint_uniform_color(color)
        line_mat = rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 3.0
        line_mat.base_color = [*color, 1.0]
        renderer.scene.add_geometry(f"box_{idx}", line_set, line_mat)

    renderer.setup_camera(60.0, lookat, eye, up)
    image = renderer.render_to_image()
    o3d.io.write_image(str(path), image)
    return path


def build_hw_int8_model(cfg, dataset, ckpt, logger):
    from pcdet.models import build_network
    from test_second_hw_qat import export_hw_payload, get_backbone

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    args = argparse.Namespace(
        weight_quant="per_channel",
        observer="max",
        observer_momentum=0.95,
        bias_bits=32,
        shift_bits=5,
        max_shift_rel_error=1.0,
        emit_binary=False,
        check_export=False,
    )
    backbone = get_backbone(model)
    backbone.enable_hw_qat(False, weight_quant=args.weight_quant, observer=args.observer,
                           observer_momentum=args.observer_momentum, fake_quant=False)
    with tempfile.TemporaryDirectory(prefix="second_hw_qat_vis_") as tmp:
        export_result = export_hw_payload(backbone, Path(tmp), args, logger)
    backbone.enable_hw_reference(True, qparams=export_result["qparams"])
    logger.info("Enabled HW-equivalent INT8 reference path")
    return model


def run_inference(model, dataset):
    from pcdet.models import load_data_to_gpu

    data_dict = dataset.collate_batch([dataset[0]])
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)

    pred = pred_dicts[0]
    boxes = pred["pred_boxes"].detach().cpu().numpy()
    scores = pred["pred_scores"].detach().cpu().numpy()
    labels = pred["pred_labels"].detach().cpu().numpy().astype(np.int32)
    return boxes, scores, labels


def main():
    ensure_project_imports()
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.utils import common_utils
    from demo import DemoDataset

    logger = common_utils.create_logger()
    cfg_from_yaml_file(str(CFG_FILE), cfg)
    dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=BIN_PATH,
        ext=".bin",
        logger=logger,
    )
    logger.info("points file: %s" % BIN_PATH)
    logger.info("ckpt: %s" % CKPT_PATH)

    model = build_hw_int8_model(cfg, dataset, CKPT_PATH, logger)
    boxes, scores, labels = run_inference(model, dataset)

    print(f"detections: {len(boxes)}")
    for class_id, name in enumerate(CLASS_NAMES, start=1):
        mask = labels == class_id
        print(f"  {name}: {int(mask.sum())}")
        for box, score in zip(boxes[mask], scores[mask]):
            print(f"    score={score:.3f} xyz=({box[0]:.2f},{box[1]:.2f},{box[2]:.2f}) "
                  f"lwh=({box[3]:.2f},{box[4]:.2f},{box[5]:.2f}) yaw={box[6]:.3f}")

    _, pts = load_kitti_bin(BIN_PATH)
    out_bin, n_src, n_box = save_det_bin(pts, boxes, labels, OUT_BIN)
    print(f"saved: {out_bin}")
    print(f"source points: {n_src}")
    print(f"box wireframe points: {n_box}")
    print(f"total points: {n_src + n_box}")
    if "--save-png" in sys.argv:
        pcd, pts = load_kitti_bin(BIN_PATH)
        save_offscreen(pcd, pts, boxes, labels, OUT_PATH)
        print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

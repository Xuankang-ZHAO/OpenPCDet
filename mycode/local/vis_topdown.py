"""Open a KITTI .bin point cloud from a forward driving view."""
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import rendering

BIN_PATH = Path(__file__).with_name("000008.bin")
OUT_PATH = Path(__file__).with_name("forward_000008.png")
BOX_COLORS = {
    1: np.array([0.05, 0.72, 0.18]),
    2: np.array([0.05, 0.62, 0.92]),
    3: np.array([0.95, 0.72, 0.08]),
}


def jet_from_z(z):
    """Open3D-style height coloring: low=blue, high=red (jet)."""
    z = np.asarray(z, dtype=np.float64)
    lo, hi = np.percentile(z, [2, 98])
    t = np.clip((z - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=1)


def load_kitti_bin(path):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    colors = jet_from_z(pts[:, 2])
    intensity = pts[:, 3]
    box_mask = intensity >= 0.995
    for label, color in BOX_COLORS.items():
        colors[box_mask & (np.round(intensity) == label)] = color
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, pts


def camera_extrinsic(eye, lookat, up):
    """OpenCV/Open3D world-to-camera matrix: look from eye toward lookat."""
    eye = np.asarray(eye, dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z_cam = lookat - eye
    z_cam /= np.linalg.norm(z_cam)
    x_cam = np.cross(z_cam, up)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)

    rot = np.stack([x_cam, y_cam, z_cam], axis=0)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot
    extrinsic[:3, 3] = -rot @ eye
    return extrinsic


def forward_camera(pts):
    """Camera near the ego vehicle, looking forward along +X, slightly downward."""
    z_ground = float(np.percentile(pts[:, 2], 15))
    eye = np.array([-5.0, 0.0, z_ground + 4.0])
    lookat = np.array([20.0, 0.0, z_ground + 0.6])
    up = np.array([0.0, 0.0, 1.0])
    return eye, lookat, up


def has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def set_forward_view(vis, pts):
    eye, lookat, up = forward_camera(pts)
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    params.extrinsic = camera_extrinsic(eye, lookat, up)
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)


def save_offscreen(pcd, pts, path):
    eye, lookat, up = forward_camera(pts)
    renderer = rendering.OffscreenRenderer(1280, 800)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 3.0
    renderer.scene.add_geometry("pcd", pcd, mat)
    renderer.setup_camera(60.0, lookat, eye, up)
    image = renderer.render_to_image()
    o3d.io.write_image(str(path), image)
    return path


def show_interactive(pcd, pts, title):
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name=title, width=1280, height=800):
        return False
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    if opt is None:
        vis.destroy_window()
        return False
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = 2.0
    set_forward_view(vis, pts)
    vis.run()
    vis.destroy_window()
    return True


def resolve_paths():
    extra = [arg for arg in sys.argv[1:] if arg != "--save-only"]
    bin_path = Path(extra[0]) if extra else BIN_PATH
    if not bin_path.is_absolute():
        local = Path(__file__).with_name(bin_path.name)
        bin_path = local if local.exists() else bin_path.resolve()
    out_path = OUT_PATH if bin_path.resolve() == BIN_PATH.resolve() else bin_path.with_name(bin_path.stem + "_vis.png")
    return bin_path, out_path


def main():
    bin_path, out_path = resolve_paths()
    pcd, pts = load_kitti_bin(bin_path)
    save_only = "--save-only" in sys.argv

    if not save_only and has_display():
        if show_interactive(pcd, pts, f"forward {bin_path.name}"):
            return

    save_offscreen(pcd, pts, out_path)
    print(f"saved: {out_path}")
    if not has_display():
        print("no DISPLAY; skipped Open3D window")


if __name__ == "__main__":
    main()

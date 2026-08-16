"""Open a KITTI .bin point cloud from a forward driving view."""
from pathlib import Path

import numpy as np
import open3d as o3d

BIN_PATH = Path(__file__).with_name("000008.bin")


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
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(jet_from_z(pts[:, 2]))
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


def set_forward_view(vis, pts):
    """Camera near the ego vehicle, looking forward along +X, slightly downward."""
    z_ground = float(np.percentile(pts[:, 2], 15))

    # A few meters behind and above the lidar, looking ~20 m down the road.
    eye = np.array([-5.0, 0.0, z_ground + 4.0])
    lookat = np.array([20.0, 0.0, z_ground + 0.6])
    up = np.array([0.0, 0.0, 1.0])

    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    params.extrinsic = camera_extrinsic(eye, lookat, up)
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)


def main():
    pcd, pts = load_kitti_bin(BIN_PATH)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"forward {BIN_PATH.name}", width=1280, height=800)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = 2.0
    set_forward_view(vis, pts)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

"""Schematic of KITTI voxelization (SECOND-style grid) with Open3D / matplotlib."""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BIN_PATH = Path(__file__).with_name("000008.bin")
OUT_PATH = Path(__file__).with_name("voxelization_000008.png")

# Same range origin as tools/cfgs/dataset_configs/kitti_dataset.yaml
PC_RANGE_MIN = np.array([0.0, -40.0, -3.0], dtype=np.float64)

# Slightly larger than SECOND's 0.05/0.05/0.1 so individual cubes read clearly.
VOXEL_SIZE = np.array([0.13, 0.13, 0.16], dtype=np.float64)
BOX_SCALE = 1.0
MAX_VOXELS = 20000
CUBE_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=np.float64,
)
CUBE_EDGES = np.array(
    [[0, 1], [1, 2], [2, 3], [3, 0],
     [4, 5], [5, 6], [6, 7], [7, 4],
     [0, 4], [1, 5], [2, 6], [3, 7]],
    dtype=np.int32,
)


def load_kitti_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def forward_crop(pts, x_max=50.0, y_half=16.0):
    xyz = pts[:, :3]
    mask = (xyz[:, 0] >= 0.0) & (xyz[:, 0] <= x_max) & (np.abs(xyz[:, 1]) <= y_half)
    crop = pts[mask]
    if len(crop) < 100:
        crop = pts
    return crop, crop[:, :3].min(axis=0), crop[:, :3].max(axis=0)


def forward_camera(pts):
    """Same chase-cam as vis_topdown.py: behind the car, looking down the road."""
    z_ground = float(np.percentile(pts[:, 2], 15))
    eye = np.array([-5.0, 0.0, z_ground + 4.0])
    lookat = np.array([20.0, 0.0, z_ground + 0.6])
    up = np.array([0.0, 0.0, 1.0])
    return eye, lookat, up


def voxelize(pts, voxel_size):
    xyz = pts[:, :3]
    ijk = np.floor((xyz - PC_RANGE_MIN) / voxel_size).astype(np.int32)
    uniq, inv, counts = np.unique(ijk, axis=0, return_inverse=True, return_counts=True)
    corners = PC_RANGE_MIN + uniq.astype(np.float64) * voxel_size
    return uniq, corners, counts


def jet_from_z(z):
    """Open3D-style height coloring: low=blue, high=red (jet)."""
    z = np.asarray(z, dtype=np.float64)
    lo, hi = np.percentile(z, [2, 98])
    t = np.clip((z - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=1)


def voxel_mesh(corners, voxel_size, colors, scale=BOX_SCALE):
    n = len(corners)
    pad = voxel_size * (1.0 - scale) / 2.0
    c0 = corners + pad
    size = voxel_size * scale
    verts = (c0[:, None, :] + CUBE_CORNERS[None, :, :] * size).reshape(-1, 3)
    faces_one = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [5, 6, 7],
            [0, 4, 5], [0, 5, 1],
            [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ],
        dtype=np.int32,
    )
    faces = (np.arange(n, dtype=np.int32)[:, None, None] * 8 + faces_one[None, :, :]).reshape(-1, 3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.repeat(colors, 8, axis=0))
    return mesh


def voxel_wireframes(corners, voxel_size, scale=BOX_SCALE):
    n = len(corners)
    pad = voxel_size * (1.0 - scale) / 2.0
    size = voxel_size * scale
    verts = (corners[:, None, :] + pad + CUBE_CORNERS[None, :, :] * size).reshape(-1, 3)
    lines = (np.arange(n, dtype=np.int32)[:, None, None] * 8 + CUBE_EDGES[None, :, :]).reshape(-1, 2)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(verts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color([0.15, 0.15, 0.15])
    return ls


def camera_extrinsic(eye, lookat, up):
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


def build_geometries(pts):
    crop, mins, maxs = forward_crop(pts)
    voxel_size = VOXEL_SIZE.copy()
    uniq, corners, counts = voxelize(crop, voxel_size)
    while len(corners) > MAX_VOXELS:
        voxel_size *= 1.25
        uniq, corners, counts = voxelize(crop, voxel_size)

    colors = jet_from_z(corners[:, 2] + 0.5 * voxel_size[2])
    mesh = voxel_mesh(corners, voxel_size, colors)
    lines = voxel_wireframes(corners, voxel_size)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(crop[:, :3])
    return mesh, lines, pcd, mins, maxs, voxel_size, len(crop), uniq, corners


def save_matplotlib(corners, voxel_size, path, eye, lookat):
    rgb = jet_from_z(corners[:, 2] + 0.5 * voxel_size[2])
    pad = voxel_size * (1.0 - BOX_SCALE) / 2.0
    size = voxel_size * BOX_SCALE
    off = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=np.float64,
    )
    face_ids = np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
         [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]],
        dtype=np.int32,
    )

    verts = corners[:, None, :] + pad + off[None, :, :] * size
    faces = verts[:, face_ids].reshape(-1, 4, 3)
    facecolors = np.repeat(np.concatenate([rgb, np.full((len(rgb), 1), 0.92)], axis=1), 6, axis=0)

    fig = plt.figure(figsize=(12.5, 7.8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    coll = Poly3DCollection(
        faces,
        facecolors=facecolors,
        edgecolors=(0.62, 0.62, 0.62, 0.40),
        linewidths=0.12,
        shade=False,
    )
    ax.add_collection3d(coll)

    lookat = np.asarray(lookat, dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64)
    ax.set_xlim(-2.0, 42.0)
    ax.set_ylim(-14.0, 14.0)
    ax.set_zlim(lookat[2] - 2.2, lookat[2] + 3.4)
    ax.set_box_aspect((44.0, 28.0, 5.6))
    vec = eye - lookat
    elev = float(np.degrees(np.arctan2(vec[2], np.hypot(vec[0], vec[1]))))
    azim = float(np.degrees(np.arctan2(vec[1], vec[0])))
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("persp", focal_length=1.2)
    ax.set_xlabel("X forward (m)", labelpad=8)
    ax.set_ylabel("Y left (m)", labelpad=8)
    ax.set_zlabel("Z up (m)", labelpad=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def show_interactive(mesh, lines, eye, lookat, up):
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name="voxelization", width=1280, height=800):
        return False
    vis.add_geometry(mesh)
    vis.add_geometry(lines)
    opt = vis.get_render_option()
    if opt is None:
        vis.destroy_window()
        return False
    opt.mesh_show_back_face = True
    opt.light_on = False

    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    params.extrinsic = camera_extrinsic(eye, lookat, up)
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    vis.run()
    vis.destroy_window()
    return True


def main():
    pts = load_kitti_bin(BIN_PATH)
    mesh, lines, pcd, mins, maxs, voxel_size, n_pts, uniq, corners = build_geometries(pts)
    eye, lookat, up = forward_camera(pts)

    print(f"crop points: {n_pts}")
    print(f"occupied voxels: {len(uniq)}")
    print(f"voxel size: {voxel_size.tolist()}")
    print(f"crop xyz min: {mins}")
    print(f"crop xyz max: {maxs}")
    print(f"eye: {eye}")
    print(f"lookat: {lookat}")

    save_matplotlib(corners, voxel_size, OUT_PATH, eye, lookat)
    print(f"saved: {OUT_PATH}")
    if "--save-only" in sys.argv:
        return
    if has_display() and show_interactive(mesh, lines, eye, lookat, up):
        return
    print("no DISPLAY; skipped Open3D window")


if __name__ == "__main__":
    main()

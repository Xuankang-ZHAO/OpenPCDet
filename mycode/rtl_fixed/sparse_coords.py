"""SECOND VoxelBackBone8x occupancy coordinates for downsample layers.

SubM layers (conv_input / conv1, and the SubM blocks after each stride-2
conv) keep the same voxel indices. SparseConv downsample layers change both
the spatial shape and the active set. These helpers follow the same output-
site rule as `accdesign` golden export (`generate_output_coords`).
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

# VoxelBackBone8x: sparse_shape = grid_size[::-1] + [1, 0, 0]
SPARSE_SHAPE_Z_PAD = 1

# First SparseConv of conv2 / conv3 / conv4. Kernel/stride/padding are ZYX.
DOWNSAMPLE_STAGES: List[Dict] = [
    {
        'name': 'conv2.0',
        'tag': 'conv2_0',
        'kernel': (3, 3, 3),
        'stride': (2, 2, 2),
        'padding': (1, 1, 1),
    },
    {
        'name': 'conv3.0',
        'tag': 'conv3_0',
        'kernel': (3, 3, 3),
        'stride': (2, 2, 2),
        'padding': (1, 1, 1),
    },
    {
        'name': 'conv4.0',
        'tag': 'conv4_0',
        'kernel': (3, 3, 3),
        'stride': (2, 2, 2),
        'padding': (0, 1, 1),
    },
]


def backbone_input_sparse_shape_zyx(grid_size_xyz: Sequence[int]) -> Tuple[int, int, int]:
    nx, ny, nz = (int(grid_size_xyz[0]), int(grid_size_xyz[1]), int(grid_size_xyz[2]))
    return (nz + SPARSE_SHAPE_Z_PAD, ny, nx)


def spatial_shape_zyx_to_xyz(shape_zyx: Sequence[int]) -> Tuple[int, int, int]:
    nz, ny, nx = (int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2]))
    return (nx, ny, nz)


def generate_output_coords(
    input_coords: np.ndarray,
    kernel: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    input_shape_zyx: Sequence[int],
    conv_type: str = 'SparseConv3d',
):
    """Return unique OFM coordinates in [z, y, x] and the ZYX output shape."""
    if conv_type == 'SubMConv3d':
        return np.asarray(input_coords, dtype=np.int64).copy(), tuple(int(v) for v in input_shape_zyx)

    kz_count, ky_count, kx_count = (int(kernel[0]), int(kernel[1]), int(kernel[2]))
    sz, sy, sx = (int(stride[0]), int(stride[1]), int(stride[2]))
    pz, py, px = (int(padding[0]), int(padding[1]), int(padding[2]))
    in_z, in_y, in_x = (int(input_shape_zyx[0]), int(input_shape_zyx[1]), int(input_shape_zyx[2]))
    out_shape = (
        (in_z + 2 * pz - (kz_count - 1) - 1) // sz + 1,
        (in_y + 2 * py - (ky_count - 1) - 1) // sy + 1,
        (in_x + 2 * px - (kx_count - 1) - 1) // sx + 1,
    )

    if input_coords is None or np.asarray(input_coords).size == 0:
        return np.zeros((0, 3), dtype=np.int64), out_shape

    coords = np.asarray(input_coords, dtype=np.int64)
    out_set = set()
    for kz in range(kz_count):
        dz = coords[:, 0] + pz - kz
        valid_z = (dz % sz) == 0
        oz = dz // sz
        for ky in range(ky_count):
            dy = coords[:, 1] + py - ky
            valid_y = (dy % sy) == 0
            oy = dy // sy
            for kx in range(kx_count):
                dx = coords[:, 2] + px - kx
                valid_x = (dx % sx) == 0
                ox = dx // sx
                valid = (
                    valid_z & valid_y & valid_x
                    & (oz >= 0) & (oz < out_shape[0])
                    & (oy >= 0) & (oy < out_shape[1])
                    & (ox >= 0) & (ox < out_shape[2])
                )
                if np.any(valid):
                    out_set.update(
                        (int(z), int(y), int(x))
                        for z, y, x in zip(oz[valid], oy[valid], ox[valid])
                    )

    if not out_set:
        return np.zeros((0, 3), dtype=np.int64), out_shape
    return np.asarray(sorted(out_set), dtype=np.int64), out_shape


def iter_downsample_stage_coords(input_coords_zyx: np.ndarray, grid_size_xyz: Sequence[int]):
    """Yield (stage_spec, ofm_coords_zyx, ofm_grid_size_xyz) for conv2.0/3.0/4.0."""
    coords = np.asarray(input_coords_zyx, dtype=np.int64)
    shape_zyx = backbone_input_sparse_shape_zyx(grid_size_xyz)
    for spec in DOWNSAMPLE_STAGES:
        coords, shape_zyx = generate_output_coords(
            coords,
            spec['kernel'],
            spec['stride'],
            spec['padding'],
            shape_zyx,
            'SparseConv3d',
        )
        yield spec, coords, spatial_shape_zyx_to_xyz(shape_zyx)

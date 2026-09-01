"""Fixed-size block partition helper.

Tiles the voxel grid with one XYZ block size and no zone lookup. Block
indices are `coord // block_size` from grid origin (0, 0, 0). Chebyshev
distance and LiDAR-center zone squares are not used.

Boundary voxels still emit RTL-style halo requests: flags are computed
once from the owner voxel, combinations 1..7 are emitted in order, and
each halo coordinate is re-binned with the same fixed block size.
"""

from typing import Iterable, Sequence, Tuple

import numpy as np


def _axis_boundary_flags(coord: int, block_size: int, coord_max: int) -> Tuple[bool, bool]:
    residue = int(coord) % int(block_size)
    is_low = (residue == 0) and (coord != 0)
    is_high = (residue == int(block_size) - 1) and (coord != coord_max)
    return is_low, is_high


def _block_index(coord: int, block_size: int, num_blocks: int) -> int:
    index = int(coord) // int(block_size)
    if not (0 <= index < num_blocks):
        raise ValueError(f'Block index {index} is outside [0, {num_blocks}) for coord {coord}')
    return index


def _linear_block_id(
    block_x: int,
    block_y: int,
    block_z: int,
    num_blocks_xyz: Tuple[int, int, int],
) -> int:
    num_bx, num_by, _ = num_blocks_xyz
    return int(block_x) + int(block_y) * num_bx + int(block_z) * (num_bx * num_by)


def grid_block_counts(grid_size: Tuple[int, int, int], block_size_xyz: Tuple[int, int, int]) -> Tuple[int, int, int]:
    nx, ny, nz = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
    bx, by, bz = (int(block_size_xyz[0]), int(block_size_xyz[1]), int(block_size_xyz[2]))
    if min(nx, ny, nz, bx, by, bz) <= 0:
        raise ValueError(f'Grid size {grid_size} and block size {block_size_xyz} must be positive')
    return (
        (nx + bx - 1) // bx,
        (ny + by - 1) // by,
        (nz + bz - 1) // bz,
    )


def _grid_block_counts(grid_size: Tuple[int, int, int], block_size_xyz: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return grid_block_counts(grid_size, block_size_xyz)


def _iter_rtl_block_ids_for_voxel(
    x_idx: int,
    y_idx: int,
    z_idx: int,
    grid_size: Tuple[int, int, int],
    block_size_xyz: Tuple[int, int, int],
    num_blocks_xyz: Tuple[int, int, int],
) -> Iterable[int]:
    nx, ny, nz = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
    bx, by, bz = (int(block_size_xyz[0]), int(block_size_xyz[1]), int(block_size_xyz[2]))
    num_bx, num_by, num_bz = num_blocks_xyz

    yield _linear_block_id(
        _block_index(x_idx, bx, num_bx),
        _block_index(y_idx, by, num_by),
        _block_index(z_idx, bz, num_bz),
        num_blocks_xyz,
    )

    x_low, x_high = _axis_boundary_flags(x_idx, bx, nx - 1)
    y_low, y_high = _axis_boundary_flags(y_idx, by, ny - 1)
    z_low, z_high = _axis_boundary_flags(z_idx, bz, nz - 1)

    x_on = x_low or x_high
    y_on = y_low or y_high
    z_on = z_low or z_high
    valid_halo = {
        1: x_on,
        2: y_on,
        3: x_on and y_on,
        4: z_on,
        5: x_on and z_on,
        6: y_on and z_on,
        7: x_on and y_on and z_on,
    }

    for halo_index in range(1, 8):
        if not valid_halo[halo_index]:
            continue

        halo_x = x_idx
        halo_y = y_idx
        halo_z = z_idx
        if halo_index & 0b001:
            halo_x += -1 if x_low else 1
        if halo_index & 0b010:
            halo_y += -1 if y_low else 1
        if halo_index & 0b100:
            halo_z += -1 if z_low else 1

        if not (0 <= halo_x < nx and 0 <= halo_y < ny and 0 <= halo_z < nz):
            continue

        yield _linear_block_id(
            _block_index(halo_x, bx, num_bx),
            _block_index(halo_y, by, num_by),
            _block_index(halo_z, bz, num_bz),
            num_blocks_xyz,
        )


def compute_rtl_fixed_partition_counts(
    coords: np.ndarray,
    grid_size: Tuple[int, int, int],
    block_size_xyz: Sequence[int] = (10, 10, 6),
):
    """Aggregate RTL-emitted fixed-block requests into per-block counts.

    Args:
        coords: voxel coordinates in [z, y, x] order with shape (N, 3)
        grid_size: full voxel grid size as (nx, ny, nz)
        block_size_xyz: fixed block size as (bx, by, bz); default 10x10x6

    Returns:
        counts: per-block request counts over the full tiled grid, including empties
        total_blocks: number of tiles covering the voxel grid
        block_voxel_limit: bx * by * bz
    """
    nx, ny, nz = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
    bx, by, bz = (int(block_size_xyz[0]), int(block_size_xyz[1]), int(block_size_xyz[2]))
    num_blocks_xyz = grid_block_counts((nx, ny, nz), (bx, by, bz))
    total_blocks = int(num_blocks_xyz[0] * num_blocks_xyz[1] * num_blocks_xyz[2])
    block_voxel_limit = bx * by * bz
    counts = np.zeros(total_blocks, dtype=np.int64)

    if coords is None or coords.size == 0:
        return counts, total_blocks, block_voxel_limit

    for z_idx, y_idx, x_idx in coords.astype(np.int64):
        x_i, y_i, z_i = int(x_idx), int(y_idx), int(z_idx)
        if not (0 <= x_i < nx and 0 <= y_i < ny and 0 <= z_i < nz):
            continue
        for block_id in _iter_rtl_block_ids_for_voxel(
            x_i,
            y_i,
            z_i,
            (nx, ny, nz),
            (bx, by, bz),
            num_blocks_xyz,
        ):
            counts[block_id] += 1

    return counts, total_blocks, block_voxel_limit

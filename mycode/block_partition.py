"""Block partition helper: compute replicated block counts for voxels.

Provides `compute_block_partition_counts(...)` as the unified entrypoint
that supports two modes:
    - 'fixed': tile the voxel grid into fixed-size blocks and apply
        boundary replication.
    - 'unfixed': variable block sizes selected from a lookup table (LUT)
        based on the voxel's Chebyshev distance to a LiDAR center; boundary
        replication is applied per-block-size similarly to fixed mode.

This module exposes the unified API and two internal implementations
`_compute_fixed_partition_counts` and `_compute_unfixed_partition_counts`.
"""
import math
from typing import Tuple
import numpy as np


def compute_block_partition_counts(coords: np.ndarray,
                                   grid_size: Tuple[int, int, int],
                                   block_size_xyz: Tuple[int, int, int],
                                   mode: str = 'fixed',
                                   lut_path: str = None,
                                   lidar_center_xy: Tuple[int, int] = None):
    """Compute per-block voxel counts with optional boundary replication modes.

    Unified API for block partitioning.

    Supports two modes selected via the `mode` argument:
      - 'fixed': use `block_size_xyz` for all blocks (original behavior).
      - 'unfixed': determine block sizes per-voxel using a LUT file and
        a LiDAR voxel-space center; in this mode `lut_path` and
        `lidar_center_xy` must be provided.

    Args:
        coords: numpy array of shape (N,3) with voxel indices in order [z,y,x]
        grid_size: (nx, ny, nz) total grid size in voxels
        block_size_xyz: (bx, by, bz) block size in voxels per axis (used for 'fixed')
        mode: 'fixed' or 'unfixed' partitioning strategy
        lut_path: path to block-size lookup table used when mode='unfixed'
        lidar_center_xy: (cx, cy) LiDAR voxel-space center used for Chebyshev distance

    Returns:
        counts: numpy array of length total_blocks with replicated counts
        total_blocks: int number of blocks
        block_voxel_limit: int capacity of a single block (without replication)
    """
    # Dispatch to the appropriate implementation
    if mode == 'fixed':
        return _compute_fixed_partition_counts(coords, grid_size, block_size_xyz)
    else:
        return _compute_unfixed_partition_counts(coords, grid_size, lut_path, lidar_center_xy)


def _compute_unfixed_partition_counts(coords: np.ndarray,
                                      grid_size: Tuple[int, int, int],
                                      lut_path: str,
                                      lidar_center_xy: Tuple[int, int]):
    """Unfixed partition: use LUT and lidar center to pick block sizes.

    Returns same tuple as compute_block_partition_counts.
    """
    # Unfixed mode requires a lookup table and a lidar center
    if lut_path is None:
        raise ValueError('lut_path must be provided for unfixed mode')
    if lidar_center_xy is None:
        raise ValueError('lidar_center_xy must be provided for unfixed mode')

    # Load lookup table: each non-empty non-comment line should be
    # of the form: zone:start-end:bx,by,bz
    # where `zone` is an integer zone id starting from 0.
    lut = []  # list of tuples (zone, start, end, (bx,by,bz))
    with open(lut_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            # Expected format: "zone:start-end:bx,by,bz"
            parts = s.split(':')
            if len(parts) != 3:
                raise ValueError(f'Invalid LUT line (expected zone:start-end:bx,by,bz): {s}')
            zone_part, range_part, size_part = parts
            try:
                zone = int(zone_part.strip())
            except Exception:
                raise ValueError(f'Invalid zone id in LUT line: {s}')
            if '-' not in range_part:
                raise ValueError(f'Invalid LUT range: {range_part}')
            start_s, end_s = range_part.split('-', 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            sizes = [int(x.strip()) for x in size_part.split(',')]
            if len(sizes) != 3:
                raise ValueError(f'Invalid block size in LUT line: {s}')
            lut.append((zone, start, end, (sizes[0], sizes[1], sizes[2])))

    # Helper to find block size for a given chebyshev distance
    def lookup_block_size(dist: int):
        for (zone, a, b, sz) in lut:
            if a <= dist <= b:
                return zone, sz
        # If not found, raise
        raise ValueError(f'No LUT entry for distance {dist}')

    cx, cy = int(lidar_center_xy[0]), int(lidar_center_xy[1])
    nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])

    # Build dynamic block ids using dict mapping
    block_id_map = {}  # key -> id
    next_id = 1 # start from 1 to reserve 0 for potential empty blocks，since bincount minlength will use 0 to meet the length

    # For each voxel determine its block size by chebyshev distance
    z_idx = coords[:, 0]
    y_idx = coords[:, 1]
    x_idx = coords[:, 2]

    replicated_block_ids = []
    nvox = coords.shape[0]
    for i in range(nvox):
        xi = int(x_idx[i])
        yi = int(y_idx[i])
        zi = int(z_idx[i])

        # Determine own zone and block indices
        dist = max(abs(xi - cx), abs(yi - cy))
        zone_i, (bx_i, by_i, bz_i) = lookup_block_size(dist)

        bx0 = xi // bx_i
        by0 = yi // by_i
        bz0 = zi // bz_i

        # Start with own block，the set is unique so we can add neighbors to it without worrying about duplicates
        target_keys = set()
        target_keys.add((zone_i, int(bx0), int(by0), int(bz0)))

        # For each axis, if voxel lies on a block boundary (low or high),
        # compute neighbor voxel coordinates by shifting +/-1 and compute
        # that neighbor's zone and block id — add to targets. Also handle
        # multi-axis combinations by forming cartesian product of shifts.
        sx_choices = [0]
        modx = xi % bx_i
        if modx == 0:
            sx_choices.append(-1)
        if modx == (bx_i - 1):
            sx_choices.append(1)

        sy_choices = [0]
        mody = yi % by_i
        if mody == 0:
            sy_choices.append(-1)
        if mody == (by_i - 1):
            sy_choices.append(1)

        sz_choices = [0]
        modz = zi % bz_i
        if modz == 0:
            sz_choices.append(-1)
        if modz == (bz_i - 1):
            sz_choices.append(1)

        # Iterate through neighbor shifts excluding (0,0,0) since own block already added
        for sx in sx_choices:
            for sy in sy_choices:
                for sz in sz_choices:
                    if sx == 0 and sy == 0 and sz == 0:
                        continue
                    xj = xi + sx
                    yj = yi + sy
                    zj = zi + sz
                    # skip out-of-bounds neighbor voxels
                    if not (0 <= xj < nx and 0 <= yj < ny and 0 <= zj < nz):
                        continue
                    # neighbor's chebyshev distance and zone/sizes
                    neigh_dist = max(abs(xj - cx), abs(yj - cy))
                    zone_j, (bx_j, by_j, bz_j) = lookup_block_size(neigh_dist)
                    # neighbor's block indices using its block sizes
                    bxj0 = xj // bx_j
                    byj0 = yj // by_j
                    bzj0 = zj // bz_j
                    target_keys.add((zone_j, int(bxj0), int(byj0), int(bzj0)))

        # Map target_keys to ids and record replicated ids
        for key in target_keys:
            if key not in block_id_map:
                block_id_map[key] = next_id
                next_id += 1
            replicated_block_ids.append(block_id_map[key])

    # Convert replicated_block_ids -> counts using numpy bincount for speed.
    # Use minlength=next_id to ensure we have an entry for every assigned block id.
    if len(replicated_block_ids) == 0:
        counts = np.zeros(0, dtype=np.int64)
    else:
        counts = np.bincount(np.array(replicated_block_ids, dtype=np.int64), minlength=next_id)

    # block_voxel_limit: pick maximum capacity among LUT entries
    block_voxel_limit = max([sz[0] * sz[1] * sz[2] for (_, _, sz) in lut])
    total_blocks = counts.size
    return counts, total_blocks, block_voxel_limit


def _compute_fixed_partition_counts(coords: np.ndarray,
                                    grid_size: Tuple[int, int, int],
                                    block_size_xyz: Tuple[int, int, int]):
    """Fixed-size block partition implementation (original logic)."""
    nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    bx, by, bz = int(block_size_xyz[0]), int(block_size_xyz[1]), int(block_size_xyz[2])

    num_blocks_x = math.ceil(nx / bx)
    num_blocks_y = math.ceil(ny / by)
    num_blocks_z = math.ceil(nz / bz)
    total_blocks = num_blocks_x * num_blocks_y * num_blocks_z

    # Handle empty coords
    if coords is None or coords.size == 0:
        counts = np.zeros(total_blocks, dtype=np.int64)
        block_voxel_limit = int(bx) * int(by) * int(bz)
        return counts, total_blocks, block_voxel_limit

    # Expect coords as [z,y,x]
    z_idx = coords[:, 0]
    y_idx = coords[:, 1]
    x_idx = coords[:, 2]

    nvox = coords.shape[0]
    replicated_block_ids = []
    for i in range(nvox):
        xi = int(x_idx[i])
        yi = int(y_idx[i])
        zi = int(z_idx[i])

        # X axis candidates
        bx_cands = set()
        bx0 = xi // bx
        if 0 <= bx0 < num_blocks_x:
            bx_cands.add(int(bx0))
        modx = xi % bx
        if modx == 0:
            bx_prev = bx0 - 1
            if 0 <= bx_prev < num_blocks_x:
                bx_cands.add(int(bx_prev))
        if modx == (bx - 1):
            bx_next = bx0 + 1
            if 0 <= bx_next < num_blocks_x:
                bx_cands.add(int(bx_next))

        # Y axis candidates
        by_cands = set()
        by0 = yi // by
        if 0 <= by0 < num_blocks_y:
            by_cands.add(int(by0))
        mody = yi % by
        if mody == 0:
            by_prev = by0 - 1
            if 0 <= by_prev < num_blocks_y:
                by_cands.add(int(by_prev))
        if mody == (by - 1):
            by_next = by0 + 1
            if 0 <= by_next < num_blocks_y:
                by_cands.add(int(by_next))

        # Z axis candidates
        bz_cands = set()
        bz0 = zi // bz
        if 0 <= bz0 < num_blocks_z:
            bz_cands.add(int(bz0))
        modz = zi % bz
        if modz == 0:
            bz_prev = bz0 - 1
            if 0 <= bz_prev < num_blocks_z:
                bz_cands.add(int(bz_prev))
        if modz == (bz - 1):
            bz_next = bz0 + 1
            if 0 <= bz_next < num_blocks_z:
                bz_cands.add(int(bz_next))

        # Cartesian product
        for bx_i in bx_cands:
            for by_i in by_cands:
                for bz_i in bz_cands:
                    block_id = bx_i + by_i * num_blocks_x + bz_i * (num_blocks_x * num_blocks_y)
                    replicated_block_ids.append(block_id)

    if len(replicated_block_ids) == 0:
        counts = np.zeros(total_blocks, dtype=np.int64)
    else:
        replicated_block_ids = np.array(replicated_block_ids, dtype=np.int64)
        # minlength=total_blocks ensures we get a count for every block even if some are zero
        counts = np.bincount(replicated_block_ids, minlength=total_blocks)

    block_voxel_limit = int(bx) * int(by) * int(bz)
    return counts, total_blocks, block_voxel_limit

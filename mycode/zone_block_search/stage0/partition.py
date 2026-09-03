"""LiDAR-centered uniform block partition with RTL-style 3D halo.

Matches mycode/rtl_unfixed/partition.py voxel-centric emission (primary +
halo directions 1..7), but uses a single Bx×By×Bz size for the whole grid
(no zone LUT). Block indices are:

    u = floor((x - x0) / Bx),  v = floor((y - y0) / By),  w = floor(z / Bz)

Fine square rings use half-open nested squares around (x0, y0).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from mycode.zone_block_search.stage0.config import (
    DELTA_D,
    GRID_SIZE_XYZ,
    LIDAR_CENTER_XY,
    TWO_PAGE_LIMIT,
)


def _ilog2(value: int) -> int:
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f'Block size must be a positive power of two, got {value}')
    return value.bit_length() - 1


def ring_index_from_core_corner(
    core_x: np.ndarray,
    core_y: np.ndarray,
    lidar_center_xy: Tuple[int, int] = LIDAR_CENTER_XY,
    delta_d: int = DELTA_D,
) -> np.ndarray:
    """Assign each block to fine square ring Z_j via its core lower corner.

    Ring j occupies inside((j+1)*delta_d) minus inside(j*delta_d), where
    inside(T) means -T <= dx < T and -T <= dy < T.
    """
    cx, cy = int(lidar_center_xy[0]), int(lidar_center_xy[1])
    dx = core_x.astype(np.int64) - cx
    dy = core_y.astype(np.int64) - cy
    # Chebyshev half-open radius: smallest T=k*delta_d with inside(T).
    # For dx in [-T, T), need T > dx when dx >= 0, and T >= -dx when dx < 0.
    # Equivalent: T > max(dx, -dx-1) wait...
    # inside(T): -T <= dx < T  <=>  dx < T and dx >= -T  <=>  T > dx and T >= -dx
    # For positive integer T multiples of delta_d:
    #   need T >= dx+1 if dx >= 0, and T >= -dx if dx < 0.
    # So needed_T = max(dx+1, -dx) for integer dx... when dx>=0: max(dx+1,-dx)=dx+1
    # when dx<0: max(dx+1,-dx)=-dx. Yes: needed = np.maximum(dx + 1, -dx)
    need_x = np.maximum(dx + 1, -dx)
    need_y = np.maximum(dy + 1, -dy)
    need = np.maximum(need_x, need_y)
    # Ring j covers T in (j*delta_d, (j+1)*delta_d], i.e. need in (j*d, (j+1)*d]
    # j = ceil(need / delta_d) - 1
    j = (need + delta_d - 1) // delta_d - 1
    return np.maximum(j, 0).astype(np.int64)


def _assert_block_fits_one_ring(
    bx: int,
    by: int,
    core_x: np.ndarray,
    core_y: np.ndarray,
    ring: np.ndarray,
    lidar_center_xy: Tuple[int, int],
    delta_d: int,
) -> None:
    """Every interior block core must lie entirely inside its assigned ring."""
    if core_x.size == 0:
        return
    # Opposite corner of the half-open core [core_x, core_x+bx) x [core_y, core_y+by)
    # The last voxel in the core is (core_x+bx-1, core_y+by-1).
    opp_x = core_x + int(bx) - 1
    opp_y = core_y + int(by) - 1
    ring_opp = ring_index_from_core_corner(opp_x, opp_y, lidar_center_xy, delta_d)
    mismatch = ring != ring_opp
    if not np.any(mismatch):
        return
    # Outer FOV-clipped blocks may span rings near the grid edge; only flag
    # mismatches where both corners are on a delta_d-aligned interior lattice.
    cx, cy = int(lidar_center_xy[0]), int(lidar_center_xy[1])
    aligned = (
        ((core_x - cx) % delta_d == 0)
        & ((core_y - cy) % delta_d == 0)
        & ((opp_x + 1 - cx) % delta_d == 0)
        & ((opp_y + 1 - cy) % delta_d == 0)
    )
    bad = mismatch & aligned
    if np.any(bad):
        idx = int(np.flatnonzero(bad)[0])
        raise RuntimeError(
            f'Interior block core crosses ring boundary: '
            f'core=({int(core_x[idx])},{int(core_y[idx])}) '
            f'size={bx}x{by} ring_lo={int(ring[idx])} ring_hi={int(ring_opp[idx])}. '
            f'Check LiDAR anchor / half-open / floor-division consistency.'
        )


def compute_materialized_block_nb(
    coords_zyx: np.ndarray,
    block_size_xyz: Sequence[int],
    grid_size_xyz: Tuple[int, int, int] = GRID_SIZE_XYZ,
    lidar_center_xy: Tuple[int, int] = LIDAR_CENTER_XY,
    delta_d: int = DELTA_D,
    validate_ring_alignment: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Emit RTL-style block requests and return per-materialized-block Nb.

    Args:
        coords_zyx: occupied voxels as (N, 3) in [z, y, x] order.
        block_size_xyz: (Bx, By, Bz), each a power of two.

    Returns:
        nb: int64 array of Nb for each materialized block
        ring: int64 ring index Z_j for each block (from core lower corner)
        block_u, block_v, block_w: optional not returned — we return
            nb, ring, core_x, core_y for diagnostics
        Actually returns: nb, ring, core_x, core_y
    """
    bx, by, bz = (int(block_size_xyz[0]), int(block_size_xyz[1]), int(block_size_xyz[2]))
    nx, ny, nz = (int(grid_size_xyz[0]), int(grid_size_xyz[1]), int(grid_size_xyz[2]))
    cx, cy = int(lidar_center_xy[0]), int(lidar_center_xy[1])
    log2_bx, log2_by, log2_bz = _ilog2(bx), _ilog2(by), _ilog2(bz)

    if coords_zyx is None or np.asarray(coords_zyx).size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty

    coords = np.asarray(coords_zyx, dtype=np.int64)
    z = coords[:, 0]
    y = coords[:, 1]
    x = coords[:, 2]
    valid = (z >= 0) & (z < nz) & (y >= 0) & (y < ny) & (x >= 0) & (x < nx)
    z, y, x = z[valid], y[valid], x[valid]
    if z.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty

    # Boundary flags relative to LiDAR-centered / z-origin block grid.
    # Matches rtl_unfixed._axis_boundary_flags with arithmetic right-shift bins.
    mask_x = bx - 1
    mask_y = by - 1
    mask_z = bz - 1
    rel_x = x - cx
    rel_y = y - cy
    # Power-of-two residue via bit mask (works for negative rel with two's complement).
    res_x = rel_x & mask_x
    res_y = rel_y & mask_y
    res_z = z & mask_z

    x_low = (res_x == 0) & (x != 0)
    x_high = (res_x == mask_x) & (x != nx - 1)
    y_low = (res_y == 0) & (y != 0)
    y_high = (res_y == mask_y) & (y != ny - 1)
    z_low = (res_z == 0) & (z != 0)
    z_high = (res_z == mask_z) & (z != nz - 1)

    x_on = x_low | x_high
    y_on = y_low | y_high
    z_on = z_low | z_high

    x_delta = np.where(x_low, -1, np.where(x_high, 1, 0)).astype(np.int64)
    y_delta = np.where(y_low, -1, np.where(y_high, 1, 0)).astype(np.int64)
    z_delta = np.where(z_low, -1, np.where(z_high, 1, 0)).astype(np.int64)

    # Collect (hx, hy, hz) emission targets: primary + up to 7 halos.
    emit_x = [x]
    emit_y = [y]
    emit_z = [z]

    def _append_halo(use_x: bool, use_y: bool, use_z: bool, active: np.ndarray) -> None:
        if not np.any(active):
            return
        hx = x[active].copy()
        hy = y[active].copy()
        hz = z[active].copy()
        if use_x:
            hx = hx + x_delta[active]
        if use_y:
            hy = hy + y_delta[active]
        if use_z:
            hz = hz + z_delta[active]
        in_grid = (hx >= 0) & (hx < nx) & (hy >= 0) & (hy < ny) & (hz >= 0) & (hz < nz)
        if np.any(in_grid):
            emit_x.append(hx[in_grid])
            emit_y.append(hy[in_grid])
            emit_z.append(hz[in_grid])

    # Halo index bits: 1=x, 2=y, 4=z (same order as rtl_unfixed).
    _append_halo(True, False, False, x_on)
    _append_halo(False, True, False, y_on)
    _append_halo(True, True, False, x_on & y_on)
    _append_halo(False, False, True, z_on)
    _append_halo(True, False, True, x_on & z_on)
    _append_halo(False, True, True, y_on & z_on)
    _append_halo(True, True, True, x_on & y_on & z_on)

    all_x = np.concatenate(emit_x)
    all_y = np.concatenate(emit_y)
    all_z = np.concatenate(emit_z)

    # Floor division via arithmetic right shift (power-of-two sizes).
    block_u = (all_x - cx) >> log2_bx
    block_v = (all_y - cy) >> log2_by
    block_w = all_z >> log2_bz

    # Pack keys into int64 for unique counting. Ranges are modest for stage0.
    # u in roughly [-176, 176], v in [-50, 50], w in [0, 2] for 64x64 — use offsets.
    u_off = block_u - block_u.min()
    v_off = block_v - block_v.min()
    w_off = block_w - block_w.min()
    u_span = int(u_off.max()) + 1 if u_off.size else 1
    v_span = int(v_off.max()) + 1 if v_off.size else 1
    packed = (w_off.astype(np.int64) * (u_span * v_span)
              + v_off.astype(np.int64) * u_span
              + u_off.astype(np.int64))

    unique_packed, nb = np.unique(packed, return_counts=True)
    # Recover one representative (u,v,w) per unique packed key.
    # Rebuild from min offsets:
    u_min = int(block_u.min())
    v_min = int(block_v.min())
    w_min = int(block_w.min())
    rec_w = unique_packed // (u_span * v_span) + w_min
    rem = unique_packed % (u_span * v_span)
    rec_v = rem // u_span + v_min
    rec_u = rem % u_span + u_min

    core_x = cx + rec_u * bx
    core_y = cy + rec_v * by
    ring = ring_index_from_core_corner(core_x, core_y, lidar_center_xy, delta_d)

    if validate_ring_alignment:
        _assert_block_fits_one_ring(
            bx, by, core_x, core_y, ring, lidar_center_xy, delta_d
        )

    return nb.astype(np.int64), ring.astype(np.int64), core_x.astype(np.int64), core_y.astype(np.int64)


def accumulate_ring_nb_samples(
    coords_zyx: np.ndarray,
    block_size_xyz: Sequence[int],
    ring_samples: Dict[int, list],
    **kwargs,
) -> int:
    """Append Nb values into ring_samples[j] lists. Returns materialized count."""
    nb, ring, _, _ = compute_materialized_block_nb(coords_zyx, block_size_xyz, **kwargs)
    for j, value in zip(ring.tolist(), nb.tolist()):
        ring_samples.setdefault(int(j), []).append(int(value))
    return int(nb.size)


def summarize_ring_coverage(
    ring_samples: Dict[int, list],
    two_page_limit: int = TWO_PAGE_LIMIT,
) -> Dict[int, dict]:
    """Per-ring pooled stats over block-frame instances."""
    summary = {}
    for j, values in ring_samples.items():
        arr = np.asarray(values, dtype=np.int64)
        n = int(arr.size)
        if n == 0:
            summary[int(j)] = {
                'n_samples': 0,
                'coverage': float('nan'),
                'p95_nb': float('nan'),
                'reshape_frac': float('nan'),
                'mean_nb': float('nan'),
                'max_nb': 0,
            }
            continue
        le = int(np.sum(arr <= two_page_limit))
        summary[int(j)] = {
            'n_samples': n,
            'coverage': float(le / n),
            'p95_nb': float(np.percentile(arr, 95)),
            'reshape_frac': float(np.mean(arr > two_page_limit)),
            'mean_nb': float(np.mean(arr)),
            'max_nb': int(arr.max()),
        }
    return summary


def validate_against_rtl_unfixed(
    coords_zyx: np.ndarray,
    block_size_xyz: Sequence[int],
    grid_size_xyz: Tuple[int, int, int] = GRID_SIZE_XYZ,
    lidar_center_xy: Tuple[int, int] = LIDAR_CENTER_XY,
) -> dict:
    """Compare Nb multiset vs single-zone rtl_unfixed partition on one frame."""
    from mycode.rtl_unfixed.partition import ZoneSpec, compute_rtl_unfixed_partition_counts

    bx, by, bz = (int(block_size_xyz[0]), int(block_size_xyz[1]), int(block_size_xyz[2]))
    zone_specs = [
        ZoneSpec(
            zone_id=0,
            inner_half_open=0,
            outer_half_open=None,
            block_size_xyz=(bx, by, bz),
            log2_block_size_xyz=(_ilog2(bx), _ilog2(by), _ilog2(bz)),
        )
    ]
    rtl_counts, rtl_n, _ = compute_rtl_unfixed_partition_counts(
        coords_zyx, grid_size_xyz, zone_specs, lidar_center_xy
    )
    nb, _, _, _ = compute_materialized_block_nb(
        coords_zyx,
        block_size_xyz,
        grid_size_xyz=grid_size_xyz,
        lidar_center_xy=lidar_center_xy,
        validate_ring_alignment=False,
    )
    rtl_sorted = np.sort(rtl_counts)
    ours_sorted = np.sort(nb)
    match = rtl_sorted.shape == ours_sorted.shape and np.array_equal(rtl_sorted, ours_sorted)
    return {
        'match': bool(match),
        'rtl_n_blocks': int(rtl_n),
        'ours_n_blocks': int(nb.size),
        'rtl_sum_nb': int(rtl_counts.sum()) if rtl_counts.size else 0,
        'ours_sum_nb': int(nb.sum()) if nb.size else 0,
        'block_size': f'{bx}x{by}x{bz}',
    }

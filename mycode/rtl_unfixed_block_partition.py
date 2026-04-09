"""RTL-aligned unfixed block partition helper.

This module implements the zone-based block partitioning semantics described in
`online_block_partitioning_algorithm_summary.md` and aggregates emitted block
requests into per-block counts for offline analysis.

Differences from the existing software-style unfixed partition helper:
    - boundary flags are computed once from the primary voxel
    - halo combinations are emitted in fixed priority order 1..7
    - each halo coordinate re-runs zone lookup and block index calculation
    - duplicate halo beats that collapse onto the same block key are preserved
      in the aggregated count to match the emitted RTL request stream
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ZoneSpec:
    zone_id: int
    start_dist: int
    end_dist: int
    block_size_xyz: Tuple[int, int, int]
    log2_block_size_xyz: Tuple[int, int, int]


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _ilog2(value: int) -> int:
    return value.bit_length() - 1


def _parse_zone_lut_line(line: str, line_no: int) -> ZoneSpec:
    parts = line.split(':')
    if len(parts) != 3:
        raise ValueError(
            f'Invalid LUT line {line_no}: expected zone:start-end:bx,by,bz, got {line!r}'
        )

    zone_text, range_text, size_text = parts
    try:
        zone_id = int(zone_text.strip())
    except ValueError as exc:
        raise ValueError(f'Invalid zone id on line {line_no}: {zone_text!r}') from exc

    if '-' not in range_text:
        raise ValueError(f'Invalid distance range on line {line_no}: {range_text!r}')

    start_text, end_text = range_text.split('-', 1)
    try:
        start_dist = int(start_text.strip())
        end_dist = int(end_text.strip())
    except ValueError as exc:
        raise ValueError(f'Invalid distance bounds on line {line_no}: {range_text!r}') from exc

    if start_dist < 0 or end_dist < 0 or end_dist < start_dist:
        raise ValueError(f'Invalid distance interval on line {line_no}: {range_text!r}')

    size_parts = [item.strip() for item in size_text.split(',')]
    if len(size_parts) != 3:
        raise ValueError(f'Invalid block size triple on line {line_no}: {size_text!r}')

    try:
        block_size_xyz = tuple(int(item) for item in size_parts)
    except ValueError as exc:
        raise ValueError(f'Invalid block size on line {line_no}: {size_text!r}') from exc

    for size in block_size_xyz:
        if not _is_power_of_two(size):
            raise ValueError(
                f'Block sizes must be powers of two for RTL semantics, line {line_no}: {block_size_xyz!r}'
            )

    return ZoneSpec(
        zone_id=zone_id,
        start_dist=start_dist,
        end_dist=end_dist,
        block_size_xyz=block_size_xyz,
        log2_block_size_xyz=tuple(_ilog2(size) for size in block_size_xyz),
    )


def _max_possible_distance(grid_size: Tuple[int, int, int], lidar_center_xy: Tuple[int, int]) -> int:
    nx, ny, _ = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
    cx, cy = int(lidar_center_xy[0]), int(lidar_center_xy[1])

    max_dx = max(abs(0 - cx), abs((nx - 1) - cx))
    max_dy = max(abs(0 - cy), abs((ny - 1) - cy))
    return max(max_dx, max_dy)


def load_zone_specs(
    lut_path: str,
    grid_size: Tuple[int, int, int],
    lidar_center_xy: Tuple[int, int],
) -> List[ZoneSpec]:
    """Load and validate zone specs for RTL-style unfixed partitioning."""
    zone_specs: List[ZoneSpec] = []
    with open(lut_path, 'r') as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            zone_specs.append(_parse_zone_lut_line(line, line_no))

    if not zone_specs:
        raise ValueError(f'No valid zone specs found in {lut_path}')

    zone_specs = sorted(zone_specs, key=lambda spec: (spec.start_dist, spec.end_dist, spec.zone_id))

    seen_zone_ids = set()
    for index, spec in enumerate(zone_specs):
        if spec.zone_id in seen_zone_ids:
            raise ValueError(f'Duplicate zone id detected in LUT: {spec.zone_id}')
        seen_zone_ids.add(spec.zone_id)

        if index == 0:
            if spec.start_dist != 0:
                raise ValueError('Zone LUT must start from distance 0')
            continue

        prev = zone_specs[index - 1]
        if spec.start_dist <= prev.end_dist:
            raise ValueError(
                f'Overlapping zone ranges detected: zone {prev.zone_id} [{prev.start_dist}, {prev.end_dist}] '
                f'and zone {spec.zone_id} [{spec.start_dist}, {spec.end_dist}]'
            )
        if spec.start_dist != prev.end_dist + 1:
            raise ValueError(
                f'Zone LUT must be contiguous without gaps: zone {prev.zone_id} ends at {prev.end_dist}, '
                f'but zone {spec.zone_id} starts at {spec.start_dist}'
            )

    required_max_dist = _max_possible_distance(grid_size, lidar_center_xy)
    if zone_specs[-1].end_dist < required_max_dist:
        raise ValueError(
            f'Zone LUT does not cover max possible distance {required_max_dist}; '
            f'last zone ends at {zone_specs[-1].end_dist}'
        )

    return zone_specs


def summarize_zone_specs(zone_specs: Sequence[ZoneSpec]) -> str:
    return ';'.join(
        f'{spec.zone_id}:{spec.start_dist}-{spec.end_dist}:{spec.block_size_xyz[0]},{spec.block_size_xyz[1]},{spec.block_size_xyz[2]}'
        for spec in zone_specs
    )


def _lookup_zone_spec(zone_specs: Sequence[ZoneSpec], distance: int) -> ZoneSpec:
    for spec in zone_specs:
        if spec.start_dist <= distance <= spec.end_dist:
            return spec
    raise ValueError(f'No zone spec found for distance {distance}')


def _chebyshev_distance(x_idx: int, y_idx: int, lidar_center_xy: Tuple[int, int]) -> int:
    cx, cy = int(lidar_center_xy[0]), int(lidar_center_xy[1])
    return max(abs(x_idx - cx), abs(y_idx - cy))


def _axis_boundary_flags(coord: int, log2_block_size: int, coord_max: int) -> Tuple[bool, bool]:
    mask = (1 << log2_block_size) - 1
    is_low = ((coord & mask) == 0) and (coord != 0)
    is_high = ((coord & mask) == mask) and (coord != coord_max)
    return is_low, is_high


def _compute_block_key(
    x_idx: int,
    y_idx: int,
    z_idx: int,
    zone_spec: ZoneSpec,
    lidar_center_xy: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    cx, cy = int(lidar_center_xy[0]), int(lidar_center_xy[1])
    log2_bx, log2_by, log2_bz = zone_spec.log2_block_size_xyz

    block_x = (x_idx - cx) >> log2_bx
    block_y = (y_idx - cy) >> log2_by
    block_z = z_idx >> log2_bz
    return zone_spec.zone_id, block_x, block_y, block_z


def _iter_rtl_block_keys_for_voxel(
    x_idx: int,
    y_idx: int,
    z_idx: int,
    grid_size: Tuple[int, int, int],
    zone_specs: Sequence[ZoneSpec],
    lidar_center_xy: Tuple[int, int],
) -> Iterable[Tuple[int, int, int, int]]:
    nx, ny, nz = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
    primary_spec = _lookup_zone_spec(zone_specs, _chebyshev_distance(x_idx, y_idx, lidar_center_xy))
    yield _compute_block_key(x_idx, y_idx, z_idx, primary_spec, lidar_center_xy)

    x_low, x_high = _axis_boundary_flags(x_idx, primary_spec.log2_block_size_xyz[0], nx - 1)
    y_low, y_high = _axis_boundary_flags(y_idx, primary_spec.log2_block_size_xyz[1], ny - 1)
    z_low, z_high = _axis_boundary_flags(z_idx, primary_spec.log2_block_size_xyz[2], nz - 1)

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

        halo_spec = _lookup_zone_spec(zone_specs, _chebyshev_distance(halo_x, halo_y, lidar_center_xy))
        yield _compute_block_key(halo_x, halo_y, halo_z, halo_spec, lidar_center_xy)


def compute_rtl_unfixed_partition_counts(
    coords: np.ndarray,
    grid_size: Tuple[int, int, int],
    zone_specs: Sequence[ZoneSpec],
    lidar_center_xy: Tuple[int, int],
):
    """Aggregate RTL-emitted block requests into per-block counts.

    Args:
        coords: voxel coordinates in [z, y, x] order with shape (N, 3)
        grid_size: full voxel grid size as (nx, ny, nz)
        zone_specs: validated zone specs from `load_zone_specs`
        lidar_center_xy: LiDAR center in voxel index space as (x, y)

    Returns:
        counts: per-block request counts ordered by sorted block key
        total_blocks: number of unique block keys observed in emitted requests
        block_voxel_limit: always -1 in unfixed mode
    """
    if coords is None or coords.size == 0:
        return np.zeros(0, dtype=np.int64), 0, -1

    counts_by_key = {}

    for z_idx, y_idx, x_idx in coords.astype(np.int64):
        for block_key in _iter_rtl_block_keys_for_voxel(
            int(x_idx),
            int(y_idx),
            int(z_idx),
            grid_size,
            zone_specs,
            lidar_center_xy,
        ):
            counts_by_key[block_key] = counts_by_key.get(block_key, 0) + 1

    ordered_keys = sorted(counts_by_key)
    counts = np.array([counts_by_key[key] for key in ordered_keys], dtype=np.int64)
    return counts, len(ordered_keys), -1
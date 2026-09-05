"""RTL-aligned unfixed block partition helper.

Zone membership uses nested signed half-open squares in XY, centered at the
LiDAR voxel origin. Z is not used to choose a zone. Each zone then tiles
voxels with a fixed XYZ block size. Halo emission still follows the RTL
priority order 1..7, but halo coordinates re-run zone lookup and block-index
calculation, and duplicate block keys are preserved in the aggregated count.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

REQUIRED_BLOCK_SIZE_Z = 16


@dataclass(frozen=True)
class ZoneSpec:
    zone_id: int
    inner_half_open: int
    outer_half_open: Optional[int]
    block_size_xyz: Tuple[int, int, int]
    log2_block_size_xyz: Tuple[int, int, int]

    def to_manifest(self):
        return {
            'zone_id': self.zone_id,
            'inner_half_open': self.inner_half_open,
            'outer_half_open': self.outer_half_open,
            'block_size_xyz': list(self.block_size_xyz),
            'log2_block_size_xyz': list(self.log2_block_size_xyz),
        }


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _ilog2(value: int) -> int:
    return value.bit_length() - 1


def _parse_outer_half_open(text: str, line_no: int) -> Optional[int]:
    token = text.strip()
    if token in ('*', 'inf', '+inf'):
        return None
    try:
        outer_t = int(token)
    except ValueError as exc:
        raise ValueError(
            f'Invalid outer half-open bound on line {line_no}: {text!r}'
        ) from exc
    if outer_t <= 0:
        raise ValueError(f'Outer half-open bound must be positive on line {line_no}: {text!r}')
    return outer_t


def _parse_zone_lut_line(line: str, line_no: int) -> Tuple[int, Optional[int], Tuple[int, int, int]]:
    parts = line.split(':')
    if len(parts) != 3:
        raise ValueError(
            f'Invalid LUT line {line_no}: expected zone:outer_T:bx,by,bz, got {line!r}'
        )

    zone_text, bound_text, size_text = parts
    try:
        zone_id = int(zone_text.strip())
    except ValueError as exc:
        raise ValueError(f'Invalid zone id on line {line_no}: {zone_text!r}') from exc

    outer_half_open = _parse_outer_half_open(bound_text, line_no)

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
    if block_size_xyz[2] != REQUIRED_BLOCK_SIZE_Z:
        raise ValueError(
            f'Z block size must be {REQUIRED_BLOCK_SIZE_Z} on line {line_no}: {block_size_xyz!r}'
        )

    return zone_id, outer_half_open, block_size_xyz


def _inside_half_open_square(dx: int, dy: int, outer_t: int) -> bool:
    return (-outer_t <= dx < outer_t) and (-outer_t <= dy < outer_t)


def _validate_nested_squares_and_block_alignment(zone_specs: Sequence[ZoneSpec]) -> None:
    if zone_specs[-1].outer_half_open is not None:
        raise ValueError('The last zone must be unbounded (outer_half_open=*) so it covers the remaining XY')

    for spec in zone_specs[:-1]:
        if spec.outer_half_open is None:
            raise ValueError(f'Only the last zone may be unbounded, but zone {spec.zone_id} is unbounded')

    for index, spec in enumerate(zone_specs):
        expected_inner = 0 if index == 0 else zone_specs[index - 1].outer_half_open
        if spec.inner_half_open != expected_inner:
            raise ValueError(
                f'Zone {spec.zone_id} inner bound {spec.inner_half_open} does not match previous outer bound {expected_inner}'
            )
        if index == 0:
            continue
        prev = zone_specs[index - 1]
        if spec.outer_half_open is not None and spec.outer_half_open <= prev.outer_half_open:
            raise ValueError(
                f'Zone outer bounds must strictly increase: zone {prev.zone_id} T={prev.outer_half_open}, '
                f'zone {spec.zone_id} T={spec.outer_half_open}'
            )

        boundary_t = prev.outer_half_open
        sizes = prev.block_size_xyz[:2] + spec.block_size_xyz[:2]
        for size in sizes:
            if boundary_t % size != 0:
                raise ValueError(
                    f'Zone boundary T={boundary_t} is not aligned to adjacent XY block size {size} '
                    f'(zones {prev.zone_id} and {spec.zone_id})'
                )


def load_zone_specs(
    lut_path: str,
    grid_size: Tuple[int, int, int],
    lidar_center_xy: Tuple[int, int],
) -> List[ZoneSpec]:
    """Load nested signed half-open square zones and their fixed block sizes."""
    nx, ny, _ = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
    cx, cy = int(lidar_center_xy[0]), int(lidar_center_xy[1])
    if not (0 <= cx < nx and 0 <= cy < ny):
        raise ValueError(
            f'LiDAR center {lidar_center_xy} is outside the voxel grid XY {(nx, ny)}'
        )

    parsed: List[Tuple[int, Optional[int], Tuple[int, int, int]]] = []
    with open(lut_path, 'r') as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            parsed.append(_parse_zone_lut_line(line, line_no))

    if not parsed:
        raise ValueError(f'No valid zone specs found in {lut_path}')

    finite = [item for item in parsed if item[1] is not None]
    unbounded = [item for item in parsed if item[1] is None]
    finite.sort(key=lambda item: item[1])
    if len(unbounded) != 1 or unbounded[0][0] != parsed[-1][0]:
        raise ValueError('Zone LUT must end with exactly one unbounded zone')

    ordered = finite + unbounded
    seen_zone_ids = set()
    zone_specs: List[ZoneSpec] = []
    for index, (zone_id, outer_half_open, block_size_xyz) in enumerate(ordered):
        if zone_id in seen_zone_ids:
            raise ValueError(f'Duplicate zone id detected in LUT: {zone_id}')
        seen_zone_ids.add(zone_id)
        inner_half_open = 0 if index == 0 else ordered[index - 1][1]
        zone_specs.append(
            ZoneSpec(
                zone_id=zone_id,
                inner_half_open=inner_half_open,
                outer_half_open=outer_half_open,
                block_size_xyz=block_size_xyz,
                log2_block_size_xyz=tuple(_ilog2(size) for size in block_size_xyz),
            )
        )

    _validate_nested_squares_and_block_alignment(zone_specs)
    return zone_specs


def summarize_zone_specs(zone_specs: Sequence[ZoneSpec]) -> str:
    parts = []
    for spec in zone_specs:
        outer = '*' if spec.outer_half_open is None else str(spec.outer_half_open)
        bx, by, bz = spec.block_size_xyz
        parts.append(f'{spec.zone_id}:{outer}:{bx},{by},{bz}')
    return ';'.join(parts)


def _lookup_zone_spec(
    zone_specs: Sequence[ZoneSpec],
    x_idx: int,
    y_idx: int,
    lidar_center_xy: Tuple[int, int],
) -> ZoneSpec:
    dx = int(x_idx) - int(lidar_center_xy[0])
    dy = int(y_idx) - int(lidar_center_xy[1])
    for spec in zone_specs:
        if spec.outer_half_open is None:
            return spec
        if _inside_half_open_square(dx, dy, spec.outer_half_open):
            return spec
    raise ValueError(f'No zone spec found for signed offset ({dx}, {dy})')


def _axis_boundary_flags(
    coord: int,
    log2_block_size: int,
    coord_max: int,
    origin: int = 0,
) -> Tuple[bool, bool]:
    rel = int(coord) - int(origin)
    mask = (1 << log2_block_size) - 1
    is_low = ((rel & mask) == 0) and (coord != 0)
    is_high = ((rel & mask) == mask) and (coord != coord_max)
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
    cx, cy = int(lidar_center_xy[0]), int(lidar_center_xy[1])
    primary_spec = _lookup_zone_spec(zone_specs, x_idx, y_idx, lidar_center_xy)
    yield _compute_block_key(x_idx, y_idx, z_idx, primary_spec, lidar_center_xy)

    x_low, x_high = _axis_boundary_flags(x_idx, primary_spec.log2_block_size_xyz[0], nx - 1, origin=cx)
    y_low, y_high = _axis_boundary_flags(y_idx, primary_spec.log2_block_size_xyz[1], ny - 1, origin=cy)
    z_low, z_high = _axis_boundary_flags(z_idx, primary_spec.log2_block_size_xyz[2], nz - 1, origin=0)

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

        halo_spec = _lookup_zone_spec(zone_specs, halo_x, halo_y, lidar_center_xy)
        yield _compute_block_key(halo_x, halo_y, halo_z, halo_spec, lidar_center_xy)


def compute_rtl_unfixed_partition_counts(
    coords: np.ndarray,
    grid_size: Tuple[int, int, int],
    zone_specs: Sequence[ZoneSpec],
    lidar_center_xy: Tuple[int, int],
    return_keys: bool = False,
):
    """Aggregate RTL-emitted block requests into per-block counts.

    Args:
        coords: voxel coordinates in [z, y, x] order with shape (N, 3)
        grid_size: full voxel grid size as (nx, ny, nz)
        zone_specs: validated zone specs from `load_zone_specs`
        lidar_center_xy: LiDAR center in voxel index space as (x, y)
        return_keys: if True, also return sorted keys as an (N, 4) int64 array
            of (zone_id, block_x, block_y, block_z)

    Returns:
        counts: per-block request counts ordered by sorted block key
        total_blocks: number of unique block keys observed in emitted requests
        block_voxel_limit: always -1 in unfixed mode
        keys: only when return_keys=True; shape (N, 4)
    """
    empty_counts = np.zeros(0, dtype=np.int64)
    empty_keys = np.zeros((0, 4), dtype=np.int64)
    if coords is None or coords.size == 0:
        if return_keys:
            return empty_counts, 0, -1, empty_keys
        return empty_counts, 0, -1

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
    if return_keys:
        keys = np.asarray(ordered_keys, dtype=np.int64).reshape(-1, 4)
        return counts, len(ordered_keys), -1, keys
    return counts, len(ordered_keys), -1

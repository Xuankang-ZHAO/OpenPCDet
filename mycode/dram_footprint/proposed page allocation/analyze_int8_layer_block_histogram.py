#!/usr/bin/env python3
"""INT8 SECOND 3D-backbone occupancy under the proposed zone/block LUT.

Halo is generated from the *consumer* (next) layer: 3-tap axes use
neg=padding, pos=2-padding; 1-tap axes emit no halo. Pages hold 64 voxels
and the page byte size is 64*(8+C) rounded up to 512B. Per-layer IFM and OFM
are counted separately; peak DRAM is IFM+OFM coresident.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mycode.kitti_frame_loader import (
    add_data_mode_args,
    build_kitti_dataset,
    choose_kitti_frame_ids,
    load_kitti_sample,
    normalize_voxel_coords,
    resolve_data_mode,
)
from mycode.rtl_unfixed.partition import (
    _compute_block_key,
    _lookup_zone_spec,
    summarize_zone_specs,
)
from mycode.zone_block_search.block_nb_analysis import (
    FINAL_LUTS,
    lut_lines_from_final,
    make_bin_edges,
    zone_specs_from_lut_lines,
)

BIN_WIDTH = 16
PAGE_VOXELS = 64
PAGE_ALIGN_BYTES = 512
COORD_BYTES = 8
FEATURE_BYTES_PER_CHANNEL = 1

STAGE_BY_ZYX = {
    (41, 1600, 1408): 0,
    (21, 800, 704): 1,
    (11, 400, 352): 2,
    (5, 200, 176): 3,
    (2, 200, 176): 3,
}

LIDAR_CENTER_BY_STAGE = {
    0: (0, 800),
    1: (0, 400),
    2: (0, 200),
    3: (0, 100),
}

HaloXYZ = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Proposed-LUT INT8 SECOND 3D-backbone IFM/OFM DRAM footprint'
    )
    parser.add_argument('--cfg', type=str, default='tools/cfgs/kitti_models/second_hw_qat.yaml')
    parser.add_argument(
        '--ckpt',
        type=str,
        default='output/kitti_models/second_hw_qat/hw_qat_10ep/ckpt/checkpoint_epoch_10.pth',
    )
    add_data_mode_args(parser)
    parser.add_argument('--frame_id', type=str, default='000216')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--weight_quant', choices=['per_channel', 'per_tensor'], default='per_channel')
    parser.add_argument('--bin_width', type=int, default=BIN_WIDTH)
    parser.add_argument(
        '--out_dir',
        type=str,
        default=str(_SCRIPT_DIR),
        help='Directory for markdown/json/csv outputs',
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def move_batch_to_device(batch_dict, device):
    skip_keys = {'frame_id', 'metadata', 'calib', 'image_paths', 'ori_shape', 'img_process_infos'}
    output = {}
    for key, value in batch_dict.items():
        if key in skip_keys:
            output[key] = value
            continue
        if isinstance(value, np.ndarray):
            if key in {'image_shape', 'voxel_coords'}:
                output[key] = torch.from_numpy(value).int().to(device)
            else:
                output[key] = torch.from_numpy(value).float().to(device)
        else:
            output[key] = value
    output['batch_size'] = batch_dict.get('batch_size', 1)
    return output


def spatial_shape_zyx(sparse_tensor) -> Tuple[int, int, int]:
    shape = [int(value) for value in list(sparse_tensor.spatial_shape)[:3]]
    if len(shape) != 3:
        raise ValueError(f'Unexpected spatial_shape: {sparse_tensor.spatial_shape}')
    return tuple(shape)


def coords_from_sparse(sparse_tensor) -> np.ndarray:
    coords = normalize_voxel_coords(sparse_tensor.indices.detach().cpu().numpy())
    if coords is None:
        return np.zeros((0, 3), dtype=np.int64)
    return coords


def stage_for_spatial(shape_zyx: Sequence[int]) -> int:
    key = tuple(int(v) for v in shape_zyx[:3])
    if key not in STAGE_BY_ZYX:
        raise KeyError(f'No proposed LUT stage for spatial_shape ZYX={key}')
    return STAGE_BY_ZYX[key]


def grid_xyz_from_zyx(shape_zyx: Sequence[int]) -> Tuple[int, int, int]:
    nz, ny, nx = (int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2]))
    return nx, ny, nz


def lut_label(stage: int) -> str:
    parts = []
    for zone_id, outer, size in FINAL_LUTS[stage]:
        outer_s = '*' if outer is None else str(outer)
        bx, by, bz = size
        parts.append(f'Z{zone_id}[{outer_s}]:{bx}x{by}x{bz}')
    return '; '.join(parts)


def as_int_triple(value) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value), int(value), int(value))
    seq = list(value)
    if len(seq) == 1:
        item = int(seq[0])
        return (item, item, item)
    return tuple(int(item) for item in seq)


def conv_geometry(conv) -> dict:
    kernel = as_int_triple(conv.kernel_size)
    padding = as_int_triple(getattr(conv, 'padding', 0))
    stride = as_int_triple(getattr(conv, 'stride', 1))
    return {
        'kernel_zyx': list(kernel),
        'padding_zyx': list(padding),
        'stride_zyx': list(stride),
        'cin': int(conv.in_channels),
        'cout': int(conv.out_channels),
        'conv_type': type(conv).__name__,
    }


def halo_extent_one(kernel: int, padding: int) -> Tuple[int, int]:
    """Return (neg, pos) along one axis."""
    kernel = int(kernel)
    padding = int(padding)
    if kernel <= 1:
        return (0, 0)
    if kernel == 3:
        return (padding, 2 - padding)
    return (padding, kernel - 1 - padding)


def halo_xyz_from_consumer(consumer: Optional[dict]) -> HaloXYZ:
    """Halo extents in XYZ as ((neg,pos) x3). No consumer => no halo."""
    if consumer is None:
        return ((0, 0), (0, 0), (0, 0))
    kz, ky, kx = consumer['kernel_zyx']
    pz, py, px = consumer['padding_zyx']
    return (
        halo_extent_one(kx, px),
        halo_extent_one(ky, py),
        halo_extent_one(kz, pz),
    )


def format_halo(halo_xyz: HaloXYZ) -> str:
    axes = ('x', 'y', 'z')
    parts = []
    for axis, (neg, pos) in zip(axes, halo_xyz):
        if neg == 0 and pos == 0:
            parts.append(f'{axis}:none')
        else:
            neg_s = f'-{neg}' if neg else '0'
            parts.append(f'{axis}:[{neg_s},+{pos}]')
    return ' '.join(parts)


def enable_hw_reference(model, project_root: Path, out_dir: Path, weight_quant: str, logger):
    tools_dir = project_root / 'tools'
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import test_second_hw_qat as hw_qat_tools

    backbone = model.backbone_3d
    export_dir = out_dir / 'hw_export'
    export_args = SimpleNamespace(
        weight_quant=weight_quant,
        observer='max',
        observer_momentum=0.95,
        bias_bits=32,
        shift_bits=5,
        max_shift_rel_error=1.0,
        emit_binary=False,
        check_export=False,
    )
    export_result = hw_qat_tools.export_hw_payload(backbone, export_dir, export_args, logger)
    backbone.enable_hw_qat(
        False,
        weight_quant=weight_quant,
        observer='max',
        observer_momentum=0.95,
        fake_quant=False,
    )
    backbone.enable_hw_reference(True, qparams=export_result['qparams'])
    return export_result


def capture_layer_outputs(backbone, batch_after_vfe) -> Tuple[List[dict], List[dict]]:
    captured: List[dict] = []
    consumers: List[dict] = []
    orig_layer = backbone._forward_hw_reference_layer

    def record(layer_id, layer_name, conv_type, sparse_tensor, geom=None):
        shape = spatial_shape_zyx(sparse_tensor)
        coords = coords_from_sparse(sparse_tensor)
        captured.append({
            'layer_id': layer_id,
            'layer_name': layer_name,
            'conv_type': conv_type,
            'spatial_shape_zyx': list(shape),
            'feature_channels': int(sparse_tensor.features.shape[1]),
            'active_voxels': int(coords.shape[0]),
            'coords_zyx': coords,
            'geom': geom,
        })

    def wrapped(sparse_tensor, layer_id, layer_info):
        name, conv, _bn, _relu, _act_key = layer_info
        geom = conv_geometry(conv)
        geom['layer_id'] = layer_id
        geom['module_name'] = name
        geom['output_shape_zyx'] = None
        consumers.append(geom)
        if layer_id == 0:
            record(-1, 'input', 'InputSparseTensor', sparse_tensor)
        output = orig_layer(sparse_tensor, layer_id, layer_info)
        geom['output_shape_zyx'] = list(spatial_shape_zyx(output))
        record(layer_id, name, type(conv).__name__, output, geom=dict(geom))
        return output

    backbone._forward_hw_reference_layer = wrapped
    try:
        with torch.no_grad():
            _ = backbone(batch_after_vfe)
    finally:
        backbone._forward_hw_reference_layer = orig_layer
    return captured, consumers


def pages_for_nb(nb: int, page_voxels: int = PAGE_VOXELS) -> int:
    if nb <= 0:
        return 0
    return int((int(nb) + page_voxels - 1) // page_voxels)


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return int(value)
    return int((int(value) + alignment - 1) // alignment * alignment)


def aligned_page_bytes(channels: int) -> Tuple[int, int]:
    raw = PAGE_VOXELS * (COORD_BYTES + int(channels) * FEATURE_BYTES_PER_CHANNEL)
    return raw, align_up(raw, PAGE_ALIGN_BYTES)


def format_mib(num_bytes: int) -> str:
    return f'{num_bytes / (1024.0 * 1024.0):.4f}'


def format_pct(ratio: float) -> str:
    return f'{100.0 * ratio:.2f}%'


def _axis_copy_dir(coord: int, origin: int, log2_block: int, coord_max: int, neg: int, pos: int) -> int:
    if neg <= 0 and pos <= 0:
        return 0
    rel = int(coord) - int(origin)
    local = rel & ((1 << log2_block) - 1)
    block = 1 << log2_block
    if pos > 0 and local < pos:
        dest = int(coord) - local - 1
        if dest >= 0:
            return -1
    if neg > 0 and local >= block - neg:
        dest = int(coord) + (block - local)
        if dest <= coord_max:
            return 1
    return 0


def _cross_boundary_coord(coord: int, origin: int, log2_block: int, direction: int) -> int:
    if direction == 0:
        return int(coord)
    rel = int(coord) - int(origin)
    local = rel & ((1 << log2_block) - 1)
    block = 1 << log2_block
    if direction < 0:
        return int(coord) - local - 1
    return int(coord) + (block - local)


def iter_consumer_halo_block_keys(
    x_idx: int,
    y_idx: int,
    z_idx: int,
    grid_size: Tuple[int, int, int],
    zone_specs: Sequence,
    lidar_center_xy: Tuple[int, int],
    halo_xyz: HaloXYZ,
) -> Iterable[Tuple[int, int, int, int]]:
    nx, ny, nz = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
    cx, cy = int(lidar_center_xy[0]), int(lidar_center_xy[1])
    primary = _lookup_zone_spec(zone_specs, x_idx, y_idx, lidar_center_xy)
    yield _compute_block_key(x_idx, y_idx, z_idx, primary, lidar_center_xy)

    log2_bx, log2_by, log2_bz = primary.log2_block_size_xyz
    dx = _axis_copy_dir(x_idx, cx, log2_bx, nx - 1, halo_xyz[0][0], halo_xyz[0][1])
    dy = _axis_copy_dir(y_idx, cy, log2_by, ny - 1, halo_xyz[1][0], halo_xyz[1][1])
    dz = _axis_copy_dir(z_idx, 0, log2_bz, nz - 1, halo_xyz[2][0], halo_xyz[2][1])

    for halo_index in range(1, 8):
        if (halo_index & 0b001) and dx == 0:
            continue
        if (halo_index & 0b010) and dy == 0:
            continue
        if (halo_index & 0b100) and dz == 0:
            continue

        dest_x = _cross_boundary_coord(x_idx, cx, log2_bx, dx if (halo_index & 0b001) else 0)
        dest_y = _cross_boundary_coord(y_idx, cy, log2_by, dy if (halo_index & 0b010) else 0)
        dest_z = _cross_boundary_coord(z_idx, 0, log2_bz, dz if (halo_index & 0b100) else 0)
        if not (0 <= dest_x < nx and 0 <= dest_y < ny and 0 <= dest_z < nz):
            continue

        halo_spec = _lookup_zone_spec(zone_specs, dest_x, dest_y, lidar_center_xy)
        yield _compute_block_key(dest_x, dest_y, dest_z, halo_spec, lidar_center_xy)


def compute_consumer_halo_partition_counts(
    coords: np.ndarray,
    grid_size: Tuple[int, int, int],
    zone_specs: Sequence,
    lidar_center_xy: Tuple[int, int],
    halo_xyz: HaloXYZ,
):
    if coords is None or coords.size == 0:
        return np.zeros(0, dtype=np.int64), 0

    counts_by_key: Dict[Tuple[int, int, int, int], int] = {}
    for z_idx, y_idx, x_idx in coords.astype(np.int64):
        for block_key in iter_consumer_halo_block_keys(
            int(x_idx),
            int(y_idx),
            int(z_idx),
            grid_size,
            zone_specs,
            lidar_center_xy,
            halo_xyz,
        ):
            counts_by_key[block_key] = counts_by_key.get(block_key, 0) + 1

    ordered_keys = sorted(counts_by_key)
    counts = np.array([counts_by_key[key] for key in ordered_keys], dtype=np.int64)
    return counts, len(ordered_keys)


def histogram_from_counts(counts: np.ndarray, bin_edges: Sequence[Tuple[int, int]]) -> List[dict]:
    total = int(counts.size)
    rows = []
    for lo, hi in bin_edges:
        n_blocks = int(np.sum((counts >= lo) & (counts <= hi))) if total else 0
        rows.append({
            'bin_lo': lo,
            'bin_hi': hi,
            'bin_label': f'{lo}-{hi}',
            'n_blocks': n_blocks,
            'pct': float(n_blocks / total) if total else 0.0,
        })
    return rows


def analyze_tensor(tensor: dict, consumer: Optional[dict], bin_width: int) -> dict:
    shape_zyx = tuple(tensor['spatial_shape_zyx'])
    stage = stage_for_spatial(shape_zyx)
    grid_xyz = grid_xyz_from_zyx(shape_zyx)
    lidar_center = LIDAR_CENTER_BY_STAGE[stage]
    zone_specs = zone_specs_from_lut_lines(lut_lines_from_final(stage))
    halo_xyz = halo_xyz_from_consumer(consumer)
    counts, n_blocks = compute_consumer_halo_partition_counts(
        tensor['coords_zyx'],
        grid_xyz,
        zone_specs,
        lidar_center,
        halo_xyz,
    )
    nonempty_counts = counts[counts > 0] if counts.size else np.zeros(0, dtype=np.int64)
    max_nb = int(nonempty_counts.max()) if nonempty_counts.size else 0
    channels = int(tensor['feature_channels'])
    bytes_per_voxel = COORD_BYTES + channels * FEATURE_BYTES_PER_CHANNEL
    raw_page_bytes, page_bytes = aligned_page_bytes(channels)
    total_pages = int(sum(pages_for_nb(int(nb)) for nb in nonempty_counts)) if nonempty_counts.size else 0
    dram_bytes = total_pages * page_bytes
    unique_voxels = int(tensor['active_voxels'])
    sum_nb = int(nonempty_counts.sum()) if nonempty_counts.size else 0
    packed = unique_voxels * bytes_per_voxel
    packed_halo = sum_nb * bytes_per_voxel
    return {
        'tensor_name': tensor['layer_name'],
        'tensor_layer_id': tensor['layer_id'],
        'conv_type': tensor['conv_type'],
        'feature_channels': channels,
        'bytes_per_voxel': bytes_per_voxel,
        'raw_page_bytes': raw_page_bytes,
        'page_bytes': page_bytes,
        'pages': total_pages,
        'dram_bytes': dram_bytes,
        'packed_dram_bytes': packed,
        'occupancy': float(packed / dram_bytes) if dram_bytes else 0.0,
        'packed_halo_dram_bytes': packed_halo,
        'occupancy_halo': float(packed_halo / dram_bytes) if dram_bytes else 0.0,
        'spatial_shape_zyx': list(shape_zyx),
        'grid_size_xyz': list(grid_xyz),
        'stage': stage,
        'lidar_center_xy': list(lidar_center),
        'lut': summarize_zone_specs(zone_specs),
        'consumer_layer_id': None if consumer is None else consumer['layer_id'],
        'consumer_name': None if consumer is None else consumer['module_name'],
        'consumer_kernel_zyx': None if consumer is None else list(consumer['kernel_zyx']),
        'consumer_padding_zyx': None if consumer is None else list(consumer['padding_zyx']),
        'consumer_stride_zyx': None if consumer is None else list(consumer['stride_zyx']),
        'consumer_output_shape_zyx': None if consumer is None else consumer.get('output_shape_zyx'),
        'halo_xyz': [list(item) for item in halo_xyz],
        'halo_label': format_halo(halo_xyz),
        'nonempty_voxels': unique_voxels,
        'nonempty_blocks': int(n_blocks),
        'mean_nb': float(np.mean(nonempty_counts)) if nonempty_counts.size else 0.0,
        'median_nb': float(np.median(nonempty_counts)) if nonempty_counts.size else 0.0,
        'max_nb': max_nb,
        'sum_nb': sum_nb,
        'histogram': histogram_from_counts(nonempty_counts, make_bin_edges(max_nb, width=bin_width)),
    }


def build_layer_rows(tensors: Sequence[dict], consumers: Sequence[dict], bin_width: int) -> List[dict]:
    tensor_stats = []
    for index, tensor in enumerate(tensors):
        consumer = consumers[index] if index < len(consumers) else None
        tensor_stats.append(analyze_tensor(tensor, consumer, bin_width))

    rows = []
    for layer_id, consumer in enumerate(consumers):
        ifm = tensor_stats[layer_id]
        ofm = tensor_stats[layer_id + 1]
        rows.append({
            'layer_id': layer_id,
            'layer_name': consumer['module_name'],
            'conv_type': consumer['conv_type'],
            'kernel_zyx': list(consumer['kernel_zyx']),
            'padding_zyx': list(consumer['padding_zyx']),
            'stride_zyx': list(consumer['stride_zyx']),
            'output_shape_zyx': list(consumer['output_shape_zyx']),
            'cin': consumer['cin'],
            'cout': consumer['cout'],
            'ifm': ifm,
            'ofm': ofm,
            'peak_dram_bytes': int(ifm['dram_bytes'] + ofm['dram_bytes']),
        })
    return rows


def md_escape(text) -> str:
    return str(text).replace('|', '\\|')


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    header_line = '| ' + ' | '.join(md_escape(h) for h in headers) + ' |'
    sep_line = '| ' + ' | '.join('---' for _ in headers) + ' |'
    body = ['| ' + ' | '.join(md_escape(cell) for cell in row) + ' |' for row in rows]
    return '\n'.join([header_line, sep_line, *body])


def tensor_stat_cells(stats: dict) -> List[object]:
    return [
        stats['feature_channels'],
        stats['halo_label'],
        stats['nonempty_voxels'],
        stats['nonempty_blocks'],
        stats['sum_nb'],
        stats['pages'],
        stats['raw_page_bytes'],
        stats['page_bytes'],
        stats['dram_bytes'],
        format_mib(stats['dram_bytes']),
        stats['packed_dram_bytes'],
        format_pct(float(stats['occupancy'])),
        stats['packed_halo_dram_bytes'],
        format_pct(float(stats['occupancy_halo'])),
    ]


def tensor_table_headers(prefix: str) -> List[str]:
    return [
        f'{prefix} C',
        f'{prefix} halo',
        f'{prefix} voxels (不含halo)',
        f'{prefix} blocks (含halo)',
        f'{prefix} Sum N_b',
        f'{prefix} pages',
        f'{prefix} raw page Byte',
        f'{prefix} aligned page Byte',
        f'{prefix} DRAM Byte',
        f'{prefix} DRAM MiB',
        f'{prefix} 紧凑 (无halo) Byte',
        f'{prefix} 占用率 (无halo)',
        f'{prefix} 紧凑 (含halo) Byte',
        f'{prefix} 占用率 (含halo)',
    ]


def build_markdown(payload: dict) -> str:
    frame = payload['frame']
    layers = payload['layers']
    peak_bytes = int(payload.get('peak_ifm_ofm_dram_bytes', 0))
    peak_layer = payload.get('peak_ifm_ofm_layer_name', '')

    lines = [
        '# Proposed Page Allocation: INT8 SECOND 3D Backbone Block Occupancy',
        '',
        f"- Frame: KITTI `{frame['split']}/{frame['frame_id']}`",
        f"- Point cloud: `{frame['point_cloud_path']}`",
        f"- Config: `{payload['cfg']}`",
        f"- Checkpoint: `{payload['ckpt']}`",
        f"- Mode: hardware-reference INT8 (`{payload['mode']}`)",
        f"- Device: `{payload['device']}`",
        f"- Data loader: `{frame['data_loader']}` (FOV_POINTS_ONLY={frame['fov_points_only']})",
        f"- Generated: `{payload['generated_at']}`",
        '',
        '## DRAM / halo rules',
        '',
        '- Halo is generated from the **consumer (next) layer** kernel / padding / output shape.',
        '- 3-tap axis: `neg=padding`, `pos=2-padding`. 1-tap axis: no halo.',
        '- Last OFM (`conv_out`) has no 3D consumer, so OFM halo is none.',
        f'- Coordinate `{COORD_BYTES}` Byte, feature `{FEATURE_BYTES_PER_CHANNEL}` Byte/channel, voxel `{COORD_BYTES}+C` Byte.',
        f'- Pages per block: `ceil(N_b / {PAGE_VOXELS})`; raw page `{PAGE_VOXELS}*(8+C)`, then **align up to {PAGE_ALIGN_BYTES} Byte**.',
        '- Layer peak DRAM = IFM allocated pages + OFM allocated pages (coresident).',
        f'- Pipeline peak: `{peak_bytes}` Byte = `{format_mib(peak_bytes)}` MiB at `{peak_layer}`',
        '',
        '## Proposed LUT',
        '',
        markdown_table(
            ['Stage', 'LiDAR ref (x,y)', 'LUT'],
            [
                [stage, f'({LIDAR_CENTER_BY_STAGE[stage][0]},{LIDAR_CENTER_BY_STAGE[stage][1]})', lut_label(stage)]
                for stage in range(4)
            ],
        ),
        '',
        '## Per-layer IFM / OFM peak DRAM',
        '',
        markdown_table(
            [
                'Layer', 'Name', 'Type',
                'Kernel ZYX', 'Pad ZYX', 'Stride ZYX', 'Output ZYX',
                'IFM C', 'OFM C',
                'IFM halo', 'OFM halo',
                'IFM DRAM Byte', 'IFM DRAM MiB',
                'OFM DRAM Byte', 'OFM DRAM MiB',
                'Peak IFM+OFM Byte', 'Peak MiB',
                'IFM 占用率(含halo)', 'OFM 占用率(含halo)',
            ],
            [
                [
                    row['layer_id'],
                    row['layer_name'],
                    row['conv_type'],
                    'x'.join(str(v) for v in row['kernel_zyx']),
                    'x'.join(str(v) for v in row['padding_zyx']),
                    'x'.join(str(v) for v in row['stride_zyx']),
                    'x'.join(str(v) for v in row['output_shape_zyx']),
                    row['ifm']['feature_channels'],
                    row['ofm']['feature_channels'],
                    row['ifm']['halo_label'],
                    row['ofm']['halo_label'],
                    row['ifm']['dram_bytes'],
                    format_mib(row['ifm']['dram_bytes']),
                    row['ofm']['dram_bytes'],
                    format_mib(row['ofm']['dram_bytes']),
                    row['peak_dram_bytes'],
                    format_mib(row['peak_dram_bytes']),
                    format_pct(float(row['ifm']['occupancy_halo'])),
                    format_pct(float(row['ofm']['occupancy_halo'])),
                ]
                for row in layers
            ],
        ),
        '',
        '## Per-layer IFM DRAM',
        '',
        markdown_table(
            ['Layer', 'Name'] + tensor_table_headers('IFM'),
            [[row['layer_id'], row['layer_name'], *tensor_stat_cells(row['ifm'])] for row in layers],
        ),
        '',
        '## Per-layer OFM DRAM',
        '',
        markdown_table(
            ['Layer', 'Name'] + tensor_table_headers('OFM'),
            [[row['layer_id'], row['layer_name'], *tensor_stat_cells(row['ofm'])] for row in layers],
        ),
        '',
        '## IFM / OFM histogram detail',
        '',
    ]

    for row in layers:
        lines.append(f"### Layer {row['layer_id']}: `{row['layer_name']}`")
        lines.append('')
        lines.append(
            f"- Kernel ZYX `{ 'x'.join(str(v) for v in row['kernel_zyx']) }`, "
            f"pad `{ 'x'.join(str(v) for v in row['padding_zyx']) }`, "
            f"stride `{ 'x'.join(str(v) for v in row['stride_zyx']) }`, "
            f"output `{ 'x'.join(str(v) for v in row['output_shape_zyx']) }`"
        )
        lines.append(
            f"- Peak `{row['peak_dram_bytes']}` Byte = `{format_mib(row['peak_dram_bytes'])}` MiB "
            f"(IFM `{format_mib(row['ifm']['dram_bytes'])}` + OFM `{format_mib(row['ofm']['dram_bytes'])}`)"
        )
        for kind, stats in (('IFM', row['ifm']), ('OFM', row['ofm'])):
            lines.append('')
            lines.append(
                f"- **{kind}** `{stats['tensor_name']}` C=`{stats['feature_channels']}` "
                f"halo `{stats['halo_label']}` consumer `{stats['consumer_name']}`"
            )
            lines.append(
                f"  voxels `{stats['nonempty_voxels']}` (no halo), blocks `{stats['nonempty_blocks']}`, "
                f"Sum N_b `{stats['sum_nb']}`, pages `{stats['pages']}`, "
                f"page `{stats['raw_page_bytes']}→{stats['page_bytes']}` Byte, "
                f"DRAM `{stats['dram_bytes']}` Byte (`{format_mib(stats['dram_bytes'])}` MiB), "
                f"occ no-halo `{format_pct(float(stats['occupancy']))}`, "
                f"occ halo `{format_pct(float(stats['occupancy_halo']))}`"
            )
            detail_rows = [
                [item['bin_label'], item['n_blocks'], f"{100.0 * item['pct']:.2f}%"]
                for item in stats['histogram']
                if item['n_blocks'] > 0
            ]
            if detail_rows:
                lines.append('')
                lines.append(markdown_table([f'{kind} N_b bin', 'Blocks', 'Share'], detail_rows))
        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


def flatten_layer_csv_row(row: dict) -> dict:
    out = {
        'layer_id': row['layer_id'],
        'layer_name': row['layer_name'],
        'conv_type': row['conv_type'],
        'kernel_zyx': 'x'.join(str(v) for v in row['kernel_zyx']),
        'padding_zyx': 'x'.join(str(v) for v in row['padding_zyx']),
        'stride_zyx': 'x'.join(str(v) for v in row['stride_zyx']),
        'output_shape_zyx': 'x'.join(str(v) for v in row['output_shape_zyx']),
        'peak_dram_bytes': row['peak_dram_bytes'],
        'peak_dram_mib': format_mib(row['peak_dram_bytes']),
    }
    for prefix, stats in (('ifm', row['ifm']), ('ofm', row['ofm'])):
        out.update({
            f'{prefix}_c': stats['feature_channels'],
            f'{prefix}_halo': stats['halo_label'],
            f'{prefix}_voxels': stats['nonempty_voxels'],
            f'{prefix}_blocks': stats['nonempty_blocks'],
            f'{prefix}_sum_nb': stats['sum_nb'],
            f'{prefix}_pages': stats['pages'],
            f'{prefix}_raw_page_bytes': stats['raw_page_bytes'],
            f'{prefix}_page_bytes': stats['page_bytes'],
            f'{prefix}_dram_bytes': stats['dram_bytes'],
            f'{prefix}_dram_mib': format_mib(stats['dram_bytes']),
            f'{prefix}_packed_bytes': stats['packed_dram_bytes'],
            f'{prefix}_occupancy': f"{float(stats['occupancy']):.6f}",
            f'{prefix}_packed_halo_bytes': stats['packed_halo_dram_bytes'],
            f'{prefix}_occupancy_halo': f"{float(stats['occupancy_halo']):.6f}",
        })
    return out


def write_layer_csv(path: Path, layers: Sequence[dict]) -> None:
    rows = [flatten_layer_csv_row(row) for row in layers]
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready_layers(layers: Sequence[dict]) -> List[dict]:
    return [json.loads(json.dumps(row)) for row in layers]


def main():
    args = parse_args()
    project_root = _PROJECT_ROOT
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.utils import common_utils

    cfg_path = project_root / args.cfg
    ckpt_path = project_root / args.ckpt
    if not cfg_path.exists():
        raise FileNotFoundError(f'Config file not found: {cfg_path}')
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    original_cwd = Path.cwd()
    try:
        os.chdir(project_root / 'tools')
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(original_cwd)

    device = resolve_device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    logger = common_utils.create_logger()
    data_mode = resolve_data_mode(cfg, args.data_mode)
    if data_mode != 'kitti':
        raise RuntimeError('This script expects KITTI FOV loading so it matches the golden 000216 path')

    dataset = build_kitti_dataset(cfg, project_root, args.kitti_root, logger)
    frame_ids = choose_kitti_frame_ids(dataset, args.frame_id, seed=0, num_frames=1)
    frame_id = frame_ids[0]

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=(device.type == 'cpu'))
    model.to(device)
    model.eval()
    enable_hw_reference(model, project_root, out_dir, args.weight_quant, logger)

    sample, frame_info = load_kitti_sample(dataset, frame_id)
    batch = dataset.collate_batch([sample])
    batch_torch = move_batch_to_device(batch, device)
    with torch.no_grad():
        batch_after_vfe = model.vfe(batch_torch)

    tensors, consumers = capture_layer_outputs(model.backbone_3d, batch_after_vfe)
    layer_rows = build_layer_rows(tensors, consumers, args.bin_width)
    peak_row = max(layer_rows, key=lambda row: int(row['peak_dram_bytes']))

    payload = {
        'cfg': str(cfg_path),
        'ckpt': str(ckpt_path),
        'device': str(device),
        'weight_quant': args.weight_quant,
        'mode': 'hw_reference_int8',
        'data_mode': data_mode,
        'halo_mode': 'consumer_kernel_padding_asymmetric',
        'bin_width': args.bin_width,
        'coord_bytes': COORD_BYTES,
        'feature_bytes_per_channel': FEATURE_BYTES_PER_CHANNEL,
        'page_voxels': PAGE_VOXELS,
        'page_align_bytes': PAGE_ALIGN_BYTES,
        'peak_ifm_ofm_dram_bytes': int(peak_row['peak_dram_bytes']),
        'peak_ifm_ofm_layer_id': peak_row['layer_id'],
        'peak_ifm_ofm_layer_name': peak_row['layer_name'],
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'frame': {
            'split': 'val',
            'frame_id': frame_id,
            'point_cloud_path': frame_info['point_cloud_path'],
            'data_loader': frame_info['data_loader'],
            'fov_points_only': frame_info['fov_points_only'],
        },
        'layers': json_ready_layers(layer_rows),
    }

    stem = f'{frame_id}_int8_layer_block_histogram'
    md_path = out_dir / f'{stem}.md'
    json_path = out_dir / f'{stem}.json'
    csv_path = out_dir / f'{stem}.csv'

    markdown = build_markdown({**payload, 'layers': layer_rows})
    md_path.write_text(markdown, encoding='utf-8')
    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    write_layer_csv(csv_path, layer_rows)

    print(markdown)
    print(f'Saved markdown: {md_path}')
    print(f'Saved JSON:     {json_path}')
    print(f'Saved CSV:      {csv_path}')


if __name__ == '__main__':
    main()

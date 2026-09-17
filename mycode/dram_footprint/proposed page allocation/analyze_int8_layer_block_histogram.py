#!/usr/bin/env python3
"""INT8 SECOND 3D-backbone occupancy under the proposed zone/block LUT.

Halo extents come from the consumer (next) layer:
  low_extent  = kernel_size - 1 - padding
  high_extent = padding
1-tap axes emit no halo. Window-corner mapping re-runs zone lookup at each
corner. Pages pack 64 voxels; page_bytes = (1 + ceil(C/8)) * 512.
Per-layer IFM and OFM are counted separately; conv_out OFM is not paged.
Peak DRAM is coresident IFM+allocated-OFM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

# 000216 allocated states INIT..L10 (L11 OFM is not paged).
EXPECTED_STATE_PAGES = [1150, 1128, 1150, 1556, 1556, 1781, 1349, 1349, 1320, 488, 488, 365]
EXPECTED_STATE_SUM_NB = [21358, 21002, 21358, 44913, 44913, 50785, 40485, 40485, 40175, 16591, 16591, 8495]
EXPECTED_PEAK_LAYER_ID = 6
EXPECTED_PEAK_PAGES = 1349 + 1349
EXPECTED_PEAK_BYTES = 12432384

ExtentXYZ = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
XYKey = Tuple[int, int, int]
BlockKey = Tuple[int, int, int, int]


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


def extent_one(kernel: int, padding: int) -> Tuple[int, int]:
    """Return (low_extent, high_extent) along one axis."""
    kernel = int(kernel)
    padding = int(padding)
    if kernel <= 1:
        return (0, 0)
    return (kernel - 1 - padding, padding)


def extent_xyz_from_consumer(consumer: Optional[dict]) -> ExtentXYZ:
    if consumer is None:
        return ((0, 0), (0, 0), (0, 0))
    kz, ky, kx = consumer['kernel_zyx']
    pz, py, px = consumer['padding_zyx']
    return (
        extent_one(kx, px),
        extent_one(ky, py),
        extent_one(kz, pz),
    )


def format_extent(extent_xyz: ExtentXYZ) -> str:
    axes = ('x', 'y', 'z')
    parts = []
    for axis, (low, high) in zip(axes, extent_xyz):
        if low == 0 and high == 0:
            parts.append(f'{axis}:none')
            continue
        low_s = f'-{low}' if low else '0'
        high_s = f'+{high}' if high else '0'
        parts.append(f'{axis}:[{low_s},{high_s}]')
    return ' '.join(parts)


def coordinate_identity(coords_zyx: np.ndarray) -> dict:
    if coords_zyx is None or np.asarray(coords_zyx).size == 0:
        payload = np.zeros((0, 3), dtype='<i4')
        return {
            'coordinate_count': 0,
            'coordinate_sha256': hashlib.sha256(payload.tobytes()).hexdigest(),
            'coordinate_unique': True,
        }
    arr = np.ascontiguousarray(np.asarray(coords_zyx)[:, :3], dtype='<i4')
    unique = np.unique(arr, axis=0)
    unique = np.ascontiguousarray(unique, dtype='<i4')
    return {
        'coordinate_count': int(unique.shape[0]),
        'coordinate_sha256': hashlib.sha256(unique.tobytes()).hexdigest(),
        'coordinate_unique': bool(unique.shape[0] == arr.shape[0]),
    }


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


def page_layout(channels: int) -> Tuple[int, int, int]:
    feature_words = (int(channels) + 7) // 8
    page_stride = 1 + feature_words
    page_bytes = page_stride * PAGE_ALIGN_BYTES
    return feature_words, page_stride, page_bytes


def format_mib(num_bytes: int) -> str:
    return f'{num_bytes / (1024.0 * 1024.0):.4f}'


def format_pct(ratio: float) -> str:
    return f'{100.0 * ratio:.2f}%'


def _map_xy(
    x_idx: int,
    y_idx: int,
    zone_specs: Sequence,
    lidar_center_xy: Tuple[int, int],
) -> Tuple[object, XYKey]:
    spec = _lookup_zone_spec(zone_specs, x_idx, y_idx, lidar_center_xy)
    zone_id, block_x, block_y, _block_z = _compute_block_key(
        x_idx, y_idx, 0, spec, lidar_center_xy
    )
    return spec, (int(zone_id), int(block_x), int(block_y))


def _window_bounds(
    coord: int,
    low_extent: int,
    high_extent: int,
    coord_max: int,
) -> Tuple[int, int]:
    lo = max(0, int(coord) - int(low_extent))
    hi = min(int(coord_max), int(coord) + int(high_extent))
    return lo, hi


def iter_window_corner_block_keys(
    x_idx: int,
    y_idx: int,
    z_idx: int,
    grid_size: Tuple[int, int, int],
    zone_specs: Sequence,
    lidar_center_xy: Tuple[int, int],
    extent_xyz: ExtentXYZ,
) -> Iterable[BlockKey]:
    nx, ny, nz = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
    (low_x, high_x), (low_y, high_y), (low_z, high_z) = extent_xyz
    x_lo, x_hi = _window_bounds(x_idx, low_x, high_x, nx - 1)
    y_lo, y_hi = _window_bounds(y_idx, low_y, high_y, ny - 1)
    z_lo, z_hi = _window_bounds(z_idx, low_z, high_z, nz - 1)

    primary_spec, primary_xy = _map_xy(x_idx, y_idx, zone_specs, lidar_center_xy)
    xy_entries = [(primary_spec, primary_xy)]
    seen_xy = {primary_xy}
    for px, py in ((x_lo, y_lo), (x_lo, y_hi), (x_hi, y_lo), (x_hi, y_hi)):
        spec, xy_key = _map_xy(px, py, zone_specs, lidar_center_xy)
        if xy_key in seen_xy:
            continue
        seen_xy.add(xy_key)
        xy_entries.append((spec, xy_key))

    seen_block_keys = set()
    for spec, xy_key in xy_entries:
        log2_bz = spec.log2_block_size_xyz[2]
        primary_z = int(z_idx) >> log2_bz
        low_bz = int(z_lo) >> log2_bz
        high_bz = int(z_hi) >> log2_bz
        z_indices = [primary_z]
        if low_bz != primary_z:
            z_indices.append(low_bz)
        elif high_bz != primary_z:
            z_indices.append(high_bz)
        for block_z in z_indices:
            key = (xy_key[0], xy_key[1], xy_key[2], int(block_z))
            if key in seen_block_keys:
                continue
            seen_block_keys.add(key)
            yield key


def compute_window_corner_partition_counts(
    coords: np.ndarray,
    grid_size: Tuple[int, int, int],
    zone_specs: Sequence,
    lidar_center_xy: Tuple[int, int],
    extent_xyz: ExtentXYZ,
):
    if coords is None or coords.size == 0:
        return np.zeros(0, dtype=np.int64), 0

    counts_by_key: Dict[BlockKey, int] = {}
    for z_idx, y_idx, x_idx in coords.astype(np.int64):
        for block_key in iter_window_corner_block_keys(
            int(x_idx),
            int(y_idx),
            int(z_idx),
            grid_size,
            zone_specs,
            lidar_center_xy,
            extent_xyz,
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


def analyze_tensor(
    tensor: dict,
    consumer: Optional[dict],
    bin_width: int,
    allocate_pages: bool = True,
) -> dict:
    shape_zyx = tuple(tensor['spatial_shape_zyx'])
    stage = stage_for_spatial(shape_zyx)
    grid_xyz = grid_xyz_from_zyx(shape_zyx)
    lidar_center = LIDAR_CENTER_BY_STAGE[stage]
    zone_specs = zone_specs_from_lut_lines(lut_lines_from_final(stage))
    extent_xyz = extent_xyz_from_consumer(consumer)
    identity = coordinate_identity(tensor['coords_zyx'])
    channels = int(tensor['feature_channels'])
    feature_words, page_stride, page_bytes = page_layout(channels)
    unique_voxels = int(tensor['active_voxels'])
    semantic_channel_bytes = unique_voxels * channels * FEATURE_BYTES_PER_CHANNEL

    if allocate_pages:
        counts, n_blocks = compute_window_corner_partition_counts(
            tensor['coords_zyx'],
            grid_xyz,
            zone_specs,
            lidar_center,
            extent_xyz,
        )
        nonempty_counts = counts[counts > 0] if counts.size else np.zeros(0, dtype=np.int64)
        max_nb = int(nonempty_counts.max()) if nonempty_counts.size else 0
        total_pages = int(sum(pages_for_nb(int(nb)) for nb in nonempty_counts)) if nonempty_counts.size else 0
        sum_nb = int(nonempty_counts.sum()) if nonempty_counts.size else 0
        dram_bytes = total_pages * page_bytes
        useful_bytes = sum_nb * page_stride * COORD_BYTES
        occupancy = float(sum_nb / (total_pages * PAGE_VOXELS)) if total_pages else 0.0
        histogram = histogram_from_counts(nonempty_counts, make_bin_edges(max_nb, width=bin_width))
        mean_nb = float(np.mean(nonempty_counts)) if nonempty_counts.size else 0.0
        median_nb = float(np.median(nonempty_counts)) if nonempty_counts.size else 0.0
    else:
        n_blocks = 0
        max_nb = 0
        total_pages = 0
        sum_nb = 0
        dram_bytes = 0
        useful_bytes = 0
        occupancy = 0.0
        histogram = []
        mean_nb = 0.0
        median_nb = 0.0

    return {
        'tensor_name': tensor['layer_name'],
        'tensor_layer_id': tensor['layer_id'],
        'conv_type': tensor['conv_type'],
        'allocated': bool(allocate_pages),
        'feature_channels': channels,
        'feature_words': feature_words,
        'page_stride': page_stride,
        'page_bytes': page_bytes,
        'pages': total_pages,
        'dram_bytes': dram_bytes,
        'useful_bytes': useful_bytes,
        'occupancy': occupancy,
        'semantic_channel_bytes': semantic_channel_bytes,
        'semantic_ratio': float(semantic_channel_bytes / dram_bytes) if dram_bytes else 0.0,
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
        'low_high_extent_xyz': [list(item) for item in extent_xyz],
        'extent_label': format_extent(extent_xyz),
        'nonempty_voxels': unique_voxels,
        'coordinate_count': identity['coordinate_count'],
        'coordinate_sha256': identity['coordinate_sha256'],
        'coordinate_unique': identity['coordinate_unique'],
        'nonempty_blocks': int(n_blocks),
        'mean_nb': mean_nb,
        'median_nb': median_nb,
        'max_nb': max_nb,
        'sum_nb': sum_nb,
        'histogram': histogram,
    }


def build_layer_rows(tensors: Sequence[dict], consumers: Sequence[dict], bin_width: int) -> Tuple[List[dict], List[dict]]:
    tensor_stats = []
    last_tensor_index = len(tensors) - 1
    for index, tensor in enumerate(tensors):
        consumer = consumers[index] if index < len(consumers) else None
        allocate = index != last_tensor_index
        tensor_stats.append(analyze_tensor(tensor, consumer, bin_width, allocate_pages=allocate))

    rows = []
    for layer_id, consumer in enumerate(consumers):
        ifm = tensor_stats[layer_id]
        ofm = tensor_stats[layer_id + 1]
        ofm_bytes = int(ofm['dram_bytes']) if ofm['allocated'] else 0
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
            'peak_dram_bytes': int(ifm['dram_bytes'] + ofm_bytes),
            'peak_pages': int(ifm['pages'] + (ofm['pages'] if ofm['allocated'] else 0)),
        })
    return rows, tensor_stats


def state_label(tensor_layer_id: int) -> str:
    if tensor_layer_id < 0:
        return 'INIT'
    return f'L{tensor_layer_id}'


def build_acceptance(tensor_stats: Sequence[dict], layer_rows: Sequence[dict]) -> dict:
    allocated_states = [stats for stats in tensor_stats if stats['allocated']]
    page_rows = []
    sum_rows = []
    for index, stats in enumerate(allocated_states):
        expected_pages = EXPECTED_STATE_PAGES[index] if index < len(EXPECTED_STATE_PAGES) else None
        expected_sum = EXPECTED_STATE_SUM_NB[index] if index < len(EXPECTED_STATE_SUM_NB) else None
        page_rows.append({
            'state': state_label(stats['tensor_layer_id']),
            'pages': stats['pages'],
            'expected_pages': expected_pages,
            'match': expected_pages is not None and stats['pages'] == expected_pages,
        })
        sum_rows.append({
            'state': state_label(stats['tensor_layer_id']),
            'sum_nb': stats['sum_nb'],
            'expected_sum_nb': expected_sum,
            'match': expected_sum is not None and stats['sum_nb'] == expected_sum,
        })

    last_ofm = tensor_stats[-1]
    peak_row = max(layer_rows, key=lambda row: int(row['peak_dram_bytes']))
    peak_ok = (
        peak_row['layer_id'] == EXPECTED_PEAK_LAYER_ID
        and peak_row['peak_pages'] == EXPECTED_PEAK_PAGES
        and peak_row['peak_dram_bytes'] == EXPECTED_PEAK_BYTES
        and last_ofm['pages'] == 0
        and not last_ofm['allocated']
    )
    pages_ok = all(item['match'] for item in page_rows)
    sum_ok = all(item['match'] for item in sum_rows)
    return {
        'pages_match': pages_ok,
        'sum_nb_match': sum_ok,
        'peak_match': peak_ok,
        'passed': pages_ok and sum_ok and peak_ok,
        'last_ofm_allocated': last_ofm['allocated'],
        'last_ofm_pages': last_ofm['pages'],
        'peak_layer_id': peak_row['layer_id'],
        'peak_layer_name': peak_row['layer_name'],
        'peak_pages': peak_row['peak_pages'],
        'peak_bytes': peak_row['peak_dram_bytes'],
        'states': page_rows,
        'copies': sum_rows,
    }


def md_escape(text) -> str:
    return str(text).replace('|', '\\|')


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    header_line = '| ' + ' | '.join(md_escape(h) for h in headers) + ' |'
    sep_line = '| ' + ' | '.join('---' for _ in headers) + ' |'
    body = ['| ' + ' | '.join(md_escape(cell) for cell in row) + ' |' for row in rows]
    return '\n'.join([header_line, sep_line, *body])


def occupancy_cell(stats: dict) -> str:
    if not stats['allocated'] or stats['pages'] == 0:
        return 'n/a'
    return format_pct(float(stats['occupancy']))


def semantic_cell(stats: dict) -> str:
    if not stats['allocated'] or stats['dram_bytes'] == 0:
        return 'n/a'
    return format_pct(float(stats['semantic_ratio']))


def tensor_stat_cells(stats: dict) -> List[object]:
    return [
        stats['feature_channels'],
        stats['extent_label'],
        stats['nonempty_voxels'],
        stats['nonempty_blocks'] if stats['allocated'] else 0,
        stats['sum_nb'] if stats['allocated'] else 0,
        stats['pages'],
        stats['feature_words'],
        stats['page_stride'],
        stats['page_bytes'],
        stats['dram_bytes'],
        format_mib(stats['dram_bytes']),
        stats['useful_bytes'],
        occupancy_cell(stats),
        stats['semantic_channel_bytes'],
        semantic_cell(stats),
    ]


def tensor_table_headers(prefix: str) -> List[str]:
    return [
        f'{prefix} C',
        f'{prefix} extent',
        f'{prefix} voxels (不含halo)',
        f'{prefix} blocks (含halo)',
        f'{prefix} Sum N_b',
        f'{prefix} pages',
        f'{prefix} feature_words',
        f'{prefix} page_stride',
        f'{prefix} page Byte',
        f'{prefix} DRAM Byte',
        f'{prefix} DRAM MiB',
        f'{prefix} useful Byte',
        f'{prefix} 占用率 (page slot)',
        f'{prefix} 有效通道 Byte',
        f'{prefix} 语义压缩率',
    ]


def build_markdown(payload: dict) -> str:
    frame = payload['frame']
    layers = payload['layers']
    peak_bytes = int(payload.get('peak_ifm_ofm_dram_bytes', 0))
    peak_layer = payload.get('peak_ifm_ofm_layer_name', '')
    peak_pages = int(payload.get('peak_ifm_ofm_pages', 0))
    acceptance = payload.get('acceptance', {})

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
        '## DRAM / extent rules',
        '',
        '- Extent is generated from the **consumer (next) layer** kernel / padding / output shape.',
        '- `low_extent = kernel_size - 1 - padding`, `high_extent = padding`. Displayed as `[-low,+high]`.',
        '- 1-tap axis: no halo (`none`). Last OFM (`conv_out`) is a logical network output and is **not paged**.',
        '- Window-corner mapping: primary XY plus the four window corners, each with a fresh zone lookup.',
        '- Alternate Z prefers `block_z(z_lo)` if it differs from primary Z, otherwise a different `block_z(z_hi)`.',
        '- Per-owner `seen_block_keys` ensures each full block key is counted at most once.',
        f'- `feature_words = ceil(C / 8)`, `page_stride = 1 + feature_words`, `page_bytes = page_stride × {PAGE_ALIGN_BYTES}`.',
        f'- Pages per block: `ceil(N_b / {PAGE_VOXELS})`. Physical occupancy: `Sum_N_b / (pages × {PAGE_VOXELS})`.',
        '- Useful bytes: `Sum_N_b × page_stride × 8`. Effective-channel bytes (`voxels × C`) are a semantic ratio, not physical occupancy.',
        '- Layer peak DRAM = IFM allocated pages + OFM allocated pages (coresident). L11 peak = IFM only.',
        f'- Pipeline peak: `{peak_pages}` pages, `{peak_bytes}` Byte = `{format_mib(peak_bytes)}` MiB at `{peak_layer}`',
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
        '## Static acceptance (000216)',
        '',
        f"- Pages match: `{acceptance.get('pages_match')}`",
        f"- Halo copies match: `{acceptance.get('sum_nb_match')}`",
        f"- Peak match: `{acceptance.get('peak_match')}` (expect L6 `{EXPECTED_PEAK_PAGES}` pages / `{EXPECTED_PEAK_BYTES}` Byte)",
        f"- L11 OFM pages: `{acceptance.get('last_ofm_pages')}` (must be 0)",
        f"- Overall: **{'PASS' if acceptance.get('passed') else 'FAIL'}**",
        '',
        markdown_table(
            ['State', 'Pages', 'Expected pages', 'Sum N_b', 'Expected Sum N_b'],
            [
                [
                    page_row['state'],
                    page_row['pages'],
                    page_row['expected_pages'],
                    copy_row['sum_nb'],
                    copy_row['expected_sum_nb'],
                ]
                for page_row, copy_row in zip(acceptance.get('states', []), acceptance.get('copies', []))
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
                'IFM extent', 'OFM extent',
                'IFM pages', 'OFM pages',
                'IFM DRAM Byte', 'IFM DRAM MiB',
                'OFM DRAM Byte', 'OFM DRAM MiB',
                'Peak pages', 'Peak Byte', 'Peak MiB',
                'IFM 占用率(page slot)', 'OFM 占用率(page slot)',
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
                    row['ifm']['extent_label'],
                    row['ofm']['extent_label'] if row['ofm']['allocated'] else 'not paged',
                    row['ifm']['pages'],
                    row['ofm']['pages'],
                    row['ifm']['dram_bytes'],
                    format_mib(row['ifm']['dram_bytes']),
                    row['ofm']['dram_bytes'],
                    format_mib(row['ofm']['dram_bytes']),
                    row['peak_pages'],
                    row['peak_dram_bytes'],
                    format_mib(row['peak_dram_bytes']),
                    occupancy_cell(row['ifm']),
                    occupancy_cell(row['ofm']),
                ]
                for row in layers
            ],
        ),
        '',
        '## Coordinate identity',
        '',
        markdown_table(
            ['Layer', 'Name', 'Tensor', 'Count', 'Unique', 'SHA-256'],
            [
                cell
                for row in layers
                for cell in (
                    [
                        row['layer_id'],
                        row['layer_name'],
                        'IFM',
                        row['ifm']['coordinate_count'],
                        row['ifm']['coordinate_unique'],
                        row['ifm']['coordinate_sha256'],
                    ],
                    [
                        row['layer_id'],
                        row['layer_name'],
                        'OFM',
                        row['ofm']['coordinate_count'],
                        row['ofm']['coordinate_unique'],
                        row['ofm']['coordinate_sha256'],
                    ],
                )
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
            f"- Peak `{row['peak_pages']}` pages / `{row['peak_dram_bytes']}` Byte = `{format_mib(row['peak_dram_bytes'])}` MiB "
            f"(IFM `{format_mib(row['ifm']['dram_bytes'])}` + OFM `{format_mib(row['ofm']['dram_bytes'])}`)"
        )
        for kind, stats in (('IFM', row['ifm']), ('OFM', row['ofm'])):
            lines.append('')
            if not stats['allocated']:
                lines.append(
                    f"- **{kind}** `{stats['tensor_name']}` C=`{stats['feature_channels']}` "
                    f"logical output only: voxels `{stats['nonempty_voxels']}`, "
                    f"sha256 `{stats['coordinate_sha256']}`, allocated pages `0`"
                )
                continue
            lines.append(
                f"- **{kind}** `{stats['tensor_name']}` C=`{stats['feature_channels']}` "
                f"extent `{stats['extent_label']}` consumer `{stats['consumer_name']}`"
            )
            lines.append(
                f"  voxels `{stats['nonempty_voxels']}` (no halo), sha256 `{stats['coordinate_sha256']}`, "
                f"blocks `{stats['nonempty_blocks']}`, Sum N_b `{stats['sum_nb']}`, pages `{stats['pages']}`, "
                f"page_stride `{stats['page_stride']}`, page `{stats['page_bytes']}` Byte, "
                f"DRAM `{stats['dram_bytes']}` Byte (`{format_mib(stats['dram_bytes'])}` MiB), "
                f"useful `{stats['useful_bytes']}` Byte, occ `{occupancy_cell(stats)}`, "
                f"semantic `{semantic_cell(stats)}`"
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
        'peak_pages': row['peak_pages'],
        'peak_dram_bytes': row['peak_dram_bytes'],
        'peak_dram_mib': format_mib(row['peak_dram_bytes']),
    }
    for prefix, stats in (('ifm', row['ifm']), ('ofm', row['ofm'])):
        out.update({
            f'{prefix}_c': stats['feature_channels'],
            f'{prefix}_extent': stats['extent_label'],
            f'{prefix}_allocated': int(stats['allocated']),
            f'{prefix}_voxels': stats['nonempty_voxels'],
            f'{prefix}_coordinate_count': stats['coordinate_count'],
            f'{prefix}_coordinate_sha256': stats['coordinate_sha256'],
            f'{prefix}_coordinate_unique': int(stats['coordinate_unique']),
            f'{prefix}_blocks': stats['nonempty_blocks'],
            f'{prefix}_sum_nb': stats['sum_nb'],
            f'{prefix}_pages': stats['pages'],
            f'{prefix}_feature_words': stats['feature_words'],
            f'{prefix}_page_stride': stats['page_stride'],
            f'{prefix}_page_bytes': stats['page_bytes'],
            f'{prefix}_dram_bytes': stats['dram_bytes'],
            f'{prefix}_dram_mib': format_mib(stats['dram_bytes']),
            f'{prefix}_useful_bytes': stats['useful_bytes'],
            f'{prefix}_occupancy': f"{float(stats['occupancy']):.6f}",
            f'{prefix}_semantic_channel_bytes': stats['semantic_channel_bytes'],
            f'{prefix}_semantic_ratio': f"{float(stats['semantic_ratio']):.6f}",
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
    layer_rows, tensor_stats = build_layer_rows(tensors, consumers, args.bin_width)
    peak_row = max(layer_rows, key=lambda row: int(row['peak_dram_bytes']))
    acceptance = build_acceptance(tensor_stats, layer_rows)

    payload = {
        'cfg': str(cfg_path),
        'ckpt': str(ckpt_path),
        'device': str(device),
        'weight_quant': args.weight_quant,
        'mode': 'hw_reference_int8',
        'data_mode': data_mode,
        'halo_mode': 'consumer_window_corner',
        'bin_width': args.bin_width,
        'coord_bytes': COORD_BYTES,
        'feature_bytes_per_channel': FEATURE_BYTES_PER_CHANNEL,
        'page_voxels': PAGE_VOXELS,
        'page_align_bytes': PAGE_ALIGN_BYTES,
        'peak_ifm_ofm_dram_bytes': int(peak_row['peak_dram_bytes']),
        'peak_ifm_ofm_pages': int(peak_row['peak_pages']),
        'peak_ifm_ofm_layer_id': peak_row['layer_id'],
        'peak_ifm_ofm_layer_name': peak_row['layer_name'],
        'acceptance': acceptance,
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
    print(f"Acceptance: {'PASS' if acceptance['passed'] else 'FAIL'}")
    if not acceptance['passed']:
        for item in acceptance['states']:
            if not item['match']:
                print(f"  pages {item['state']}: got {item['pages']} expected {item['expected_pages']}")
        for item in acceptance['copies']:
            if not item['match']:
                print(f"  sum_nb {item['state']}: got {item['sum_nb']} expected {item['expected_sum_nb']}")
        if not acceptance['peak_match']:
            print(
                f"  peak: L{acceptance['peak_layer_id']} {acceptance['peak_pages']} pages "
                f"{acceptance['peak_bytes']} Byte; last OFM pages {acceptance['last_ofm_pages']}"
            )


if __name__ == '__main__':
    main()

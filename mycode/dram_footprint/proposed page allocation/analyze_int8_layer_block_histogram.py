#!/usr/bin/env python3
"""INT8 SECOND 3D-backbone occupancy under the proposed zone/block LUT.

Runs hardware-reference INT8 inference of VoxelBackBone8x_HWQAT on KITTI
val/000216, then partitions each layer's active voxels with the closed final
zone LUT (including RTL boundary/halo copies). Writes markdown tables for:

  - nonempty voxel count per layer
  - nonempty (materialized) block count per layer
  - histogram of per-block voxel counts N_b, where N_b includes halo copies
  - page-allocated DRAM: ceil(N_b / 64) pages per block, page = 64 * (8 + C) bytes
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
from typing import Dict, List, Sequence, Tuple

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
from mycode.rtl_unfixed.partition import compute_rtl_unfixed_partition_counts, summarize_zone_specs
from mycode.zone_block_search.block_nb_analysis import (
    FINAL_LUTS,
    lut_lines_from_final,
    make_bin_edges,
    zone_specs_from_lut_lines,
)

BIN_WIDTH = 16
PAGE_VOXELS = 64
COORD_BYTES = 8
FEATURE_BYTES_PER_CHANNEL = 1

# Output sparse-shape ZYX → proposed stage LUT.
# conv_out keeps Stage-3 XY and LiDAR ref, but Z becomes 2.
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Proposed-LUT INT8 SECOND 3D-backbone nonempty voxel/block histogram'
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


def capture_layer_outputs(backbone, batch_after_vfe) -> List[dict]:
    captured: List[dict] = []
    orig_layer = backbone._forward_hw_reference_layer

    def record(layer_id, layer_name, conv_type, sparse_tensor):
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
        })

    def wrapped(sparse_tensor, layer_id, layer_info):
        name, conv, _bn, _relu, act_key = layer_info
        if layer_id == 0:
            record(-1, 'input', 'InputSparseTensor', sparse_tensor)
        output = orig_layer(sparse_tensor, layer_id, layer_info)
        record(layer_id, name, type(conv).__name__, output)
        return output

    backbone._forward_hw_reference_layer = wrapped
    try:
        with torch.no_grad():
            _ = backbone(batch_after_vfe)
    finally:
        backbone._forward_hw_reference_layer = orig_layer
    return captured


def pages_for_nb(nb: int, page_voxels: int = PAGE_VOXELS) -> int:
    if nb <= 0:
        return 0
    return int((int(nb) + page_voxels - 1) // page_voxels)


def format_mib(num_bytes: int) -> str:
    return f'{num_bytes / (1024.0 * 1024.0):.4f}'


def format_pct(ratio: float) -> str:
    return f'{100.0 * ratio:.2f}%'


def attach_page_dram(layer_rows: Sequence[dict]) -> int:
    """Add IFM+OFM coresident DRAM and return the pipeline peak."""
    peak = 0
    for index, row in enumerate(layer_rows):
        if index == 0:
            coresident = int(row['dram_bytes'])
        else:
            coresident = int(layer_rows[index - 1]['dram_bytes'] + row['dram_bytes'])
        row['ifm_ofm_dram_bytes'] = coresident
        peak = max(peak, coresident)
    return peak


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


def analyze_layer(layer: dict, bin_width: int) -> dict:
    shape_zyx = tuple(layer['spatial_shape_zyx'])
    stage = stage_for_spatial(shape_zyx)
    grid_xyz = grid_xyz_from_zyx(shape_zyx)
    lidar_center = LIDAR_CENTER_BY_STAGE[stage]
    zone_specs = zone_specs_from_lut_lines(lut_lines_from_final(stage))
    counts, n_blocks, _limit = compute_rtl_unfixed_partition_counts(
        layer['coords_zyx'],
        grid_xyz,
        zone_specs,
        lidar_center,
    )
    nonempty_counts = counts[counts > 0] if counts.size else np.zeros(0, dtype=np.int64)
    max_nb = int(nonempty_counts.max()) if nonempty_counts.size else 0
    bin_edges = make_bin_edges(max_nb, width=bin_width)
    channels = int(layer['feature_channels'])
    bytes_per_voxel = COORD_BYTES + channels * FEATURE_BYTES_PER_CHANNEL
    page_bytes = PAGE_VOXELS * bytes_per_voxel
    if nonempty_counts.size:
        total_pages = int(sum(pages_for_nb(int(nb)) for nb in nonempty_counts))
    else:
        total_pages = 0
    dram_bytes = total_pages * page_bytes
    packed_dram_bytes = int(layer['active_voxels']) * bytes_per_voxel
    occupancy = float(packed_dram_bytes / dram_bytes) if dram_bytes else 0.0
    return {
        'layer_id': layer['layer_id'],
        'layer_name': layer['layer_name'],
        'conv_type': layer['conv_type'],
        'feature_channels': channels,
        'bytes_per_voxel': bytes_per_voxel,
        'page_bytes': page_bytes,
        'pages': total_pages,
        'dram_bytes': dram_bytes,
        'packed_dram_bytes': packed_dram_bytes,
        'occupancy': occupancy,
        'spatial_shape_zyx': list(shape_zyx),
        'grid_size_xyz': list(grid_xyz),
        'stage': stage,
        'lidar_center_xy': list(lidar_center),
        'lut': summarize_zone_specs(zone_specs),
        'lut_pretty': lut_label(stage),
        'nonempty_voxels': int(layer['active_voxels']),
        'nonempty_blocks': int(n_blocks),
        'mean_nb': float(np.mean(nonempty_counts)) if nonempty_counts.size else 0.0,
        'median_nb': float(np.median(nonempty_counts)) if nonempty_counts.size else 0.0,
        'max_nb': max_nb,
        'sum_nb': int(nonempty_counts.sum()) if nonempty_counts.size else 0,
        'histogram': histogram_from_counts(nonempty_counts, bin_edges),
        'block_voxel_counts': [int(v) for v in nonempty_counts.tolist()],
    }


def md_escape(text) -> str:
    return str(text).replace('|', '\\|')


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    header_line = '| ' + ' | '.join(md_escape(h) for h in headers) + ' |'
    sep_line = '| ' + ' | '.join('---' for _ in headers) + ' |'
    body = ['| ' + ' | '.join(md_escape(cell) for cell in row) + ' |' for row in rows]
    return '\n'.join([header_line, sep_line, *body])


def unified_bin_edges(layer_rows: Sequence[dict], bin_width: int) -> List[Tuple[int, int]]:
    max_nb = max((int(row['max_nb']) for row in layer_rows), default=0)
    return make_bin_edges(max_nb, width=bin_width)


def histogram_lookup(layer_row: dict) -> Dict[str, int]:
    return {item['bin_label']: int(item['n_blocks']) for item in layer_row['histogram']}


def build_markdown(payload: dict) -> str:
    frame = payload['frame']
    layers = payload['layers']
    bin_width = int(payload['bin_width'])
    bin_edges = unified_bin_edges(layers, bin_width)
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
        f"- Halo / boundary copy: **enabled** (RTL 1..7 neighbor copies)",
        f"- Histogram bin width: `{bin_width}` (closed intervals `1-{bin_width}`, `{bin_width+1}-{2*bin_width}`, ...)",
        f"- Generated: `{payload['generated_at']}`",
        '',
        'Block partitioning uses the closed proposed LUT. `N_b` of a nonempty block is the number of stored voxels **including halo copies**. Halo-only blocks are counted as nonempty.',
        '',
        '## DRAM page allocation',
        '',
        f"- Coordinate: `{COORD_BYTES}` Byte / voxel",
        f"- Feature: `{FEATURE_BYTES_PER_CHANNEL}` Byte / channel; channels follow SECOND 3D backbone (`accdesign` / HW-QAT)",
        f"- Voxel record: `{COORD_BYTES} + C` Byte",
        f"- Pages per block: `ceil(N_b / {PAGE_VOXELS})` (N_b includes halo copies)",
        f"- Page capacity: `{PAGE_VOXELS} * ( {COORD_BYTES} + C )` Byte",
        f"- Layer DRAM: `sum_blocks ceil(N_b / {PAGE_VOXELS}) * page_capacity`",
        f"- Packed DRAM: `Nonempty voxels × (8 + C)`，不含边界复制、也不按页对齐",
        f"- Occupancy: `Packed DRAM / 本层 page DRAM`",
        f"- Pipeline peak (IFM+OFM coresident): `{peak_bytes}` Byte = `{format_mib(peak_bytes)}` MiB at `{peak_layer}`",
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
        '`conv_out` keeps Stage 3 XY / LiDAR reference, with spatial Z reduced to 2.',
        '',
        '## Per-layer nonempty voxels, blocks, and DRAM',
        '',
        markdown_table(
            [
                'Layer',
                'Name',
                'Type',
                'Stage',
                'Spatial ZYX',
                '通道数 C',
                '单个体素 Byte (8+C)',
                'Nonempty voxels (不含边界复制)',
                'Nonempty blocks (含边界复制)',
                'Mean N_b',
                'Median N_b',
                'Max N_b',
                'Sum N_b (with halo)',
                'Pages (ceil(N_b/64) 求和)',
                'Page 容量 Byte',
                '本层 DRAM Byte',
                '本层 DRAM MiB',
                '紧凑 DRAM Byte (体素数×(8+C)，不含边界复制)',
                '占用率 (紧凑/按页)',
                'IFM+OFM 驻留 Byte',
                'IFM+OFM 驻留 MiB',
            ],
            [
                [
                    row['layer_id'],
                    row['layer_name'],
                    row['conv_type'],
                    row['stage'],
                    'x'.join(str(v) for v in row['spatial_shape_zyx']),
                    row['feature_channels'],
                    row['bytes_per_voxel'],
                    row['nonempty_voxels'],
                    row['nonempty_blocks'],
                    f"{row['mean_nb']:.2f}",
                    f"{row['median_nb']:.1f}",
                    row['max_nb'],
                    row['sum_nb'],
                    row['pages'],
                    row['page_bytes'],
                    row['dram_bytes'],
                    format_mib(row['dram_bytes']),
                    row['packed_dram_bytes'],
                    format_pct(float(row['occupancy'])),
                    row.get('ifm_ofm_dram_bytes', ''),
                    format_mib(int(row.get('ifm_ofm_dram_bytes', 0))),
                ]
                for row in layers
            ],
        ),
        '',
        '## Nonempty-block N_b histogram',
        '',
        'Each cell is the number of nonempty blocks whose voxel count (including halo) falls in that bin.',
        '',
    ]

    hist_headers = ['Layer', 'Name', 'Stage', 'Nonempty blocks'] + [f'{lo}-{hi}' for lo, hi in bin_edges]
    hist_rows = []
    for row in layers:
        lookup = histogram_lookup(row)
        cells = [
            row['layer_id'],
            row['layer_name'],
            row['stage'],
            row['nonempty_blocks'],
        ]
        for lo, hi in bin_edges:
            cells.append(lookup.get(f'{lo}-{hi}', 0))
        hist_rows.append(cells)
    lines.append(markdown_table(hist_headers, hist_rows))
    lines.append('')

    lines.append('## Per-layer histogram detail')
    lines.append('')
    for row in layers:
        lines.append(
            f"### Layer {row['layer_id']}: `{row['layer_name']}` (stage {row['stage']})"
        )
        lines.append('')
        lines.append(
            f"- Spatial ZYX `{ 'x'.join(str(v) for v in row['spatial_shape_zyx']) }`, "
            f"LiDAR `({row['lidar_center_xy'][0]},{row['lidar_center_xy'][1]})`"
        )
        lines.append(f"- LUT: `{row['lut_pretty']}`")
        lines.append(
            f"- 通道数 C `{row['feature_channels']}`, voxel `{row['bytes_per_voxel']}` Byte, "
            f"page `{row['page_bytes']}` Byte, pages `{row['pages']}`"
        )
        lines.append(
            f"- Nonempty voxels `{row['nonempty_voxels']}` (不含边界复制), "
            f"nonempty blocks `{row['nonempty_blocks']}` (含边界复制)"
        )
        lines.append(
            f"- 本层 DRAM `{row['dram_bytes']}` Byte = `{format_mib(row['dram_bytes'])}` MiB; "
            f"紧凑 DRAM `{row['packed_dram_bytes']}` Byte; "
            f"占用率 `{format_pct(float(row['occupancy']))}`; "
            f"IFM+OFM 驻留 `{row.get('ifm_ofm_dram_bytes', 0)}` Byte = "
            f"`{format_mib(int(row.get('ifm_ofm_dram_bytes', 0)))}` MiB"
        )
        lines.append('')
        if not row['histogram']:
            lines.append('_No nonempty blocks._')
            lines.append('')
            continue
        detail_rows = [
            [item['bin_label'], item['n_blocks'], f"{100.0 * item['pct']:.2f}%"]
            for item in row['histogram']
            if item['n_blocks'] > 0
        ]
        if not detail_rows:
            lines.append('_No nonempty blocks._')
        else:
            lines.append(markdown_table(['N_b bin', 'Nonempty blocks', 'Share'], detail_rows))
        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


def json_ready_layers(layers: Sequence[dict]) -> List[dict]:
    out = []
    for row in layers:
        item = dict(row)
        item.pop('block_voxel_counts', None)
        out.append(item)
    return out


def write_histogram_csv(path: Path, layers: Sequence[dict], bin_edges: Sequence[Tuple[int, int]]) -> None:
    fieldnames = [
        'layer_id', 'layer_name', 'stage', 'channels', 'bytes_per_voxel', 'page_bytes',
        'pages', 'dram_bytes', 'dram_mib', 'packed_dram_bytes', 'occupancy',
        'ifm_ofm_dram_bytes', 'ifm_ofm_dram_mib',
        'nonempty_voxels', 'nonempty_blocks', 'mean_nb', 'max_nb',
    ]
    fieldnames.extend(f'bin_{lo}_{hi}' for lo, hi in bin_edges)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in layers:
            lookup = histogram_lookup(row)
            csv_row = {
                'layer_id': row['layer_id'],
                'layer_name': row['layer_name'],
                'stage': row['stage'],
                'channels': row['feature_channels'],
                'bytes_per_voxel': row['bytes_per_voxel'],
                'page_bytes': row['page_bytes'],
                'pages': row['pages'],
                'dram_bytes': row['dram_bytes'],
                'dram_mib': format_mib(row['dram_bytes']),
                'packed_dram_bytes': row['packed_dram_bytes'],
                'occupancy': f"{float(row['occupancy']):.6f}",
                'ifm_ofm_dram_bytes': row.get('ifm_ofm_dram_bytes', ''),
                'ifm_ofm_dram_mib': format_mib(int(row.get('ifm_ofm_dram_bytes', 0))),
                'nonempty_voxels': row['nonempty_voxels'],
                'nonempty_blocks': row['nonempty_blocks'],
                'mean_nb': f"{row['mean_nb']:.4f}",
                'max_nb': row['max_nb'],
            }
            for lo, hi in bin_edges:
                csv_row[f'bin_{lo}_{hi}'] = lookup.get(f'{lo}-{hi}', 0)
            writer.writerow(csv_row)


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

    captured = capture_layer_outputs(model.backbone_3d, batch_after_vfe)
    layer_rows = [analyze_layer(layer, args.bin_width) for layer in captured]
    peak_bytes = attach_page_dram(layer_rows)
    peak_row = max(layer_rows, key=lambda row: int(row['ifm_ofm_dram_bytes']))
    bin_edges = unified_bin_edges(layer_rows, args.bin_width)

    payload = {
        'cfg': str(cfg_path),
        'ckpt': str(ckpt_path),
        'device': str(device),
        'weight_quant': args.weight_quant,
        'mode': 'hw_reference_int8',
        'data_mode': data_mode,
        'halo': True,
        'bin_width': args.bin_width,
        'coord_bytes': COORD_BYTES,
        'feature_bytes_per_channel': FEATURE_BYTES_PER_CHANNEL,
        'page_voxels': PAGE_VOXELS,
        'peak_ifm_ofm_dram_bytes': peak_bytes,
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
    write_histogram_csv(csv_path, layer_rows, bin_edges)

    print(markdown)
    print(f'Saved markdown: {md_path}')
    print(f'Saved JSON:     {json_path}')
    print(f'Saved CSV:      {csv_path}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Compare three DRAM schemes on KITTI filename 000200-000219.

Reuses the 000216 / 000000-000019 analysis path: FOV KITTI loading,
hardware-reference INT8 SECOND 3D backbone, consumer window-corner halo.
Writes one markdown with per-frame Feature Map tables and IFM+OFM
coresident peaks. Frame IDs are consecutive filenames after the first
200 profiling frames (000000-000199); official KITTI split is mixed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mycode.kitti_frame_loader import (
    build_kitti_dataset,
    load_kitti_sample,
    resolve_data_mode,
)

SCHEME_META = [
    {
        'key': 'fixed_capacity',
        'label': '固定块 + 固定容量',
        'hash_from': 'blocks',
        'module_path': _SCRIPT_DIR / 'nonempty-block fixed allocation' / 'analyze_int8_layer_block_histogram.py',
        'module_name': 'fixed_capacity_hist',
    },
    {
        'key': 'fixed_page',
        'label': '固定块 + Page',
        'hash_from': 'pages',
        'module_path': _SCRIPT_DIR / 'page_with_fixed_block' / 'analyze_int8_layer_block_histogram.py',
        'module_name': 'fixed_page_hist',
    },
    {
        'key': 'proposed',
        'label': 'Proposed 可变块 + Page',
        'hash_from': 'pages',
        'module_path': _SCRIPT_DIR / 'proposed page allocation' / 'analyze_int8_layer_block_histogram.py',
        'module_name': 'proposed_hist',
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description='20-frame DRAM scheme comparison')
    parser.add_argument('--cfg', type=str, default='tools/cfgs/kitti_models/second_hw_qat.yaml')
    parser.add_argument(
        '--ckpt',
        type=str,
        default='output/kitti_models/second_hw_qat/hw_qat_10ep/ckpt/checkpoint_epoch_10.pth',
    )
    parser.add_argument('--kitti_root', type=str, default='data/kitti')
    parser.add_argument('--id_start', type=int, default=200)
    parser.add_argument('--id_end', type=int, default=219)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--weight_quant', choices=['per_channel', 'per_tensor'], default='per_channel')
    parser.add_argument(
        '--out_md',
        type=str,
        default=str(_SCRIPT_DIR / 'train_000200_000219_feature_map_dram_hash_entry_comparison.md'),
    )
    parser.add_argument(
        '--out_json',
        type=str,
        default=str(_SCRIPT_DIR / 'train_000200_000219_feature_map_dram_hash_entry_comparison.json'),
    )
    return parser.parse_args()


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attach_trainval_infos(dataset, project_root: Path):
    infos = []
    seen = set()
    split_by_id = {}
    for split, name in (('train', 'kitti_infos_train.pkl'), ('val', 'kitti_infos_val.pkl')):
        path = project_root / 'data/kitti' / name
        with path.open('rb') as handle:
            for info in pickle.load(handle):
                lidar_idx = str(info['point_cloud']['lidar_idx']).zfill(6)
                if lidar_idx in seen:
                    continue
                seen.add(lidar_idx)
                info['point_cloud']['lidar_idx'] = lidar_idx
                infos.append(info)
                split_by_id[lidar_idx] = split
    dataset.kitti_infos = infos
    info_by_id = {info['point_cloud']['lidar_idx']: idx for idx, info in enumerate(infos)}
    return info_by_id, split_by_id


def format_mib(num_bytes: int) -> str:
    return f'{num_bytes / (1024.0 * 1024.0):.4f}'


def mib_value(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def blocks_gt_128(stats: dict) -> int:
    return int(sum(item['n_blocks'] for item in stats.get('histogram', []) if int(item['bin_lo']) >= 129))


def hash_entries(stats: dict, hash_from: str) -> int:
    if not stats.get('allocated', True):
        return 0
    if hash_from == 'blocks':
        return int(stats['nonempty_blocks'])
    return int(stats['pages'])


def feature_map_rows(layers: Sequence[dict], hash_from: str) -> List[dict]:
    rows = []
    first = layers[0]['ifm']
    rows.append({
        'name': '初始输入',
        'name_html': '初始输入',
        'voxels': int(first['nonempty_voxels']),
        'channels': int(first['feature_channels']),
        'dram_bytes': int(first['dram_bytes']),
        'blocks': int(first['nonempty_blocks']),
        'hash_entries': hash_entries(first, hash_from),
        'blocks_gt_128': blocks_gt_128(first),
    })
    for layer in layers:
        ofm = layer['ofm']
        if not ofm.get('allocated', True):
            continue
        name = layer['layer_name']
        rows.append({
            'name': name,
            'name_html': f'<code>{name}</code>',
            'voxels': int(ofm['nonempty_voxels']),
            'channels': int(ofm['feature_channels']),
            'dram_bytes': int(ofm['dram_bytes']),
            'blocks': int(ofm['nonempty_blocks']),
            'hash_entries': hash_entries(ofm, hash_from),
            'blocks_gt_128': blocks_gt_128(ofm),
        })
    return rows


def coresident_peaks(layers: Sequence[dict], hash_from: str) -> dict:
    dram_peak = -1
    hash_peak = -1
    dram_layer = ''
    hash_layer = ''
    dram_layer_id = -1
    hash_layer_id = -1
    for layer in layers:
        ifm = layer['ifm']
        ofm = layer['ofm']
        ofm_bytes = int(ofm['dram_bytes']) if ofm.get('allocated', True) else 0
        dram = int(ifm['dram_bytes']) + ofm_bytes
        he = hash_entries(ifm, hash_from) + (hash_entries(ofm, hash_from) if ofm.get('allocated', True) else 0)
        if dram > dram_peak:
            dram_peak = dram
            dram_layer = layer['layer_name']
            dram_layer_id = layer['layer_id']
        if he > hash_peak:
            hash_peak = he
            hash_layer = layer['layer_name']
            hash_layer_id = layer['layer_id']
    return {
        'dram_bytes': dram_peak,
        'dram_mib': format_mib(dram_peak),
        'dram_layer_id': dram_layer_id,
        'dram_layer_name': dram_layer,
        'hash_entries': hash_peak,
        'hash_layer_id': hash_layer_id,
        'hash_layer_name': hash_layer,
    }


def column_max(rows: Sequence[dict], key: str):
    if key == 'dram_bytes':
        return max(int(row[key]) for row in rows)
    return max(int(row[key]) for row in rows)


def reduction_pct(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return 100.0 * (1.0 - float(new) / float(old))


def html_feature_table(scheme_rows: Dict[str, List[dict]]) -> str:
    cap = scheme_rows['fixed_capacity']
    page = scheme_rows['fixed_page']
    prop = scheme_rows['proposed']
    lines = [
        '<table>',
        '  <thead>',
        '    <tr>',
        '      <th rowspan="2">Feature map（产生它的层）</th>',
        '      <th rowspan="2">有效体素数</th>',
        '      <th rowspan="2">通道数</th>',
        '      <th colspan="4">固定块 + 固定容量</th>',
        '      <th colspan="4">固定块 + Page</th>',
        '      <th colspan="4">Proposed 可变块 + Page</th>',
        '    </tr>',
        '    <tr>',
        '      <th>DRAM / MiB</th>',
        '      <th>Blocks</th>',
        '      <th>Hash entries</th>',
        '      <th>Blocks &gt; 128 voxels</th>',
        '      <th>DRAM / MiB</th>',
        '      <th>Blocks</th>',
        '      <th>Hash entries</th>',
        '      <th>Blocks &gt; 128 voxels</th>',
        '      <th>DRAM / MiB</th>',
        '      <th>Blocks</th>',
        '      <th>Hash entries</th>',
        '      <th>Blocks &gt; 128 voxels</th>',
        '    </tr>',
        '  </thead>',
        '  <tbody>',
    ]
    for cap_row, page_row, prop_row in zip(cap, page, prop):
        lines.append(
            '<tr>'
            f'<td>{cap_row["name_html"]}</td>'
            f'<td>{cap_row["voxels"]}</td>'
            f'<td>{cap_row["channels"]}</td>'
            f'<td>{format_mib(cap_row["dram_bytes"])}</td>'
            f'<td>{cap_row["blocks"]}</td>'
            f'<td>{cap_row["hash_entries"]}</td>'
            f'<td>{cap_row["blocks_gt_128"]}</td>'
            f'<td>{format_mib(page_row["dram_bytes"])}</td>'
            f'<td>{page_row["blocks"]}</td>'
            f'<td>{page_row["hash_entries"]}</td>'
            f'<td>{page_row["blocks_gt_128"]}</td>'
            f'<td>{format_mib(prop_row["dram_bytes"])}</td>'
            f'<td>{prop_row["blocks"]}</td>'
            f'<td>{prop_row["hash_entries"]}</td>'
            f'<td>{prop_row["blocks_gt_128"]}</td>'
            '</tr>'
        )
    lines.append(
        '<tr>'
        '<td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td>'
        f'<td><strong>{format_mib(column_max(cap, "dram_bytes"))}</strong></td>'
        f'<td><strong>{column_max(cap, "blocks")}</strong></td>'
        f'<td><strong>{column_max(cap, "hash_entries")}</strong></td>'
        f'<td><strong>{column_max(cap, "blocks_gt_128")}</strong></td>'
        f'<td><strong>{format_mib(column_max(page, "dram_bytes"))}</strong></td>'
        f'<td><strong>{column_max(page, "blocks")}</strong></td>'
        f'<td><strong>{column_max(page, "hash_entries")}</strong></td>'
        f'<td><strong>{column_max(page, "blocks_gt_128")}</strong></td>'
        f'<td><strong>{format_mib(column_max(prop, "dram_bytes"))}</strong></td>'
        f'<td><strong>{column_max(prop, "blocks")}</strong></td>'
        f'<td><strong>{column_max(prop, "hash_entries")}</strong></td>'
        f'<td><strong>{column_max(prop, "blocks_gt_128")}</strong></td>'
        '</tr>'
    )
    lines.extend(['  </tbody>', '</table>'])
    return '\n'.join(lines)


def md_peak_table(peaks: dict) -> str:
    cap = peaks['fixed_capacity']
    page = peaks['fixed_page']
    prop = peaks['proposed']
    return '\n'.join([
        '| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |',
        '| --- | ---: | ---: | ---: |',
        f'| DRAM / MiB | {cap["dram_mib"]} | {page["dram_mib"]} | {prop["dram_mib"]} |',
        f'| DRAM 峰值层 | `{cap["dram_layer_name"]}` | `{page["dram_layer_name"]}` | `{prop["dram_layer_name"]}` |',
        f'| Hash entries | {cap["hash_entries"]} | {page["hash_entries"]} | {prop["hash_entries"]} |',
        f'| Hash 峰值层 | `{cap["hash_layer_name"]}` | `{page["hash_layer_name"]}` | `{prop["hash_layer_name"]}` |',
    ])


def summarize_reductions(peaks: dict) -> dict:
    cap = peaks['fixed_capacity']
    page = peaks['fixed_page']
    prop = peaks['proposed']
    return {
        'dram_vs_capacity_pct': reduction_pct(prop['dram_bytes'], cap['dram_bytes']),
        'hash_vs_capacity_pct': reduction_pct(prop['hash_entries'], cap['hash_entries']),
        'dram_vs_page_pct': reduction_pct(prop['dram_bytes'], page['dram_bytes']),
        'hash_vs_page_pct': reduction_pct(prop['hash_entries'], page['hash_entries']),
    }


def analyze_layers(modules: dict, tensors, consumers, bin_width: int, block_size_xyz):
    cap_layers, _ = modules['fixed_capacity'].build_layer_rows(
        tensors, consumers, bin_width, block_size_xyz
    )
    page_layers, _ = modules['fixed_page'].build_layer_rows(
        tensors, consumers, bin_width, block_size_xyz
    )
    prop_layers, _ = modules['proposed'].build_layer_rows(tensors, consumers, bin_width)
    scheme_layers = {
        'fixed_capacity': cap_layers,
        'fixed_page': page_layers,
        'proposed': prop_layers,
    }
    maps = {}
    peaks = {}
    for meta in SCHEME_META:
        key = meta['key']
        maps[key] = feature_map_rows(scheme_layers[key], meta['hash_from'])
        peaks[key] = coresident_peaks(scheme_layers[key], meta['hash_from'])
    return {
        'feature_maps': maps,
        'peaks': peaks,
        'reductions': summarize_reductions(peaks),
        'coordinate_sha256': {
            'input': tensors[0]['active_voxels'],
            'input_sha256': cap_layers[0]['ifm']['coordinate_sha256'],
            'hit_voxel_cap': int(tensors[0]['active_voxels']) >= 15000,
        },
    }


def build_markdown(payload: dict) -> str:
    frames = payload['frames']
    lines = [
        '# 三种 Block Structuring 方案在 filename 000200–000219 上的稳定性与泛化',
        '',
        '本文件复现 `feature_map_dram_hash_entry_comparison.md` 在 KITTI `val/000216` 上的口径，',
        '以及 `train_000000_000019_feature_map_dram_hash_entry_comparison.md` 的算法与实验设置。',
        '000000–000019 属于前 200 帧 profiling 集合（filename `000000`–`000199`）；',
        '本实验从其后再选连续 20 帧：filename `000200`–`000219`。',
        '这些 ID 按 KITTI 官方 train/val 划分交错出现，加载时仍合并 train/val infos。',
        '',
        '- 加载：KITTI FOV（`FOV_POINTS_ONLY=True`），与 golden 导出一致。',
        '- 模型：hardware-reference INT8 SECOND 3D backbone，checkpoint `checkpoint_epoch_10.pth`。',
        '- Halo：由下一层 kernel/padding 决定的窗口角点复制；`conv_out` 逻辑输出不分配 DRAM。',
        '- 三种方案：固定块+固定容量（`10x10x6`、每块 600 slot）、固定块+Page、Proposed 可变块+Page。',
        '- 上一层 OFM 与下一层 IFM 是同一 feature map，表中只统计一次。',
        '- Hash entries：固定容量方案等于物化 block 数；两种 Page 方案等于 page 数。',
        '- 执行时峰值按 IFM 与 OFM 同时驻留求和；DRAM 峰值层与 hash 峰值层可能不同。',
        f'- Generated: `{payload["generated_at"]}`',
        '',
        '## 20 帧执行时峰值总表',
        '',
        '| Frame | Split | 输入体素 | 触达 15000 上限 | 固定容量 DRAM | 固定块 Page DRAM | Proposed DRAM | 固定容量 Hash | 固定块 Page Hash | Proposed Hash | DRAM vs 固定容量 | Hash vs 固定容量 | DRAM vs 固定块 Page | Hash vs 固定块 Page |',
        '| --- | --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    dram_vs_cap = []
    hash_vs_cap = []
    dram_vs_page = []
    hash_vs_page = []
    for frame in frames:
        red = frame['reductions']
        peaks = frame['peaks']
        dram_vs_cap.append(red['dram_vs_capacity_pct'])
        hash_vs_cap.append(red['hash_vs_capacity_pct'])
        dram_vs_page.append(red['dram_vs_page_pct'])
        hash_vs_page.append(red['hash_vs_page_pct'])
        hit = 'Y' if frame['coordinate_sha256']['hit_voxel_cap'] else 'N'
        lines.append(
            f"| `{frame['frame_id']}` | {frame['split']} | {frame['coordinate_sha256']['input']} | {hit} | "
            f"{peaks['fixed_capacity']['dram_mib']} | {peaks['fixed_page']['dram_mib']} | {peaks['proposed']['dram_mib']} | "
            f"{peaks['fixed_capacity']['hash_entries']} | {peaks['fixed_page']['hash_entries']} | {peaks['proposed']['hash_entries']} | "
            f"{red['dram_vs_capacity_pct']:.2f}% | {red['hash_vs_capacity_pct']:.2f}% | "
            f"{red['dram_vs_page_pct']:.2f}% | {red['hash_vs_page_pct']:.2f}% |"
        )

    def stats(values):
        return min(values), sum(values) / len(values), max(values)

    dvc = stats(dram_vs_cap)
    hvc = stats(hash_vs_cap)
    dvp = stats(dram_vs_page)
    hvp = stats(hash_vs_page)
    lines.extend([
        '',
        '相对固定容量 / 固定块分页，Proposed 在 20 帧上的降低比例（正值表示 Proposed 更小）：',
        '',
        '| 指标 | 最小 | 平均 | 最大 |',
        '| --- | ---: | ---: | ---: |',
        f'| DRAM vs 固定容量 | {dvc[0]:.2f}% | {dvc[1]:.2f}% | {dvc[2]:.2f}% |',
        f'| Hash vs 固定容量 | {hvc[0]:.2f}% | {hvc[1]:.2f}% | {hvc[2]:.2f}% |',
        f'| DRAM vs 固定块 Page | {dvp[0]:.2f}% | {dvp[1]:.2f}% | {dvp[2]:.2f}% |',
        f'| Hash vs 固定块 Page | {hvp[0]:.2f}% | {hvp[1]:.2f}% | {hvp[2]:.2f}% |',
        '',
    ])
    for frame in frames:
        lines.extend([
            f"## Frame `{frame['split']}/{frame['frame_id']}`",
            '',
            f"- Point cloud: `{frame['point_cloud_path']}`",
            f"- 输入体素: `{frame['coordinate_sha256']['input']}`，坐标 SHA-256 `{frame['coordinate_sha256']['input_sha256']}`",
            f"- 触达 15000 voxel 上限: `{frame['coordinate_sha256']['hit_voxel_cap']}`",
            '',
            '### 按 Feature Map 去重后的对比',
            '',
            html_feature_table(frame['feature_maps']),
            '',
            '“单个 feature map 最大值”一行对每个指标独立取最大值。',
            '',
            f"- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **{frame['reductions']['dram_vs_capacity_pct']:.2f}%**，将 hash entry 峰值降低 **{frame['reductions']['hash_vs_capacity_pct']:.2f}%**。",
            f"- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **{frame['reductions']['dram_vs_page_pct']:.2f}%**，将 hash entry 峰值降低 **{frame['reductions']['hash_vs_page_pct']:.2f}%**。",
            '',
            '### 执行时 IFM 与 OFM 同时驻留峰值',
            '',
            md_peak_table(frame['peaks']),
            '',
        ])
    return '\n'.join(lines).rstrip() + '\n'


def main():
    args = parse_args()
    project_root = _PROJECT_ROOT
    frame_ids = [f'{idx:06d}' for idx in range(int(args.id_start), int(args.id_end) + 1)]

    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.utils import common_utils

    modules = {}
    for meta in SCHEME_META:
        modules[meta['key']] = load_module(meta['module_name'], meta['module_path'])
    proposed = modules['proposed']

    cfg_path = project_root / args.cfg
    ckpt_path = project_root / args.ckpt
    original_cwd = Path.cwd()
    try:
        os.chdir(project_root / 'tools')
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(original_cwd)

    device = proposed.resolve_device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    logger = common_utils.create_logger()
    data_mode = resolve_data_mode(cfg, 'kitti')
    if data_mode != 'kitti':
        raise RuntimeError('Expected KITTI FOV loading')

    dataset = build_kitti_dataset(cfg, project_root, args.kitti_root, logger)
    info_by_id, split_by_id = attach_trainval_infos(dataset, project_root)
    missing = [frame_id for frame_id in frame_ids if frame_id not in info_by_id]
    if missing:
        raise RuntimeError(f'Frames not found in KITTI train/val infos: {missing}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=(device.type == 'cpu'))
    model.to(device)
    model.eval()
    hw_dir = _SCRIPT_DIR / 'train_000200_000219_hw_export'
    hw_dir.mkdir(parents=True, exist_ok=True)
    proposed.enable_hw_reference(model, project_root, hw_dir, args.weight_quant, logger)

    block_size_xyz = (10, 10, 6)
    bin_width = 16
    frame_results = []
    for frame_id in frame_ids:
        print(f"Analyzing {split_by_id[frame_id]}/{frame_id} ...", flush=True)
        sample, frame_info = load_kitti_sample(dataset, frame_id)
        batch = dataset.collate_batch([sample])
        batch_torch = proposed.move_batch_to_device(batch, device)
        with torch.no_grad():
            batch_after_vfe = model.vfe(batch_torch)
        tensors, consumers = proposed.capture_layer_outputs(model.backbone_3d, batch_after_vfe)
        result = analyze_layers(modules, tensors, consumers, bin_width, block_size_xyz)
        result['frame_id'] = frame_id
        result['split'] = split_by_id[frame_id]
        result['point_cloud_path'] = frame_info['point_cloud_path']
        result['fov_points_only'] = frame_info['fov_points_only']
        frame_results.append(result)
        print(
            f"  voxels={result['coordinate_sha256']['input']} "
            f"proposed DRAM {result['peaks']['proposed']['dram_mib']} MiB "
            f"hash {result['peaks']['proposed']['hash_entries']}",
            flush=True,
        )

    payload = {
        'cfg': str(cfg_path),
        'ckpt': str(ckpt_path),
        'device': str(device),
        'mode': 'hw_reference_int8',
        'split': 'trainval_filename',
        'id_start': int(args.id_start),
        'id_end': int(args.id_end),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'frames': frame_results,
    }
    markdown = build_markdown(payload)
    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    if not out_md.is_absolute():
        out_md = project_root / out_md
    if not out_json.is_absolute():
        out_json = project_root / out_json
    out_md.write_text(markdown, encoding='utf-8')
    with out_json.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(f'Saved markdown: {out_md}')
    print(f'Saved JSON:     {out_json}')


if __name__ == '__main__':
    main()

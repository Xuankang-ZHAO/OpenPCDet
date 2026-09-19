#!/usr/bin/env python3
"""Compare three DRAM schemes on 200 dispersed KITTI val frames.

Samples official val IDs that have infos + velodyne on disk, using a fixed
seed and one draw from each of N equal strata so the subset is spread across
the split. Reuses the 000000-000019 / 000200-000219 analysis path and writes
the same markdown / JSON tables.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
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

DEFAULT_SEED = 20260919
DEFAULT_NUM_FRAMES = 200


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load_module(
    'dram_cmp_ref_000200',
    _SCRIPT_DIR / 'analyze_train_000200_000219_comparison.py',
)
SCHEME_META = ref.SCHEME_META


def parse_args():
    parser = argparse.ArgumentParser(description='200-frame val DRAM scheme comparison')
    parser.add_argument('--cfg', type=str, default='tools/cfgs/kitti_models/second_hw_qat.yaml')
    parser.add_argument(
        '--ckpt',
        type=str,
        default='output/kitti_models/second_hw_qat/hw_qat_10ep/ckpt/checkpoint_epoch_10.pth',
    )
    parser.add_argument('--kitti_root', type=str, default='data/kitti')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--num_frames', type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--weight_quant', choices=['per_channel', 'per_tensor'], default='per_channel')
    parser.add_argument(
        '--frame_list',
        type=str,
        default='',
        help='Optional existing ID list (one per line). Skips sampling when set.',
    )
    parser.add_argument(
        '--sample_only',
        action='store_true',
        help='Write the sampled frame IDs and exit without running the model.',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Reuse completed frames already stored in --out_json (default: on).',
    )
    parser.add_argument('--no_resume', action='store_false', dest='resume')
    parser.add_argument(
        '--out_md',
        type=str,
        default=str(_SCRIPT_DIR / 'val_sampled_200_feature_map_dram_hash_entry_comparison.md'),
    )
    parser.add_argument(
        '--out_json',
        type=str,
        default=str(_SCRIPT_DIR / 'val_sampled_200_feature_map_dram_hash_entry_comparison.json'),
    )
    parser.add_argument(
        '--out_ids',
        type=str,
        default=str(_SCRIPT_DIR / 'val_sampled_200_frame_ids.txt'),
    )
    parser.add_argument(
        '--out_sample_json',
        type=str,
        default=str(_SCRIPT_DIR / 'val_sampled_200_frame_ids.json'),
    )
    return parser.parse_args()


def resolve_out_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def normalize_val_infos(dataset, velodyne_dir: Path) -> List[str]:
    ids = []
    seen = set()
    infos = []
    for info in dataset.kitti_infos:
        lidar_idx = str(info['point_cloud']['lidar_idx']).zfill(6)
        if lidar_idx in seen:
            continue
        if not (velodyne_dir / f'{lidar_idx}.bin').exists():
            continue
        seen.add(lidar_idx)
        info['point_cloud']['lidar_idx'] = lidar_idx
        infos.append(info)
        ids.append(lidar_idx)
    dataset.kitti_infos = infos
    return ids


def sample_dispersed(ids: Sequence[str], num_frames: int, seed: int) -> List[str]:
    """Pick one ID from each of num_frames equal strata over the sorted pool."""
    pool = sorted(ids)
    if num_frames <= 0:
        raise ValueError('num_frames must be positive')
    if num_frames > len(pool):
        raise ValueError(f'Requested {num_frames} frames, but only {len(pool)} val frames are available')
    if num_frames == len(pool):
        return pool
    rng = random.Random(seed)
    selected = []
    for idx in range(num_frames):
        start = idx * len(pool) // num_frames
        end = (idx + 1) * len(pool) // num_frames
        selected.append(rng.choice(pool[start:end]))
    return selected


def load_frame_list(path: Path) -> List[str]:
    ids = []
    seen = set()
    for line in path.read_text(encoding='ascii').splitlines():
        token = line.strip()
        if not token or token.startswith('#'):
            continue
        frame_id = token.zfill(6)
        if frame_id in seen:
            continue
        seen.add(frame_id)
        ids.append(frame_id)
    return ids


def write_frame_ids(path: Path, frame_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(f'{frame_id}\n' for frame_id in frame_ids), encoding='ascii')


def write_sample_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
    tmp_path.replace(path)


def load_completed_frames(path: Path, expected_ids: Sequence[str]) -> Dict[str, dict]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    expected = set(expected_ids)
    completed = {}
    for frame in payload.get('frames', []):
        frame_id = str(frame.get('frame_id', '')).zfill(6)
        if frame_id in expected:
            completed[frame_id] = frame
    return completed


def build_markdown(payload: dict) -> str:
    frames = payload['frames']
    sampling = payload['sampling']
    n_frames = len(frames)
    id_preview = ', '.join(f'`{frame_id}`' for frame_id in sampling['frame_ids'][:20])
    more = '' if n_frames <= 20 else f' … 共 {n_frames} 帧，完整列表见 `{Path(payload["frame_id_list"]).name}`'
    lines = [
        f'# 三种 Block Structuring 方案在 {n_frames} 帧 KITTI val 抽样上的稳定性与泛化',
        '',
        '本文件复现 `feature_map_dram_hash_entry_comparison.md` 在 KITTI `val/000216` 上的口径，',
        '以及 `train_000000_000019` / `train_000200_000219` 的算法与实验设置。',
        '现有 40 帧是 filename 连续段（`000000`–`000019`、`000200`–`000219`）；',
        f'本实验从官方 val 可用帧（infos + velodyne）中用固定种子分层分散抽取 {sampling["num_frames"]} 帧。',
        '',
        f'- 抽样：`{sampling["method"]}`，seed=`{sampling["seed"]}`，池大小=`{sampling["pool_size"]}`。',
        f'- 帧 ID 列表：`{payload["frame_id_list"]}`',
        f'- 抽样记录：前 20 个 ID 为 {id_preview}{more}',
        '- 加载：KITTI FOV（`FOV_POINTS_ONLY=True`），与 golden 导出一致。',
        '- 模型：hardware-reference INT8 SECOND 3D backbone，checkpoint `checkpoint_epoch_10.pth`。',
        '- Halo：由下一层 kernel/padding 决定的窗口角点复制；`conv_out` 逻辑输出不分配 DRAM。',
        '- 三种方案：固定块+固定容量（`10x10x6`、每块 600 slot）、固定块+Page、Proposed 可变块+Page。',
        '- 上一层 OFM 与下一层 IFM 是同一 feature map，表中只统计一次。',
        '- Hash entries：固定容量方案等于物化 block 数；两种 Page 方案等于 page 数。',
        '- 执行时峰值按 IFM 与 OFM 同时驻留求和；DRAM 峰值层与 hash 峰值层可能不同。',
        f'- Generated: `{payload["generated_at"]}`',
        '',
        f'## {n_frames} 帧执行时峰值总表',
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
        f'相对固定容量 / 固定块分页，Proposed 在 {n_frames} 帧上的降低比例（正值表示 Proposed 更小）：',
        '',
        '| 指标 | 最小 | 平均 | 最大 |',
        '| --- | ---: | ---: | ---: |',
        f'| DRAM vs 固定容量 | {dvc[0]:.2f}% | {dvc[1]:.2f}% | {dvc[2]:.2f}% |',
        f'| Hash vs 固定容量 | {hvc[0]:.2f}% | {hvc[1]:.2f}% | {hvc[2]:.2f}% |',
        f'| DRAM vs 固定块 Page | {dvp[0]:.2f}% | {dvp[1]:.2f}% | {dvp[2]:.2f}% |',
        f'| Hash vs 固定块 Page | {hvp[0]:.2f}% | {hvp[1]:.2f}% | {hvp[2]:.2f}% |',
        '',
        '## 抽样帧 ID',
        '',
        '| # | Frame |',
        '| ---: | --- |',
    ])
    for idx, frame_id in enumerate(sampling['frame_ids'], start=1):
        lines.append(f'| {idx} | `{frame_id}` |')
    lines.append('')
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
            ref.html_feature_table(frame['feature_maps']),
            '',
            '“单个 feature map 最大值”一行对每个指标独立取最大值。',
            '',
            f"- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **{frame['reductions']['dram_vs_capacity_pct']:.2f}%**，将 hash entry 峰值降低 **{frame['reductions']['hash_vs_capacity_pct']:.2f}%**。",
            f"- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **{frame['reductions']['dram_vs_page_pct']:.2f}%**，将 hash entry 峰值降低 **{frame['reductions']['hash_vs_page_pct']:.2f}%**。",
            '',
            '### 执行时 IFM 与 OFM 同时驻留峰值',
            '',
            ref.md_peak_table(frame['peaks']),
            '',
        ])
    return '\n'.join(lines).rstrip() + '\n'


def main():
    args = parse_args()
    project_root = _PROJECT_ROOT
    kitti_root = project_root / args.kitti_root
    velodyne_dir = kitti_root / 'training' / 'velodyne'
    out_md = resolve_out_path(args.out_md, project_root)
    out_json = resolve_out_path(args.out_json, project_root)
    out_ids = resolve_out_path(args.out_ids, project_root)
    out_sample_json = resolve_out_path(args.out_sample_json, project_root)

    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.utils import common_utils

    cfg_path = project_root / args.cfg
    ckpt_path = project_root / args.ckpt
    original_cwd = Path.cwd()
    try:
        os.chdir(project_root / 'tools')
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(original_cwd)

    logger = common_utils.create_logger()
    data_mode = resolve_data_mode(cfg, 'kitti')
    if data_mode != 'kitti':
        raise RuntimeError('Expected KITTI FOV loading')

    dataset = build_kitti_dataset(cfg, project_root, args.kitti_root, logger)
    pool_ids = normalize_val_infos(dataset, velodyne_dir)
    if not pool_ids:
        raise RuntimeError(f'No available val frames under {velodyne_dir}')

    if args.frame_list:
        frame_ids = load_frame_list(resolve_out_path(args.frame_list, project_root))
        sampling_method = 'from_list'
    else:
        frame_ids = sample_dispersed(pool_ids, int(args.num_frames), int(args.seed))
        sampling_method = 'stratified_random'

    missing = [frame_id for frame_id in frame_ids if frame_id not in set(pool_ids)]
    if missing:
        raise RuntimeError(f'Sampled/listed frames not in available val pool: {missing}')

    sampling = {
        'method': sampling_method,
        'seed': int(args.seed),
        'num_frames': len(frame_ids),
        'requested_num_frames': int(args.num_frames),
        'pool_size': len(pool_ids),
        'pool_split': 'val',
        'pool_source': 'kitti_infos_val.pkl + training/velodyne',
        'frame_ids': frame_ids,
    }
    write_frame_ids(out_ids, frame_ids)
    write_sample_record(out_sample_json, {
        **sampling,
        'frame_id_list': str(out_ids),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
    })
    print(f'Sampled {len(frame_ids)} / {len(pool_ids)} val frames with seed={args.seed}', flush=True)
    print(f'Saved frame IDs: {out_ids}', flush=True)
    print(f'Saved sample record: {out_sample_json}', flush=True)
    if args.sample_only:
        return

    modules = {}
    for meta in SCHEME_META:
        modules[meta['key']] = load_module(meta['module_name'], meta['module_path'])
    proposed = modules['proposed']

    device = proposed.resolve_device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    from pcdet.models import build_network

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=(device.type == 'cpu'))
    model.to(device)
    model.eval()
    hw_dir = _SCRIPT_DIR / 'val_sampled_200_hw_export'
    hw_dir.mkdir(parents=True, exist_ok=True)
    proposed.enable_hw_reference(model, project_root, hw_dir, args.weight_quant, logger)

    block_size_xyz = (10, 10, 6)
    bin_width = 16
    completed = load_completed_frames(out_json, frame_ids) if args.resume else {}
    if completed:
        print(f'Resuming with {len(completed)} completed frames from {out_json}', flush=True)

    payload = {
        'cfg': str(cfg_path),
        'ckpt': str(ckpt_path),
        'device': str(device),
        'mode': 'hw_reference_int8',
        'split': 'val',
        'sampling': sampling,
        'frame_id_list': str(out_ids),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'frames': [],
    }

    for idx, frame_id in enumerate(frame_ids, start=1):
        if frame_id in completed:
            result = completed[frame_id]
            payload['frames'].append(result)
            print(
                f'[{idx}/{len(frame_ids)}] skip cached val/{frame_id} '
                f"voxels={result['coordinate_sha256']['input']}",
                flush=True,
            )
        else:
            print(f'[{idx}/{len(frame_ids)}] Analyzing val/{frame_id} ...', flush=True)
            sample, frame_info = load_kitti_sample(dataset, frame_id)
            batch = dataset.collate_batch([sample])
            batch_torch = proposed.move_batch_to_device(batch, device)
            with torch.no_grad():
                batch_after_vfe = model.vfe(batch_torch)
            tensors, consumers = proposed.capture_layer_outputs(model.backbone_3d, batch_after_vfe)
            result = ref.analyze_layers(modules, tensors, consumers, bin_width, block_size_xyz)
            result['frame_id'] = frame_id
            result['split'] = 'val'
            result['point_cloud_path'] = frame_info['point_cloud_path']
            result['fov_points_only'] = frame_info['fov_points_only']
            payload['frames'].append(result)
            print(
                f"  voxels={result['coordinate_sha256']['input']} "
                f"proposed DRAM {result['peaks']['proposed']['dram_mib']} MiB "
                f"hash {result['peaks']['proposed']['hash_entries']}",
                flush=True,
            )
        payload['generated_at'] = datetime.now().isoformat(timespec='seconds')
        write_payload(out_json, payload)

    markdown = build_markdown(payload)
    out_md.write_text(markdown, encoding='utf-8')
    print(f'Saved markdown: {out_md}')
    print(f'Saved JSON:     {out_json}')
    print(f'Saved frame IDs: {out_ids}')


if __name__ == '__main__':
    main()

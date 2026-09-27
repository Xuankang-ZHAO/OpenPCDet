#!/usr/bin/env python3
"""Compare three DRAM schemes on KITTI val/000216 at a 40000 voxel cap."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

VOXEL_CAP = 40000
FRAME_ID = '000216'


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load_module(
    'dram_cmp_ref_000000',
    _SCRIPT_DIR / 'analyze_train_000000_000019_comparison.py',
)


def voxel_cap(cfg) -> int:
    for processor in cfg.DATA_CONFIG.DATA_PROCESSOR:
        if processor.NAME == 'transform_points_to_voxels':
            return int(processor.MAX_NUMBER_OF_VOXELS['test'])
    raise RuntimeError('transform_points_to_voxels is missing from DATA_PROCESSOR')


def build_markdown(payload: dict) -> str:
    frame = payload['frames'][0]
    red = frame['reductions']
    lines = [
        '# 三种 Block Structuring 方案的 Feature Map 存储与 Hash Entry 对比',
        '',
        '本表基于 KITTI `val/000216`、INT8 SECOND 3D backbone，对三种 Block Structuring 方案做统一比较。',
        f'体素上限是 `kitti_dataset.yaml` 的 `MAX_NUMBER_OF_VOXELS.test={VOXEL_CAP}`。',
        '',
        '- 加载：KITTI FOV（`FOV_POINTS_ONLY=True`）。',
        '- 模型：hardware-reference INT8 SECOND 3D backbone，checkpoint `checkpoint_epoch_10.pth`。',
        '- Halo：由下一层 kernel/padding 决定的窗口角点复制；`conv_out` 逻辑输出不分配 DRAM。',
        '- 三种方案：固定块+固定容量（`10x10x6`、每块 600 slot）、固定块+Page、Proposed 可变块+Page。',
        '- 上一层 OFM 与下一层 IFM 是同一 feature map，表中只统计一次。',
        '- Hash entries：固定容量方案等于物化 block 数；两种 Page 方案等于 page 数。',
        '- 执行时峰值按 IFM 与 OFM 同时驻留求和。',
        f'- Generated: `{payload["generated_at"]}`',
        '',
        f"- Point cloud: `{frame['point_cloud_path']}`",
        f"- 输入体素: `{frame['coordinate_sha256']['input']}`，坐标 SHA-256 `{frame['coordinate_sha256']['input_sha256']}`",
        f"- 触达 {VOXEL_CAP} voxel 上限: `{frame['coordinate_sha256']['hit_voxel_cap']}`",
        '',
        '## 按 Feature Map 去重后的对比',
        '',
        ref.html_feature_table(frame['feature_maps']),
        '',
        '“单个 feature map 最大值”一行对每个指标独立取最大值。',
        '',
        f"- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **{red['dram_vs_capacity_pct']:.2f}%**，将 hash entry 峰值降低 **{red['hash_vs_capacity_pct']:.2f}%**。",
        f"- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **{red['dram_vs_page_pct']:.2f}%**，将 hash entry 峰值降低 **{red['hash_vs_page_pct']:.2f}%**。",
        '',
        '## 执行时 IFM 与 OFM 同时驻留峰值',
        '',
        ref.md_peak_table(frame['peaks']),
        '',
    ]
    return '\n'.join(lines)


def main():
    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.utils import common_utils
    from mycode.kitti_frame_loader import build_kitti_dataset, load_kitti_sample, resolve_data_mode

    cfg_path = _PROJECT_ROOT / 'tools/cfgs/kitti_models/second_hw_qat.yaml'
    ckpt_path = _PROJECT_ROOT / 'output/kitti_models/second_hw_qat/hw_qat_10ep/ckpt/checkpoint_epoch_10.pth'
    original_cwd = Path.cwd()
    try:
        os.chdir(_PROJECT_ROOT / 'tools')
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(original_cwd)
    loaded_cap = voxel_cap(cfg)
    if loaded_cap != VOXEL_CAP:
        raise RuntimeError(f'MAX_NUMBER_OF_VOXELS.test={loaded_cap}, expected {VOXEL_CAP}')

    modules = {}
    for meta in ref.SCHEME_META:
        modules[meta['key']] = ref.load_module(meta['module_name'] + '_216', meta['module_path'])
    proposed = modules['proposed']
    device = proposed.resolve_device('auto')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    logger = common_utils.create_logger()
    if resolve_data_mode(cfg, 'kitti') != 'kitti':
        raise RuntimeError('Expected KITTI FOV loading')
    dataset = build_kitti_dataset(cfg, _PROJECT_ROOT, 'data/kitti', logger)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=(device.type == 'cpu'))
    model.to(device)
    model.eval()
    hw_dir = _SCRIPT_DIR / 'val_000216_hw_export'
    hw_dir.mkdir(parents=True, exist_ok=True)
    proposed.enable_hw_reference(model, _PROJECT_ROOT, hw_dir, 'per_channel', logger)

    print(f'Analyzing val/{FRAME_ID} ...', flush=True)
    sample, frame_info = load_kitti_sample(dataset, FRAME_ID)
    batch = dataset.collate_batch([sample])
    batch_torch = proposed.move_batch_to_device(batch, device)
    with torch.no_grad():
        batch_after_vfe = model.vfe(batch_torch)
    tensors, consumers = proposed.capture_layer_outputs(model.backbone_3d, batch_after_vfe)
    result = ref.analyze_layers(modules, tensors, consumers, 16, (10, 10, 6))
    result['frame_id'] = FRAME_ID
    result['split'] = 'val'
    result['point_cloud_path'] = frame_info['point_cloud_path']
    result['fov_points_only'] = frame_info['fov_points_only']
    print(
        f"  voxels={result['coordinate_sha256']['input']} "
        f"hit_{VOXEL_CAP}={result['coordinate_sha256']['hit_voxel_cap']} "
        f"proposed DRAM {result['peaks']['proposed']['dram_mib']} MiB",
        flush=True,
    )

    payload = {
        'cfg': str(cfg_path),
        'ckpt': str(ckpt_path),
        'device': str(device),
        'mode': 'hw_reference_int8',
        'max_number_of_voxels_test': loaded_cap,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'frames': [result],
    }
    out_md = _SCRIPT_DIR / 'feature_map_dram_hash_entry_comparison.md'
    out_json = _SCRIPT_DIR / 'feature_map_dram_hash_entry_comparison.json'
    out_md.write_text(build_markdown(payload), encoding='utf-8')
    out_json.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f'Saved markdown: {out_md}')
    print(f'Saved JSON:     {out_json}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Count 3x3x3 kernel-offset usage in the SECOND 3D backbone.

Runs the same KITTI inference path as ``mycode/second_layer_sparsity.py``
(VFE, then the 3D sparse backbone) and, for every sparse conv whose kernel
is 3x3x3, records how many valid input-output pairs use each kernel offset.

The spatial offset ``(dz, dy, dx)`` is the spconv rulebook offset:

    input_zyx = output_zyx * stride_zyx + (dz, dy, dx)
    (dz, dy, dx) = kernel_index_zyx * dilation_zyx - padding_zyx

For the usual padding of 1 this is the centered set ``{-1, 0, +1}^3``.
Submanifold layers use effective padding ``ksize // 2`` even when the module
attribute is 0. ``conv4.0`` uses padding ``(0, 1, 1)``, so its Z offsets are
``{0, 1, 2}``.

``pair_count`` is the number of valid pairs. ``pair_frequency`` is that count
divided by the layer's total valid pairs. Submanifold center pairs are
included; spconv's ``indice_pair_num`` omits some of them, so counts are
taken from the pair tensor itself.

Default frame is KITTI ``000216``, the accdesign golden frame.
``conv_out`` uses a ``3x1x1`` kernel and is not part of this table.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Per-layer 3x3x3 kernel-offset usage for the SECOND 3D backbone'
    )
    parser.add_argument('--cfg', type=str, default='tools/cfgs/kitti_models/second_hw_qat.yaml')
    parser.add_argument(
        '--ckpt',
        type=str,
        default='output/kitti_models/second_hw_qat/hw_qat_10ep/ckpt/checkpoint_epoch_10.pth',
    )
    parser.add_argument('--kitti_root', type=str, default='data/kitti')
    parser.add_argument(
        '--data_mode', type=str, default='auto', choices=['auto', 'kitti', 'raw'],
        help='auto uses KittiDataset so FOV filtering matches normal inference.',
    )
    parser.add_argument(
        '--frame_id', type=str, default='000216',
        help='KITTI frame id. 216 and 000216 both select the golden frame.',
    )
    parser.add_argument('--device', type=str, default='auto', help='auto|cpu|cuda|cuda:0')
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='mycode/output/second_kernel_offset')
    parser.add_argument('--log_file', type=str, default='')
    return parser.parse_args()


def resolve_project_root():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def normalize_frame_id(token):
    token = Path(token).stem if str(token).endswith('.bin') else str(token).strip()
    if token.isdigit():
        return f'{int(token):06d}'
    return token


def kernel_index_to_zyx(index, kernel_zyx):
    kx = int(kernel_zyx[2])
    ky = int(kernel_zyx[1])
    kz = index // (ky * kx)
    rem = index % (ky * kx)
    ky_i = rem // kx
    kx_i = rem % kx
    return int(kz), int(ky_i), int(kx_i)


def spatial_offset_zyx(kernel_zyx_index, dilation_zyx, padding_zyx):
    return tuple(
        int(kernel_zyx_index[axis] * dilation_zyx[axis] - padding_zyx[axis])
        for axis in range(3)
    )


def effective_padding_zyx(kernel_zyx, dilation_zyx, padding_zyx, subm):
    """Padding that actually enters the spconv rulebook.

    Submanifold conv ignores ``module.padding`` and uses ``(ksize // 2) * dilation``.
    ``post_act_block`` builds SubMConv3d without passing padding, so the module
    attribute is often 0 even though the geometric padding is 1.
    """
    if subm:
        return [int((kernel_zyx[axis] // 2) * dilation_zyx[axis]) for axis in range(3)]
    return [int(v) for v in padding_zyx]


def pair_counts_from_rulebook(pair):
    """Count valid pairs per kernel index.

    ``pair[0, k]`` holds input indices and uses -1 for empty slots. Submanifold
    conv leaves ``indice_pair_num`` at 0 for the center (and sometimes other
    offsets) even when those pairs are present, so the pair tensor is the
    source of truth.
    """
    if pair.dim() != 3 or pair.shape[0] < 1:
        raise RuntimeError(f'Unexpected indice pair shape {tuple(pair.shape)}')
    return (pair[0] >= 0).sum(dim=-1).to(dtype=torch.int64)


def _assert_offset_matches_pairs(pair, in_indices, out_indices, stride_zyx, dilation_zyx, padding_zyx, kernel_zyx):
    stride = torch.tensor(stride_zyx, device=pair.device, dtype=torch.int32)
    counts = pair_counts_from_rulebook(pair)
    kv = int(np.prod(kernel_zyx))
    if int(counts.numel()) != kv:
        raise AssertionError(f'pair count length {int(counts.numel())} != kernel volume {kv}')
    for index in range(kv):
        valid = pair[0, index] >= 0
        if not bool(valid.any()):
            continue
        kz, ky, kx = kernel_index_to_zyx(index, kernel_zyx)
        offset = spatial_offset_zyx((kz, ky, kx), dilation_zyx, padding_zyx)
        offset_t = torch.tensor(offset, device=pair.device, dtype=torch.int32)
        in_idx = pair[0, index][valid].long()
        out_idx = pair[1, index][valid].long()
        got = in_indices[in_idx, 1:]
        expect = out_indices[out_idx, 1:] * stride + offset_t
        if not torch.equal(got, expect):
            raise AssertionError(
                f'offset check failed at kernel index {index} offset {offset}: '
                f'got {got[:4].tolist()} expect {expect[:4].tolist()}'
            )


def self_test_offset_convention():
    """Fail fast if spconv's 3x3x3 kernel-index order changes."""
    from spconv.core import ConvAlgo
    from spconv.pytorch import ops

    indices = torch.tensor([
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 2, 2, 3],
        [0, 4, 5, 6],
    ], dtype=torch.int32)
    spatial = [8, 8, 8]
    kernel = [3, 3, 3]
    dilation = [1, 1, 1]
    cases = [
        # SubM ignores the padding argument and always uses ksize // 2.
        dict(stride=[1, 1, 1], padding=[0, 0, 0], effective_padding=[1, 1, 1], subm=True),
        dict(stride=[2, 2, 2], padding=[1, 1, 1], effective_padding=[1, 1, 1], subm=False),
        dict(stride=[2, 2, 2], padding=[0, 1, 1], effective_padding=[0, 1, 1], subm=False),
    ]
    for case in cases:
        out_indices, pair, _num = ops.get_indice_pairs(
            indices, 1, spatial, ConvAlgo.Native, kernel,
            case['stride'], case['padding'], dilation, [0, 0, 0],
            case['subm'], False,
        )
        _assert_offset_matches_pairs(
            pair, indices, out_indices, case['stride'], dilation, case['effective_padding'], kernel,
        )
        counts = pair_counts_from_rulebook(pair)
        if case['subm']:
            center = 13  # kz=ky=kx=1, x fastest
            if int(counts[center]) != int(indices.shape[0]):
                raise AssertionError(
                    f'subm center offset count {int(counts[center])} != voxel count {int(indices.shape[0])}'
                )


def collect_layer_offsets(module, sparse_input, ops, conv_algo):
    kernel = [int(v) for v in module.kernel_size]
    if kernel != [3, 3, 3]:
        return None

    stride = [int(v) for v in module.stride]
    module_padding = [int(v) for v in module.padding]
    dilation = [int(v) for v in module.dilation]
    padding = effective_padding_zyx(kernel, dilation, module_padding, bool(module.subm))
    indices = sparse_input.indices
    if not indices.is_contiguous():
        indices = indices.contiguous()

    out_indices, pair, _num = ops.get_indice_pairs(
        indices,
        int(sparse_input.batch_size),
        [int(v) for v in sparse_input.spatial_shape],
        conv_algo.Native,
        kernel,
        stride,
        padding,
        dilation,
        [int(v) for v in module.output_padding],
        bool(module.subm),
        bool(module.transposed),
    )
    _assert_offset_matches_pairs(pair, indices, out_indices, stride, dilation, padding, kernel)
    counts = pair_counts_from_rulebook(pair).detach().cpu().numpy().astype(np.int64)
    total_pairs = int(counts.sum())
    num_input = int(indices.shape[0])
    num_output = int(out_indices.shape[0])
    offsets = []
    order = np.argsort(-counts, kind='mergesort')
    rank_of = {int(index): rank for rank, index in enumerate(order, start=1)}
    cumulative = 0
    cumulative_at = {}
    for index in order.tolist():
        cumulative += int(counts[index])
        cumulative_at[int(index)] = cumulative

    for index in range(int(counts.shape[0])):
        kz, ky, kx = kernel_index_to_zyx(index, kernel)
        dz, dy, dx = spatial_offset_zyx((kz, ky, kx), dilation, padding)
        count = int(counts[index])
        offsets.append({
            'kernel_index': index,
            'kz': kz,
            'ky': ky,
            'kx': kx,
            'dz': dz,
            'dy': dy,
            'dx': dx,
            'is_kernel_center': bool(kz == ky == kx == 1),
            'pair_count': count,
            'pair_frequency': (float(count) / total_pairs) if total_pairs else 0.0,
            'cumulative_frequency': (float(cumulative_at[index]) / total_pairs) if total_pairs else 0.0,
            'rank': int(rank_of[index]),
            'output_hit_rate': (float(count) / num_output) if num_output else 0.0,
            'input_hit_rate': (float(count) / num_input) if num_input else 0.0,
        })

    used = [item for item in offsets if item['pair_count'] > 0]
    return {
        'kernel_zyx': kernel,
        'stride_zyx': stride,
        'padding_zyx': padding,
        'module_padding_zyx': module_padding,
        'dilation_zyx': dilation,
        'subm': bool(module.subm),
        'input_spatial_shape_zyx': [int(v) for v in sparse_input.spatial_shape],
        'input_voxels': num_input,
        'output_voxels': num_output,
        'total_pairs': total_pairs,
        'used_offsets': len(used),
        'unused_offsets': 27 - len(used),
        'mean_pairs_per_output': (float(total_pairs) / num_output) if num_output else 0.0,
        'mean_pairs_per_input': (float(total_pairs) / num_input) if num_input else 0.0,
        'offsets': offsets,
    }


class KernelOffsetRecorder:
    def __init__(self, spconv_module, ops, conv_algo):
        self.spconv_module = spconv_module
        self.ops = ops
        self.conv_algo = conv_algo
        self.records = []
        self.skipped = []
        self.handles = []

    def register(self, backbone):
        allowed = (self.spconv_module.SubMConv3d, self.spconv_module.SparseConv3d)
        for name, module in backbone.named_modules():
            if isinstance(module, allowed):
                self.handles.append(module.register_forward_hook(self._make_hook(name, module)))

    def _make_hook(self, name, module):
        def hook(_module, inputs, output):
            kernel = [int(v) for v in module.kernel_size]
            if kernel != [3, 3, 3]:
                self.skipped.append({
                    'layer_name': name,
                    'layer_type': module.__class__.__name__,
                    'indice_key': getattr(module, 'indice_key', '') or '',
                    'kernel_zyx': kernel,
                    'reason': 'not a 3x3x3 kernel',
                })
                return
            stats = collect_layer_offsets(module, inputs[0], self.ops, self.conv_algo)
            if int(output.indices.shape[0]) != stats['output_voxels']:
                raise RuntimeError(
                    f'{name}: rulebook output voxels {stats["output_voxels"]} '
                    f'!= forward output voxels {int(output.indices.shape[0])}'
                )
            stats.update({
                'layer_name': name,
                'layer_type': module.__class__.__name__,
                'indice_key': getattr(module, 'indice_key', '') or '',
                'output_spatial_shape_zyx': [int(v) for v in output.spatial_shape],
                'forward_algo': str(getattr(module, 'algo', '')),
            })
            self.records.append(stats)

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _fmt_xyz(values):
    return 'x'.join(str(int(v)) for v in values)


def _fmt_signed(value):
    return f'{int(value):+d}'


def coverage_rank(offsets, threshold):
    ranked = sorted(offsets, key=lambda item: (-item['pair_count'], item['kernel_index']))
    total = sum(item['pair_count'] for item in ranked)
    if total == 0:
        return 0
    running = 0
    for rank, item in enumerate(ranked, start=1):
        running += item['pair_count']
        if running / total >= threshold:
            return rank
    return len(ranked)


def format_report(frame_id, device, records, skipped):
    lines = [
        f'Frame: {frame_id}',
        f'Device: {device}',
        'Spatial offset definition: input_zyx = output_zyx * stride_zyx + (dz, dy, dx)',
        '(dz, dy, dx) = (kz, ky, kx) * dilation_zyx - padding_zyx, kernel index with X fastest.',
        '',
    ]
    if skipped:
        skipped_desc = ', '.join(
            f"{item['layer_name']}({_fmt_xyz(item['kernel_zyx'])})" for item in skipped
        )
        lines.append(f'Skipped non-3x3x3 sparse convs: {skipped_desc}')
        lines.append('')

    for record in records:
        padding_note = ''
        if record['padding_zyx'] != record['module_padding_zyx']:
            padding_note = f"  module_pad={_fmt_xyz(record['module_padding_zyx'])}"
        lines.append(
            f"{record['layer_name']}  {record['layer_type']}  key={record['indice_key'] or '-'}  "
            f"k={_fmt_xyz(record['kernel_zyx'])}  s={_fmt_xyz(record['stride_zyx'])}  "
            f"p={_fmt_xyz(record['padding_zyx'])}{padding_note}  "
            f"in={record['input_voxels']}  out={record['output_voxels']}  "
            f"pairs={record['total_pairs']}  used={record['used_offsets']}/27  "
            f"mean_per_out={record['mean_pairs_per_output']:.4f}"
        )
        lines.append(
            f"  cover50={coverage_rank(record['offsets'], 0.50)}  "
            f"cover90={coverage_rank(record['offsets'], 0.90)}  "
            f"cover99={coverage_rank(record['offsets'], 0.99)} offsets"
        )
        lines.append(
            f"  {'idx':>3} {'kz':>3} {'ky':>3} {'kx':>3} {'dz':>4} {'dy':>4} {'dx':>4} "
            f"{'count':>10} {'freq%':>8} {'cum%':>8} {'rank':>4} {'out_hit%':>9}"
        )
        for item in record['offsets']:
            lines.append(
                f"  {item['kernel_index']:>3d} {item['kz']:>3d} {item['ky']:>3d} {item['kx']:>3d} "
                f"{_fmt_signed(item['dz']):>4} {_fmt_signed(item['dy']):>4} {_fmt_signed(item['dx']):>4} "
                f"{item['pair_count']:>10d} {100.0 * item['pair_frequency']:>8.4f} "
                f"{100.0 * item['cumulative_frequency']:>8.4f} {item['rank']:>4d} "
                f"{100.0 * item['output_hit_rate']:>9.4f}"
            )
        lines.append('  frequency grid (rows dy, cols dx, percent of pairs):')
        by_offset = {(item['dz'], item['dy'], item['dx']): item for item in record['offsets']}
        dz_values = sorted({item['dz'] for item in record['offsets']})
        dy_values = sorted({item['dy'] for item in record['offsets']})
        dx_values = sorted({item['dx'] for item in record['offsets']})
        header = ' ' * 10 + ''.join(f'{("dx"+_fmt_signed(dx)):>10}' for dx in dx_values)
        for dz in dz_values:
            lines.append(f'  dz={_fmt_signed(dz)}')
            lines.append(header)
            for dy in dy_values:
                cells = []
                for dx in dx_values:
                    item = by_offset[(dz, dy, dx)]
                    cells.append(f'{100.0 * item["pair_frequency"]:>10.4f}')
                lines.append(f'  {"dy"+_fmt_signed(dy):<8}{"".join(cells)}')
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'


def save_outputs(out_dir, frame_id, payload, report_text):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / f'{frame_id}_second_kernel_offset.json'
    csv_path = out_path / f'{frame_id}_second_kernel_offset.csv'
    log_path = out_path / f'{frame_id}_second_kernel_offset.log'

    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)

    fieldnames = [
        'frame_id', 'layer_name', 'layer_type', 'indice_key',
        'kernel_zyx', 'stride_zyx', 'padding_zyx', 'dilation_zyx',
        'input_voxels', 'output_voxels', 'total_pairs', 'used_offsets',
        'kernel_index', 'kz', 'ky', 'kx', 'dz', 'dy', 'dx', 'is_kernel_center',
        'pair_count', 'pair_frequency', 'cumulative_frequency', 'rank',
        'output_hit_rate', 'input_hit_rate',
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in payload['records']:
            for item in record['offsets']:
                writer.writerow({
                    'frame_id': frame_id,
                    'layer_name': record['layer_name'],
                    'layer_type': record['layer_type'],
                    'indice_key': record['indice_key'],
                    'kernel_zyx': _fmt_xyz(record['kernel_zyx']),
                    'stride_zyx': _fmt_xyz(record['stride_zyx']),
                    'padding_zyx': _fmt_xyz(record['padding_zyx']),
                    'dilation_zyx': _fmt_xyz(record['dilation_zyx']),
                    'input_voxels': record['input_voxels'],
                    'output_voxels': record['output_voxels'],
                    'total_pairs': record['total_pairs'],
                    'used_offsets': record['used_offsets'],
                    'kernel_index': item['kernel_index'],
                    'kz': item['kz'],
                    'ky': item['ky'],
                    'kx': item['kx'],
                    'dz': item['dz'],
                    'dy': item['dy'],
                    'dx': item['dx'],
                    'is_kernel_center': int(item['is_kernel_center']),
                    'pair_count': item['pair_count'],
                    'pair_frequency': f"{item['pair_frequency']:.8f}",
                    'cumulative_frequency': f"{item['cumulative_frequency']:.8f}",
                    'rank': item['rank'],
                    'output_hit_rate': f"{item['output_hit_rate']:.8f}",
                    'input_hit_rate': f"{item['input_hit_rate']:.8f}",
                })

    with log_path.open('w', encoding='utf-8') as handle:
        handle.write(report_text)
    return json_path, csv_path, log_path


def analyze_frame(model, backbone, spconv_module, ops, conv_algo, batch_torch):
    recorder = KernelOffsetRecorder(spconv_module, ops, conv_algo)
    recorder.register(backbone)
    with torch.no_grad():
        batch_torch = model.vfe(batch_torch)
        try:
            _ = backbone(batch_torch)
        finally:
            recorder.remove()
    return recorder.records, recorder.skipped


def main():
    args = parse_args()
    self_test_offset_convention()
    project_root = resolve_project_root()

    from mycode.second_layer_sparsity import (
        build_inference_dataset,
        choose_kitti_frames,
        choose_raw_frames,
        load_kitti_batch,
        load_raw_batch,
        move_batch_to_device,
        resolve_data_mode,
        resolve_device,
    )
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.utils import common_utils
    from pcdet.utils.spconv_utils import spconv
    from spconv.core import ConvAlgo
    from spconv.pytorch import ops

    original_cwd = Path.cwd()
    try:
        os.chdir(project_root / 'tools')
        cfg_from_yaml_file(str(project_root / args.cfg), cfg)
    finally:
        os.chdir(original_cwd)

    device = resolve_device(args.device)
    logger = common_utils.create_logger()
    data_mode = resolve_data_mode(cfg, args.data_mode)
    dataset = build_inference_dataset(cfg, project_root, args, data_mode, logger)

    frame_id_arg = ','.join(normalize_frame_id(token) for token in args.frame_id.split(',') if token.strip())
    if data_mode == 'kitti':
        frame_refs = choose_kitti_frames(dataset, frame_id_arg, args.seed, args.num_frames)
    else:
        frame_refs = choose_raw_frames(project_root / 'data/kitti/training/velodyne', frame_id_arg, args.seed, args.num_frames)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    ckpt_path = project_root / args.ckpt
    if ckpt_path.exists():
        model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=(device.type == 'cpu'))
    else:
        logger.info(f'Checkpoint not found, continue without loading weights: {ckpt_path}')

    model.to(device)
    model.eval()
    backbone = model.backbone_3d

    out_dir = project_root / args.out_dir
    for frame_ref in frame_refs:
        if data_mode == 'kitti':
            batch, frame_info = load_kitti_batch(dataset, frame_ref)
        else:
            batch, frame_info = load_raw_batch(dataset, frame_ref)
        batch_torch = move_batch_to_device(batch, device)
        records, skipped = analyze_frame(model, backbone, spconv, ops, ConvAlgo, batch_torch)
        payload = {
            'frame_id': frame_info['frame_id'],
            'point_cloud_path': frame_info['point_cloud_path'],
            'data_loader': frame_info['data_loader'],
            'fov_points_only': frame_info['fov_points_only'],
            'cfg': str(project_root / args.cfg),
            'ckpt': str(ckpt_path),
            'device': str(device),
            'data_mode': data_mode,
            'offset_definition': 'input_zyx = output_zyx * stride_zyx + (dz, dy, dx)',
            'kernel_index_order': 'X fastest, then Y, then Z',
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'skipped_layers': skipped,
            'records': records,
        }
        report = format_report(frame_info['frame_id'], device, records, skipped)
        print(report)
        json_path, csv_path, log_path = save_outputs(out_dir, frame_info['frame_id'], payload, report)
        print(f'Saved JSON: {json_path}')
        print(f'Saved CSV:  {csv_path}')
        print(f'Saved log:  {log_path}')

        if args.log_file:
            extra_log = project_root / args.log_file
            extra_log.parent.mkdir(parents=True, exist_ok=True)
            extra_log.write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()

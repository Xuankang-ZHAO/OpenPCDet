#!/usr/bin/env python3
"""Export one KITTI SECOND HW-QAT frame as an RTL golden-vector package."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import struct
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


RTL_SCHEDULE = [
    (4, 16), (16, 16), (16, 32), (32, 32), (32, 32), (32, 64),
    (64, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 128),
]

ACT_KEYS = [
    'input',
    'l00_conv_input',
    'l01_conv1',
    'l02_conv2_0',
    'l03_conv2_1',
    'l04_conv2_2',
    'l05_conv3_0',
    'l06_conv3_1',
    'l07_conv3_2',
    'l08_conv4_0',
    'l09_conv4_1',
    'l10_conv4_2',
    'l11_conv_out',
]


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    default_run = repo_root / 'output/kitti_models/second_hw_qat/hw_qat_10ep'
    parser = argparse.ArgumentParser(description='Export SECOND HW-QAT RTL golden-vector package')
    parser.add_argument('--cfg_file', default=str(repo_root / 'tools/cfgs/kitti_models/second_hw_qat.yaml'))
    parser.add_argument('--ckpt', default=str(default_run / 'ckpt/checkpoint_epoch_10.pth'))
    parser.add_argument('--hw_export_dir', default=str(default_run / 'hw_export'))
    parser.add_argument('--split', default='val', choices=['train', 'val', 'trainval'])
    parser.add_argument('--frame_pool_size', type=int, default=100)
    parser.add_argument('--random_seed', type=int, default=20260625)
    parser.add_argument('--output_root', default=str(repo_root / 'accdesign/second_rtl_golden_packages'))
    parser.add_argument('--zone_lut', default=str(repo_root / 'mycode/block_size_lut_rtl_unfixed.txt'))
    parser.add_argument('--lidar_center', default='0,800')
    parser.add_argument('--max_voxels', type=int, default=15000)
    parser.add_argument('--overwrite', action='store_true', default=False)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def read_csv_rows(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def parse_tuple(text):
    return tuple(int(part.strip()) for part in text.strip().strip('()').split(',') if part.strip())


def ensure_project_imports(repo_root):
    for path in (repo_root, repo_root / 'tools'):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def make_logger(log_path):
    from pcdet.utils import common_utils

    return common_utils.create_logger(log_path, rank=0)


def output_channel_axis(conv, weight):
    if weight.shape[0] == conv.out_channels:
        return 0
    if weight.shape[-1] == conv.out_channels:
        return weight.dim() - 1
    return 0


def view_channel_vector(conv, weight, vector):
    if vector.dim() == 0:
        return vector
    shape = [1] * weight.dim()
    shape[output_channel_axis(conv, weight)] = vector.numel()
    return vector.view(*shape)


def per_channel_scale(conv, weight):
    axis = output_channel_axis(conv, weight)
    reduce_dims = [idx for idx in range(weight.dim()) if idx != axis]
    return (weight.detach().abs().amax(dim=reduce_dims) / 127.0).clamp_min(1e-8)


def fold_bn(conv, bn):
    weight = conv.weight.detach().float().cpu()
    gamma = bn.weight.detach().float().cpu()
    beta = bn.bias.detach().float().cpu()
    mean = bn.running_mean.detach().float().cpu()
    var = bn.running_var.detach().float().cpu()
    bn_scale = gamma / torch.sqrt(var + bn.eps)
    folded_weight = weight * view_channel_vector(conv, weight, bn_scale)
    folded_bias = beta - mean * bn_scale
    return folded_weight, folded_bias


def quantize_weight(conv, folded_weight):
    scale = per_channel_scale(conv, folded_weight).cpu()
    q = (folded_weight / view_channel_vector(conv, folded_weight, scale)).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def kernel_tuple(conv):
    return tuple(int(v) for v in conv.kernel_size)


def stride_tuple(conv):
    return tuple(int(v) for v in conv.stride)


def padding_tuple(conv):
    padding = getattr(conv, 'padding', 0)
    if isinstance(padding, int):
        return (padding, padding, padding)
    return tuple(int(v) for v in padding)


def normalize_scale(scale, cout):
    if isinstance(scale, torch.Tensor):
        if scale.dim() == 0:
            return [float(scale)] * cout
        return [float(v) for v in scale.view(-1)]
    return [float(scale)] * cout


def load_activation_scales(path):
    rows = read_csv_rows(path)
    scales = {}
    manifest_rows = []
    for row in rows:
        name = row['name']
        signed = row['signed'] in ('True', 'true', '1')
        if not signed or int(row['zero_point']) != 0 or int(row['qmin']) != -127 or int(row['qmax']) != 127:
            raise RuntimeError(f'Activation quantization contract mismatch for {name}')
        scales[name] = float(row['scale'])
        manifest_rows.append({
            'activation_id': int(row['activation_id']),
            'name': name,
            'scale': float(row['scale']),
            'zero_point': 0,
            'qmin': -127,
            'qmax': 127,
            'seen': row.get('seen', 'True') in ('True', 'true', '1'),
        })
    missing = [key for key in ACT_KEYS if key not in scales]
    if missing:
        raise RuntimeError(f'Missing activation scales: {missing}')
    return scales, manifest_rows


def check_max_voxels(cfg, expected):
    for proc_cfg in cfg.DATA_CONFIG.DATA_PROCESSOR:
        if proc_cfg.NAME == 'transform_points_to_voxels':
            value = int(proc_cfg.MAX_NUMBER_OF_VOXELS['test'])
            if value != expected:
                raise RuntimeError(f'MAX_NUMBER_OF_VOXELS.test={value}, expected {expected}')
            return value
    raise RuntimeError('transform_points_to_voxels processor not found')


def choose_frame(repo_root, split, pool_size, seed):
    split_path = repo_root / 'data/kitti/ImageSets' / f'{split}.txt'
    frame_ids = [
        line.strip() for line in split_path.read_text(encoding='ascii').splitlines()
        if line.strip()
    ]
    if len(frame_ids) < pool_size:
        raise RuntimeError(f'{split_path} has only {len(frame_ids)} IDs, need {pool_size}')
    pool = frame_ids[:pool_size]
    frame_id = random.Random(seed).choice(pool)
    return split_path, pool, frame_id


def coord32_from_zyx(z, y, x, is_halo=False):
    x = int(x)
    y = int(y)
    z = int(z)
    if not (0 <= x < (1 << 11)):
        raise ValueError(f'x cannot be encoded in 11 bits: {x}')
    if not (0 <= y < (1 << 11)):
        raise ValueError(f'y cannot be encoded in 11 bits: {y}')
    if not (0 <= z < (1 << 6)):
        raise ValueError(f'z cannot be encoded in 6 bits: {z}')
    return (int(bool(is_halo)) << 31) | (z << 22) | (y << 11) | x


def pack_int8_tile(values):
    if len(values) > 8:
        raise ValueError('A feature tile can contain at most 8 lanes')
    word = 0
    for lane in range(8):
        value = int(values[lane]) if lane < len(values) else 0
        if not (-128 <= value <= 127):
            raise ValueError(f'INT8 lane out of range: {value}')
        word |= (value & 0xFF) << (8 * lane)
    return word


def pack_u64(word):
    return struct.pack('<Q', int(word) & 0xFFFFFFFFFFFFFFFF)


class ZoneSpec:
    def __init__(self, zone_id, start_dist, end_dist, block_size_xyz):
        self.zone_id = int(zone_id)
        self.start_dist = int(start_dist)
        self.end_dist = int(end_dist)
        self.block_size_xyz = tuple(int(v) for v in block_size_xyz)
        self.log2_block_size_xyz = tuple(int(math.log2(v)) for v in self.block_size_xyz)

    def to_manifest(self):
        return {
            'zone_id': self.zone_id,
            'start_dist': self.start_dist,
            'end_dist': self.end_dist,
            'block_size_xyz': list(self.block_size_xyz),
            'log2_block_size_xyz': list(self.log2_block_size_xyz),
        }


def load_zone_specs(path):
    specs = []
    for raw_line in Path(path).read_text(encoding='ascii').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        zone_text, dist_text, size_text = line.split(':')
        start_text, end_text = dist_text.split('-')
        sizes = tuple(int(part.strip()) for part in size_text.split(','))
        specs.append(ZoneSpec(int(zone_text), int(start_text), int(end_text), sizes))
    if not specs:
        raise RuntimeError(f'No zone specs loaded from {path}')
    return specs


def lookup_zone(specs, distance):
    for spec in specs:
        if spec.start_dist <= distance <= spec.end_dist:
            return spec
    raise ValueError(f'Distance {distance} is outside zone LUT range')


def chebyshev_distance(x, y, lidar_center_xy):
    return max(abs(int(x) - lidar_center_xy[0]), abs(int(y) - lidar_center_xy[1]))


def axis_boundary_flags(coord, log2_block_size, coord_max):
    mask = (1 << log2_block_size) - 1
    is_low = ((coord & mask) == 0) and coord != 0
    is_high = ((coord & mask) == mask) and coord != coord_max
    return is_low, is_high


def compute_block_key(x, y, z, zone_spec, lidar_center_xy):
    log2_bx, log2_by, log2_bz = zone_spec.log2_block_size_xyz
    block_x = (int(x) - lidar_center_xy[0]) >> log2_bx
    block_y = (int(y) - lidar_center_xy[1]) >> log2_by
    block_z = int(z) >> log2_bz
    return zone_spec.zone_id, block_x, block_y, block_z


def pack_block_id(block_key):
    zone_id, block_x, block_y, block_z = block_key
    return ((zone_id & 0x3) << 16) | ((block_x & 0x7F) << 9) | ((block_y & 0x7F) << 2) | (block_z & 0x3)


def iter_block_beats_for_voxel(z, y, x, grid_size_xyz, zone_specs, lidar_center_xy):
    nx, ny, nz = grid_size_xyz
    primary_spec = lookup_zone(zone_specs, chebyshev_distance(x, y, lidar_center_xy))
    yield compute_block_key(x, y, z, primary_spec, lidar_center_xy), False

    x_low, x_high = axis_boundary_flags(int(x), primary_spec.log2_block_size_xyz[0], nx - 1)
    y_low, y_high = axis_boundary_flags(int(y), primary_spec.log2_block_size_xyz[1], ny - 1)
    z_low, z_high = axis_boundary_flags(int(z), primary_spec.log2_block_size_xyz[2], nz - 1)
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
        halo_x = int(x)
        halo_y = int(y)
        halo_z = int(z)
        if halo_index & 0b001:
            halo_x += -1 if x_low else 1
        if halo_index & 0b010:
            halo_y += -1 if y_low else 1
        if halo_index & 0b100:
            halo_z += -1 if z_low else 1
        if not (0 <= halo_x < nx and 0 <= halo_y < ny and 0 <= halo_z < nz):
            continue
        halo_spec = lookup_zone(zone_specs, chebyshev_distance(halo_x, halo_y, lidar_center_xy))
        yield compute_block_key(halo_x, halo_y, halo_z, halo_spec, lidar_center_xy), True


def append_page(pages, page_addr, voxel_records, word_offset):
    payload = bytearray()
    seen = set()
    for coord32, qfeat in voxel_records:
        if coord32 in seen:
            raise RuntimeError('Duplicate coord32 in one input page')
        seen.add(coord32)
        payload += pack_u64(coord32)
        payload += pack_u64(pack_int8_tile(qfeat))
    pages.append({
        'page_addr': page_addr,
        'voxel_count': len(voxel_records),
        'word_offset': word_offset,
    })
    return bytes(payload)


def build_vfe_input_pages(coords_zyx, q_features, grid_size_xyz, zone_specs, lidar_center_xy):
    blocks = OrderedDict()
    for index, (z, y, x) in enumerate(coords_zyx.astype(np.int64)):
        for block_key, is_halo in iter_block_beats_for_voxel(z, y, x, grid_size_xyz, zone_specs, lidar_center_xy):
            block_id = pack_block_id(block_key)
            if block_id not in blocks:
                blocks[block_id] = {
                    'block_id': block_id,
                    'block_key': list(block_key),
                    'records': [],
                }
            coord32 = coord32_from_zyx(z, y, x, is_halo=is_halo)
            blocks[block_id]['records'].append((coord32, q_features[index, :4].tolist()))

    payload = bytearray()
    manifest_blocks = []
    page_addr = 0
    for block in blocks.values():
        pages = []
        current_page = []
        current_seen = set()
        for record in block['records']:
            coord32 = record[0]
            if len(current_page) == 64 or coord32 in current_seen:
                payload += append_page(pages, page_addr, current_page, len(payload) // 8)
                page_addr += 1
                current_page = []
                current_seen = set()
            current_page.append(record)
            current_seen.add(coord32)
        if current_page:
            payload += append_page(pages, page_addr, current_page, len(payload) // 8)
            page_addr += 1
        total_voxels = sum(page['voxel_count'] for page in pages)
        manifest_blocks.append({
            'block_id': block['block_id'],
            'block_key': block['block_key'],
            'total_voxel_count': total_voxels,
            'pages': pages,
        })

    return bytes(payload), manifest_blocks


def pack_weights_layer(weight_int8, cin, cout):
    weight = weight_int8.cpu().numpy().astype(np.int16)
    if weight.shape[0] != cout:
        raise RuntimeError(f'Unexpected weight Cout axis shape {weight.shape}, expected Cout={cout}')
    _, kz_count, ky_count, kx_count, cin_weight = weight.shape
    if cin_weight != cin:
        raise RuntimeError(f'Unexpected weight Cin={cin_weight}, expected {cin}')

    payload = bytearray()
    cin_tiles = math.ceil(cin / 8)
    cout_tiles = math.ceil(cout / 8)
    for kernel_idx in range(27):
        kx = kernel_idx // 9
        ky = (kernel_idx % 9) // 3
        kz = kernel_idx % 3
        kernel_present = kz < kz_count and ky < ky_count and kx < kx_count
        for cin_tile in range(cin_tiles):
            for cout_tile in range(cout_tiles):
                for co_lane in range(8):
                    co = cout_tile * 8 + co_lane
                    lanes = []
                    for ci_lane in range(8):
                        ci = cin_tile * 8 + ci_lane
                        value = 0
                        if kernel_present and co < cout and ci < cin:
                            value = int(weight[co, kz, ky, kx, ci])
                        lanes.append(value)
                    payload += pack_u64(pack_int8_tile(lanes))
    return bytes(payload)


def pack_params_layer(bias_int, shifts):
    payload = bytearray()
    cout = len(bias_int)
    if cout % 8 != 0:
        raise RuntimeError(f'Cout must be a multiple of 8 for params packing, got {cout}')
    for cout_tile in range(cout // 8):
        for pair_id in range(4):
            c0 = cout_tile * 8 + pair_id * 2
            c1 = c0 + 1
            b0 = int(bias_int[c0])
            b1 = int(bias_int[c1])
            s0 = int(shifts[c0])
            s1 = int(shifts[c1])
            if not (-32768 <= b0 <= 32767 and -32768 <= b1 <= 32767):
                raise RuntimeError(f'Bias out of INT16 range at channels {c0}/{c1}: {b0}/{b1}')
            if not (0 <= s0 <= 15 and 0 <= s1 <= 15):
                raise RuntimeError(f'Shift out of UINT4 range at channels {c0}/{c1}: {s0}/{s1}')
            carrier = (s0 & 0xF) | ((s1 & 0xF) << 4) | ((b0 & 0xFFFF) << 8) | ((b1 & 0xFFFF) << 24)
            payload += pack_u64(carrier)
    return bytes(payload)


def qparams_from_model(backbone, act_scales, hw_export_dir):
    params_csv_cache = {}
    qparams = []
    layer_manifest = []
    for layer_id, (name, conv, bn, relu, act_key) in enumerate(backbone.get_layer_modules()):
        rtl_cin, rtl_cout = RTL_SCHEDULE[layer_id]
        if int(conv.in_channels) != rtl_cin or int(conv.out_channels) != rtl_cout:
            raise RuntimeError(f'Layer {layer_id} channel mismatch')
        if bn is None:
            raise RuntimeError(f'Layer {layer_id} is missing BN')

        input_key = 'input' if layer_id == 0 else backbone.get_layer_modules()[layer_id - 1][4]
        sx = float(act_scales[input_key])
        sy = float(act_scales[act_key])
        folded_weight, folded_bias = fold_bn(conv, bn)
        weight_int8, weight_scale = quantize_weight(conv, folded_weight)
        if int(weight_int8.min()) < -127 or int(weight_int8.max()) > 127 or bool((weight_int8 == -128).any()):
            raise RuntimeError(f'Layer {layer_id} has invalid INT8 weight payload')
        sw_values = normalize_scale(weight_scale, int(conv.out_channels))

        bias_int = []
        shifts = []
        for channel, bias_fp32 in enumerate(folded_bias.tolist()):
            sw = sw_values[channel]
            denom = max(sx * sw, 1e-12)
            bias_value = int(round(float(bias_fp32) / denom))
            real_multiplier = max(sx * sw / max(sy, 1e-12), 1e-12)
            shift_value = max(0, int(round(-math.log(real_multiplier, 2))))
            bias_int.append(bias_value)
            shifts.append(shift_value)

        params_path = hw_export_dir / f'params_layer{layer_id:02d}.csv'
        params_rows = read_csv_rows(params_path)
        params_csv_cache[layer_id] = params_rows
        csv_bias = [int(row['bias_int']) for row in params_rows]
        csv_shift = [int(row['shift']) for row in params_rows]
        if csv_bias != bias_int or csv_shift != shifts:
            raise RuntimeError(f'Layer {layer_id} recomputed bias/shift does not match {params_path}')

        layer_q = {
            'layer_id': layer_id,
            'module_name': name,
            'conv_type': type(conv).__name__,
            'kernel': kernel_tuple(conv),
            'stride': stride_tuple(conv),
            'padding': padding_tuple(conv),
            'cin': int(conv.in_channels),
            'cout': int(conv.out_channels),
            'act_key': act_key,
            'input_act_key': input_key,
            'sx': sx,
            'sy': sy,
            'weight_int8': weight_int8.contiguous(),
            'weight_scale': [float(v) for v in sw_values],
            'bias_int': np.asarray(bias_int, dtype=np.int64),
            'shift': np.asarray(shifts, dtype=np.int64),
            'relu_en': relu is not None,
            'rounding_en': True,
        }
        qparams.append(layer_q)
        layer_manifest.append(layer_q)
    return qparams, layer_manifest


def generate_output_coords(input_coords, kernel, stride, padding, input_shape, conv_type):
    if conv_type == 'SubMConv3d':
        return np.asarray(input_coords, dtype=np.int64).copy(), tuple(input_shape)

    kz_count, ky_count, kx_count = kernel
    sz, sy, sx = stride
    pz, py, px = padding
    in_z, in_y, in_x = input_shape
    out_shape = (
        (in_z + 2 * pz - (kz_count - 1) - 1) // sz + 1,
        (in_y + 2 * py - (ky_count - 1) - 1) // sy + 1,
        (in_x + 2 * px - (kx_count - 1) - 1) // sx + 1,
    )
    out_set = set()
    coords = np.asarray(input_coords, dtype=np.int64)
    for kz in range(kz_count):
        dz = coords[:, 0] + pz - kz
        valid_z = (dz % sz) == 0
        oz = dz // sz
        for ky in range(ky_count):
            dy = coords[:, 1] + py - ky
            valid_y = (dy % sy) == 0
            oy = dy // sy
            for kx in range(kx_count):
                dx = coords[:, 2] + px - kx
                valid_x = (dx % sx) == 0
                ox = dx // sx
                valid = (
                    valid_z & valid_y & valid_x &
                    (oz >= 0) & (oz < out_shape[0]) &
                    (oy >= 0) & (oy < out_shape[1]) &
                    (ox >= 0) & (ox < out_shape[2])
                )
                if np.any(valid):
                    out_set.update((int(z), int(y), int(x)) for z, y, x in zip(oz[valid], oy[valid], ox[valid]))
    if not out_set:
        return np.zeros((0, 3), dtype=np.int64), out_shape
    return np.asarray(sorted(out_set), dtype=np.int64), out_shape


def sparse_int_layer(input_coords, input_features, input_shape, layer_q):
    kernel = layer_q['kernel']
    stride = layer_q['stride']
    padding = layer_q['padding']
    out_coords, out_shape = generate_output_coords(
        input_coords, kernel, stride, padding, input_shape, layer_q['conv_type']
    )
    cout = layer_q['cout']
    if out_coords.shape[0] == 0:
        return out_coords, np.zeros((0, cout), dtype=np.int16), out_shape

    input_index = {tuple(coord): idx for idx, coord in enumerate(np.asarray(input_coords, dtype=np.int64))}
    acc = np.zeros((out_coords.shape[0], cout), dtype=np.int64)
    weight = layer_q['weight_int8'].cpu().numpy().astype(np.int32)
    features = input_features.astype(np.int32, copy=False)
    kz_count, ky_count, kx_count = kernel
    sz, sy, sx = stride
    pz, py, px = padding
    for kz in range(kz_count):
        for ky in range(ky_count):
            for kx in range(kx_count):
                out_ids = []
                in_ids = []
                for out_idx, (oz, oy, ox) in enumerate(out_coords):
                    in_coord = (int(oz * sz + kz - pz), int(oy * sy + ky - py), int(ox * sx + kx - px))
                    in_idx = input_index.get(in_coord)
                    if in_idx is not None:
                        out_ids.append(out_idx)
                        in_ids.append(in_idx)
                if not out_ids:
                    continue
                w = weight[:, kz, ky, kx, :].T
                partial = features[np.asarray(in_ids, dtype=np.int64)] @ w
                acc[np.asarray(out_ids, dtype=np.int64)] += partial.astype(np.int64)

    shifted_input = acc + layer_q['bias_int'].reshape(1, -1)
    shifts = layer_q['shift'].reshape(1, -1)
    rounding = np.where(shifts > 0, np.left_shift(1, shifts - 1), 0)
    shifted = np.right_shift(shifted_input + rounding, shifts)
    if layer_q['relu_en']:
        shifted = np.maximum(shifted, 0)
    q = np.clip(shifted, 0, 127).astype(np.int16)
    return out_coords, q, out_shape


def pack_ofm_records(coords, features):
    order = np.argsort(np.asarray([coord32_from_zyx(z, y, x, False) for z, y, x in coords], dtype=np.uint32))
    payload = bytearray()
    for idx in order:
        z, y, x = coords[idx]
        payload += pack_u64(coord32_from_zyx(z, y, x, False))
        for start in range(0, features.shape[1], 8):
            payload += pack_u64(pack_int8_tile(features[idx, start:start + 8].tolist()))
    return bytes(payload), order


def build_binaries(qparams, input_coords, input_qfeatures, input_shape):
    weights_payload = bytearray()
    params_payload = bytearray()
    ofm_payload = bytearray()
    layer_entries = []
    coords = np.asarray(input_coords, dtype=np.int64)
    features = np.asarray(input_qfeatures[:, :4], dtype=np.int16)
    shape = tuple(int(v) for v in input_shape)

    for layer_q in qparams:
        layer_id = layer_q['layer_id']
        weight_offset = len(weights_payload) // 8
        layer_weights = pack_weights_layer(layer_q['weight_int8'], layer_q['cin'], layer_q['cout'])
        weights_payload += layer_weights

        param_offset = len(params_payload) // 8
        layer_params = pack_params_layer(layer_q['bias_int'], layer_q['shift'])
        params_payload += layer_params

        coords, features, shape = sparse_int_layer(coords, features, shape, layer_q)
        if features.size and (int(features.min()) < 0 or int(features.max()) > 127):
            raise RuntimeError(f'Layer {layer_id} OFM feature outside [0,127]')
        if len({tuple(coord) for coord in coords.tolist()}) != coords.shape[0]:
            raise RuntimeError(f'Layer {layer_id} produced duplicate coordinates')
        layer_ofm, _ = pack_ofm_records(coords, features)
        ofm_offset = len(ofm_payload) // 8
        ofm_payload += layer_ofm

        layer_entries.append({
            'layer_id': layer_id,
            'module_name': layer_q['module_name'],
            'conv_kind': layer_q['conv_type'],
            'kernel': list(layer_q['kernel']),
            'stride': list(layer_q['stride']),
            'padding': list(layer_q['padding']),
            'cin': layer_q['cin'],
            'cout': layer_q['cout'],
            'relu': bool(layer_q['relu_en']),
            'rounding': bool(layer_q['rounding_en']),
            'input_activation': {
                'name': layer_q['input_act_key'],
                'scale': layer_q['sx'],
            },
            'output_activation': {
                'name': layer_q['act_key'],
                'scale': layer_q['sy'],
            },
            'weight_scale': layer_q['weight_scale'],
            'weights': {
                'word_offset': weight_offset,
                'word_length': len(layer_weights) // 8,
                'record_count': len(layer_weights) // 8,
                'sha256': sha256_bytes(layer_weights),
            },
            'params': {
                'word_offset': param_offset,
                'word_length': len(layer_params) // 8,
                'record_count': len(layer_params) // 8,
                'sha256': sha256_bytes(layer_params),
            },
            'ofm_golden': {
                'word_offset': ofm_offset,
                'word_length': len(layer_ofm) // 8,
                'record_count': int(coords.shape[0]),
                'sha256': sha256_bytes(layer_ofm),
            },
        })
        print(
            f'Layer {layer_id:02d} {layer_q["module_name"]}: '
            f'coords={coords.shape[0]} cout={features.shape[1]}'
        )

    return bytes(weights_payload), bytes(params_payload), bytes(ofm_payload), layer_entries


def write_binary(path, payload):
    if len(payload) % 8 != 0:
        raise RuntimeError(f'{path.name} is not 64-bit word aligned')
    path.write_bytes(payload)
    return {
        'path': path.name,
        'bytes': len(payload),
        'word_length': len(payload) // 8,
        'sha256': sha256_file(path),
    }


def build_manifest(args, package_dir, sample, provenance, quantization, partition_geometry,
                   input_pages, layer_entries, file_entries):
    return {
        'format_version': 'second_rtl_golden_v1',
        'word_bits': 64,
        'byte_order': 'little',
        'sample': sample,
        'provenance': provenance,
        'quantization': quantization,
        'partition_geometry': partition_geometry,
        'input_pages': input_pages,
        'files': file_entries,
        'layers': layer_entries,
        'rtl_contract': {
            'cin_tile': 8,
            'cout_tile': 8,
            'mac_lane': 4,
            'bias_bits': 16,
            'shift_bits': 4,
            'weight_kernel_order': 'x*9+y*3+z',
            'layer11_status': [
                'conv_out.0 is 3x1x1 with z-only stride 2; current RTL top-level semantics are a known blocker.',
                'conv_out.0 exports all 128 Cout channels; RTL weight replay for 16 Cout tiles is a known blocker.',
            ],
        },
    }


def write_report(path, manifest, selected_index, actual_voxels, max_voxels):
    sample = manifest['sample']
    lines = [
        '# SECOND RTL Golden Export Report',
        '',
        f'- Selected split: `{sample["split"]}`',
        f'- Selected sample ID: `{sample["sample_id"]}`',
        f'- Selected index in first-100 pool: `{selected_index}`',
        f'- Random seed: `{sample["random_seed"]}`',
        f'- Raw point cloud: `{sample["point_cloud_relpath"]}`',
        f'- Actual voxel count: `{actual_voxels}`',
        f'- MAX_NUMBER_OF_VOXELS: `{max_voxels}`',
        f'- Hit voxel cap: `{actual_voxels >= max_voxels}`',
        f'- Package directory: `{path.parent}`',
        '',
        '## Files',
    ]
    for name, entry in manifest['files'].items():
        lines.append(f'- `{name}`: {entry["bytes"]} bytes, sha256 `{entry["sha256"]}`')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    exporter_relpath = Path(__file__).resolve().relative_to(repo_root)
    ensure_project_imports(repo_root)

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    from pcdet.datasets.dataset import DatasetTemplate
    from pcdet.models import build_network

    os.chdir(repo_root / 'tools')
    cfg_from_yaml_file(args.cfg_file, cfg)
    check_max_voxels(cfg, args.max_voxels)

    hw_export_dir = Path(args.hw_export_dir)
    act_scales, act_manifest_rows = load_activation_scales(hw_export_dir / 'activation_scales.csv')
    split_path, frame_pool, frame_id = choose_frame(repo_root, args.split, args.frame_pool_size, args.random_seed)
    selected_index = frame_pool.index(frame_id)
    lidar_relpath = Path('data/kitti/training/velodyne') / f'{frame_id}.bin'
    lidar_path = repo_root / lidar_relpath
    if not lidar_path.exists():
        raise FileNotFoundError(lidar_path)

    output_root = Path(args.output_root)
    package_dir = output_root / f'second_{args.split}_{frame_id}_golden'
    if package_dir.exists() and not args.overwrite:
        raise RuntimeError(f'Output package already exists: {package_dir}; pass --overwrite to replace files')
    package_dir.mkdir(parents=True, exist_ok=True)
    log_path = package_dir / 'export.log'
    logger = make_logger(log_path)

    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=repo_root / 'data/kitti',
        logger=logger,
    )
    info_by_id = {info['point_cloud']['lidar_idx']: idx for idx, info in enumerate(dataset.kitti_infos)}
    if frame_id not in info_by_id:
        raise RuntimeError(f'Selected frame {frame_id} was not found in KITTI infos')
    sample_dict = dataset[info_by_id[frame_id]]
    batch_dict = DatasetTemplate.collate_batch([sample_dict])
    batch_dict['voxels'] = torch.from_numpy(batch_dict['voxels']).float()
    batch_dict['voxel_num_points'] = torch.from_numpy(batch_dict['voxel_num_points']).float()
    batch_dict['voxel_coords'] = torch.from_numpy(batch_dict['voxel_coords']).int()

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.eval()
    with torch.no_grad():
        batch_dict = model.vfe(batch_dict)

    voxel_features = batch_dict['voxel_features'].cpu().numpy()
    voxel_coords_bzyx = batch_dict['voxel_coords'].cpu().numpy().astype(np.int64)
    voxel_coords_zyx = voxel_coords_bzyx[:, 1:4]
    actual_voxel_count = int(voxel_coords_zyx.shape[0])
    if actual_voxel_count > args.max_voxels:
        raise RuntimeError(f'Actual voxel count {actual_voxel_count} exceeds max {args.max_voxels}')

    input_scale = act_scales['input']
    input_qfeatures = np.rint(voxel_features / input_scale).clip(-127, 127).astype(np.int16)
    if np.any(input_qfeatures == -128):
        raise RuntimeError('VFE input quantized payload contains -128')

    backbone = model.backbone_3d
    qparams, _ = qparams_from_model(backbone, act_scales, hw_export_dir)

    lidar_center_xy = tuple(int(part.strip()) for part in args.lidar_center.split(','))
    if len(lidar_center_xy) != 2:
        raise RuntimeError('--lidar_center must be x,y')
    grid_size_xyz = tuple(int(v) for v in dataset.grid_size.tolist())
    sparse_shape_zyx = tuple(int(v) for v in backbone.sparse_shape)
    zone_specs = load_zone_specs(args.zone_lut)
    vfe_payload, input_blocks = build_vfe_input_pages(
        voxel_coords_zyx, input_qfeatures, grid_size_xyz, zone_specs, lidar_center_xy
    )

    weights_payload, params_payload, ofm_payload, layer_entries = build_binaries(
        qparams, voxel_coords_zyx, input_qfeatures, sparse_shape_zyx
    )

    file_entries = {
        'vfe_input_pages.bin': write_binary(package_dir / 'vfe_input_pages.bin', vfe_payload),
        'weights.bin': write_binary(package_dir / 'weights.bin', weights_payload),
        'params.bin': write_binary(package_dir / 'params.bin', params_payload),
        'ofm_golden.bin': write_binary(package_dir / 'ofm_golden.bin', ofm_payload),
    }

    sample = {
        'split': args.split,
        'sample_id': frame_id,
        'frame_pool': {
            'source': str(split_path.relative_to(repo_root)),
            'pool_size': args.frame_pool_size,
            'selected_index': selected_index,
        },
        'random_seed': args.random_seed,
        'point_cloud_relpath': str(lidar_relpath),
        'point_cloud_sha256': sha256_file(lidar_path),
        'max_number_of_voxels': args.max_voxels,
        'actual_voxel_count': actual_voxel_count,
        'hit_voxel_cap': actual_voxel_count >= args.max_voxels,
        'fov_points_only': bool(cfg.DATA_CONFIG.FOV_POINTS_ONLY),
    }
    provenance = {
        'qat_checkpoint': str(Path(args.ckpt).relative_to(repo_root)),
        'qat_checkpoint_sha256': sha256_file(args.ckpt),
        'config': str(Path(args.cfg_file).relative_to(repo_root)),
        'config_sha256': sha256_file(args.cfg_file),
        'exporter': str(exporter_relpath),
        'activation_scales_csv': str((hw_export_dir / 'activation_scales.csv').relative_to(repo_root)),
        'activation_scales_sha256': sha256_file(hw_export_dir / 'activation_scales.csv'),
        'weight_scales_csv': str((hw_export_dir / 'weight_scales.csv').relative_to(repo_root)),
        'weight_scales_sha256': sha256_file(hw_export_dir / 'weight_scales.csv'),
        'layer_inventory_csv': str((hw_export_dir / 'layer_inventory.csv').relative_to(repo_root)),
        'layer_inventory_sha256': sha256_file(hw_export_dir / 'layer_inventory.csv'),
        'zone_lut': str(Path(args.zone_lut).relative_to(repo_root)),
        'zone_lut_sha256': sha256_file(args.zone_lut),
    }
    quantization = {
        'signed': True,
        'zero_point': 0,
        'qmin': -127,
        'qmax': 127,
        'activations': act_manifest_rows,
    }
    partition_geometry = {
        'mode': 'rtl_unfixed_zone',
        'grid_size_xyz': list(grid_size_xyz),
        'sparse_shape_zyx': list(sparse_shape_zyx),
        'lidar_center_xy': list(lidar_center_xy),
        'zone_specs': [spec.to_manifest() for spec in zone_specs],
        'coord32': {
            'x_bits': [0, 10],
            'y_bits': [11, 21],
            'z_bits': [22, 27],
            'reserved_bits': [28, 30],
            'is_halo_bit': 31,
        },
    }
    input_pages = {
        'file': 'vfe_input_pages.bin',
        'block_count': len(input_blocks),
        'total_records': sum(block['total_voxel_count'] for block in input_blocks),
        'unique_source_voxel_count': actual_voxel_count,
        'blocks': input_blocks,
    }
    manifest = build_manifest(
        args, package_dir, sample, provenance, quantization, partition_geometry,
        input_pages, layer_entries, file_entries
    )
    manifest_path = package_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + '\n', encoding='utf-8')
    report_path = package_dir / 'export_report.md'
    write_report(report_path, manifest, selected_index, actual_voxel_count, args.max_voxels)
    (package_dir / 'export_report.json').write_text(
        json.dumps({
            'package_dir': str(package_dir),
            'sample_id': frame_id,
            'actual_voxel_count': actual_voxel_count,
            'max_number_of_voxels': args.max_voxels,
            'hit_voxel_cap': actual_voxel_count >= args.max_voxels,
            'manifest_sha256': sha256_file(manifest_path),
        }, indent=2) + '\n',
        encoding='utf-8',
    )

    print(json.dumps({
        'package_dir': str(package_dir),
        'sample_id': frame_id,
        'actual_voxel_count': actual_voxel_count,
        'max_number_of_voxels': args.max_voxels,
        'manifest': str(manifest_path),
    }, indent=2))


if __name__ == '__main__':
    main()

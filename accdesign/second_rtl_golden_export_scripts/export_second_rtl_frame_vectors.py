#!/usr/bin/env python3
"""Export slim per-frame SECOND RTL golden vectors.

Each frame package contains only the frame-varying RTL input and OFM golden.
Weights, params, scales, and layer topology stay in the existing model package
(second_val_000216_golden).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import export_second_rtl_golden as golden


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    default_run = repo_root / 'output/kitti_models/second_hw_qat/hw_qat_10ep'
    parser = argparse.ArgumentParser(description='Export slim per-frame SECOND RTL golden vectors')
    parser.add_argument('--cfg_file', default=str(repo_root / 'tools/cfgs/kitti_models/second_hw_qat.yaml'))
    parser.add_argument('--ckpt', default=str(default_run / 'ckpt/checkpoint_epoch_10.pth'))
    parser.add_argument('--hw_export_dir', default=str(default_run / 'hw_export'))
    parser.add_argument('--split', default='train', choices=['train', 'val', 'trainval'])
    parser.add_argument('--num_frames', type=int, default=20)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--id_start', type=int, default=None, help='Inclusive consecutive filename ID, e.g. 0 for 000000.bin')
    parser.add_argument('--id_end', type=int, default=None, help='Inclusive consecutive filename ID, e.g. 19 for 000019.bin')
    parser.add_argument(
        '--model_package',
        default=str(repo_root / 'accdesign/second_rtl_golden_packages/second_val_000216_golden'),
    )
    parser.add_argument(
        '--output_root',
        default=str(repo_root / 'accdesign/second_rtl_golden_packages/frames'),
    )
    parser.add_argument('--max_voxels', type=int, default=15000)
    parser.add_argument('--overwrite', action='store_true', default=False)
    return parser.parse_args()


def load_imageset_membership(repo_root):
    membership = {}
    for split_name in ('train', 'val'):
        split_path = repo_root / 'data/kitti/ImageSets' / f'{split_name}.txt'
        for line in split_path.read_text(encoding='ascii').splitlines():
            frame_id = line.strip()
            if frame_id:
                membership[frame_id] = split_name
    return membership


def load_frame_ids(repo_root, args):
    if args.id_start is None and args.id_end is None:
        split_path = repo_root / 'data/kitti/ImageSets' / f'{args.split}.txt'
        frame_ids = [
            line.strip() for line in split_path.read_text(encoding='ascii').splitlines()
            if line.strip()
        ]
        selected = frame_ids[args.start_index:args.start_index + args.num_frames]
        if len(selected) != args.num_frames:
            raise RuntimeError(
                f'{split_path} has {len(frame_ids)} IDs; cannot take {args.num_frames} from index {args.start_index}'
            )
        return {
            'selection': 'imageset_prefix',
            'split_path': split_path,
            'frame_ids': selected,
            'start_index': args.start_index,
        }
    if args.id_start is None or args.id_end is None:
        raise RuntimeError('Both --id_start and --id_end are required for consecutive filename selection')
    if args.id_end < args.id_start:
        raise RuntimeError(f'id_end {args.id_end} is before id_start {args.id_start}')
    selected = [f'{frame_idx:06d}' for frame_idx in range(args.id_start, args.id_end + 1)]
    return {
        'selection': 'consecutive_filename',
        'split_path': repo_root / 'data/kitti/training/velodyne',
        'frame_ids': selected,
        'start_index': args.id_start,
    }


def attach_trainval_infos(dataset, repo_root):
    import pickle

    infos = []
    seen = set()
    for name in ('kitti_infos_train.pkl', 'kitti_infos_val.pkl'):
        path = repo_root / 'data/kitti' / name
        with path.open('rb') as handle:
            for info in pickle.load(handle):
                lidar_idx = info['point_cloud']['lidar_idx']
                if lidar_idx in seen:
                    continue
                seen.add(lidar_idx)
                infos.append(info)
    dataset.kitti_infos = infos
    return {info['point_cloud']['lidar_idx']: idx for idx, info in enumerate(infos)}


def model_package_ref(repo_root, model_package, ckpt):
    model_dir = Path(model_package).resolve()
    required = ['manifest.json', 'weights.bin', 'params.bin']
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f'Model package {model_dir} is missing {missing}')
    rel_dir = model_dir.relative_to(repo_root)
    model_manifest = json.loads((model_dir / 'manifest.json').read_text(encoding='utf-8'))
    return {
        'path': str(rel_dir),
        'format_version': model_manifest.get('format_version'),
        'manifest_sha256': golden.sha256_file(model_dir / 'manifest.json'),
        'weights_sha256': golden.sha256_file(model_dir / 'weights.bin'),
        'params_sha256': golden.sha256_file(model_dir / 'params.bin'),
        'qat_checkpoint': str(Path(ckpt).resolve().relative_to(repo_root)),
        'qat_checkpoint_sha256': golden.sha256_file(ckpt),
    }


def pack_coord_keys(coords):
    packed = np.asarray(coords, dtype=np.int64)
    return (packed[:, 0] << 32) | (packed[:, 1] << 16) | packed[:, 2]


def sparse_int_layer(input_coords, input_features, input_shape, layer_q):
    """Bit-exact with export_second_rtl_golden.sparse_int_layer, vectorized gather."""
    kernel = layer_q['kernel']
    stride = layer_q['stride']
    padding = layer_q['padding']
    out_coords, out_shape = golden.generate_output_coords(
        input_coords, kernel, stride, padding, input_shape, layer_q['conv_type']
    )
    cout = layer_q['cout']
    if out_coords.shape[0] == 0:
        return out_coords, np.zeros((0, cout), dtype=np.int16), out_shape

    in_coords = np.asarray(input_coords, dtype=np.int64)
    in_keys = pack_coord_keys(in_coords)
    order = np.argsort(in_keys, kind='mergesort')
    sorted_keys = in_keys[order]
    acc = np.zeros((out_coords.shape[0], cout), dtype=np.int64)
    weight = layer_q['weight_int8'].cpu().numpy().astype(np.int32)
    features = input_features.astype(np.int32, copy=False)
    kz_count, ky_count, kx_count = kernel
    sz, sy, sx = stride
    pz, py, px = padding
    out_z = out_coords[:, 0]
    out_y = out_coords[:, 1]
    out_x = out_coords[:, 2]
    key_count = sorted_keys.shape[0]
    for kz in range(kz_count):
        for ky in range(ky_count):
            for kx in range(kx_count):
                query = pack_coord_keys(np.stack((
                    out_z * sz + kz - pz,
                    out_y * sy + ky - py,
                    out_x * sx + kx - px,
                ), axis=1))
                pos = np.searchsorted(sorted_keys, query)
                valid = pos < key_count
                if not np.any(valid):
                    continue
                pos_valid = pos[valid]
                found_mask = np.zeros(out_coords.shape[0], dtype=bool)
                found_mask[valid] = sorted_keys[pos_valid] == query[valid]
                if not np.any(found_mask):
                    continue
                out_ids = np.nonzero(found_mask)[0]
                in_ids = order[pos[found_mask]]
                partial = features[in_ids] @ weight[:, kz, ky, kx, :].T
                acc[out_ids] += partial.astype(np.int64)

    shifted_input = acc + layer_q['bias_int'].reshape(1, -1)
    shifts = layer_q['shift'].reshape(1, -1)
    rounding = np.where(shifts > 0, np.left_shift(1, shifts - 1), 0)
    shifted = np.right_shift(shifted_input + rounding, shifts)
    if layer_q['relu_en']:
        shifted = np.maximum(shifted, 0)
    q = np.clip(shifted, 0, 127).astype(np.int16)
    return out_coords, q, out_shape


def build_ofm_payload(qparams, input_coords, input_qfeatures, input_shape):
    ofm_payload = bytearray()
    layer_entries = []
    coords = np.asarray(input_coords, dtype=np.int64)
    features = np.asarray(input_qfeatures[:, :4], dtype=np.int16)
    shape = tuple(int(v) for v in input_shape)

    for layer_q in qparams:
        layer_id = layer_q['layer_id']
        coords, features, shape = sparse_int_layer(coords, features, shape, layer_q)
        if features.size and (int(features.min()) < 0 or int(features.max()) > 127):
            raise RuntimeError(f'Layer {layer_id} OFM feature outside [0,127]')
        if len({tuple(coord) for coord in coords.tolist()}) != coords.shape[0]:
            raise RuntimeError(f'Layer {layer_id} produced duplicate coordinates')
        layer_ofm, _ = golden.pack_ofm_records(coords, features)
        ofm_offset = len(ofm_payload) // 8
        ofm_payload += layer_ofm
        layer_entries.append({
            'layer_id': layer_id,
            'ofm_golden': {
                'word_offset': ofm_offset,
                'word_length': len(layer_ofm) // 8,
                'record_count': int(coords.shape[0]),
                'sha256': golden.sha256_bytes(layer_ofm),
            },
        })
        print(
            f'  layer {layer_id:02d} {layer_q["module_name"]}: '
            f'coords={coords.shape[0]} cout={features.shape[1]}'
        )
    return bytes(ofm_payload), layer_entries


def export_one_frame(
    frame_id,
    split_index,
    args,
    repo_root,
    selection,
    dataset,
    model,
    qparams,
    act_scales,
    model_ref,
    info_by_id,
    imageset_membership,
):
    from pcdet.datasets.dataset import DatasetTemplate

    lidar_relpath = Path('data/kitti/training/velodyne') / f'{frame_id}.bin'
    lidar_path = repo_root / lidar_relpath
    if not lidar_path.exists():
        raise FileNotFoundError(lidar_path)
    if frame_id not in info_by_id:
        raise RuntimeError(f'Frame {frame_id} was not found in KITTI infos')

    package_dir = Path(args.output_root) / f'second_{args.split}_{frame_id}'
    if package_dir.exists() and not args.overwrite:
        raise RuntimeError(f'Output package already exists: {package_dir}; pass --overwrite to replace files')
    package_dir.mkdir(parents=True, exist_ok=True)

    sample_dict = dataset[info_by_id[frame_id]]
    batch_dict = DatasetTemplate.collate_batch([sample_dict])
    batch_dict['voxels'] = torch.from_numpy(batch_dict['voxels']).float()
    batch_dict['voxel_num_points'] = torch.from_numpy(batch_dict['voxel_num_points']).float()
    batch_dict['voxel_coords'] = torch.from_numpy(batch_dict['voxel_coords']).int()
    with torch.no_grad():
        batch_dict = model.vfe(batch_dict)

    voxel_features = batch_dict['voxel_features'].cpu().numpy()
    voxel_coords_zyx = batch_dict['voxel_coords'].cpu().numpy().astype(np.int64)[:, 1:4]
    actual_voxel_count = int(voxel_coords_zyx.shape[0])
    if actual_voxel_count > args.max_voxels:
        raise RuntimeError(f'Frame {frame_id} voxel count {actual_voxel_count} exceeds max {args.max_voxels}')

    input_qfeatures = np.rint(voxel_features / act_scales['input']).clip(-127, 127).astype(np.int16)
    if np.any(input_qfeatures == -128):
        raise RuntimeError(f'Frame {frame_id} VFE input quantized payload contains -128')

    sparse_shape_zyx = tuple(int(v) for v in model.backbone_3d.sparse_shape)
    vfe_payload = golden.build_raw_vfe_voxel_stream(voxel_coords_zyx, input_qfeatures)
    ofm_payload, layer_entries = build_ofm_payload(
        qparams, voxel_coords_zyx, input_qfeatures, sparse_shape_zyx
    )

    file_entries = {
        'raw_vfe_voxel_stream.bin': golden.write_binary(package_dir / 'raw_vfe_voxel_stream.bin', vfe_payload),
        'ofm_golden.bin': golden.write_binary(package_dir / 'ofm_golden.bin', ofm_payload),
    }
    manifest = {
        'format_version': 'second_rtl_golden_v2',
        'package_kind': 'frame_vector',
        'word_bits': 64,
        'byte_order': 'little',
        'model_package': model_ref,
        'sample': {
            'split': args.split,
            'sample_id': frame_id,
            'split_index': split_index,
            'selection': selection['selection'],
            'source': str(selection['split_path'].relative_to(repo_root)),
            'kitti_imageset': imageset_membership.get(frame_id),
            'point_cloud_relpath': str(lidar_relpath),
            'point_cloud_sha256': golden.sha256_file(lidar_path),
            'max_number_of_voxels': args.max_voxels,
            'actual_voxel_count': actual_voxel_count,
            'hit_voxel_cap': actual_voxel_count >= args.max_voxels,
            'fov_points_only': bool(dataset.dataset_cfg.FOV_POINTS_ONLY),
        },
        'raw_vfe_voxel_stream': {
            'file': 'raw_vfe_voxel_stream.bin',
            'record_count': actual_voxel_count,
            'record_bytes': 16,
            'word_length_per_record': 2,
            'total_bytes': len(vfe_payload),
            'word_length': len(vfe_payload) // 8,
            'sha256': file_entries['raw_vfe_voxel_stream.bin']['sha256'],
            'sort_order': 'vfe_output_order',
            'owner_voxel_only': True,
            'halo_expanded': False,
            'bm_page_preallocated': False,
        },
        'files': file_entries,
        'layers': layer_entries,
        'exporter': str(Path(__file__).resolve().relative_to(repo_root)),
    }
    manifest_path = package_dir / 'frame_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + '\n', encoding='utf-8')
    return {
        'sample_id': frame_id,
        'split_index': split_index,
        'package_dir': str(package_dir.relative_to(repo_root)),
        'actual_voxel_count': actual_voxel_count,
        'hit_voxel_cap': actual_voxel_count >= args.max_voxels,
        'raw_vfe_voxel_stream_bytes': len(vfe_payload),
        'ofm_golden_bytes': len(ofm_payload),
        'layer_ofm_counts': [entry['ofm_golden']['record_count'] for entry in layer_entries],
        'frame_manifest_sha256': golden.sha256_file(manifest_path),
        'raw_vfe_voxel_stream_sha256': file_entries['raw_vfe_voxel_stream.bin']['sha256'],
        'ofm_golden_sha256': file_entries['ofm_golden.bin']['sha256'],
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    golden.ensure_project_imports(repo_root)

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    from pcdet.models import build_network

    os.chdir(repo_root / 'tools')
    cfg_from_yaml_file(args.cfg_file, cfg)
    golden.check_max_voxels(cfg, args.max_voxels)

    selection = load_frame_ids(repo_root, args)
    frame_ids = selection['frame_ids']
    imageset_membership = load_imageset_membership(repo_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / 'export.log'
    logger = golden.make_logger(log_path)
    model_ref = model_package_ref(repo_root, args.model_package, args.ckpt)
    act_scales, _ = golden.load_activation_scales(Path(args.hw_export_dir) / 'activation_scales.csv')

    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=repo_root / 'data/kitti',
        logger=logger,
    )
    info_by_id = attach_trainval_infos(dataset, repo_root)
    missing = [frame_id for frame_id in frame_ids if frame_id not in info_by_id]
    if missing:
        raise RuntimeError(f'Frames not found in KITTI train/val infos: {missing}')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.eval()
    qparams, _ = golden.qparams_from_model(model.backbone_3d, act_scales, Path(args.hw_export_dir))

    summaries = []
    for offset, frame_id in enumerate(frame_ids):
        split_index = selection['start_index'] + offset
        print(f'[{offset + 1}/{len(frame_ids)}] exporting {args.split}/{frame_id}')
        logger.info('Exporting %s/%s (%d/%d)', args.split, frame_id, offset + 1, len(frame_ids))
        summary = export_one_frame(
            frame_id,
            split_index,
            args,
            repo_root,
            selection,
            dataset,
            model,
            qparams,
            act_scales,
            model_ref,
            info_by_id,
            imageset_membership,
        )
        summaries.append(summary)
        print(json.dumps({k: summary[k] for k in ('sample_id', 'actual_voxel_count', 'hit_voxel_cap')}, indent=2))

    index = {
        'format_version': 'second_rtl_golden_v2',
        'package_kind': 'frame_vector_index',
        'split': args.split,
        'selection': selection['selection'],
        'source': str(selection['split_path'].relative_to(repo_root)),
        'id_start': args.id_start,
        'id_end': args.id_end,
        'start_index': selection['start_index'],
        'num_frames': len(frame_ids),
        'model_package': model_ref,
        'output_root': str(output_root.relative_to(repo_root)),
        'frames': summaries,
    }
    index_path = output_root / 'index.json'
    index_path.write_text(json.dumps(index, indent=2, sort_keys=False) + '\n', encoding='utf-8')
    print(json.dumps({
        'index': str(index_path),
        'num_frames': len(summaries),
        'sample_ids': [item['sample_id'] for item in summaries],
    }, indent=2))


if __name__ == '__main__':
    main()

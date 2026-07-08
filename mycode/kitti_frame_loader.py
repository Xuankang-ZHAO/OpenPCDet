"""Shared KITTI frame loading helpers for mycode analysis scripts."""

import random
import sys
from pathlib import Path

import numpy as np


def resolve_project_root():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


DEFAULT_KITTI_CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']


def get_dataset_cfg(cfg_obj):
    if hasattr(cfg_obj, 'DATA_CONFIG'):
        return cfg_obj.DATA_CONFIG
    return cfg_obj


def get_class_names(cfg_obj):
    if hasattr(cfg_obj, 'CLASS_NAMES'):
        return list(cfg_obj.CLASS_NAMES)
    return DEFAULT_KITTI_CLASS_NAMES.copy()


def resolve_data_mode(cfg_obj, requested_mode):
    if requested_mode != 'auto':
        return requested_mode
    dataset_cfg = get_dataset_cfg(cfg_obj)
    return 'kitti' if dataset_cfg.get('DATASET', '') == 'KittiDataset' else 'raw'


def normalize_frame_token(token):
    return Path(token).stem if token.endswith('.bin') else token


def normalize_voxel_coords(coords):
    if coords is None:
        return None

    if isinstance(coords, list):
        coords = coords[0]

    if hasattr(coords, 'cpu'):
        coords = coords.cpu().numpy()

    coords = coords.astype(np.int64)
    if coords.ndim == 2 and coords.shape[1] == 4:
        coords = coords[:, 1:4]
    return coords


def add_data_mode_args(parser, default_mode='auto'):
    parser.add_argument('--kitti_root', type=str, default='data/kitti')
    parser.add_argument(
        '--data_mode',
        type=str,
        default=default_mode,
        choices=['auto', 'kitti', 'raw'],
        help='auto uses KittiDataset for KITTI configs so FOV filtering matches normal inference.',
    )


def build_kitti_dataset(cfg_obj, project_root, kitti_root, logger):
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset

    return KittiDataset(
        dataset_cfg=get_dataset_cfg(cfg_obj),
        class_names=get_class_names(cfg_obj),
        training=False,
        root_path=project_root / kitti_root,
        logger=logger,
    )


def build_template_dataset(cfg_obj):
    from pcdet.datasets.dataset import DatasetTemplate

    return DatasetTemplate(
        dataset_cfg=get_dataset_cfg(cfg_obj),
        class_names=get_class_names(cfg_obj),
        training=False,
    )


def resolve_frame_ids_from_list(list_file):
    with open(list_file, 'r', encoding='ascii') as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def choose_frame_ids(velodyne_dir, list_file=None):
    if list_file:
        return resolve_frame_ids_from_list(list_file)
    return sorted(path.stem for path in Path(velodyne_dir).glob('*.bin'))


def choose_raw_frame_paths(velodyne_dir, frame_id, seed, num_frames):
    velodyne_path = Path(velodyne_dir)
    if not velodyne_path.exists():
        raise FileNotFoundError(f'Velodyne directory not found: {velodyne_path}')

    if frame_id:
        frame_tokens = [token.strip() for token in frame_id.split(',') if token.strip()]
        frames = []
        for token in frame_tokens:
            frame_name = token if token.endswith('.bin') else f'{token}.bin'
            frame_path = velodyne_path / frame_name
            if not frame_path.exists():
                raise FileNotFoundError(f'Frame not found: {frame_path}')
            frames.append(frame_path)
        if num_frames != len(frames):
            raise ValueError('When --frame_id is provided, --num_frames must match the number of frame ids.')
        return frames

    candidates = sorted(velodyne_path.glob('*.bin'))
    if not candidates:
        raise RuntimeError(f'No .bin files found in {velodyne_path}')
    if num_frames > len(candidates):
        raise ValueError(f'Requested {num_frames} frames, but only found {len(candidates)} point clouds.')

    rng = random.Random(seed)
    return rng.sample(candidates, num_frames)


def choose_kitti_frame_ids(dataset, frame_id, seed, num_frames):
    available_ids = [str(info['point_cloud']['lidar_idx']) for info in dataset.kitti_infos]
    available = set(available_ids)

    if frame_id:
        frames = [normalize_frame_token(token.strip()) for token in frame_id.split(',') if token.strip()]
        missing = [frame for frame in frames if frame not in available]
        if missing:
            raise FileNotFoundError(f'Frame id(s) not found in KITTI split: {missing}')
        if num_frames != len(frames):
            raise ValueError('When --frame_id is provided, --num_frames must match the number of frame ids.')
        return frames

    if num_frames > len(available_ids):
        raise ValueError(f'Requested {num_frames} frames, but only found {len(available_ids)} KITTI samples.')
    rng = random.Random(seed)
    return rng.sample(available_ids, num_frames)


def load_kitti_sample(dataset, frame_id):
    info_by_id = {str(info['point_cloud']['lidar_idx']): idx for idx, info in enumerate(dataset.kitti_infos)}
    if frame_id not in info_by_id:
        raise FileNotFoundError(f'Frame id not found in KITTI split: {frame_id}')
    sample = dataset[info_by_id[frame_id]]
    points = sample.get('points')
    metadata = {
        'frame_id': frame_id,
        'point_cloud_path': str(dataset.root_split_path / 'velodyne' / f'{frame_id}.bin'),
        'data_loader': 'KittiDataset.__getitem__',
        'fov_points_only': bool(dataset.dataset_cfg.get('FOV_POINTS_ONLY', False)),
        'data_mode': 'kitti',
    }
    if points is not None:
        metadata['num_raw_points'] = int(np.asarray(points).shape[0])
    return sample, metadata


def load_raw_sample(dataset, frame_path):
    points = np.fromfile(str(frame_path), dtype=np.float32).reshape(-1, 4)
    sample = dataset.prepare_data({'points': points, 'frame_id': frame_path.stem})
    metadata = {
        'frame_id': frame_path.stem,
        'point_cloud_path': str(frame_path),
        'data_loader': 'DatasetTemplate.prepare_data(raw_points)',
        'fov_points_only': False,
        'data_mode': 'raw',
        'num_raw_points': int(points.shape[0]),
    }
    return sample, metadata, int(points.shape[0])


def load_raw_voxels_via_data_processor(bin_path, data_proc):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    data_dict = {'points': points, 'use_lead_xyz': True}
    for proc in data_proc.data_processor_queue:
        data_dict = proc(data_dict)

    coords = normalize_voxel_coords(data_dict.get('voxel_coords', None))
    metadata = {
        'frame_id': Path(bin_path).stem,
        'point_cloud_path': str(bin_path),
        'data_loader': 'DataProcessor(raw_points)',
        'fov_points_only': False,
        'data_mode': 'raw',
        'num_raw_points': int(points.shape[0]),
    }
    return coords, metadata


def load_kitti_voxels(dataset, frame_id):
    sample, metadata = load_kitti_sample(dataset, frame_id)
    coords = normalize_voxel_coords(sample.get('voxel_coords', None))
    points = sample.get('points')
    if points is not None:
        metadata['num_raw_points'] = int(np.asarray(points).shape[0])
    return coords, metadata


def metadata_csv_fields():
    return ['data_loader', 'fov_points_only', 'data_mode']


def attach_metadata_to_row(row, metadata):
    row['data_loader'] = metadata.get('data_loader', '')
    row['fov_points_only'] = metadata.get('fov_points_only', '')
    row['data_mode'] = metadata.get('data_mode', '')
    return row

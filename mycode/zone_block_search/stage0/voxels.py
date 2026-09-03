"""Load and cache stage-0 KITTI FOV voxels for profiling."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mycode.zone_block_search.stage0.config import (
    DEFAULT_CFG,
    DEFAULT_KITTI_ROOT,
    FRAME_LIST_PATH,
    GRID_SIZE_XYZ,
    LIDAR_CENTER_XY,
    PROJECT_ROOT,
    VOXEL_CACHE_PATH,
)


def _ensure_project_on_path():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def attach_trainval_infos(dataset):
    """Keep eval preprocessing, but index every training/velodyne sample ID."""
    infos = []
    seen = set()
    for name in ('kitti_infos_train.pkl', 'kitti_infos_val.pkl'):
        path = Path(dataset.root_path) / name
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open('rb') as handle:
            for info in pickle.load(handle):
                lidar_idx = str(info['point_cloud']['lidar_idx'])
                if lidar_idx in seen:
                    continue
                seen.add(lidar_idx)
                infos.append(info)
    dataset.kitti_infos = infos
    return {str(info['point_cloud']['lidar_idx']): idx for idx, info in enumerate(infos)}


def load_frame_ids(list_file: Path = FRAME_LIST_PATH) -> List[str]:
    from mycode.kitti_frame_loader import normalize_frame_token, resolve_frame_ids_from_list

    ids = [normalize_frame_token(fid) for fid in resolve_frame_ids_from_list(str(list_file))]
    if not ids:
        raise RuntimeError(f'No frame ids in {list_file}')
    return ids


def load_stage0_voxel_frames(
    list_file: Path = FRAME_LIST_PATH,
    cache_path: Optional[Path] = None,
    cfg_file: str = DEFAULT_CFG,
    kitti_root: str = DEFAULT_KITTI_ROOT,
    force_reload: bool = False,
) -> Tuple[List[str], List[np.ndarray], dict]:
    """Return (frame_ids, coords_list[z,y,x], metadata).

    Caches to npz so repeated profiling skips voxelization.
    """
    _ensure_project_on_path()
    if cache_path is None:
        cache_path = VOXEL_CACHE_PATH

    frame_ids = load_frame_ids(list_file)
    if cache_path.suffix != '.pkl':
        cache_path = cache_path.with_suffix('.pkl')
    if cache_path.exists() and not force_reload:
        with cache_path.open('rb') as handle:
            data = pickle.load(handle)
        cached_ids = [str(x) for x in data['frame_ids']]
        if cached_ids == frame_ids:
            coords_list = data['coords_list']
            meta = {
                'from_cache': True,
                'cache_path': str(cache_path),
                'n_frames': len(frame_ids),
                'grid_size_xyz': tuple(int(v) for v in data['grid_size_xyz']),
                'lidar_center_xy': tuple(int(v) for v in data['lidar_center_xy']),
                'data_mode': str(data['data_mode']),
                'fov_points_only': bool(data['fov_points_only']),
            }
            return frame_ids, coords_list, meta
        print(f'Cache frame list mismatch; rebuilding {cache_path}')

    from mycode.kitti_frame_loader import (
        build_kitti_dataset,
        get_dataset_cfg,
        load_kitti_voxels,
        resolve_project_root,
    )
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.utils import common_utils

    project_root = resolve_project_root()
    cfg_path = Path(cfg_file)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    original_cwd = Path.cwd()
    try:
        import os
        os.chdir(project_root / 'tools')
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(original_cwd)

    dataset_cfg = get_dataset_cfg(cfg)
    fov = bool(dataset_cfg.get('FOV_POINTS_ONLY', False))
    if not fov:
        raise RuntimeError('FOV_POINTS_ONLY must be True')

    logger = common_utils.create_logger()
    dataset = build_kitti_dataset(cfg, project_root, kitti_root, logger)
    info_by_id = attach_trainval_infos(dataset)
    missing = [fid for fid in frame_ids if fid not in info_by_id]
    if missing:
        raise RuntimeError(f'Frames not found in KITTI train/val infos: {missing[:8]}')

    # Confirm voxelizer / backbone grid expectations.
    grid_xyz = tuple(int(v) for v in dataset.grid_size.tolist())
    # dataset.grid_size is typically (1408, 1600, 40); profiling uses sparse shape Z=41.
    print(
        f'Loading {len(frame_ids)} frames; dataset.grid_size={grid_xyz}; '
        f'profiling grid={GRID_SIZE_XYZ}; lidar_center={LIDAR_CENTER_XY}; FOV={fov}'
    )

    from tqdm import tqdm

    coords_list: List[np.ndarray] = []
    for frame_id in tqdm(frame_ids, desc='Voxelize'):
        coords, _metadata = load_kitti_voxels(dataset, frame_id)
        if coords is None:
            coords = np.zeros((0, 3), dtype=np.int64)
        else:
            coords = np.asarray(coords, dtype=np.int64)
        coords_list.append(coords)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open('wb') as handle:
        pickle.dump(
            {
                'frame_ids': frame_ids,
                'coords_list': coords_list,
                'grid_size_xyz': GRID_SIZE_XYZ,
                'lidar_center_xy': LIDAR_CENTER_XY,
                'data_mode': 'kitti',
                'fov_points_only': True,
                'dataset_grid_size_xyz': grid_xyz,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    meta = {
        'from_cache': False,
        'cache_path': str(cache_path),
        'n_frames': len(frame_ids),
        'grid_size_xyz': GRID_SIZE_XYZ,
        'lidar_center_xy': LIDAR_CENTER_XY,
        'data_mode': 'kitti',
        'fov_points_only': True,
        'dataset_grid_size_xyz': grid_xyz,
    }
    return frame_ids, coords_list, meta

"""Load Stage-0 KITTI FOV voxels and map them to conv3 (stage-2) occupancy."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from mycode.rtl_fixed.sparse_coords import (
    DOWNSAMPLE_STAGES,
    backbone_input_sparse_shape_zyx,
    generate_output_coords,
    spatial_shape_zyx_to_xyz,
)
from mycode.zone_block_search.stage2.config import (
    CONV2_GRID_SIZE_XYZ,
    FRAME_LIST_PATH,
    GRID_SIZE_XYZ,
    LIDAR_CENTER_XY,
    PROJECT_ROOT,
    STAGE0_VOXEL_CACHE_PATH,
    STAGE0_VOXELIZER_GRID_XYZ,
    STAGE1_VOXEL_CACHE_PATH,
    VOXEL_CACHE_PATH,
)


def _ensure_project_on_path():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def downsample_conv2_to_conv3(coords_zyx: np.ndarray) -> np.ndarray:
    """Map conv2 occupied voxels to conv3.0 SparseConv OFM coordinates."""
    spec = DOWNSAMPLE_STAGES[1]
    in_shape = (CONV2_GRID_SIZE_XYZ[2], CONV2_GRID_SIZE_XYZ[1], CONV2_GRID_SIZE_XYZ[0])
    ofm, out_shape = generate_output_coords(
        coords_zyx,
        spec['kernel'],
        spec['stride'],
        spec['padding'],
        in_shape,
        'SparseConv3d',
    )
    expected = GRID_SIZE_XYZ
    got = spatial_shape_zyx_to_xyz(out_shape)
    if got != expected:
        raise RuntimeError(
            f'conv3.0 grid mismatch: got XYZ {got}, expected {expected}; '
            f'ZYX in={in_shape} out={out_shape}'
        )
    return np.asarray(ofm, dtype=np.int64)


def downsample_stage0_to_conv3(coords_zyx: np.ndarray) -> np.ndarray:
    """Map stage-0 occupied voxels through conv2.0 then conv3.0."""
    spec2 = DOWNSAMPLE_STAGES[0]
    in_shape = backbone_input_sparse_shape_zyx(STAGE0_VOXELIZER_GRID_XYZ)
    conv2, conv2_shape = generate_output_coords(
        coords_zyx,
        spec2['kernel'],
        spec2['stride'],
        spec2['padding'],
        in_shape,
        'SparseConv3d',
    )
    got2 = spatial_shape_zyx_to_xyz(conv2_shape)
    if got2 != CONV2_GRID_SIZE_XYZ:
        raise RuntimeError(
            f'conv2.0 grid mismatch: got XYZ {got2}, expected {CONV2_GRID_SIZE_XYZ}; '
            f'ZYX in={in_shape} out={conv2_shape}'
        )
    return downsample_conv2_to_conv3(conv2)


def load_stage2_voxel_frames(
    list_file: Path = FRAME_LIST_PATH,
    cache_path: Optional[Path] = None,
    stage0_cache_path: Optional[Path] = None,
    stage1_cache_path: Optional[Path] = None,
    force_reload: bool = False,
) -> Tuple[List[str], List[np.ndarray], dict]:
    """Return (frame_ids, conv3 coords_list[z,y,x], metadata)."""
    _ensure_project_on_path()
    if cache_path is None:
        cache_path = VOXEL_CACHE_PATH
    if stage0_cache_path is None:
        stage0_cache_path = STAGE0_VOXEL_CACHE_PATH
    if stage1_cache_path is None:
        stage1_cache_path = STAGE1_VOXEL_CACHE_PATH

    from mycode.zone_block_search.stage0.voxels import load_frame_ids, load_stage0_voxel_frames

    frame_ids = load_frame_ids(list_file)
    if cache_path.suffix != '.pkl':
        cache_path = cache_path.with_suffix('.pkl')

    if cache_path.exists() and not force_reload:
        with cache_path.open('rb') as handle:
            data = pickle.load(handle)
        cached_ids = [str(x) for x in data['frame_ids']]
        cached_grid = tuple(int(v) for v in data['grid_size_xyz'])
        if cached_ids == frame_ids and cached_grid == GRID_SIZE_XYZ:
            meta = {
                'from_cache': True,
                'cache_path': str(cache_path),
                'n_frames': len(frame_ids),
                'grid_size_xyz': cached_grid,
                'lidar_center_xy': tuple(int(v) for v in data['lidar_center_xy']),
                'data_mode': str(data.get('data_mode', 'kitti')),
                'fov_points_only': bool(data.get('fov_points_only', True)),
                'downsample': str(data.get('downsample', 'conv3.0')),
            }
            return frame_ids, data['coords_list'], meta
        print(f'Cache mismatch; rebuilding {cache_path}')

    from tqdm import tqdm

    coords_list: List[np.ndarray] = []
    source = 'stage0'
    s0_meta = {}

    if stage1_cache_path.exists() and not force_reload:
        from mycode.zone_block_search.stage1.voxels import load_stage1_voxel_frames

        print(f'Loading stage-1 conv2 voxels then mapping to conv3.0 ({len(frame_ids)} frames)')
        s1_ids, s1_coords, s1_meta = load_stage1_voxel_frames(
            list_file=list_file,
            cache_path=stage1_cache_path,
            stage0_cache_path=stage0_cache_path,
            force_reload=False,
        )
        if s1_ids != frame_ids:
            raise RuntimeError('Stage-1 frame list does not match Stage-2 request')
        for coords in tqdm(s1_coords, desc='Downsample conv3.0'):
            coords_list.append(downsample_conv2_to_conv3(coords))
        source = 'stage1'
        s0_meta = {'stage1_from_cache': bool(s1_meta.get('from_cache'))}
    else:
        print(f'Loading stage-0 voxels then mapping to conv3.0 ({len(frame_ids)} frames)')
        s0_ids, s0_coords, s0_meta = load_stage0_voxel_frames(
            list_file=list_file,
            cache_path=stage0_cache_path,
            force_reload=False,
        )
        if s0_ids != frame_ids:
            raise RuntimeError('Stage-0 frame list does not match Stage-2 request')
        for coords in tqdm(s0_coords, desc='Downsample conv2.0+conv3.0'):
            coords_list.append(downsample_stage0_to_conv3(coords))
        source = 'stage0'
        s0_meta = {'stage0_from_cache': bool(s0_meta.get('from_cache'))}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'frame_ids': frame_ids,
        'coords_list': coords_list,
        'grid_size_xyz': GRID_SIZE_XYZ,
        'lidar_center_xy': LIDAR_CENTER_XY,
        'data_mode': 'kitti',
        'fov_points_only': True,
        'downsample': 'conv3.0',
        'source': source,
        'stage0_cache': str(stage0_cache_path),
        'stage1_cache': str(stage1_cache_path),
        **s0_meta,
    }
    with cache_path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        'from_cache': False,
        'cache_path': str(cache_path),
        'n_frames': len(frame_ids),
        'grid_size_xyz': GRID_SIZE_XYZ,
        'lidar_center_xy': LIDAR_CENTER_XY,
        'data_mode': 'kitti',
        'fov_points_only': True,
        'downsample': 'conv3.0',
        'source': source,
        **s0_meta,
    }
    return frame_ids, coords_list, meta

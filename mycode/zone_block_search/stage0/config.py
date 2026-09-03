"""Stage-0 fixed inputs for zone / block-size profiling."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]  # OpenPCDet root (…/zone_block_search/stage0 → …)

# Backbone sparse shape ZYX = [41, 1600, 1408] → XYZ grid below.
# Voxelizer grid_size_z is 40; backbone pads +1. Occupancy coords use z in [0, 40).
GRID_SIZE_XYZ = (1408, 1600, 41)
LIDAR_CENTER_XY = (0, 800)

BZ = 16
DELTA_D = 64
PAGE_CAPACITY = 64
TWO_PAGE_LIMIT = 128  # Nb <= 128 still fits two pages
COVERAGE_Q = 0.95

# Rings with fewer materialized block-frame samples than this are flagged low_sample.
LOW_SAMPLE_THRESHOLD = 50

# First orientation menu: non-square sizes satisfy Bx < By; squares shared.
MENU_BX_LT_BY = (
    (8, 8, 16),
    (8, 16, 16),
    (16, 16, 16),
    (16, 32, 16),
    (32, 32, 16),
    (32, 64, 16),
    (64, 64, 16),
)

FRAME_LIST_PATH = PACKAGE_DIR / 'frame_list_200.txt'
CACHE_DIR = PACKAGE_DIR / 'cache'
VOXEL_CACHE_PATH = CACHE_DIR / 'stage0_voxels.pkl'
RESULTS_DIR = PACKAGE_DIR / 'results' / 'stage0_bxltby'

DEFAULT_CFG = 'tools/cfgs/kitti_models/second.yaml'
DEFAULT_VELODYNE_DIR = 'data/kitti/training/velodyne'
DEFAULT_KITTI_ROOT = 'data/kitti'


def size_label(bx, by, bz=BZ):
    return f'{int(bx)}x{int(by)}x{int(bz)}'


def menu_volume_rank(menu=MENU_BX_LT_BY):
    """Larger volume → higher rank; used to pick the largest passing size."""
    return {size_label(*shape): int(shape[0] * shape[1] * shape[2]) for shape in menu}

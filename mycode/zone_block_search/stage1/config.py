"""Stage-1 fixed inputs for zone / block-size profiling (conv2 grid)."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]  # OpenPCDet root
STAGE0_DIR = PACKAGE_DIR.parent / 'stage0'

# VoxelBackBone8x conv2: sparse ZYX [21, 800, 704] → XYZ below.
# Input sparse shape is [41, 1600, 1408]; stride-2 SparseConv (pad 1) yields this.
GRID_SIZE_XYZ = (704, 800, 21)
LIDAR_CENTER_XY = (0, 400)

# Voxelizer grid used to form backbone input sparse_shape = grid[::-1] + [1,0,0].
STAGE0_VOXELIZER_GRID_XYZ = (1408, 1600, 40)

BZ = 8
DELTA_D = 32
PAGE_CAPACITY = 64
TWO_PAGE_LIMIT = 128  # Nb <= 128 still fits two pages
COVERAGE_Q = 0.97

# Rings with fewer materialized block-frame samples than this are flagged low_sample.
LOW_SAMPLE_THRESHOLD = 50

# Orientation menus: non-square sizes satisfy Bx>By or Bx<By; squares shared.
MENU_BX_GT_BY = (
    (4, 4, 8),
    (8, 4, 8),
    (8, 8, 8),
    (16, 8, 8),
    (16, 16, 8),
    (32, 16, 8),
    (32, 32, 8),
)
MENU_BX_LT_BY = (
    (4, 4, 8),
    (4, 8, 8),
    (8, 8, 8),
    (8, 16, 8),
    (16, 16, 8),
    (16, 32, 8),
    (32, 32, 8),
)

FRAME_LIST_PATH = STAGE0_DIR / 'frame_list_200.txt'
STAGE0_VOXEL_CACHE_PATH = STAGE0_DIR / 'cache' / 'stage0_voxels.pkl'
CACHE_DIR = PACKAGE_DIR / 'cache'
VOXEL_CACHE_PATH = CACHE_DIR / 'stage1_voxels.pkl'
RESULTS_DIR = PACKAGE_DIR / 'results' / 'stage1_bxgtby'
RESULTS_DIR_BX_LT_BY = PACKAGE_DIR / 'results' / 'stage1_bxltby'


def size_label(bx, by, bz=BZ):
    return f'{int(bx)}x{int(by)}x{int(bz)}'


def menu_volume_rank(menu=MENU_BX_GT_BY):
    """Larger volume → higher rank; used to pick the largest passing size."""
    return {size_label(*shape): int(shape[0] * shape[1] * shape[2]) for shape in menu}

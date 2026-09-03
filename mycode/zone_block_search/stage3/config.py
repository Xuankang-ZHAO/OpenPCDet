"""Stage-3 fixed inputs for zone / block-size profiling (conv4 grid)."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]  # OpenPCDet root
STAGE0_DIR = PACKAGE_DIR.parent / 'stage0'
STAGE1_DIR = PACKAGE_DIR.parent / 'stage1'
STAGE2_DIR = PACKAGE_DIR.parent / 'stage2'

# VoxelBackBone8x conv4: sparse ZYX [5, 200, 176] → XYZ below.
# conv3 grid is [11, 400, 352]; stride-2 SparseConv (pad 0,1,1) yields this.
GRID_SIZE_XYZ = (176, 200, 5)
LIDAR_CENTER_XY = (0, 100)

# Intermediate grids used when downsampling from earlier occupancy.
CONV3_GRID_SIZE_XYZ = (352, 400, 11)
CONV2_GRID_SIZE_XYZ = (704, 800, 21)

# Voxelizer grid used to form backbone input sparse_shape = grid[::-1] + [1,0,0].
STAGE0_VOXELIZER_GRID_XYZ = (1408, 1600, 40)

BZ = 8
DELTA_D = 16
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
)
MENU_BX_LT_BY = (
    (4, 4, 8),
    (4, 8, 8),
    (8, 8, 8),
    (8, 16, 8),
    (16, 16, 8),
)

FRAME_LIST_PATH = STAGE0_DIR / 'frame_list_200.txt'
STAGE0_VOXEL_CACHE_PATH = STAGE0_DIR / 'cache' / 'stage0_voxels.pkl'
STAGE1_VOXEL_CACHE_PATH = STAGE1_DIR / 'cache' / 'stage1_voxels.pkl'
STAGE2_VOXEL_CACHE_PATH = STAGE2_DIR / 'cache' / 'stage2_voxels.pkl'
CACHE_DIR = PACKAGE_DIR / 'cache'
VOXEL_CACHE_PATH = CACHE_DIR / 'stage3_voxels.pkl'
RESULTS_DIR = PACKAGE_DIR / 'results' / 'stage3_bxgtby'
RESULTS_DIR_BX_LT_BY = PACKAGE_DIR / 'results' / 'stage3_bxltby'


def size_label(bx, by, bz=BZ):
    return f'{int(bx)}x{int(by)}x{int(bz)}'


def menu_volume_rank(menu=MENU_BX_GT_BY):
    """Larger volume → higher rank; used to pick the largest passing size."""
    return {size_label(*shape): int(shape[0] * shape[1] * shape[2]) for shape in menu}

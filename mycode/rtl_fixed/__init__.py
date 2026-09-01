"""Fixed-size RTL block partitioning analysis."""

from .partition import compute_rtl_fixed_partition_counts, grid_block_counts
from .sparse_coords import DOWNSAMPLE_STAGES, iter_downsample_stage_coords

__all__ = [
    'compute_rtl_fixed_partition_counts',
    'grid_block_counts',
    'DOWNSAMPLE_STAGES',
    'iter_downsample_stage_coords',
]

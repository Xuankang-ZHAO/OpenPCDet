#!/usr/bin/env python3
"""Stage-2 final-config materialized-block N_b histogram (halo on, 200 frames)."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mycode.zone_block_search.block_nb_analysis import run_stage_cli
from mycode.zone_block_search.stage2.config import (
    FRAME_LIST_PATH,
    GRID_SIZE_XYZ,
    LIDAR_CENTER_XY,
    PACKAGE_DIR,
    TWO_PAGE_LIMIT,
    VOXEL_CACHE_PATH,
)
from mycode.zone_block_search.stage2.voxels import load_stage2_voxel_frames


def main():
    run_stage_cli(
        stage=2,
        load_frames=load_stage2_voxel_frames,
        grid_size=GRID_SIZE_XYZ,
        lidar_center=LIDAR_CENTER_XY,
        default_out_dir=PACKAGE_DIR / 'results' / 'final_nb_hist',
        default_cache=VOXEL_CACHE_PATH,
        default_list_file=FRAME_LIST_PATH,
        two_page_limit=TWO_PAGE_LIMIT,
    )


if __name__ == '__main__':
    main()

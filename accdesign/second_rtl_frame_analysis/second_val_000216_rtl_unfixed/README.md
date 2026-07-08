# SECOND val/000216 Frame Analysis

This directory contains analysis outputs for the KITTI point-cloud frame used by the current accdesign golden package:

- Golden package: `accdesign/second_rtl_golden_packages/second_val_000216_golden`
- Manifest: `accdesign/second_rtl_golden_packages/second_val_000216_golden/manifest.json`
- Frame: `val/000216`
- Point cloud: `data/kitti/training/velodyne/000216.bin`
- Point cloud SHA-256: `ced3ed1a871cff9a739e22ed8982e922969d177f6661ce348c530b219d84bd66`
- Golden format: `second_rtl_golden_v2`
- Golden raw VFE stream: 15000 records, 240000 bytes
- Grid size XYZ: `[1408, 1600, 40]`
- Sparse shape ZYX: `[41, 1600, 1408]`
- LiDAR center XY: `[0, 800]`
- RTL unfixed zone LUT: `mycode/block_size_lut_rtl_unfixed.txt`
- Zone LUT SHA-256: `d5864716816058089c525af005d430a1c063a1fb4eba2a53990dda1b21ed856a`

All commands were run from the repository root with the `openpcd` conda environment.

## Generated Files

- `frame_list.txt`: single-frame list containing `000216`
- `analysis_summary.json`: machine-readable summary extracted from all outputs
- `block_v2_rtl_unfixed_zone4_000216.csv`: RTL unfixed block distribution from `voxel_analyze_with_boudary_rtl_unfixed.py`
- `block_vis2d_fixed/000216.png`: fixed-block XY heatmap from `block_voxel_vis2d.py`
- `chebyshev_stats_000216.csv`: Chebyshev-distance histogram from `chebyshev_analyze.py`
- `logs/voxel_neighbour_count_000216.log`: 3x3x3 neighbor-count histogram from `voxel_neighbour_count.py`
- `second_layer_input_feature_sparsity/000216_second_input_feature_sparsity.{csv,json}`: per-layer input feature zero-channel analysis
- `second_layer_sparsity/000216_second_layer_sparsity.{csv,json}`: per-layer sparse tensor active-voxel analysis
- `logs/*.log`: stdout/stderr or script combined logs for each run

## Script Sources And Commands

`mycode/voxel_analyze_with_boudary_rtl_unfixed.py`

```bash
conda run -n openpcd python mycode/voxel_analyze_with_boudary_rtl_unfixed.py \
  --list_file accdesign/second_rtl_frame_analysis/second_val_000216_rtl_unfixed/frame_list.txt \
  --max_files 1 \
  --zone_lut mycode/block_size_lut_rtl_unfixed.txt \
  --lidar_center 0,800 \
  --out accdesign/second_rtl_frame_analysis/second_val_000216_rtl_unfixed/block_v2_rtl_unfixed_zone4_000216.csv
```

`mycode/block_voxel_vis2d.py`

```bash
conda run -n openpcd python mycode/block_voxel_vis2d.py \
  --list_file accdesign/second_rtl_frame_analysis/second_val_000216_rtl_unfixed/frame_list.txt \
  --out_dir accdesign/second_rtl_frame_analysis/second_val_000216_rtl_unfixed/block_vis2d_fixed
```

`mycode/voxel_neighbour_count.py`

```bash
conda run -n openpcd python mycode/voxel_neighbour_count.py \
  --list_file accdesign/second_rtl_frame_analysis/second_val_000216_rtl_unfixed/frame_list.txt
```

`mycode/chebyshev_analyze.py`

```bash
conda run -n openpcd python mycode/chebyshev_analyze.py \
  --list_file accdesign/second_rtl_frame_analysis/second_val_000216_rtl_unfixed/frame_list.txt \
  --lidar_center 0,800 \
  --out accdesign/second_rtl_frame_analysis/second_val_000216_rtl_unfixed/chebyshev_stats_000216.csv
```

`mycode/second_layer_input_feature_sparsity.py`

```bash
conda run -n openpcd python mycode/second_layer_input_feature_sparsity.py \
  --cfg tools/cfgs/kitti_models/second_hw_qat.yaml \
  --ckpt output/kitti_models/second_hw_qat/hw_qat_10ep/ckpt/checkpoint_epoch_10.pth \
  --frame_id 000216 \
  --num_frames 1 \
  --out_dir accdesign/second_rtl_frame_analysis/second_val_000216_rtl_unfixed/second_layer_input_feature_sparsity \
  --log_file accdesign/second_rtl_frame_analysis/second_val_000216_rtl_unfixed/logs/second_layer_input_feature_sparsity_000216.log
```

`mycode/second_layer_sparsity.py`

```bash
conda run -n openpcd python mycode/second_layer_sparsity.py \
  --cfg tools/cfgs/kitti_models/second_hw_qat.yaml \
  --ckpt output/kitti_models/second_hw_qat/hw_qat_10ep/ckpt/checkpoint_epoch_10.pth \
  --frame_id 000216 \
  --num_frames 1 \
  --out_dir accdesign/second_rtl_frame_analysis/second_val_000216_rtl_unfixed/second_layer_sparsity \
  --log_file accdesign/second_rtl_frame_analysis/second_val_000216_rtl_unfixed/logs/second_layer_sparsity_000216.log
```

## Key Results

RTL unfixed initial block distribution:

- Non-empty voxels: 15000
- Voxel sparsity over the full grid: `0.00016645951704545453`
- Non-empty blocks: 1113
- Empty blocks among emitted blocks: 0
- Max voxels in one block: 287
- Mean voxels per block: `18.315363881401616`
- Most common block occupancies by block count: 2 voxels/block -> 115 blocks, 1 -> 110, 3 -> 65, 5 -> 62, 4 -> 57

3x3x3 occupied-neighbor count:

- Valid voxels: 15000
- Most common neighbor bins: 1 neighbor -> 6705 voxels, 2 -> 3063, 3 -> 1652, 4 -> 1083, 5 -> 526
- Highest nonzero bin: 26 neighbors -> 1 voxel; 27 neighbors -> 0 voxels

Chebyshev distance from LiDAR center:

- Non-empty voxels: 15000
- Nonzero distance range: 113 to 1407
- Peak distance bin: 524 with 267 voxels
- Voxel sparsity: `0.00016645951704545453`

SECOND layer active-voxel analysis from `second_layer_sparsity.py`:

- `input_voxelized`: 15000 active voxels
- Data loader: `KittiDataset.__getitem__` with `FOV_POINTS_ONLY=True`
- `conv2.0.0`: 25762 active voxels
- `conv3.0.0`: 19089 active voxels
- `conv4.0.0`: 8495 active voxels
- `conv_out.0`: 6612 active voxels
- `conv_out.0` active density: `0.09392045454545454`

SECOND layer input feature sparsity from `second_layer_input_feature_sparsity.py`:

- Raw points before voxelization: 122676
- Voxels after VFE: 15000
- `input_voxel_features` zero element ratio: `0.05418333333333333`
- Highest observed layer-input zero ratio: `conv4.0` / `conv4.0.0` at `0.7190618515956921`
- `conv4.0` average zero channels per voxel: `46.01995849609375` of 64

## Notes

- `voxel_analyze_with_boudary_rtl_unfixed.py` uses the same RTL unfixed zone LUT and LiDAR center recorded by the golden manifest.
- `block_voxel_vis2d.py` is the script's default fixed `10x10x6` block visualization. It is useful as a 2D occupancy view, but it is not the RTL unfixed block allocator result.
- `second_layer_sparsity.py` was regenerated with the HW-QAT golden config/checkpoint and the KITTI dataset path so its FOV filtering matches the golden package.
- The SECOND sparsity outputs are model-hook analysis artifacts; the bit-exact RTL golden outputs remain in `accdesign/second_rtl_golden_packages/second_val_000216_golden/ofm_golden.bin`.

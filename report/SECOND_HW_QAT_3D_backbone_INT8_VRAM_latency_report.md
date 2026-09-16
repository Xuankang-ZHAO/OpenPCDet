# SECOND VoxelBackBone8x INT8 推理：3D Backbone 逐层显存与延时报告

| 项 | 值 |
|---|---|
| 日期 | 2026-09-16 |
| GPU | NVIDIA GeForce RTX 3080（10240 MiB） |
| Python 环境 | `/home/vipuser/miniconda3/envs/openpcd` |
| 模型配置 | `tools/cfgs/kitti_models/second_hw_qat.yaml` |
| Checkpoint | `output/kitti_models/second_hw_qat/hw_qat_10ep/ckpt/checkpoint_epoch_10.pth` |
| Backbone | `VoxelBackBone8x_HWQAT` |
| 推理模式 | HW-reference INT8（数值 INT8，spconv CUDA kernel 仍为 FP32） |
| 权重量化 | per-channel |
| 测试帧 | KITTI `000216`（FOV 过滤，VFE 后 15000 voxels，打到上限） |
| Batch size | 1 |

## 1. 实验目的

在本机 RTX 3080 上，对量化后的 SECOND 3D backbone（INT8 HW-reference 路径）测量：

1. 各稀疏卷积层的 GPU 显存占用（净增、峰值、逻辑 INT8 负载）
2. 各稀疏卷积层的 GPU 处理延时（CUDA Event）

## 2. 方法说明

### 2.1 INT8 推理路径

使用 `VoxelBackBone8x_HWQAT.enable_hw_reference()`：

- 激活 / 权重按 signed symmetric INT8 量化（`qmin=-127, qmax=127`，zero point = 0）
- BN fold 后做 INT8 weight + integer bias + shift-only requant
- **注意**：当前 GPU 路径上，INT8 数值仍存放在 FP32 CUDA tensor 中，spconv 未走 Tensor Core INT8。因此：
  - 显存中的 feature 约为逻辑 INT8 的 4 倍
  - 延时是「INT8 数值语义 + FP32 spconv kernel」的 GPU 时间，不是 RTL 真 INT8 MAC 延时

### 2.2 显存测量

脚本：`mycode/second_backbone_int8_layer_vram.py`

在每层 `_forward_hw_reference_layer` 前后记录：

| 指标 | 含义 |
|---|---|
| `DeltaMB` | 层返回后 live allocated 净变化 |
| `Peak+MB` | 该层执行期间相对进入时的峰值额外 allocated（含 GEMM/indice workspace） |
| `LiveMB` | 该层结束后累计 live allocated |
| `OutFP32MB` | 输出 feature 在 GPU 上的实际字节（FP32） |
| `OutINT8MB` | 逻辑 INT8 feature：`N_out × C_out` |
| `IndiceMB` | `SparseConvTensor.indice_dict` 累计占用 |
| `Wint8MB` | 该层 INT8 权重 payload |

### 2.3 延时测量

脚本：`mycode/second_backbone_int8_layer_latency.py`

- CUDA Event 包住每一层 `_forward_hw_reference_layer`
- Warmup：10 次；计时：50 次
- 报告 mean / p50 / p90 / std，以及占 backbone 总延时的 Share%

## 3. 逐层显存结果

整段 3D backbone（frame `000216`）：

| 汇总项 | 值 |
|---|---:|
| 结束后 live allocated | 66.85 MB |
| 过程峰值 allocated | 78.80 MB |
| CUDA reserved | 96.00 MB |
| VFE 后 voxel 数 | 15000 |

| Id | 层 | 类型 | Nin | Cin | Nout | Cout | Out FP32 (MB) | 逻辑 INT8 (MB) | Indice (MB) | Weight INT8 (MB) | Delta (MB) | Peak+ (MB) | Live (MB) | 保留为 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | `conv_input.0` | SubMConv3d | 15000 | 4 | 15000 | 16 | 0.916 | 0.229 | 1.888 | 0.002 | 2.900 | 5.892 | 31.039 | — |
| 1 | `conv1.0.0` | SubMConv3d | 15000 | 16 | 15000 | 16 | 0.916 | 0.229 | 1.888 | 0.007 | 0.916 | 4.633 | 31.955 | `x_conv1` |
| 2 | `conv2.0.0` | SparseConv3d | 15000 | 16 | 25762 | 32 | 3.145 | 0.786 | 6.791 | 0.013 | 8.050 | 18.509 | 39.089 | — |
| 3 | `conv2.1.0` | SubMConv3d | 25762 | 32 | 25762 | 32 | 3.145 | 0.786 | 9.641 | 0.026 | 5.996 | 18.790 | 45.085 | — |
| 4 | `conv2.2.0` | SubMConv3d | 25762 | 32 | 25762 | 32 | 3.145 | 0.786 | 9.641 | 0.026 | 3.145 | 15.938 | 45.085 | `x_conv2` |
| 5 | `conv3.0.0` | SparseConv3d | 25762 | 32 | 19089 | 64 | 4.660 | 1.165 | 14.894 | 0.053 | 11.065 | **29.327** | 53.006 | — |
| 6 | `conv3.1.0` | SubMConv3d | 19089 | 64 | 19089 | 64 | 4.660 | 1.165 | 17.005 | 0.105 | 7.461 | 27.660 | 60.467 | — |
| 7 | `conv3.2.0` | SubMConv3d | 19089 | 64 | 19089 | 64 | 4.660 | 1.165 | 17.005 | 0.105 | 5.371 | 24.860 | 61.177 | `x_conv3` |
| 8 | `conv4.0.0` | SparseConv3d | 19089 | 64 | 8495 | 64 | 2.074 | 0.518 | 20.187 | 0.105 | 5.770 | 17.499 | 62.286 | — |
| 9 | `conv4.1.0` | SubMConv3d | 8495 | 64 | 8495 | 64 | 2.074 | 0.518 | 21.126 | 0.105 | 3.527 | 12.670 | 65.813 | — |
| 10 | `conv4.2.0` | SubMConv3d | 8495 | 64 | 8495 | 64 | 2.074 | 0.518 | 21.126 | 0.105 | 2.586 | 11.729 | 65.813 | `x_conv4` |
| 11 | `conv_out.0` | SparseConv3d | 8495 | 64 | 6612 | 128 | 3.229 | 0.807 | 21.515 | 0.023 | 3.619 | 15.569 | 66.846 | — |

### 3.1 显存结论

1. **权重可忽略**：单层 INT8 权重最大约 0.11 MB。
2. **逻辑 INT8 激活也不大**：最大约 1.17 MB（`conv3`）；GPU 上因 FP32 存放约为其 4 倍。
3. **显存大头是 indice / pair / hash**：`IndiceMB` 从约 1.9 MB 累加到约 21.5 MB。
4. **峰值发生在新 `indice_key` 的下采样层**：`conv3.0` 额外峰值约 29.3 MB。
5. **同 `indice_key` 的后续 SubM 更便宜**：例如 `conv1`、`conv2.2`、`conv4.2`，净增接近一份新 feature。
6. **`LiveMB` 单调上升**：OpenPCDet 保留 `x_conv1/2/3/4` multi-scale features，前 stage 不会立刻释放。
7. 相对 RTX 3080 的 10 GB，整段 3D backbone live/peak（约 67 / 79 MB）很小。

原始数据：

- `mycode/output/second_backbone_int8_layer_vram/second_backbone_int8_layer_vram_NVIDIAGeForceRTX3080_20260916_151241.csv`
- 同目录 `.json` / `.log`

## 4. 逐层延时结果

设置：warmup = 10，repeat = 50。

| 汇总项 | 值 |
|---|---:|
| Backbone 总延时 mean | 15.347 ms |
| p50 | 15.293 ms |
| p90 | 15.712 ms |
| std | 0.273 ms |

| Id | 层 | 类型 | Nin | Cin | Nout | Cout | Mean (ms) | P50 (ms) | P90 (ms) | Std (ms) | Share |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `conv_input.0` | SubMConv3d | 15000 | 4 | 15000 | 16 | 1.235 | 1.236 | 1.274 | 0.028 | 8.0% |
| 1 | `conv1.0.0` | SubMConv3d | 15000 | 16 | 15000 | 16 | 0.892 | 0.894 | 0.922 | 0.025 | 5.8% |
| 2 | `conv2.0.0` | SparseConv3d | 15000 | 16 | 25762 | 32 | 1.624 | 1.615 | 1.668 | 0.068 | 10.6% |
| 3 | `conv2.1.0` | SubMConv3d | 25762 | 32 | 25762 | 32 | 1.251 | 1.246 | 1.278 | 0.055 | 8.2% |
| 4 | `conv2.2.0` | SubMConv3d | 25762 | 32 | 25762 | 32 | 0.902 | 0.904 | 0.930 | 0.027 | 5.9% |
| 5 | `conv3.0.0` | SparseConv3d | 25762 | 32 | 19089 | 64 | 1.672 | 1.666 | 1.706 | 0.071 | 10.9% |
| 6 | `conv3.1.0` | SubMConv3d | 19089 | 64 | 19089 | 64 | 1.286 | 1.279 | 1.318 | 0.052 | 8.4% |
| 7 | `conv3.2.0` | SubMConv3d | 19089 | 64 | 19089 | 64 | 0.951 | 0.950 | 0.975 | 0.019 | 6.2% |
| 8 | `conv4.0.0` | SparseConv3d | 19089 | 64 | 8495 | 64 | **1.682** | 1.667 | 1.716 | 0.106 | **11.0%** |
| 9 | `conv4.1.0` | SubMConv3d | 8495 | 64 | 8495 | 64 | 1.276 | 1.271 | 1.314 | 0.043 | 8.3% |
| 10 | `conv4.2.0` | SubMConv3d | 8495 | 64 | 8495 | 64 | 0.957 | 0.950 | 0.981 | 0.042 | 6.2% |
| 11 | `conv_out.0` | SparseConv3d | 8495 | 64 | 6612 | 128 | 1.619 | 1.598 | 1.711 | 0.091 | 10.6% |

### 4.1 延时结论

1. 最慢层是新 `indice_key` / 下采样 SparseConv：`conv2.0`、`conv3.0`、`conv4.0`、`conv_out`，约 1.6–1.7 ms，合计约占 backbone 的 43%。
2. 同 `indice_key` 的后续 SubM 更便宜（约 0.9–1.0 ms），因为 pair 可复用。
3. 单帧 3D backbone 总延时约 **15.3 ms**（RTX 3080，本路径）。
4. 该延时**不能直接当作加速器 INT8 MAC 延时**；加速器侧需用同样 voxel/layer workload 另行建模。

原始数据：

- `mycode/output/second_backbone_int8_layer_latency/second_backbone_int8_layer_latency_NVIDIAGeForceRTX3080_20260916_153830.csv`
- 同目录 `.json` / `.log`

## 5. 复现命令

```bash
# 显存
/home/vipuser/miniconda3/envs/openpcd/bin/python mycode/second_backbone_int8_layer_vram.py \
  --device cuda:0 --frame_id 000216 --num_frames 1 --warmup 1 --repeat 1

# 延时
/home/vipuser/miniconda3/envs/openpcd/bin/python mycode/second_backbone_int8_layer_latency.py \
  --device cuda:0 --frame_id 000216 --num_frames 1 --warmup 10 --repeat 50
```

## 6. 使用建议

| 目标 | 应对齐的指标 |
|---|---|
| 加速器 INT8 feature / weight 存储规划 | `OutINT8MB`、`Weight INT8`、坐标字节 |
| 判断 3080 上 PyTorch/spconv 是否会爆显存 | `Peak+MB`、`LiveMB`、`IndiceMB` |
| GPU 侧瓶颈层排序 | 延时表 Share% / Mean |
| RTL / 真 INT8 加速器延时 | 需另建周期模型；本报告 GPU 延时仅作参考 |

## 7. 相关脚本与产物

| 类型 | 路径 |
|---|---|
| 显存剖析脚本 | `mycode/second_backbone_int8_layer_vram.py` |
| 延时剖析脚本 | `mycode/second_backbone_int8_layer_latency.py` |
| 显存输出目录 | `mycode/output/second_backbone_int8_layer_vram/` |
| 延时输出目录 | `mycode/output/second_backbone_int8_layer_latency/` |
| 本报告 | `report/SECOND_HW_QAT_3D_backbone_INT8_VRAM_latency_report.md` |

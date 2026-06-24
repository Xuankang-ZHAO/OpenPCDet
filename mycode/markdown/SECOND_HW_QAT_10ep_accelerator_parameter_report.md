# SECOND VoxelBackBone8x HW-QAT 10ep 结果与加速器参数建议

本文总结 `VoxelBackBone8x_HWQAT` 在 10 epoch QAT 后的两种量化方案结果：`per-channel weight + per-channel shift` 主方案，以及 `per-tensor weight + per-layer shift + per-channel bias` 极简硬件对比方案。报告同时给出硬件等价 INT8 推理精度、真实 checkpoint 导出的 `bias_int` / `shift` 统计，以及对应的加速器参数选择建议。

## 1. 实验版本

本次实验使用的算法范围如下：

- 只量化 SECOND 的 `VoxelBackBone8x` 3D sparse backbone。
- VFE、HeightCompression、2D backbone、dense head 保持 FP32。
- activation / weight 使用 signed symmetric INT8。
- zero point 固定为 0。
- BN 在导出阶段离线 fold 到 conv weight / bias。
- MAC 语义为 `INT8 x INT8 -> INT32 psum`。
- postprocess 语义为 `psum + bias_int -> rounding -> arithmetic shift -> clamp_s8 -> ReLU`。

本报告区分两种已经训练 / 导出的 10 epoch QAT 版本：

1. **per-channel 主方案**：
   - weight 是 **per-output-channel** 量化，每个输出通道一个 `Sw[c]`。
   - activation 是 **per-layer / per-tensor** 静态 scale，每层 sparse feature tensor 一个 `Sx` 或 `Sy`。
   - bias 是 **per-output-channel** integer bias，每个输出通道一个 `bias_int[c]`。
   - requant shift 是 **per-output-channel**，每个输出通道一个 `shift[c]`。
   - 这是当前精度最稳、推荐作为论文主 benchmark 的方案。

2. **per-tensor 极简硬件方案**：
   - weight 是 **per-layer / per-tensor** 量化，每层 sparse conv 只有一个 `Sw_layer`。
   - activation 仍然是 **per-layer / per-tensor** 静态 scale。
   - `real_multiplier = Sx * Sw_layer / Sy` 在同一层内不再随输出通道变化，因此 `shift` 可以自然简化为 **per-layer**。
   - BN folding 后的 `b_fold[c]` 仍随输出通道变化，因此 `bias_int[c]` 仍然是 **per-output-channel**。
   - 这是硬件参数加载最少的对比方案，但精度明显弱于 per-channel 主方案。

两种方案都使用 symmetric INT8、zero point 固定为 0；导出使用 `qmin=-127, qmax=127`，没有使用 `-128`，避免 signed symmetric INT8 的负端不对称问题。

两种方案的硬件语义可以概括为：

```text
per-channel main:
  activation scale: per layer / per tensor
  weight scale    : per output channel
  bias_int        : per output channel
  requant shift   : per output channel
  zero point      : always 0

per-tensor ablation:
  activation scale: per layer / per tensor
  weight scale    : per layer / per tensor
  bias_int        : per output channel
  requant shift   : per layer
  zero point      : always 0
```

per-channel 主方案训练命令使用：

```bash
python train_second_hw_qat.py \
  --cfg_file cfgs/kitti_models/second_hw_qat.yaml \
  --pretrained_model ../output/kitti_models/second/default/ckpt/checkpoint_epoch_80.pth \
  --epochs 10 \
  --extra_tag hw_qat_10ep \
  --export_qparams
```

per-tensor 对比方案训练命令使用：

```bash
python train_second_hw_qat.py \
  --cfg_file cfgs/kitti_models/second_hw_qat.yaml \
  --pretrained_model ../output/kitti_models/second/default/ckpt/checkpoint_epoch_80.pth \
  --epochs 10 \
  --extra_tag hw_qat_10ep_per_tensor \
  --weight_quant per_tensor \
  --export_qparams
```

关键输出目录：

```text
per-channel: output/kitti_models/second_hw_qat/hw_qat_10ep/
per-tensor : output/kitti_models/second_hw_qat/hw_qat_10ep_per_tensor/
```

硬件等价检查与导出目录：

```text
per-channel: output/kitti_models/second_hw_qat/hw_qat_10ep/hw_export/
per-tensor : output/kitti_models/second_hw_qat/hw_qat_10ep_per_tensor/hw_export/
```

## 2. 训练收敛情况

两个 10 epoch QAT 训练都完整结束。

per-channel 主方案生成：

```text
output/kitti_models/second_hw_qat/hw_qat_10ep/ckpt/checkpoint_epoch_10.pth
```

per-tensor 对比方案生成：

```text
output/kitti_models/second_hw_qat/hw_qat_10ep_per_tensor/ckpt/checkpoint_epoch_10.pth
```

从日志看，per-channel 主方案训练过程稳定：

- epoch: 10
- batch size: 4
- pretrained checkpoint: `second/default/ckpt/checkpoint_epoch_80.pth`
- QAT scope: `MODEL.BACKBONE_3D only`
- weight quantization: per-output-channel
- observer: max
- 初始平均 loss 约 `0.290`
- 最终第 10 epoch 平均 loss 约 `0.242`
- 无 loss 发散、NaN 或训练中断

per-tensor 对比方案训练也稳定：

- epoch: 10
- batch size: 8
- pretrained checkpoint: `second/default/ckpt/checkpoint_epoch_80.pth`
- QAT scope: `MODEL.BACKBONE_3D only`
- weight quantization: per-layer / per-tensor
- observer: max
- 最终第 10 epoch 平均 loss 约 `0.244`
- 无 loss 发散、NaN 或训练中断

普通 OpenPCDet eval 中，per-tensor 版本的 3D AP R40 为：Car `88.5724 / 78.6881 / 75.5639`，Pedestrian `51.7374 / 46.0814 / 40.9870`，Cyclist `83.5184 / 65.1312 / 60.9527`。该结果只说明 checkpoint 正常、训练稳定；是否适合硬件设计必须看后文的 HW-reference eval。

因此，10 epoch 对从 pretrained SECOND 微调 HW-QAT 是足够的。per-channel 版本已经足够作为论文主 benchmark；per-tensor 版本则适合用于硬件精简对比与 ablation。

## 3. 完整量化方案细节

### 3.1 训练时 QAT 图

训练阶段使用 `VoxelBackBone8x_HWQAT`，保持 OpenPCDet 原始 SECOND 3D backbone 拓扑不变：

```text
conv_input -> conv1 -> conv2.0 -> conv2.1 -> conv2.2
           -> conv3.0 -> conv3.1 -> conv3.2
           -> conv4.0 -> conv4.1 -> conv4.2
           -> conv_out
```

训练时插入 fake quant 的位置：

- backbone input voxel feature
- 每个 Conv-BN-ReLU block 输出之后

训练时使用 STE：

```text
x_qdq = round(x / Sx) * Sx
forward  uses x_qdq
backward uses STE gradient to x
```

weight fake quant 使用 folded 前的 conv weight，在训练中模拟 INT8 权重量化误差。两种方案只在 weight scale 粒度上不同：

```text
per-channel:
  Sw[c]      = max(abs(w_fp32[c])) / 127
  w_int8[c]  = clamp(round(w_fp32[c] / Sw[c]), -127, 127)
  w_qdq[c]   = w_int8[c] * Sw[c]

per-tensor:
  Sw_layer   = max(abs(w_fp32)) / 127
  w_int8     = clamp(round(w_fp32 / Sw_layer), -127, 127)
  w_qdq      = w_int8 * Sw_layer
```

其中 `Sw[c]` 是每个输出通道独立的 weight scale，`Sw_layer` 是整层 sparse conv 共享的 weight scale。

### 3.2 导出时 BN folding

推理 / 导出阶段不让硬件执行 BN。每个 sparse conv 后的 `BatchNorm1d` 被 fold 到 conv weight 和 bias：

```text
bn_scale[c] = gamma[c] / sqrt(running_var[c] + eps)
w_fold[c]  = w_fp32[c] * bn_scale[c]
b_fold[c]  = beta[c] - running_mean[c] * bn_scale[c]
```

当前 sparse conv 本身 `bias=False`，因此没有 conv bias 项。

导出时再对 `w_fold` 做 INT8 量化。per-channel 主方案按输出通道量化；per-tensor 对比方案整层共享一个 weight scale：

```text
per-channel:
  Sw[c]          = max(abs(w_fold[c])) / 127
  weight_int8[c] = clamp(round(w_fold[c] / Sw[c]), -127, 127)

per-tensor:
  Sw_layer       = max(abs(w_fold)) / 127
  weight_int8    = clamp(round(w_fold / Sw_layer), -127, 127)
```

这意味着硬件实际消费的是 BN-fold 后的 `weight_int8` 和 `bias_int`，不是训练图里的原始 conv weight + BN。

### 3.3 Integer bias 与 shift-only requant

对第 `l` 层、第 `c` 个输出通道，per-channel 主方案为：

```text
bias_int[l][c] = round(b_fold[l][c] / (Sx[l] * Sw[l][c]))
real_multiplier[l][c] = Sx[l] * Sw[l][c] / Sy[l]
shift[l][c] = round(-log2(real_multiplier[l][c]))
pow2_multiplier[l][c] = 2 ^ (-shift[l][c])
```

per-tensor 对比方案为：

```text
bias_int[l][c] = round(b_fold[l][c] / (Sx[l] * Sw_layer[l]))
real_multiplier[l] = Sx[l] * Sw_layer[l] / Sy[l]
shift_layer[l] = round(-log2(real_multiplier[l]))
pow2_multiplier[l] = 2 ^ (-shift_layer[l])
```

因此，per-tensor 方案可以把 `shift` 从每个输出通道一个值简化为每层一个值；但由于 `b_fold[l][c]` 仍随输出通道变化，`bias_int[l][c]` 仍然必须按输出通道保存。

硬件后处理等价为：

```text
psum_int32 = sum(x_int8 * weight_int8)
wide       = psum_int32 + bias_int
rounded    = wide + (1 << (shift - 1))   if rounding enabled
y_int8     = clamp_s8(rounded >>> shift)
y_int8     = max(y_int8, 0)              if ReLU enabled
```

### 3.4 本次导出的 activation scale

per-channel 主方案 8-batch calibration 得到的 per-layer activation scale：

| ID | Name | Scale |
|---:|---|---:|
| 0 | input | 0.554299 |
| 1 | l00_conv_input | 0.032009 |
| 2 | l01_conv1 | 0.050224 |
| 3 | l02_conv2_0 | 0.062827 |
| 4 | l03_conv2_1 | 0.063757 |
| 5 | l04_conv2_2 | 0.072726 |
| 6 | l05_conv3_0 | 0.098016 |
| 7 | l06_conv3_1 | 0.044539 |
| 8 | l07_conv3_2 | 0.050631 |
| 9 | l08_conv4_0 | 0.114660 |
| 10 | l09_conv4_1 | 0.057989 |
| 11 | l10_conv4_2 | 0.066146 |
| 12 | l11_conv_out | 0.080831 |

这些 scale 是硬件等价参考路径中每层输入 / 输出 INT8 数值域的依据。

per-tensor 对比方案 8-batch calibration 得到的 per-layer activation scale：

| ID | Name | Scale |
|---:|---|---:|
| 0 | input | 0.554299 |
| 1 | l00_conv_input | 0.039831 |
| 2 | l01_conv1 | 0.053187 |
| 3 | l02_conv2_0 | 0.066215 |
| 4 | l03_conv2_1 | 0.071315 |
| 5 | l04_conv2_2 | 0.084602 |
| 6 | l05_conv3_0 | 0.099908 |
| 7 | l06_conv3_1 | 0.047352 |
| 8 | l07_conv3_2 | 0.058556 |
| 9 | l08_conv4_0 | 0.123647 |
| 10 | l09_conv4_1 | 0.059049 |
| 11 | l10_conv4_2 | 0.065308 |
| 12 | l11_conv_out | 0.083950 |

## 4. 精度结果

注意：`train_second_hw_qat.py` 训练结束后自动跑的 eval 是普通 OpenPCDet eval。因为 fake quant 主要在训练模式启用，所以它不能完全代表硬件路径。论文中应优先使用 `test_second_hw_qat.py --eval_hw_ref` 的结果。

### 4.1 FP32 epoch80 baseline

来自：

```text
output/kitti_models/second/default/eval/epoch_80/val/default/log_eval_20260304-145828.txt
```

KITTI 3D AP R40：

| Class | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Car | 86.7850 | 76.6670 | 72.4570 |
| Pedestrian | 55.3908 | 48.6978 | 43.3576 |
| Cyclist | 79.6481 | 61.3241 | 56.9973 |

### 4.2 per-channel 主方案 HW-equivalent INT8 result

来自：

```text
output/kitti_models/second_hw_qat/hw_qat_10ep/eval/epoch_10/val/hw_qat/hw_ref/
```

该路径使用 int8-valued sparse features / weights、integer bias、per-channel shift-only requant、clamp 和 ReLU，再反量化回 FP32 接后续 OpenPCDet 模块。

KITTI 3D AP R40：

| Class | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Car | 87.7663 | 75.5503 | 70.7609 |
| Pedestrian | 56.8533 | 50.4542 | 45.0460 |
| Cyclist | 81.7871 | 61.4372 | 57.3116 |

完整 R40 指标如下。

Car:

| Metric | Easy | Moderate | Hard |
|---|---:|---:|---:|
| bbox AP_R40 | 95.3522 | 90.8335 | 88.1501 |
| BEV AP_R40 | 92.1135 | 87.2082 | 84.4129 |
| 3D AP_R40 | 87.7663 | 75.5503 | 70.7609 |
| AOS AP_R40 | 95.2570 | 90.4798 | 87.6279 |

Pedestrian:

| Metric | Easy | Moderate | Hard |
|---|---:|---:|---:|
| bbox AP_R40 | 70.5059 | 64.8346 | 60.6955 |
| BEV AP_R40 | 61.1479 | 55.3956 | 50.1921 |
| 3D AP_R40 | 56.8533 | 50.4542 | 45.0460 |
| AOS AP_R40 | 65.7359 | 59.6964 | 55.4418 |

Cyclist:

| Metric | Easy | Moderate | Hard |
|---|---:|---:|---:|
| bbox AP_R40 | 88.6345 | 70.1911 | 66.5198 |
| BEV AP_R40 | 88.5363 | 65.6813 | 61.1556 |
| 3D AP_R40 | 81.7871 | 61.4372 | 57.3116 |
| AOS AP_R40 | 88.4041 | 69.8166 | 66.1047 |

Recall:

| Threshold | ROI recall | RCNN recall |
|---:|---:|---:|
| 0.3 | 0.0000 | 0.9048 |
| 0.5 | 0.0000 | 0.8435 |
| 0.7 | 0.0000 | 0.5944 |

相对 FP32 epoch80：

| Class | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Car | +0.9813 | -1.1167 | -1.6961 |
| Pedestrian | +1.4625 | +1.7564 | +1.6884 |
| Cyclist | +2.1390 | +0.1131 | +0.3143 |

结论：

- per-channel HW-equivalent INT8 benchmark 精度下降可控。
- Car hard 有约 `1.70 AP` 下降，是当前最明显的代价。
- Pedestrian / Cyclist 在该 baseline 对比下没有下降，反而略高。
- 对论文表述，可写为：只量化 SECOND 3D sparse backbone 的硬件约束 INT8 版本，在 KITTI val 上保持与 FP32 baseline 接近的检测精度。

### 4.3 per-tensor 极简方案 HW-equivalent INT8 result

来自：

```text
output/kitti_models/second_hw_qat/hw_qat_10ep_per_tensor/eval/epoch_10/val/hw_qat/hw_ref/
```

该路径使用 per-tensor folded weight scale，因此同一层内 `real_multiplier = Sx * Sw_layer / Sy` 为单一值，`shift` 实际退化为 per-layer。软件导出文件中仍按 channel 重复写出同一个 shift，方便复用现有 `params_layerXX.csv` 读取逻辑。

KITTI 3D AP R40：

| Class | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Car | 86.1575 | 76.2363 | 72.8132 |
| Pedestrian | 44.7804 | 38.2011 | 34.0467 |
| Cyclist | 78.3606 | 61.6026 | 57.6272 |

完整 R40 指标如下。

Car:

| Metric | Easy | Moderate | Hard |
|---|---:|---:|---:|
| bbox AP_R40 | 95.3313 | 90.7962 | 89.1380 |
| BEV AP_R40 | 91.1423 | 86.2201 | 83.6983 |
| 3D AP_R40 | 86.1575 | 76.2363 | 72.8132 |
| AOS AP_R40 | 95.2929 | 90.5393 | 88.7627 |

Pedestrian:

| Metric | Easy | Moderate | Hard |
|---|---:|---:|---:|
| bbox AP_R40 | 62.7328 | 56.5729 | 53.4974 |
| BEV AP_R40 | 51.3871 | 44.6304 | 40.7352 |
| 3D AP_R40 | 44.7804 | 38.2011 | 34.0467 |
| AOS AP_R40 | 57.8004 | 51.4842 | 48.4263 |

Cyclist:

| Metric | Easy | Moderate | Hard |
|---|---:|---:|---:|
| bbox AP_R40 | 86.6443 | 72.1422 | 68.6028 |
| BEV AP_R40 | 83.0495 | 66.4620 | 62.2521 |
| 3D AP_R40 | 78.3606 | 61.6026 | 57.6272 |
| AOS AP_R40 | 86.0742 | 71.4288 | 67.8854 |

Recall:

| Threshold | ROI recall | RCNN recall |
|---:|---:|---:|
| 0.3 | 0.0000 | 0.9397 |
| 0.5 | 0.0000 | 0.8675 |
| 0.7 | 0.0000 | 0.6249 |

相对 FP32 epoch80：

| Class | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Car | -0.6275 | -0.4307 | +0.3562 |
| Pedestrian | -10.6104 | -10.4967 | -9.3109 |
| Cyclist | -1.2875 | +0.2785 | +0.6299 |

相对 per-channel 主方案：

| Class | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Car | -1.6088 | +0.6860 | +2.0523 |
| Pedestrian | -12.0729 | -12.2531 | -10.9993 |
| Cyclist | -3.4265 | +0.1654 | +0.3156 |

结论：

- per-tensor 方案确实实现了 per-layer shift 的硬件简化。
- Car 类别整体接近 FP32 baseline，Cyclist moderate / hard 也没有明显退化。
- Pedestrian 明显下降，3D AP R40 相比 FP32 下降约 `9-11 AP`，相比 per-channel 主方案下降约 `11-12 AP`。
- 该方案不建议作为论文主 benchmark；更适合作为“极简硬件参数加载方案”的 ablation，用于说明进一步减少参数带宽会牺牲行人等小目标类别精度。

### 4.4 普通 QAT eval 与 HW-reference eval 的区别

本实验中有两个 eval 结果，论文中需要区分：

1. `train_second_hw_qat.py` 训练结束后自动触发的 eval：
   - 使用 OpenPCDet 标准 eval。
   - 主要验证 checkpoint 正常、训练未破坏 FP32 检测图。
   - 由于当前 fake quant 主要在 training mode 生效，它不能作为最终硬件路径结果。

2. `test_second_hw_qat.py --eval_hw_ref`：
   - 使用导出的 `weight_int8 / bias_int / shift / activation scale`。
   - 在 3D backbone 内模拟 `INT8 MAC + INT32 bias + shift-only requant + clamp + ReLU`。
   - 这是本文推荐写入论文主表的 HW-equivalent INT8 benchmark。

因此，后续论文表格中建议使用 `HW-reference INT8`，普通 QAT eval 只作为训练 sanity check。

## 5. 硬件导出检查结果

per-channel 主方案来自：

```text
output/kitti_models/second_hw_qat/hw_qat_10ep/hw_export/quant_scheme_report.md
```

per-channel 检查结果：

```text
Hard failures: 0
Warnings: 0
```

导出量化语义：

- activation: signed symmetric INT8, zero point = 0
- weight: signed symmetric INT8, zero point = 0, per-output-channel
- BN: offline folding
- accumulator: INT32
- requantization: shift-only nearest power-of-two
- checked bias width: 32-bit
- checked shift width: 5-bit

per-tensor 对比方案来自：

```text
output/kitti_models/second_hw_qat/hw_qat_10ep_per_tensor/hw_export/quant_scheme_report.md
```

per-tensor 检查结果：

```text
Hard failures: 0
Warnings: 0
```

导出量化语义：

- activation: signed symmetric INT8, zero point = 0
- weight: signed symmetric INT8, zero point = 0, per-tensor
- BN: offline folding
- accumulator: INT32
- requantization: shift-only nearest power-of-two
- checked bias width: 32-bit
- checked shift width: 5-bit

两个结果都说明 checkpoint 可以被解释为硬件可执行的 `bias + shift` INT8 sparse backbone benchmark。差异在于 per-channel 主方案需要每个输出通道独立的 `shift[c]`，而 per-tensor 对比方案每层所有输出通道共享同一个 `shift_layer`。

## 6. Bias / shift 真实分布

### 6.1 per-channel 主方案统计

统计文件：

```text
output/kitti_models/second_hw_qat/hw_qat_10ep/hw_export/bias_int_stats.csv
output/kitti_models/second_hw_qat/hw_qat_10ep/hw_export/requant_shift_stats.csv
```

per-channel 全局统计：

```text
bias_int min      = -23764
bias_int max      =  18947
bias_int max_abs  =  23764
shift min         =  1
shift max         =  11
relative error max  = 0.413440
relative error mean = 0.180018
relative error p99  = 0.407202
```

逐层最大 bias 与 shift 范围：

| Layer | Module | Cout | max_abs_bias | bits_needed | shift_min | shift_max |
|---:|---|---:|---:|---:|---:|---:|
| 0 | conv_input.0 | 16 | 3267 | 13 | 1 | 7 |
| 1 | conv1.0.0 | 16 | 18947 | 16 | 8 | 10 |
| 2 | conv2.0.0 | 32 | 5728 | 14 | 8 | 10 |
| 3 | conv2.1.0 | 32 | 14991 | 15 | 9 | 11 |
| 4 | conv2.2.0 | 32 | 11207 | 15 | 9 | 11 |
| 5 | conv3.0.0 | 64 | 9183 | 15 | 9 | 11 |
| 6 | conv3.1.0 | 64 | 9723 | 15 | 8 | 10 |
| 7 | conv3.2.0 | 64 | 15577 | 15 | 9 | 11 |
| 8 | conv4.0.0 | 64 | 11483 | 15 | 10 | 11 |
| 9 | conv4.1.0 | 64 | 7833 | 14 | 8 | 10 |
| 10 | conv4.2.0 | 64 | 23764 | 16 | 10 | 10 |
| 11 | conv_out.0 | 128 | 5634 | 14 | 8 | 9 |

由此可得：

- per-channel checkpoint 的 `bias_int` 实际只需要 signed 16-bit。
- per-channel checkpoint 的 shift 实际只需要 unsigned 4-bit。
- 原先 32-bit bias + 5-bit shift 明显保守。
- 27/28-bit bias 都有极大余量。

### 6.2 per-tensor 对比方案统计

统计文件：

```text
output/kitti_models/second_hw_qat/hw_qat_10ep_per_tensor/hw_export/bias_int_stats.csv
output/kitti_models/second_hw_qat/hw_qat_10ep_per_tensor/hw_export/requant_shift_stats.csv
```

per-tensor 全局统计：

```text
bias_int min        = -17875
bias_int max        =  10135
bias_int max_abs    =  17875
shift min           =  1
shift max           =  10
relative error max  = 0.406229
relative error mean = 0.149674
```

逐层最大 bias 与 shift 范围：

| Layer | Module | Cout | max_abs_bias | bits_needed | shift_min | shift_max |
|---:|---|---:|---:|---:|---:|---:|
| 0 | conv_input.0 | 16 | 65 | 8 | 1 | 1 |
| 1 | conv1.0.0 | 16 | 4282 | 14 | 8 | 8 |
| 2 | conv2.0.0 | 32 | 4877 | 14 | 8 | 8 |
| 3 | conv2.1.0 | 32 | 10124 | 15 | 9 | 9 |
| 4 | conv2.2.0 | 32 | 7562 | 14 | 9 | 9 |
| 5 | conv3.0.0 | 64 | 4946 | 14 | 9 | 9 |
| 6 | conv3.1.0 | 64 | 4880 | 14 | 8 | 8 |
| 7 | conv3.2.0 | 64 | 8807 | 15 | 9 | 9 |
| 8 | conv4.0.0 | 64 | 4862 | 14 | 10 | 10 |
| 9 | conv4.1.0 | 64 | 4626 | 14 | 8 | 8 |
| 10 | conv4.2.0 | 64 | 17875 | 16 | 9 | 9 |
| 11 | conv_out.0 | 128 | 3319 | 13 | 8 | 8 |

由此可得：

- per-tensor 版本的 `bias_int` 仍然实际只需要 signed 16-bit。
- per-tensor 版本的 shift 实际只需要 unsigned 4-bit。
- 每层 `shift_min == shift_max`，说明该版本确实可以在硬件中使用单个 `shift_layer`，不需要按输出通道读取 `shift[c]`。
- per-tensor 的 `bias_int` 最大绝对值 `17875` 小于 per-channel 主方案的 `23764`，但这没有带来精度优势；精度瓶颈主要来自 per-tensor weight 对输出通道动态范围的压缩。

## 7. 当前 per-channel weight + per-channel shift 方案的硬件含义

per-channel 10 epoch 主实验使用的是：

```text
weight_quant = per_channel
```

因此，per-channel 导出的硬件参数粒度是：

```text
per layer:
  activation input scale  Sx
  activation output scale Sy
  relu_en
  rounding_en

per output channel:
  weight_int8[c]
  weight scale Sw[c]       (导出分析使用，硬件运行时不一定需要显式加载)
  bias_int[c]
  shift[c]
```

这里需要特别注意：虽然 activation 是 per-layer / per-tensor scale，但这并不意味着 `bias/shift` 也是 per-layer。原因是：

```text
real_multiplier[c] = Sx * Sw[c] / Sy
bias_int[c]        = round(b_fold[c] / (Sx * Sw[c]))
```

其中 `Sx` 和 `Sy` 是每层 1 个，但 `Sw[c]` 与 `b_fold[c]` 都随输出通道变化。因此 per-channel 主方案下：

```text
shift[c]    仍然是 per-output-channel
bias_int[c] 仍然是 per-output-channel
```

### 7.1 对 MAC / param preload 的影响

当前硬件在计算某一层时，可以把 activation scale 作为 layer-level context：

```text
Sx / Sy: layer-level metadata
```

但每次切换输出通道或 Cout tile 时，仍然需要加载对应输出通道的 param：

```text
for each Cout tile:
  preload bias_int[c]
  preload shift[c]
  preload weight tile
  run INT8 MAC
  apply bias + shift + clamp + ReLU
```

也就是说，当前 per-channel weight + per-channel shift 方案可以简化 activation scale 管理，但不能取消 per-channel param SRAM / preload。它对硬件设计的合理组织方式是：

```text
Activation scale SRAM/register:
  one Sx and one Sy per layer

Param SRAM:
  bias_int and shift indexed by output channel

Weight SRAM:
  weight_int8 indexed by output channel tile and kernel/Cin tile
```

这也是本文推荐 `2 output channels = 64-bit param pair` 的原因：当前算法仍需要 per-channel param，但可以把它压成规整的 64-bit pair，从而减少 DRAM word 浪费和 Param SRAM 控制复杂度。

### 7.2 per-channel 方案对硬件设计的指导

per-channel 主方案最匹配的硬件执行模型是：

```text
Descriptor / layer metadata:
  layer_id
  sparse conv geometry
  Sx
  Sy
  relu_en
  rounding_en

Param payload:
  bias_int[c]
  shift[c]

Weight payload:
  weight_int8[kz][ky][kx][cin][cout]
```

推荐硬件实现上采用 Cout tile 级参数预取：

```text
1. Load input sparse activation tile
2. Load output-channel weight tile
3. Load output-channel param tile: bias_int + shift
4. Accumulate INT32 psum
5. Apply per-channel bias and shift
6. Clamp to signed INT8
7. Apply ReLU if enabled
8. Emit output sparse feature
```

这样设计的优点是：

- 与当前 QAT / export 语义完全一致。
- 保留 per-channel weight scale 带来的精度优势。
- 不需要硬件 multiplier，只需要 shift-only requant。
- param pair 可以压到 64-bit / 2 channels，避免当前 74-bit pair 的 2-word fetch 问题。

缺点是：

- 每个 Cout tile 仍需要 param preload。
- shift SRAM / shift lane 仍然按 output channel 存在。
- descriptor 中如果已有 single `quant_shift` 字段，它不能代表当前 per-channel shift，只能作为 reserved 或 debug/default 字段。

## 8. per-tensor weight 版本的硬件简化与代价

当前脚本已经完成 per-tensor weight 版本的 10 epoch QAT、HW export 和 HW-reference eval。训练命令为：

```bash
python train_second_hw_qat.py \
  --cfg_file cfgs/kitti_models/second_hw_qat.yaml \
  --pretrained_model ../output/kitti_models/second/default/ckpt/checkpoint_epoch_80.pth \
  --epochs 10 \
  --extra_tag hw_qat_10ep_per_tensor \
  --weight_quant per_tensor \
  --export_qparams
```

代码路径上，`--weight_quant per_tensor` 会让每一层 sparse conv 的 folded weight 只使用一个 scale：

```text
Sw[c] -> Sw_layer
```

此时：

```text
real_multiplier = Sx * Sw_layer / Sy
```

在同一层内不再随输出通道变化。因此本次导出中每层所有输出通道得到同一个 shift，可以把 shift 从 per-output-channel 简化成 per-layer：

```text
shift[c] -> shift_layer
```

这会带来更强的硬件简化：

```text
per layer:
  Sx
  Sy
  Sw_layer
  shift_layer

per output channel:
  weight_int8[c]
  bias_int[c]
```

注意，即使使用 per-tensor weight，`bias_int` 仍然不能自然变成 per-layer。因为 BN folding 后的 `b_fold[c]` 仍然是每个输出通道不同：

```text
bias_int[c] = round(b_fold[c] / (Sx * Sw_layer))
```

所以 per-tensor weight 最多可以自然消除 per-channel shift，不能消除 per-channel bias。

本次 per-tensor 导出已经验证：

```text
Hard failures: 0
Warnings: 0

bias_int max_abs = 17875
bias bits needed = signed 16-bit
shift range      = 1..10
shift bits needed= unsigned 4-bit
```

逐层 `shift_min == shift_max`，说明 shift 在数学上已经是 per-layer。当前 `test_second_hw_qat.py` 仍会在 `params_layerXX.csv` 里按 channel 写出重复的 shift，这是为了复用现有读取逻辑；若硬件 RTL 要真正减少参数读取，建议后续增加一个导出选项：

```text
--shift_granularity per_layer
```

并导出：

```text
params_layerXX.csv:
  bias_int[c] per output channel
  shift_layer once per layer
```

### 8.1 per-tensor 方案的硬件加载要求

per-tensor 方案推荐的硬件执行模型为：

```text
Descriptor / layer metadata:
  layer_id
  sparse conv geometry
  Sx
  Sy
  Sw_layer                 (可只用于离线分析；硬件运行时可不加载)
  shift_layer
  relu_en
  rounding_en

Param payload:
  bias_int[c]

Weight payload:
  weight_int8[kz][ky][kx][cin][cout]
```

与 per-channel 主方案相比，per-tensor 方案在 Cout tile 切换时：

```text
still preload:
  weight tile
  bias_int[c]

no longer preload per-channel:
  shift[c]
```

这意味着 Param SRAM 可以只按 output channel 存 `bias_int[c]`，`shift_layer` 放在 layer descriptor / small register file 中即可。对于 RTL 来说，这是最直接的硬件简化点：减少每个输出通道的参数字段、减少 shift SRAM 端口或 shift lane 的索引需求，并降低 Cout tile param preload 带宽。

### 8.2 per-tensor 方案的精度代价

per-tensor HW-reference 3D AP R40 为：

| Class | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Car | 86.1575 | 76.2363 | 72.8132 |
| Pedestrian | 44.7804 | 38.2011 | 34.0467 |
| Cyclist | 78.3606 | 61.6026 | 57.6272 |

相对 per-channel 主方案，Pedestrian 下降约 `11-12 AP`，这是不可忽略的精度代价。该结果说明：

- per-tensor weight 可以作为极简硬件方案。
- per-layer shift 的硬件简化是成立的。
- 当前 checkpoint 下，它不适合作为主 benchmark。
- 论文中更合理的写法是：per-channel 作为主设计，per-tensor 作为 design-space / ablation，用于展示进一步减少参数加载时的精度代价。

## 9. per-channel 主方案的合理加速器参数

推荐作为论文主设计参数：

```text
activation       : signed INT8, zero point = 0
weight           : signed INT8, zero point = 0, per-output-channel scale
psum             : signed INT32
bias             : signed 27-bit per output channel
shift            : unsigned 5-bit per output channel
requant          : ((psum + bias + rounding) >>> shift)
output           : signed INT8 clamp, then optional ReLU
param per channel: 27 + 5 = 32 bit
param per pair   : 64 bit for 2 output channels
```

选择 `bias=27b, shift=5b` 的原因：

- 该格式与当前 RTL 语义最接近。
- `shift=5b` 保留 `0..31`，对未来 checkpoint / calibration / 数据集变化更稳。
- `bias=27b` 相对当前 `16b` 真实需求有 11-bit 安全余量。
- 两个 channel 的 param pair 正好 64-bit，可避免当前 74-bit pair 需要两个 64-bit DRAM word 的不规整问题。

这版参数适合作为论文中的主张：

```text
The accelerator uses a compact 64-bit two-channel parameter pair:
two signed 27-bit integer biases and two 5-bit requant shifts.
```

对应硬件收益：

- param pair 从当前 `74b` 降到 `64b`
- 每 2 个 output channel 参数正好一个 64-bit word
- Param SRAM / DMA / preload 控制更规整
- 保留足够算法余量，避免被质疑只对单次 checkpoint 过拟合

## 10. per-channel 主方案的硬件精简极限参数

当前数据支持的极限压缩格式：

```text
activation       : signed INT8
weight           : signed INT8
psum             : signed INT32
bias             : signed 16-bit per output channel
shift            : unsigned 4-bit per output channel
param per channel: 16 + 4 = 20 bit
param per pair   : 40 bit for 2 output channels
```

选择依据：

- `max_abs_bias = 23764`
- signed 16-bit 范围为 `[-32768, 32767]`，可以覆盖当前所有层 / 通道。
- `shift_max = 11`
- unsigned 4-bit 范围为 `[0, 15]`，可以覆盖当前所有层 / 通道。

进一步可考虑的 packing：

```text
2 channels: 40 bit
3 channels: 60 bit
4 channels: 80 bit
```

如果系统 DRAM / SRAM 以 64-bit word 为主，`3 channels = 60 bit` 是理论上非常紧凑的 packing；如果 compute tile / Cout lane 以 2 或 4 通道成组，`2 channels = 40 bit` 或 `4 channels = 80 bit` 更容易接硬件 lane。

但这个极限格式不建议直接作为论文主设计，原因：

- 16-bit bias 是根据当前 checkpoint 和当前 calibration 得到的最小结论，余量较小。
- 换 calibration batch、换 seed、换训练 epoch、换数据集后，bias 可能超过 16-bit。
- 非 64-bit 对齐的 40-bit pair 会增加 packing / unpacking 复杂度。
- 若要主张 16-bit bias，需要更多 checkpoint 和更大 calibration set 的统计支撑。

因此，`bias=16b, shift=4b` 更适合写成：

```text
an aggressive lower-bound format supported by the current checkpoint statistics
```

而不是默认硬件规格。

## 11. 可选折中参数

如果希望比 27b 更激进，但又不想压到 16b，可以考虑：

```text
bias             : signed 24-bit
shift            : unsigned 4-bit
param per channel: 28 bit
param per pair   : 56 bit
```

优点：

- bias 比当前最大需求多 8-bit 余量。
- shift 4-bit 已由当前 `shift_max=11` 支撑。
- 比 `27b + 5b` 更省。

缺点：

- 2-channel pair 为 56-bit，虽然可放进 64-bit，但有 8-bit unused / reserved。
- 相比 `27b + 5b = 64b pair`，论文解释上不如 64-bit exact pair 简洁。

也可以考虑：

```text
bias             : signed 28-bit
shift            : unsigned 4-bit
param per channel: 32 bit
param per pair   : 64 bit
```

这个格式与 `27b + 5b` 一样是 64-bit pair，但把余量更多给 bias，减少 shift 范围。由于当前 `shift_max=11`，4-bit shift 足够，因此这是另一个很合理的主设计候选。

## 12. per-tensor 极简方案的硬件参数

per-tensor 方案不建议作为论文主 benchmark，但它可以作为极简硬件设计对比。其参数粒度为：

```text
activation       : signed INT8, zero point = 0
weight           : signed INT8, zero point = 0, per-layer scale
psum             : signed INT32
bias             : per output channel
shift            : per layer
requant          : ((psum + bias[c] + rounding) >>> shift_layer)
output           : signed INT8 clamp, then optional ReLU
```

基于当前 per-tensor checkpoint 的真实统计：

```text
bias max_abs     : 17875
bias minimum     : signed 16-bit per output channel
shift range      : 1..10
shift minimum    : unsigned 4-bit per layer
```

若只写极限格式：

```text
bias             : signed 16-bit per output channel
shift            : unsigned 4-bit per layer
```

若希望作为更稳健的硬件 ablation：

```text
bias             : signed 24-bit or 27-bit per output channel
shift            : unsigned 4-bit or 5-bit per layer
```

该方案的硬件收益是减少 per-output-channel shift 字段和 preload；代价是 Pedestrian 精度明显下降。因此建议将它写为：

```text
an aggressive per-layer-shift design-space point, not the main benchmark
```

## 13. 参数选择建议

推荐分四档表述：

### 13.1 论文主设计，建议采用

```text
scheme = per-channel weight + per-channel shift
bias  = signed 27-bit
shift = unsigned 5-bit
pair  = 64-bit / 2 channels
```

理由：最稳健，和当前 RTL shift 设计一致，同时解决 74-bit param pair 不规整问题。

### 13.2 同样合理的精简主设计候选

```text
scheme = per-channel weight + per-channel shift
bias  = signed 28-bit
shift = unsigned 4-bit
pair  = 64-bit / 2 channels
```

理由：当前真实 shift 最大只有 11，4-bit 足够；bias 余量比 27-bit 更大。若 RTL 愿意把 shift 从 5-bit 改为 4-bit，这是很干净的 64-bit pair 方案。

### 13.3 per-channel 当前 checkpoint 支持的极限压缩

```text
scheme = per-channel weight + per-channel shift
bias  = signed 16-bit
shift = unsigned 4-bit
param = 20-bit / channel
```

理由：当前统计完全支持，但论文主文不宜直接采用为默认规格，除非后续补充更多 checkpoint / calibration / accuracy 验证。

### 13.4 per-tensor 极简对比方案

```text
scheme = per-tensor weight + per-layer shift + per-channel bias
bias   = signed 16-bit minimum, or 24/27-bit robust
shift  = unsigned 4-bit minimum, per layer
```

理由：硬件参数加载最少，但 Pedestrian 3D AP R40 明显下降，应作为 ablation / design-space discussion，而不是主 benchmark。

## 14. 最终建议结论

面向当前 AI 加速器设计论文，建议采用以下结论：

```text
For the accelerator-facing SECOND benchmark, only VoxelBackBone8x is quantized.
The recommended main design uses symmetric INT8 activations and per-channel INT8
weights, INT32 accumulation, offline BN folding, and per-channel integer bias plus
per-channel shift-only requantization. After 10-epoch HW-QAT, this per-channel
hardware-equivalent reference path keeps KITTI accuracy close to the FP32 baseline.
The measured integer bias requires at most 16 signed bits and the requant shift
requires at most 4 bits. For a robust accelerator design, we choose a 64-bit
two-channel parameter pair, either 27-bit bias + 5-bit shift or 28-bit bias + 4-bit
shift.

The per-tensor weight variant further simplifies hardware by using one weight scale
and one requant shift per layer while keeping bias per output channel. This reduces
per-channel parameter loading, but the HW-reference result shows a large Pedestrian
accuracy drop. Therefore, it should be reported as an aggressive design-space point
rather than the main benchmark.
```

如果希望论文表述更保守，主方案优先写 `per-channel weight + 27b bias + 5b per-channel shift`。如果希望突出硬件精简和 64-bit pair 的整齐性，同时愿意把 shift 改成 4-bit，则写 `per-channel weight + 28b bias + 4b per-channel shift`。如果要展示极限探索，可以把 `per-channel 16b bias + 4b shift` 和 `per-tensor weight + per-layer shift` 放在 ablation / design-space discussion 中。

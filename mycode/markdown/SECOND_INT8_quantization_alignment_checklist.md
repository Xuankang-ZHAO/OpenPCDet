# SECOND INT8 Quantization Alignment Checklist

本文用于把当前 RTL 加速器设计与 SECOND / `VoxelBackBone8x_INT8` 的算法量化流程对齐。重点不是重新设计量化算法，而是确认硬件正在假设的 INT8 数据格式、BN folding、bias/shift payload 位宽、requantization 形式和导出文件格式是否真实成立。

这份文档可以复制到算法量化工作目录，让 Codex 或算法侧脚本继续检查模型代码、checkpoint、QAT/PTQ 导出结果和真实参数分布。

## 1. 当前硬件事实

当前 RTL 的 MAC 与量化路径是固定的：

```text
input activation : signed INT8
weight           : signed INT8
psum             : signed INT32
bias             : signed INT32 per output channel
shift            : unsigned 5-bit per output channel
postprocess      : (psum + bias + rounding) >>> shift
output           : signed INT8 clamp, optional ReLU
```

对应 RTL 位置：

- `rtl/compute/compute_pkg.sv`
  - `COMPUTE_DATA_W = 8`
  - `COMPUTE_ACC_W = 32`
  - `COMPUTE_BIAS_TILE_W = COMPUTE_COUT_TILE * COMPUTE_ACC_W`
  - `COMPUTE_SHIFT_TILE_W = COMPUTE_COUT_TILE * 5`
  - `COMPUTE_PARAM_PAIR_W = 2 * 32 + 2 * 5 = 74`
- `rtl/compute/mac/mac_quantize.sv`
  - `wide = psum + bias`
  - optional rounding offset
  - arithmetic right shift by per-channel `shift`
  - ReLU and signed INT8 saturation
- `rtl/memory/memory_pkg.sv`
  - one param pair stores 2 output channels
  - current pair width is 74 bits
  - because 74 bits exceeds one 64-bit DRAM word, each pair consumes two 64-bit words during layer param DMA

Important: 当前硬件没有以下能力：

- no zero-point add/subtract path
- no asymmetric activation or weight quantization support
- no per-channel scale multiplier
- no fixed-point requant multiplier such as `(x * M) >> n`
- no runtime use of descriptor field `quant_shift` in MAC postprocess

因此算法侧必须确认当前模型导出的 INT8 推理格式是否能等价为：

```text
y_int8 = clamp_s8(((psum_int32 + bias_int32) + rounding) >> shift)
```

其中 `shift` 最好是 per-output-channel power-of-two requant scale。

## 2. 当前疑虑

### 2.1 Param 不是主创新，但可能变成隐藏负担

论文主线关注 sparse conv 数据流、block partition、RMCAM/event generation、memory/compute 协同，而不是 param payload 本身。因此 param 设计应该尽量朴素、紧凑、可证明，不应成为新的瓶颈。

当前格式的问题是：

```text
bias  = 32 bit/channel
shift = 5 bit/channel
total = 37 bit/channel
2 channels = 74 bit
```

这使得一个 2-channel param pair 无法落在一个 64-bit word 内。虽然每层 param DRAM 总量远小于 weight，但在本地 MAC lane 中，每个输出坐标、每个 Cout group 都要重新 preload param。对低 event-count 的 sparse segment，param preload latency/bandwidth 会更显眼。

### 2.2 当前硬件只支持 shift-only requant

常见 INT8 推理流程可能需要：

```text
psum_int32 * (Sx * Sw / Sy)
```

如果 `Sx * Sw / Sy` 不是 `2^-shift`，当前硬件的 shift-only requant 会产生系统性误差。算法侧需要确认当前 QAT/PTQ 是否已经把 requant scale 约束成 power-of-two，或者是否只是普通 fake quant 后导出了 arbitrary scale。

### 2.3 Bias 位宽不应凭感觉决定

INT8 乘加 psum 使用 32-bit accumulator 是合理的，因为最大累加项数可能达到：

```text
kernel volume * Cin = 27 * 64 = 1728 terms
max product ~= 128 * 128 = 16384
max raw psum ~= 28.3M, needs about signed 26 bits
```

但这不自动意味着 bias payload 必须用 32 bits。BN folding 后的 integer bias 分布应该由真实 checkpoint 和量化 scale 决定。

需要统计：

```text
bias_int[c] = round(bias_fold_fp32[c] / (Sx * Sw[c]))
```

然后观察所有层、所有输出通道的 max abs、bit width、outlier 和精度敏感性。

## 3. 算法侧必须确认的问题

### 3.1 网络结构与层表

请确认当前算法仓库实际部署的是不是以下硬件假定：

- backbone: `VoxelBackBone8x_INT8`
- sparse conv layer count: 12
- no residual addition path in the 3D backbone
- each sparse conv is followed by BN/ReLU or equivalent folded inference form
- layer channel schedule matches RTL scheduler:

```text
Layer 0 : Cin 4  -> Cout 16
Layer 1 : Cin 16 -> Cout 16
Layer 2 : Cin 16 -> Cout 32
Layer 3 : Cin 32 -> Cout 32
Layer 4 : Cin 32 -> Cout 32
Layer 5 : Cin 32 -> Cout 64
Layer 6 : Cin 64 -> Cout 64
Layer 7 : Cin 64 -> Cout 64
Layer 8 : Cin 64 -> Cout 64
Layer 9 : Cin 64 -> Cout 64
Layer 10: Cin 64 -> Cout 64
Layer 11: Cin 64 -> Cout 128
```

输出文件建议：

- `layer_inventory.csv`
- columns: `layer_id, module_name, conv_type, kernel, stride, cin, cout, has_bn, has_relu, has_residual`

### 3.2 Activation / weight quantization 是否对称

必须确认：

- activation 是否 signed INT8
- weight 是否 signed INT8
- activation zero point 是否恒为 0
- weight zero point 是否恒为 0
- per-tensor activation scale 还是 per-channel activation scale
- per-tensor weight scale 还是 per-output-channel weight scale
- first layer input feature 是否也已经 INT8，并且 scale 与硬件输入格式一致

当前 RTL 最自然支持：

```text
activation: signed symmetric INT8, per-layer/per-tensor scale
weight    : signed symmetric INT8, preferably per-output-channel scale
psum      : sum(x_int8 * w_int8)
```

输出文件建议：

- `quant_scheme_report.md`
- `activation_scales.csv`
- `weight_scales.csv`

### 3.3 BN folding 公式是否与硬件 bias 对齐

需要明确训练图中 Conv/BN/ReLU 的推理 folding 公式。

典型形式：

```text
w_fold[c] = w_fp32[c] * gamma[c] / sqrt(var[c] + eps)
b_fold[c] = beta[c] + (b_conv[c] - mean[c]) * gamma[c] / sqrt(var[c] + eps)
```

如果原 sparse conv 没有 conv bias，则 `b_conv[c] = 0`。

必须确认：

- BN 是否在导出前 fold 到 conv weight/bias
- fold 后 ReLU 是在 requant 之前还是之后表达
- training/eval mode 的 BN 参数是否固定
- eps 与 PyTorch/spconv 实现一致
- folded weight 再量化，还是 int8 weight 之后再 fold

输出文件建议：

- `bn_folding_report.md`
- `folded_bias_fp32.csv`
- `folded_weight_range.csv`

### 3.4 Integer bias 的真实位宽

请按硬件语义生成 integer bias：

```text
bias_int[c] = round(b_fold[c] / (Sx * Sw[c]))
```

其中：

- `Sx` 是该层输入 activation scale
- `Sw[c]` 是该输出通道 weight scale
- 若 weight 是 per-tensor scale，则 `Sw[c] = Sw`

需要统计所有层：

- max positive bias
- min negative bias
- max absolute bias
- signed bit width needed
- 99.9 percentile abs
- 是否存在 outlier channel
- outlier 是否来自 BN gamma/variance 异常

建议输出：

- `bias_int_stats.csv`
- columns:
  `layer_id, cout, min, max, max_abs, signed_bits_needed, p99_abs, p999_abs, outlier_channels`
- `bias_int_histograms/`

判定目标：

- 如果所有 `bias_int` 在 signed 27-bit 内，硬件可考虑 `bias=27b, shift=5b`
- 如果所有 `bias_int` 在 signed 28-bit 内且 shift 可压到 4-bit，硬件可考虑 `bias=28b, shift=4b`
- 如果确实需要 32-bit bias，必须给出层/通道证据，而不是默认沿用 32-bit

### 3.5 Requant scale 是否能被 shift-only 表达

当前硬件等价于：

```text
y_int = (psum + bias_int) >> shift
```

因此需要确认真实 requant scale：

```text
real_multiplier[c] = Sx * Sw[c] / Sy
```

是否可以近似为：

```text
real_multiplier[c] ~= 2^-shift[c]
```

必须统计：

- 每层每通道 ideal `real_multiplier`
- nearest `shift = round(-log2(real_multiplier))`
- relative error between `real_multiplier` and `2^-shift`
- shift min/max
- shift 是否超过 15
- shift 是否超过 31
- 使用 shift-only 后的 feature SQNR / cosine similarity / mAP 影响

输出文件建议：

- `requant_shift_stats.csv`
- columns:
  `layer_id, channel, sx, sw, sy, real_multiplier, nearest_shift, multiplier_pow2, relative_error`
- `shift_only_accuracy_report.md`

判定目标：

- 若 `shift <= 15` 对所有通道成立，可考虑 4-bit shift
- 若 `shift <= 31` 成立但超过 15，保留 5-bit shift
- 若 shift-only 误差不可接受，需要改算法 QAT 约束，或硬件增加 multiplier/scale path

### 3.6 QAT 是否加入硬件约束

如果当前训练只是普通 INT8 fake quant，算法侧需要评估是否重训或微调：

- constrain activation/weight zero point to 0
- constrain requant multiplier to power-of-two
- optionally constrain shift to 4-bit range `0..15`
- optionally constrain integer bias range to signed 27/28-bit
- export hardware-ready `bias_int` and `shift` directly

建议实验分组：

```text
Baseline A: current FP32 or existing INT8
Baseline B: existing INT8 export, no hardware constraint
Experiment C: symmetric INT8 + shift-only requant
Experiment D: symmetric INT8 + shift-only requant + 4-bit shift constraint
Experiment E: same as D + bias range regularization/clipping if needed
```

每组至少记录：

- validation mAP / NDS or project-specific metric
- per-layer activation error
- final detection error
- saturation ratio
- zero ratio after ReLU
- worst channels/layers

## 4. 建议的硬件 param 格式候选

### Option 0: 当前格式，功能安全但不经济

```text
per channel: bias[31:0] + shift[4:0] = 37b
per pair   : 74b
DRAM fetch : 2 x 64b per pair
```

优点：

- bias 位宽保守
- shift 范围 0..31
- RTL 已实现

缺点：

- pair 超过 64b，搬运和 SRAM 宽度不规整
- param payload 容易在论文设计审查中显得没有被认真约束

### Option A: 27-bit bias + 5-bit shift

```text
per channel: bias[26:0] + shift[4:0] = 32b
per pair   : 64b
```

优点：

- 保留 shift 0..31
- 每两个输出通道正好一个 64-bit word
- 对当前 RTL 改动相对直接

需要算法证明：

- all `bias_int` fit signed 27-bit
- no accuracy loss from bias clipping/sign-extension

### Option B: 28-bit bias + 4-bit shift

```text
per channel: bias[27:0] + shift[3:0] = 32b
per pair   : 64b
```

优点：

- bias 余量比 Option A 更大
- 每两个输出通道仍正好一个 64-bit word

需要算法证明：

- all shift fit 0..15
- shift-only 4-bit range 对精度无明显影响

### Option C: 32-bit bias + layer/global shift

```text
per channel: bias[31:0]
shift      : one per layer or one per Cout tile
```

适用条件：

- per-channel shift 分布很集中
- layer-level shift 精度损失可接受

风险：

- 可能牺牲精度
- 与 per-channel weight scale 的表达能力冲突

### Option D: multiplier + shift

```text
per channel: bias + multiplier + shift
```

只有在 shift-only QAT 无法达到精度要求时考虑。该方案会引入乘法器或更复杂 postprocess，不适合作为当前论文主线的默认设计。

## 5. 需要算法侧导出的最终硬件 payload

若采用 shift-only 方案，算法侧最终应能导出：

```text
layer_id
cout_channel
weight_int8[kz][ky][kx][cin]
bias_int
shift
relu_en
rounding_en
input_activation_scale
weight_scale
output_activation_scale
```

Param payload 建议最终导出为硬件可直接装载的 packed binary，并附带 human-readable CSV：

```text
params_layerXX.bin
params_layerXX.csv
weights_layerXX.bin
weights_layerXX_meta.csv
```

CSV 至少包括：

```text
layer_id, cout_channel, bias_int, bias_bits_needed, shift, sx, sw, sy, real_multiplier, pow2_multiplier, rel_error
```

## 6. 检查脚本建议

请在算法工作目录中新增或运行类似脚本：

```text
tools/export_second_int8_payload.py
tools/check_second_int8_payload.py
tools/compare_shift_only_inference.py
```

`check_second_int8_payload.py` 至少完成：

- load trained checkpoint
- fold BN
- compute int8 weights and scales
- compute integer bias
- compute ideal requant multiplier
- choose nearest shift
- check bit-width constraints
- emit CSV stats
- optionally run validation with hardware-equivalent quantization

建议 hard fail 条件：

```text
activation_zero_point != 0
weight_zero_point != 0
any bias_int outside selected signed bias width
any shift outside selected shift width
shift-only relative error above threshold without accuracy waiver
missing BN fold for any sparse conv
layer table mismatch with RTL scheduler
```

## 7. 与 RTL 设计对齐的开放问题

### Q1. `quant_shift` descriptor 字段是否应该删除、保留还是接入？

当前 RTL descriptor 有 `quant_shift` 字段，但 MAC lane 使用 Param SRAM 中的 per-channel shift。算法侧若导出 per-layer shift，则 RTL 可以简化 Param payload；若导出 per-channel shift，则 descriptor `quant_shift` 需要明确标注为 unused/reserved。

### Q2. Param SRAM 是否应改为 64-bit pair？

若算法证明 Option A 或 Option B 成立，建议 RTL 后续把 param pair 改成 64-bit，以避免 74-bit pair 导致的 2-word DRAM fetch。

### Q3. Layer 11 的 128 Cout 是否与 WT SRAM 容量一致？

当前架构文档已提示：Param/OFM path 支持 16 Cout tiles，但 WT resident depth 当前只建模 8 Cout tiles。算法侧和 RTL 侧需要共同确认 `64 -> 128` 的 `conv_out` 是如何分批装载、是否暂不执行、还是需要扩展 WT SRAM/loader。

### Q4. 输出 INT8 是否始终 signed？

RTL clamp 范围是 signed INT8 `[-128, 127]`。如果算法侧使用 unsigned activation 或 ReLU 后 `[0, 255]`，必须重新对齐。

## 8. 推荐结论目标

面向当前 AI 加速器论文，建议最终形成以下可陈述结论：

```text
The accelerator uses symmetric INT8 activation/weight with INT32 accumulation.
BatchNorm is folded offline. The postprocess is hardware-constrained to
integer bias addition plus power-of-two per-channel requantization.
The exported bias/shift payload is range-checked against the selected compact
64-bit two-channel param format, avoiding param bandwidth from becoming a
secondary bottleneck.
```

如果算法实验不支持 shift-only 或 compact bias，则应在论文中诚实写为限制或未来工作，而不是让 param payload 在硬件评审中成为未解释的隐性瓶颈。


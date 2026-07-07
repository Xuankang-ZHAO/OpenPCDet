# SECOND 固定帧 RTL Golden-Vector Package 规范

## 1. 目的与边界

本规范定义一个**固定 KITTI 样本**的最小 golden-vector package，用于对当前
SECOND `VoxelBackBone8x` INT8 RTL 加速器做逐层、逐整数值的回归。

加速器边界从 VFE 输出开始，到 `conv_out.0` 的 INT8 输出结束；VFE、
HeightCompression、2D backbone 和 detection head 不在本 package 的计算范围内。
因此，RTL 的真实输入不是原始点云，而是 VFE 输出量化后的稀疏特征。

本 package 是硬件负载与比对格式，不是 checkpoint、ONNX 或 PyTorch tensor
交换格式。它遵循奥卡姆剃刀原则：只保存 RTL 执行或判定正确性所需的信息，
不保存可以由这些信息唯一推出的副本。

具体地说：

- 固定 KITTI 帧只在 `manifest.json` 中以 split、sample ID、原始 `.bin` 的
  SHA-256 标识；**不复制**原始点云。
- 保存第一层的量化 VFE raw stream；这是 RTL 唯一外部 IFM 输入。
- 保存所有层的硬件格式权重、参数和整数输出 golden。
- 不重复保存每层 IFM：第 `l` 层的 IFM 就是第 `l-1` 层的 OFM。
- 不保存 FP32 VFE feature、FP32 OFM、反量化 feature、ONNX、checkpoint，也不
  保存全量 psum trace。发生失配时可由同一输入重跑到首个失配层定位。

一个 package 对应且只对应一个：QAT checkpoint、校准结果、KITTI 样本和 RTL
契约版本。任一项改变均须重新导出，不得混用文件。

## 2. 当前 RTL / 模型数值契约

| 项目 | 固定值 |
|---|---|
| 加速层 | 12 层：`conv_input.0` 至 `conv_out.0` |
| 输入、权重 | signed INT8；算法有效码值 `[-127, 127]`；zero point = 0 |
| 激活 scale | 每层 / 每 tensor；由 `models/hw_export/activation_scales.csv` 导出 |
| 权重 scale | 每输出通道；由 `models/hw_export/weight_scales.csv` 导出 |
| MAC | `INT8 × INT8 -> INT32` |
| bias | 每输出通道 signed INT16 |
| shift | 每输出通道 unsigned INT4 |
| Cin/Cout tile | 8 / 8 通道 |
| MAC lane | 4 条；一个权重 tile 为 `8 Cout × 8 Cin × 8 bit = 512 bit` |
| ReLU、rounding | 当前 12 层均为 enable |

对某层 `l`、输出通道 `c`，导出器的整数参考必须精确实现当前 RTL：

```text
psum    = int32(sum(x_q * w_q))
wide    = sign_extend(psum) + sign_extend(bias_int[c])
rounded = wide + (1 << (shift[c] - 1))    if shift[c] != 0
shifted = rounded >>> shift[c]
y_q     = max(shifted, 0)                 # 当前 12 层 ReLU 均 enable
y_q     = min(y_q, 127)
```

这里的右移是算术右移；对负数仍是 RTL 的“加正半 LSB 后右移”语义，不得替换为
框架默认的 `round`。因为本模型每层均有 ReLU，最终 OFM 必为 `[0,127]`，所以
当前 RTL `[-128,127]` 的物理饱和下界不会出现在 OFM 中。输入和权重 payload 中
也不得出现 `0x80`（即 `-128`）。若未来出现无 ReLU 的层，必须先统一 RTL 与
模型的下饱和契约，再复用本规范。

BN 必须在导出前 fold 到权重和 bias；RTL 不执行 BN。

## 3. 最小文件集

```text
second_<split>_<sample_id>_golden/
  manifest.json
  raw_vfe_voxel_stream.bin
  weights.bin
  params.bin
  ofm_golden.bin
```

所有 `.bin` 均是由连续 **64-bit word** 组成的二进制文件：一个 word 的 bit `[7:0]`
是文件中的第一个 byte，即 little-endian `uint64` 序列化。所有多字节字段均按此
规则解释。文件没有自描述 header；长度、分段 offset、记录数和 SHA-256 均由
`manifest.json` 提供，避免在每个文件重复存 header。

### 3.1 `manifest.json`（唯一控制面）

必须包含下列字段：

- `format_version: "second_rtl_golden_v2"`、`word_bits: 64`、
  `byte_order: "little"`；
- `sample`：KITTI split、sample ID、原始点云相对路径、原始点云 SHA-256；
- `provenance`：QAT checkpoint SHA-256、导出器 git revision、
  `activation_scales.csv` 和 `weight_scales.csv` 的 SHA-256；
- `quantization`：signed、zero point、`qmin=-127`、`qmax=127`，以及所有 13 个
  activation scale（`input` 与 `l00` 至 `l11`）；
- `partition_geometry`：当前 Block Manager 使用的 partition geometry，仅描述 RTL
  建目录所需的几何契约，不作为 golden 预分配 page 结果；
- `raw_vfe_voxel_stream`：真实 accelerator ingress stream。它只描述唯一 owner
  voxel 的输入记录、记录大小、排序和 SHA-256，不包含 halo 预展开、BM page 预分配
  或初始 sparse block directory；
- `layers[0..11]`：`layer_id`、模块名、conv kind、kernel、stride、logical
  Cin/Cout、ReLU、rounding、input/output activation scale、每通道 weight scale、
  以及 `weights.bin`、`params.bin`、`ofm_golden.bin` 中各自的 offset、长度、
  record count 和 SHA-256；
- `rtl_contract`：`cin_tile=8`、`cout_tile=8`、`bias_bits=16`、`shift_bits=4`、
  `weight_kernel_order="x*9+y*3+z"`、初始构建阶段
  `write_voxel_bytes=16`，以及本规范第 6 节列出的 layer-11 状态。

`manifest.json` 是唯一的层配置来源。它必须与
`rtl/scheduler/scheduler_descriptor_table.sv` 和
`models/hw_export/layer_inventory.csv` 一致；不再额外复制 CSV 或写第二份 layer
description 文件。

### 3.2 `raw_vfe_voxel_stream.bin`（真实 VFE ingress）

它是送入 RTL 初始构建阶段的量化 VFE 稀疏输入，按 VFE 输出的唯一 owner voxel
顺序顺序拼接。每个 voxel 恰有两个 64-bit word，即 16B：

```text
word 0 = {32'h00000000, coord32}
word 1 = {q[7], q[6], q[5], q[4], q[3], q[2], q[1], q[0]}
```

- `q[0..3]` 是 VFE feature 的四个 signed INT8 通道，量化 scale 为当前
  `input` scale（冻结 checkpoint 中为 `0.5542992353439331`）。
- `q[4..7]` 必须为零；这是第一层 `Cin=4` 填充到 8-lane IFM tile 的唯一 padding。
- `coord32` 使用当前硬件的全局坐标编码：`x[10:0]`、`y[21:11]`、
  `z[27:22]`、`reserved[30:28]=0`、`is_halo[31]`。
- `is_halo` 必须为 0。golden package 不导出预展开 halo，也不为 layer 0 预分配
  BM page。
- 坐标必须在硬件可表达范围内；整个 raw stream 内不得有重复 owner `coord32`。

RTL 启动时，Block Manager 必须沿用现有 write hash path 接收这些 raw VFE voxel，
按照 partition、hash 和 page allocator 自行建立初始 sparse block directory。配套的
VFE/input commit path 根据 BM 返回的 write descriptor 写入 layer0 IFM payload。
初始构建阶段的 `write_voxel_bytes` 固定为 layer0 IFM resident layout，即
`coord word + 1 feature word = 16B`；这不是 layer0 输出的 C16 `24B` 记录布局。
初始目录和 payload 构建完成后执行 role swap，使该 bank 成为 layer0 read
directory，随后 layer0 按正常 read scan、load、compute、writeback 流程执行。
每一层输出时继续复用同一套 BM write hash path，为下一层建立 block/page 管理结构。

若导出器或调试工具保留 `input_pages`，它只能作为算法侧派生参考或 debug oracle；
RTL testbench 不得把它作为直接输入 ABI，也不得绕过 BM 的 directory/page placement。

### 3.3 `weights.bin`（硬件 WT DMA 顺序）

每层 payload 紧密拼接，层起始位置由 manifest 指出。层内顺序固定为：

```text
kernel_idx -> cin_tile -> cout_tile -> word_idx(0..7)
```

其中每个 `word_idx` 是一个 64-bit word，正好对应一个输出通道的 8 个输入通道
权重：

```text
word_idx = co_lane
word[8*ci_lane +: 8] = w_q[cout_tile*8 + co_lane,
                              kernel_z, kernel_y, kernel_x,
                              cin_tile*8 + ci_lane]
```

对 3×3×3 层，RTL 的 kernel index 是 x-major：

```text
kernel_idx = kernel_x * 9 + kernel_y * 3 + kernel_z
```

算法导出 tensor 的次序为 `(Cout, Kz, Ky, Kx, Cin)`，故打包时不得把 `Kz/Ky/Kx`
直接当成 RTL 的 `kernel_idx` 次序。越过 logical Cin/Cout 的 lane 必须填零；本模型
只有 layer 0 的 Cin lane 4..7 需要这种填充。

当前 layer 0 至 layer 10 的 kernel 均为 3×3×3。对 layer 11 的算法 kernel
3×1×1，payload 仍须完整导出：将其 `(kz=0,1,2, ky=0, kx=0)` 分别放入
`kernel_idx=12,13,14`，其余 24 个 kernel tile 填零。这只定义权重编码，不表示
当前 RTL 已经具备该层的完整执行语义，见第 6 节。

### 3.4 `params.bin`（bias / shift DMA 顺序）

每层 payload 紧密拼接，层内顺序为：

```text
cout_tile -> line_pair_id(0..3)
```

每个 pair 恰使用一个 64-bit carrier，低 40 bit 固定如下：

```text
carrier[3:0]   = shift[cout_tile*8 + 2*line_pair_id]
carrier[7:4]   = shift[cout_tile*8 + 2*line_pair_id + 1]
carrier[23:8]  = bias_int[cout_tile*8 + 2*line_pair_id]
carrier[39:24] = bias_int[cout_tile*8 + 2*line_pair_id + 1]
carrier[63:40] = 0
```

`bias_int` 是 16-bit 二补码，`shift` 是无符号 4-bit。导出器不得截断：
`bias_int` 必须属于 `[-32768,32767]`，`shift` 必须属于 `[0,15]`。当前冻结模型的
统计范围为 `bias_int=[-23764,18947]`、`shift=[1,11]`。

### 3.5 `ofm_golden.bin`（唯一的逐层期望结果）

文件按 `layer_id=0..11` 分段。每层段由该层的全部有效稀疏输出坐标组成（即使该点
所有 feature 恰好都为零也必须保留）；记录按无符号
`coord32` 升序排列。这是**比较用 canonical 顺序**，并不要求 RTL 的 OFM 写回
stream 采用同一顺序。

一个输出点的记录为：

```text
word 0                           = {32'h00000000, coord32}
word 1 .. ceil(Cout_logical / 8) = INT8 feature tiles, channel ascending
```

feature tile 的 byte lane 0 是最低有效 byte，对应最低通道号。最后一个 tile 未使用
lane 必须为零；当前各层 Cout 都是 8 的倍数，因此没有 Cout 尾部 padding。比较器先
从 RTL 写回流恢复全局 `coord32` 和 feature，再按 `coord32` 排序，逐坐标逐 byte
比对本文件。不得比较反量化 FP32，也不得使用误差阈值。

## 4. 导出检查（任何失败均拒绝生成 package）

导出器必须在写文件前完成以下 hard-fail 检查：

1. 检查 checkpoint、calibration、layer table 和 RTL 契约版本一致。
2. 对所有激活和权重检查 signed、zero point = 0、`qmin=-127`、`qmax=127`；
   输入和权重的实际 INT8 payload 不得含 `-128`。
3. 对每个 sparse conv 完成 BN folding，并按本规范的 kernel / channel tile 顺序
   打包权重；所有 padding lane 为 0。
4. 检查每个 parameter carrier 的 `[63:40]` 为 0，bias/shift 均可精确表示。
5. 检查 raw VFE stream 的 record count 等于 `actual_voxel_count`，每条记录为
   16B，所有 owner `coord32` 可编码、唯一且 `is_halo=0`。
6. 用第 2 节的整数算术生成全部 12 层 `ofm_golden`；对每层检查坐标唯一、feature
   范围为 `[0,127]`、INT32 `psum` 未溢出。
7. 检查文件中每个 manifest 分段的长度、record count 和 SHA-256。

量化后的 VFE 输入是已经冻结的测试向量。RTL 回归只消费其 INT8 值，不重新执行 VFE
量化，也不依赖 PyTorch 对 tie 的舍入规则。

## 5. 回归方法与通过条件

1. 将 `weights.bin` 和 `params.bin` 依据 manifest 加载至层 DMA 的相应地址。
2. 将 `raw_vfe_voxel_stream.bin` 送入 RTL 的 VFE/input ingress。测试平台只提供
   raw owner voxel stream，不预置 `input_pages.blocks[]`、block directory 或 page
   allocator 结果。
3. Block Manager 通过 write hash path 建立 layer0 初始 sparse directory，并由
   VFE/input commit path 按 write descriptor 写入 16B layer0 IFM resident payload。
   完成后执行 role swap，使该 bank 成为 layer0 read directory。
4. 让硬件连续执行 layer 0 至 layer 11。层间 OFM 由硬件自身写回并再次读取；测试
   平台不得把 `ofm_golden.bin` 回灌为下一层输入。
5. 每层结束后，将 RTL 产生的 OFM 解码为 `{coord32, INT8[Cout]}`，按 `coord32`
   升序排序，并与该层 `ofm_golden` 分段做完全相等比较。
6. 通过标准是：点数相等、坐标集合相等、每一个 INT8 byte 相等；没有 epsilon、
   没有反量化容差。首个失配须报告 `layer_id`、`coord32`、channel、期望值和实际值。

这套最小数据足以定位量化、BN fold、kernel order、Cin/Cout tile、bias/shift、
rounding、坐标转换和稀疏点生成错误；全量 psum 或重复 IFM 文件不会提高该回归的
判定能力，因此不属于 package。

## 6. 当前 RTL 的已知阻断项

本 package 必须导出 layer 11 的全部 128 个输出通道，**不得为了适应 RTL 而截断为
64 通道**。但在当前代码状态下，完整 12 层 top-level bit-exact signoff 仍有两个
明确阻断项：

1. `conv_out.0` 是 3×1×1、仅 z 方向 stride 2；当前 `compute_top` 只接收单一
   `stride2_en`，顶层以 `stride_x == 2` 驱动它，尚未证明 z-only kernel / stride
   语义正确。
2. `conv_out.0` 的 128 Cout 需要 16 个 Cout tile，而当前
   `MEMORY_MAX_WT_COUT_TILES=8`；WT 的分批装载 / replay 控制尚未实现或验证。

因此，此 package 可立即用于 layer 0 至 layer 10 的精确回归，并为 layer 11 保留
完整、不可篡改的输入参数与输出 golden；在上述两个 RTL 问题关闭前，不得将 layer
11 的不匹配归因于模型或量化，也不得宣称全 backbone 已通过硬件等价验证。

## 7. 非目标

以下内容刻意不属于本 package：

- 原始 KITTI `.bin` 的副本、voxelizer/VFE 的 FP32 中间结果；
- 每层重复 IFM、FP32/QDQ tensor、完整 accumulator trace；
- ONNX、TorchScript、`.pth` checkpoint；
- 2D backbone、head 或最终 AP 指标。

它们对于训练、端到端算法复现或性能分析有价值，但不是当前 sparse-backbone RTL
精确数值回归的最小必要实体。

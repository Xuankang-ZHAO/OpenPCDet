# 面向体素稀疏卷积 AI 加速器的 Block Partition 分析与分层实施方案

## 1. 报告目的

本文以 AI 加速器与数字集成电路博士生的实验分析与论文规划视角，系统整理以下问题，并给出可执行的技术路线：

1. 初始体素化后的 block partition 是否能够直接沿用到后续多层稀疏卷积。
2. stride、padding、SubMConv 与 SparseConv 对 block partition 的影响分别是什么。
3. 是否需要每层重新 block partition，以及更合理的粒度应该是什么。
4. 如何面向硬件设计，统计并确定每个 block 需要容纳的体素上限、halo 开销、请求带宽和 overflow 风险。
5. 如何针对当前模型代码 [pcdet/models/backbones_3d/spconv_backbone_qat.py](../../pcdet/models/backbones_3d/spconv_backbone_qat.py) 中的每一层稀疏卷积给出具体实现建议。

本文同时结合现有分析脚本：

1. [mycode/voxel_analyze_with_boudary_rtl_unfixed.py](../voxel_analyze_with_boudary_rtl_unfixed.py)
2. [mycode/rtl_unfixed_block_partition.py](../rtl_unfixed_block_partition.py)
3. [mycode/markdown/online_block_partitioning_algorithm_summary.md](online_block_partitioning_algorithm_summary.md)

并面向当前 SECOND backbone 配置：

1. [tools/cfgs/kitti_models/second.yaml](../../tools/cfgs/kitti_models/second.yaml)
2. [pcdet/models/backbones_3d/spconv_backbone_qat.py](../../pcdet/models/backbones_3d/spconv_backbone_qat.py)

---

## 2. 关键结论

### 2.1 初始 block partition 不能直接代表后续所有层

这是因为在 SECOND 这类 backbone 中，稀疏坐标集合会在不同 stage 发生变化：

1. `SubMConv3d` 通常保持 active coordinate set 不变。
2. `SparseConv3d` 且 `stride > 1` 会改变 active coordinate set、空间分辨率和输出坐标分布。
3. 因此，初始 voxel grid 上确定的 block occupancy，只能准确描述输入层以及同分辨率的 `SubMConv` stage，不能直接代表 stride 之后的新 stage。

### 2.2 不需要“每一层都物理重分块”，但需要“按 stage 做逻辑重映射”

更合理的做法不是在每层都做完整数据重排，而是：

1. 在同一 active set 内复用 block metadata 与调度关系。
2. 在 `stride` 层后重建下一 stage 的 block ownership、block occupancy 和 halo 依赖。
3. 物理数据存储尽量避免大规模搬移，优先通过地址映射、索引表和工作队列更新来实现逻辑 repartition。

### 2.3 你当前脚本统计的是“请求压力”，不是“唯一驻留体素数”

根据 [mycode/rtl_unfixed_block_partition.py](../rtl_unfixed_block_partition.py) 当前实现，统计语义更接近：

1. primary block request
2. halo duplication 产生的额外 block request
3. 保留重复 request collapse 到同一 block key 的计数

这对硬件前端请求带宽、入口 FIFO 和 NoC 注入压力是有意义的；但它不等于片上 block buffer 需要容纳的唯一体素数量。

因此后续实验必须至少输出两套指标：

1. `Unique occupancy per block`：每个 block 内唯一 active voxel 数。
2. `Request count per block`：考虑 primary + halo duplication 的 block request 数。

如果只用第二个指标去定 block SRAM 容量，往往会过度保守，甚至混淆“存储瓶颈”和“带宽瓶颈”。

### 2.4 论文中推荐采用“分辨率 stage 级 block partition”

对当前 SECOND backbone，最自然的分析粒度是：

1. 输入 voxel 坐标
2. `x_conv1`
3. `x_conv2`
4. `x_conv3`
5. `x_conv4`
6. `encoded_spconv_tensor`

其中：

1. `conv_input + conv1` 对应 stage-1。
2. `conv2` 对应 stage-2。
3. `conv3` 对应 stage-4。
4. `conv4` 对应 stage-8。
5. `conv_out` 主要改变 z 方向压缩，而不是新的 xy stage。

---

## 3. 当前代码与网络结构的直接含义

根据 [pcdet/models/backbones_3d/spconv_backbone_qat.py](../../pcdet/models/backbones_3d/spconv_backbone_qat.py) 与 [pcdet/models/backbones_3d/spconv_backbone.py](../../pcdet/models/backbones_3d/spconv_backbone.py)，当前 backbone 为：

| 模块 | 类型 | kernel | stride | padding | 输出语义 |
|---|---|---:|---:|---:|---|
| `conv_input` | `SubMConv3d` | 3 | 1 | 1 | 坐标不变，进入 stage-1 |
| `conv1[0]` | `SubMConv3d` | 3 | 1 | 1 | 坐标不变，仍为 stage-1 |
| `conv2[0]` | `SparseConv3d` | 3 | 2 | 1 | 产生新的 stage-2 输出坐标 |
| `conv2[1]` | `SubMConv3d` | 3 | 1 | 1 | 坐标不变，复用 stage-2 |
| `conv2[2]` | `SubMConv3d` | 3 | 1 | 1 | 坐标不变，复用 stage-2 |
| `conv3[0]` | `SparseConv3d` | 3 | 2 | 1 | 产生新的 stage-4 输出坐标 |
| `conv3[1]` | `SubMConv3d` | 3 | 1 | 1 | 坐标不变，复用 stage-4 |
| `conv3[2]` | `SubMConv3d` | 3 | 1 | 1 | 坐标不变，复用 stage-4 |
| `conv4[0]` | `SparseConv3d` | 3 | 2 | `(0,1,1)` | 产生新的 stage-8 输出坐标，z 方向非对称 |
| `conv4[1]` | `SubMConv3d` | 3 | 1 | 1 | 坐标不变，复用 stage-8 |
| `conv4[2]` | `SubMConv3d` | 3 | 1 | 1 | 坐标不变，复用 stage-8 |
| `conv_out` | `SparseConv3d` | `(3,1,1)` | `(2,1,1)` | `last_pad` | 主要沿 z 方向压缩 |

由此得到一个非常明确的工程结论：

1. `conv_input` 与 `conv1` 可以看作同一个分辨率 stage。
2. `conv2[0]` 是第一次必须重建 block metadata 的层。
3. `conv3[0]`、`conv4[0]` 同理。
4. `conv_out` 是否需要重分块，取决于你的 block 是否显式包含 z 方向分块。

---

## 4. Block Partition 问题应拆成哪三层

为了避免论文和硬件设计表述混乱，建议将 block partition 分成三个层次。

### 4.1 逻辑分块

定义当前 active voxel 属于哪个 block。

这对应：

1. block ID 的计算方式
2. zone LUT 的查找方式
3. 当前 layer 或 stage 的 block occupancy 统计

### 4.2 访问扩展

定义为计算卷积所需要额外访问哪些邻域 voxel。

这对应：

1. halo duplication
2. block 边界跨越时的邻块访问
3. 多层同 stage 融合时 halo 半径累积

### 4.3 物理重排

定义数据是否真的被搬运到新的片上 block buffer 组织下。

这对应：

1. SRAM layout 是否变化
2. DRAM stream 是否重排
3. NoC 目的地址是否重映射

论文建议强调：

1. stride 后通常必须更新逻辑分块。
2. 不一定必须做完整物理重排。
3. 可以采用“逻辑新 block + 物理旧存储 + 索引映射表”的折中方案。

---

## 5. stride、padding 与 halo 的本质影响

### 5.1 `SubMConv3d` 的影响

`SubMConv3d` 不会扩展输出坐标集合，输出仅发生在原有 active site 上。

因此：

1. block ownership 可以保持不变。
2. 如果逐层执行，每层只需要 1-hop halo。
3. 如果在硬件中将多个 `SubMConv` 融合为一个 stage 连续执行，则 halo 半径会累积。

若连续执行 $m$ 个 $3 \times 3 \times 3$ 的同分辨率卷积，halo 半径近似为：

$$
r = m
$$

若 block 大小为 $B_x \times B_y \times B_z$，则 halo duplication 的体积膨胀近似为：

$$
\rho \approx \frac{(B_x + 2r)(B_y + 2r)(B_z + 2r)}{B_x B_y B_z} - 1
$$

这说明小 block 虽然减小单块存储，但会显著增加 halo 冗余。

### 5.2 `SparseConv3d(stride=2)` 的影响

这是最关键的变化点。

它带来的不是简单的 halo 扩张，而是：

1. 输出坐标集合改变。
2. 输出分辨率降低。
3. 输出 block occupancy 分布与输入层显著不同。
4. block 归属必须以“输出活跃坐标”为准重新计算。

因此，面对 stride 层，更合理的做法不是继续沿用旧 block ID，而是：

1. 先生成当前 stride 层的输出坐标。
2. 在新分辨率网格上重新做 block ownership。
3. 再对下一 stage 的 `SubMConv` 复用该 partition。

### 5.3 padding 的影响

padding 更准确地说是改变“邻域依赖边界”，不是必然要求重分块。

它主要影响：

1. block 边界附近是否需要额外取数。
2. rulebook 或 gather list 的规模。
3. halo duplication 的边界条件。

尤其对 `conv4[0]` 的 `(0,1,1)` padding，需要注意：

1. x/y 仍表现为对称 3x3 邻域。
2. z 方向不引入同等形式的边界扩展。
3. 若 z 方向维度本身很小，可以考虑不对 z 做独立 block 切分，或者将 z 作为 block 内局部维度而不是主要分块轴。

---

## 6. 论文工作中最应该补上的分析维度

目前仅分析输入层体素 block occupancy 还不够，建议补充为“逐 stage 的多指标画像”。

### 6.1 每个 stage 需要统计的指标

对于每个 stage，建议统计以下指标：

1. `Unique occupancy per block`
2. `Request count per block with halo duplication`
3. `Rulebook edge count per block`
4. `Cross-block dependency ratio`
5. `Empty block ratio`
6. `Non-empty block ratio`
7. `P50/P90/P95/P99/P99.9/max` 的 occupancy 分位数
8. `Overflow rate`：给定 block 容量上限时，超过上限的 block 比例

### 6.2 为什么必须加入 rulebook 统计

仅统计 block 内 voxel 数不够，因为硬件负载更直接受以下量影响：

1. 邻域映射边数
2. gather/scatter 请求数
3. MAC 触发次数

在稀疏卷积中，真实瓶颈往往不是单纯 voxel 存储，而是索引访问和不规则邻接。

因此论文中最好把：

1. occupancy
2. request count
3. rulebook edges

作为三类并列指标。

### 6.3 world-locked 与 grid-locked 两种 zone/block 设计

这是论文答辩中很容易被问到的问题。

#### 方案 A：grid-locked partition

各层沿用同样的索引空间阈值和 block 大小。

优点：

1. 硬件实现简单。
2. LUT 不需要按层变化。

缺点：

1. 深层每个 block 覆盖的物理世界范围越来越大。
2. zone 的几何含义会漂移。

#### 方案 B：world-locked partition

按物理空间覆盖范围保持不变，对每个 stage 的 block size 和 zone threshold 按累计 stride 缩放。

若第 $l$ 层累计 stride 为 $S_l$，则建议：

$$
T_l = \left\lceil \frac{T_0}{S_l} \right\rceil
$$

block 大小也按相同原则缩放到 coarse grid 上。

优点：

1. zone 的物理意义稳定。
2. 不同 stage 间统计更可比。

缺点：

1. LUT 需要按 stage 生成。
2. 硬件控制更复杂。

对于论文，建议两者都做对比，并明确说明最终采用哪一种。

---

## 7. 面向当前 backbone 的逐层具体建议

本节对 [pcdet/models/backbones_3d/spconv_backbone_qat.py](../../pcdet/models/backbones_3d/spconv_backbone_qat.py) 中每一层给出具体建议，目标是直接指导实验实现和硬件方案设计。

### 7.1 `conv_input`

**层信息**

1. 类型：`SubMConv3d`
2. 参数：`kernel=3, stride=1, padding=1`
3. 输出坐标：与输入 voxel 坐标相同

**建议做法**

1. 以输入 voxel grid 上的现有 block partition 作为 stage-1 的基础 partition。
2. 对该层统计两种指标：
   1. block 内唯一 voxel 数
   2. 为该层卷积访问产生的 halo request 数
3. 若硬件中 `conv_input` 与 `conv1[0]` 分开执行，则只需准备 1-hop halo。
4. 若二者在硬件中融合为一个 stage，则 halo 半径建议按 2-hop 评估。

**硬件含义**

1. 该层适合作为前端 block 调度与缓存预取的起点。
2. 该层 occupancy 分布可用于估计 stage-1 输入 buffer 容量。
3. 若 block occupancy 长尾严重，可在 stage-1 就引入 block split 或 overflow spill 机制。

### 7.2 `conv1[0]`

**层信息**

1. 类型：`SubMConv3d`
2. 参数：`kernel=3, stride=1, padding=1`
3. `indice_key='subm1'`
4. 输出坐标：与 `conv_input` 相同

**建议做法**

1. 完全复用 `conv_input` 的 block partition，不重新分块。
2. 若软件仿真要贴近硬件，建议在 stage-1 内做一次“统一统计”：
   1. 单层执行统计
   2. 两层融合统计
3. 重点比较两种执行模式下的 halo duplication 开销与片上复用收益。

**硬件含义**

1. 适合与 `conv_input` 构成同一 pipeline stage。
2. 如果 SRAM 面积吃紧，可以优先研究 stage-1 内是否能“块内多层复用”来减少写回。
3. 若采用融合执行，必须扩大边界 halo，不能只用单层 halo 假设。

### 7.3 `conv2[0]`

**层信息**

1. 类型：`SparseConv3d`
2. 参数：`kernel=3, stride=2, padding=1`
3. `indice_key='spconv2'`
4. 输出坐标：变为新的 stage-2 active set

**建议做法**

1. 这是第一次必须重新做逻辑 block partition 的层。
2. 建议采用“输出驱动”的统计方式：
   1. 先得到该层输出 active coordinates
   2. 在 coarse grid 上对输出坐标重新映射 block ID
   3. 再统计下一 stage 的 occupancy
3. 对该层额外统计：
   1. 输入 block 到输出 block 的映射扇出
   2. 每个输出 block 需要触达的输入 block 数量
   3. rulebook edge 数分布

**硬件含义**

1. 不建议沿用输入层 block ID 直接算 stage-2。
2. 更合理的是在该层末尾输出新的 block metadata，包括：
   1. stage-2 block ID
   2. stage-2 occupancy
   3. stage-2 block pointer 或 index list
3. 物理上可以不立即重排特征数据，但调度上必须切换到新的 stage-2 block 视图。

**论文重点**

1. 这是“是否每层重分块”问题的关键例子。
2. 论文应强调：不是每层重分块，而是在 stride 边界重建 stage 级分块。

### 7.4 `conv2[1]`

**层信息**

1. 类型：`SubMConv3d`
2. 参数：`kernel=3, stride=1, padding=1`
3. `indice_key='subm2'`

**建议做法**

1. 完全复用 `conv2[0]` 输出的 stage-2 partition。
2. 统计该层 block 内有效邻域触达比例，验证 stage-2 的 block size 是否合理。
3. 若 stage-2 occupancy 已显著下降，可适当缩小 block 大小以降低空块比例。

**硬件含义**

1. 可与 `conv2[2]` 组成同一 stage 内连续执行段。
2. 若 block 边界过多导致 halo 成本过高，说明 stage-2 block 尺寸偏小。

### 7.5 `conv2[2]`

**层信息**

1. 类型：`SubMConv3d`
2. 参数：`kernel=3, stride=1, padding=1`
3. `indice_key='subm2'`

**建议做法**

1. 与 `conv2[1]` 共享完全相同的 block ownership。
2. 若硬件计划将 `conv2[1]` 和 `conv2[2]` 融合，则 stage-2 的 halo 半径需要按 2-hop 重新评估。
3. 报告中建议单独比较：
   1. 不融合时的带宽
   2. 融合时的额外 halo
   3. 融合带来的写回减少

**硬件含义**

1. 这是评估“多层同 stage 融合是否值得”的理想位置。
2. 若 stage-2 在 occupancy 上已经明显变稀疏，融合往往更划算。

### 7.6 `conv3[0]`

**层信息**

1. 类型：`SparseConv3d`
2. 参数：`kernel=3, stride=2, padding=1`
3. `indice_key='spconv3'`
4. 输出坐标：新的 stage-4 active set

**建议做法**

1. 再次执行 stage 级逻辑 repartition。
2. 建议与 `conv2[0]` 使用同一套分析模板，以便论文中跨 stage 横向对比。
3. 重点分析 stage-4 的：
   1. occupancy 收缩速度
   2. 空块比例
   3. block 间负载不均衡是否变严重

**硬件含义**

1. 若 stage-4 的非空 block 数已经大幅减少，可以考虑增加 block 粒度，降低调度开销。
2. 若长尾 block 仍存在，则需保留 overflow 机制，不能仅按均值设计 SRAM。

### 7.7 `conv3[1]`

**层信息**

1. 类型：`SubMConv3d`
2. 参数：`kernel=3, stride=1, padding=1`
3. `indice_key='subm3'`

**建议做法**

1. 复用 stage-4 partition。
2. 记录 stage-4 中跨 block 访问的比例，评估 stage-4 的 halo 开销。
3. 如果 stage-4 空间已经非常小，可尝试将多个 block 合并映射到一个计算 cluster 上。

### 7.8 `conv3[2]`

**层信息**

1. 类型：`SubMConv3d`
2. 参数：`kernel=3, stride=1, padding=1`
3. `indice_key='subm3'`

**建议做法**

1. 与 `conv3[1]` 共享同一 partition。
2. 如果考虑两层融合，继续按 stage 内多层 halo 半径累积分析。
3. 若 stage-4 的 active voxel 总量已可被整片 SRAM 容纳，可考虑在该 stage 切换到“整 stage 驻留”模式，而非严格 block 流式模式。

### 7.9 `conv4[0]`

**层信息**

1. 类型：`SparseConv3d`
2. 参数：`kernel=3, stride=2, padding=(0,1,1)`
3. `indice_key='spconv4'`
4. 输出坐标：新的 stage-8 active set

**建议做法**

1. 必须重建 stage-8 partition。
2. 因 z 方向 padding 为 0，建议将此层单独分析，不能简单沿用前两次 stride 层的对称假设。
3. 若你的硬件 block 设计主要面向 x/y 平面，可将该层视为：
   1. x/y 继续按二维主分块
   2. z 方向作为 block 内局部维度处理
4. 对该层建议重点统计：
   1. z 维压缩前后的 occupancy 分布
   2. x/y block 与 z 局部 depth 的联合分布

**硬件含义**

1. 该层往往标志着 3D sparse backbone 后期阶段，空间尺寸更小，通道更高。
2. 在这一层之后，block 调度开销可能开始超过 block 复用收益，需要评估是否改为更粗粒度执行。

### 7.10 `conv4[1]`

**层信息**

1. 类型：`SubMConv3d`
2. 参数：`kernel=3, stride=1, padding=1`
3. `indice_key='subm4'`

**建议做法**

1. 复用 stage-8 partition。
2. 若 stage-8 非空 block 总量已经很少，建议评估是否直接取消复杂的 block stream 调度，而改为片上全驻留或 super-block 调度。

### 7.11 `conv4[2]`

**层信息**

1. 类型：`SubMConv3d`
2. 参数：`kernel=3, stride=1, padding=1`
3. `indice_key='subm4'`

**建议做法**

1. 与 `conv4[1]` 复用同一 partition。
2. 若要进行 stage-8 两层或三层融合，需要在报告中给出 halo 半径扩大的上限分析。
3. 对这一级别，建议特别关注通道数增加带来的 feature SRAM 压力，而不是只看 occupancy。

### 7.12 `conv_out`

**层信息**

1. 类型：`SparseConv3d`
2. 参数：`kernel=(3,1,1), stride=(2,1,1), padding=last_pad`
3. 输出语义：主要沿 z 方向压缩

**建议做法**

1. 若你的 block partition 主要定义在 x/y 平面，则该层通常不需要重建 xy block ownership。
2. 若你的 block 是严格三维 `(bx, by, bz)`，则应至少更新 z 方向 block 归属。
3. 对该层统计重点应从“空间块 occupancy”逐步转向：
   1. 通道并行度
   2. 压缩后 feature tensor 片上重用
   3. 与后续 BEV 映射的数据接口开销

**硬件含义**

1. 这是 3D sparse backbone 到 BEV 压缩的过渡层。
2. 若论文主打“从 voxel 开始的 3D sparse conv 加速”，该层可作为 3D block engine 的终点，并说明后续交给专门的 BEV/2D 后端。

---

## 8. 当前实验脚本的不足与建议补强

### 8.1 当前脚本的价值

[mycode/voxel_analyze_with_boudary_rtl_unfixed.py](../voxel_analyze_with_boudary_rtl_unfixed.py) 已经完成了以下有价值工作：

1. 在输入 voxel 层进行 block partition 统计。
2. 支持 zone-based unfixed partition。
3. 支持 RTL 风格 halo duplication 计数。
4. 输出 CSV，适合作为初步统计基础。

### 8.2 当前脚本的不足

它还缺少以下能力：

1. 只能分析输入 voxel 层，不能分析 backbone 中间 stage。
2. 统计的是 request count，不是 unique occupancy。
3. 不能直接获取每层真实输出 active coordinates。
4. 不能对 stride 后的 coarse grid 自动做 stage 级 repartition。
5. 不能估计 rulebook / 邻接边规模。

### 8.3 建议新增的脚本能力

建议新增或扩展以下功能：

1. 运行 backbone 一次前向，抓取：
   1. 输入 voxel coords
   2. `x_conv1.indices`
   3. `x_conv2.indices`
   4. `x_conv3.indices`
   5. `x_conv4.indices`
   6. `encoded_spconv_tensor.indices`
2. 对每个 stage 输出：
   1. unique occupancy
   2. request count
   3. rulebook edge count
3. 支持两种 partition 模式：
   1. grid-locked
   2. world-locked
4. 支持 block size sweep 与 zone LUT sweep。
5. 支持输出分位数统计和 overflow rate。

---

## 9. 建议的实验计划

### 9.1 实验目标

本课题建议围绕以下三个研究问题组织实验：

1. 输入层 block partition 与逐 stage repartition 在统计结果上差异有多大。
2. 不同 stage 的最优 block size 是否一致。
3. 在给定片上 SRAM 面积约束下，哪种 stage 级 block 策略具有最优的 overflow-rate / 带宽 / 调度复杂度折中。

### 9.2 实验步骤

#### 步骤 1：建立逐 stage 坐标提取链路

目标：从真实网络前向中提取所有 stage 的 active coordinates。

输出：

1. 每层 sparse tensor indices
2. 每层 spatial shape
3. 每层非零体素数

#### 步骤 2：建立双指标 block 统计

对每个 stage 分别计算：

1. unique occupancy per block
2. request count per block

并输出：

1. mean
2. median
3. P90
4. P99
5. max
6. empty block ratio

#### 步骤 3：建立 stride-aware repartition

在每个 stride 层后：

1. 读取输出坐标
2. 在 coarse grid 重新映射 block ID
3. 比较 grid-locked 与 world-locked 两种模式

#### 步骤 4：建立硬件容量评估模型

给定 block buffer 上限 $C$，统计：

$$
\text{overflow rate}(C) = \frac{\#\{b \mid occ(b) > C\}}{\#\{b\}}
$$

并对不同 stage 分别画出容量-溢出率曲线。

#### 步骤 5：建立融合执行与非融合执行对比

对于每个 stage 内连续 `SubMConv`，比较：

1. 不融合：更小 halo，更多写回
2. 融合：更大 halo，更少写回

从而为硬件架构选择提供定量依据。

### 9.3 建议的实验输出图表

建议论文中至少包含以下图表：

1. 各 stage 的 occupancy histogram
2. 各 stage 的 request histogram
3. 各 stage 的 overflow-rate 曲线
4. block size sweep 下的 P99 occupancy
5. block size sweep 下的 halo duplication ratio
6. grid-locked 与 world-locked 的对比图
7. stage 内融合与非融合的带宽对比图

---

## 10. 面向硬件架构的落地建议

### 10.1 建议的总体策略

建议采用以下架构策略：

1. 输入层做一次初始 block partition。
2. 在每个 stride 层后重建 stage 级 block metadata。
3. 同 stage 的 `SubMConv` 尽可能复用 block partition。
4. 是否融合多个 `SubMConv`，由 halo 增长与写回减少的 trade-off 决定。
5. buffer 容量按高分位数设计，并保留 overflow path。

### 10.2 不建议的策略

不建议简单采用以下做法：

1. 全网络只做一次输入层 block partition，然后一路沿用。
2. 用单一最大 occupancy 直接决定所有 stage 的 buffer 容量。
3. 只看体素数，不看 request count 和 rulebook edge。
4. 忽略 stride 后 active set 的变化。

### 10.3 论文表述建议

论文中建议明确表述为：

1. 本工作采用 stage-level sparse block execution，而非 layer-agnostic static block mapping。
2. block metadata 在 stride boundary 处更新，而在同一 active-set stage 内复用。
3. 通过 unique occupancy、request count 与 rulebook edge 三类指标共同确定硬件资源配置。

这比“每层都重新 block partition”更专业，也更符合硬件实现逻辑。

---

## 11. 当前最优的后续执行顺序

建议下一阶段按以下顺序推进：

1. 修改分析链路，抓取 `x_conv1/x_conv2/x_conv3/x_conv4/encoded_spconv_tensor` 的 `indices`。
2. 为每个 stage 同时统计 unique occupancy 与 request count。
3. 增加 world-locked 与 grid-locked 两种 repartition 模式。
4. 对 block size 和 zone LUT 做 sweep。
5. 用 P99 与 overflow-rate 反推各 stage 的 block buffer 容量。
6. 评估是否需要 stage 内多层融合，以及融合后 halo 的额外代价。

---

## 12. 最终建议摘要

针对“初始 block partition 是否后续也在变化、是否每层都要重新 partition、硬件如何适配”这三个问题，最终建议如下：

1. 初始 block partition 会在 stride 层后失效，因此不能直接沿用到整个网络。
2. 不需要每层都做完整物理重分块，但需要在每个 stride 边界做 stage 级逻辑重映射。
3. 同一 stage 内的 `SubMConv` 可以复用 partition，并根据是否融合执行决定 halo 半径。
4. 你的硬件容量设计不能只依据输入层 block occupancy，而应依据逐 stage 的高分位 occupancy、request count 和 overflow-rate。
5. 当前最值得补充的论文工作，是建立逐 stage 稀疏坐标统计链路，并完成从输入层到 `x_conv4` 的 block 行为画像。

这将把你的论文从“输入层 block occupancy 观察”提升为“面向真实多层稀疏卷积执行过程的 stage 级 block 架构设计”。
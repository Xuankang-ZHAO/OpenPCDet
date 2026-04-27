# 面向 SECOND Backbone 的稀疏卷积 AI 加速器：静态软件参数与 Block Partition 分层实施方案

## 1. 文档定位与适用范围

本文不再只是讨论 block partition 的抽象方法，而是明确面向当前仓库中的 SECOND backbone 实现，整理出一份硬件设计阶段可直接查阅的静态软件参考文档。

本文绑定的代码与配置范围如下：

1. 模型配置：[tools/cfgs/kitti_models/second.yaml](../../tools/cfgs/kitti_models/second.yaml)
2. 数据配置：[tools/cfgs/dataset_configs/kitti_dataset.yaml](../../tools/cfgs/dataset_configs/kitti_dataset.yaml)
3. 3D backbone 实现：[pcdet/models/backbones_3d/spconv_backbone_qat.py](../../pcdet/models/backbones_3d/spconv_backbone_qat.py)
4. 标准对照实现：[pcdet/models/backbones_3d/spconv_backbone.py](../../pcdet/models/backbones_3d/spconv_backbone.py)
5. VFE 实现：[pcdet/models/backbones_3d/vfe/mean_vfe.py](../../pcdet/models/backbones_3d/vfe/mean_vfe.py)

本文默认的目标 backbone 为：

1. `BACKBONE_3D.NAME = VoxelBackBone8x_INT8`
2. `VFE.NAME = MeanVFE`
3. 研究边界止于 3D sparse backbone 的 `conv_out`

因此，本文重点覆盖：

1. 输入体素化后的软件对象定义
2. SECOND backbone 每个稀疏卷积层的尺寸、通道、权重逻辑形状和 active-set 变化
3. 稀疏卷积层间的归一化、激活、残差与量化辅助逻辑
4. 面向 block-based AI 加速器的 stage-level 分块与 metadata 更新策略

本文不展开的部分如下：

1. `MAP_TO_BEV`、2D backbone 与 dense head 的内部实现
2. 数据集级动态统计结果
3. 训练超参数、损失函数与检测头细节

---

## 2. 先给硬件设计结论

### 2.1 当前 SECOND backbone 一共有 12 个 3D sparse conv 算子

按执行顺序分别为：

1. `conv_input`
2. `conv1[0]`
3. `conv2[0]`
4. `conv2[1]`
5. `conv2[2]`
6. `conv3[0]`
7. `conv3[1]`
8. `conv3[2]`
9. `conv4[0]`
10. `conv4[1]`
11. `conv4[2]`
12. `conv_out`

### 2.2 当前主干中只有 3 个 stride boundary 会重建新的主要 stage active set

也就是：

1. `conv2[0]`
2. `conv3[0]`
3. `conv4[0]`

这三个位置分别将 xy 空间分辨率从：

1. `1408 x 1600`
2. 降到 `704 x 800`
3. 再降到 `352 x 400`
4. 再降到 `176 x 200`

`conv_out` 也是 `SparseConv3d`，但它只沿 z 方向做 `(3,1,1)` 卷积和 `(2,1,1)` stride 压缩，不创建新的 xy stage。

### 2.3 当前选定 backbone 中，每个 sparse conv 后面都有 `BatchNorm1d + ReLU`

这条结论来自两个代码事实：

1. `conv_input` 是 `SubMConv3d + BatchNorm1d + ReLU`
2. `post_act_block` 统一封装为 `conv + BatchNorm1d + ReLU`
3. `conv_out` 也是 `SparseConv3d + BatchNorm1d + ReLU`

因此，硬件视角不能把这些层当成“只有卷积、没有归一化/激活”的裸稀疏卷积链。

### 2.4 当前选定 backbone 不是残差版

本文绑定的是 `VoxelBackBone8x_INT8`，不是 `VoxelResBackBone8x`。因此当前 SECOND 主干没有 `SparseBasicBlock` 的残差加法旁路，这会直接影响：

1. 是否需要保存 identity path
2. 是否需要额外的旁路读写带宽
3. 是否需要加法单元和残差同步控制

### 2.5 输入端的物理 voxel 网格与 sparse tensor 的代码级空间尺寸并不完全相同

这是一个很容易在硬件实现时遗漏的细节。

1. 数据配置中的物理 voxel grid 是 `1408 x 1600 x 40`，按 `XYZ` 表示
2. 代码中的 `self.sparse_shape = grid_size[::-1] + [1, 0, 0]`
3. 因为 `grid_size` 是数组，所以这里是逐元素相加，最终有效 `spatial_shape` 为 `[41, 1600, 1408]`，按 `ZYX` 表示

也就是说：

1. 物理 voxelization 的 z bin 数是 `40`
2. sparse backbone 在软件中使用的地址空间 z 深度是 `41`

如果硬件想精确镜像当前 spconv 索引生成行为，应按代码级 `spatial_shape` 理解各 stage 的空间尺寸；如果只是做物理世界范围映射，则要知道真实输入活跃体素 z 坐标仍来自 `40` 个物理 bin。

因为这两个数字描述的是**两个不同层面的“网格”**：

- `40`：物理体素化时的真实 z 方向 bin 数（来自点云范围和 voxel size 计算）。
- `41`：给 `spconv.SparseConvTensor` 的**索引地址空间深度**，代码里故意做了 `grid_size[::-1] + [1, 0, 0]`，只在 z 上加 1。

核心原因是：主干里 z 方向有多次 stride=2 的稀疏卷积，且其中一层 z padding=0。

用卷积尺寸公式 `out = floor((in + 2p - k)/s) + 1` 算一下：

- 若 z 从 `41` 开始：`41 -> 21 -> 11 -> 5`（与代码注释/后续层设计一致）
- 若 z 从 `40` 开始：`40 -> 20 -> 10 -> 4`（会少一层深度，后续 `conv_out` 的 z 输出也会变小）

所以 `+1` 不是“物理上多了一层体素”，而是为了让 spconv 的离散坐标/下采样链路在软件中得到期望的特征图尺寸（尤其保证后面 z 维还能按设计变成 `5 -> 2`）。


## 3. 当前 SECOND 的静态软件前提

### 3.1 模型、数据与输入前提

| 项目 | 当前值 | 代码来源 | 对硬件的含义 |
|---|---|---|---|
| 数据集 | KITTI | [tools/cfgs/dataset_configs/kitti_dataset.yaml](../../tools/cfgs/dataset_configs/kitti_dataset.yaml) | 本文所有尺寸均绑定 KITTI 配置 |
| 点云范围 | `[0, -40, -3, 70.4, 40, 1]` | 同上 | 决定体素网格的物理覆盖范围 |
| 体素尺寸 | `[0.05, 0.05, 0.1]`，按 `XYZ` 米 | 同上 | 决定输入网格和每层感受野的物理尺度 |
| 物理 voxel grid | `1408 x 1600 x 40`，按 `XYZ` | 范围差除以体素尺寸 | 这是输入体素化的物理离散网格 |
| 代码级 sparse spatial shape | `[41, 1600, 1408]`，按 `ZYX` | `grid_size[::-1] + [1,0,0]` | 这是 backbone 建立 `SparseConvTensor` 时使用的空间地址大小 |
| 每 voxel 最大点数 | `5` | `MAX_POINTS_PER_VOXEL` | 决定 VFE 输入缓存的单 voxel 点上限 |
| 单帧最大 voxel 数 | train=`16000`，test=`15000` | `MAX_NUMBER_OF_VOXELS` | 决定输入端最坏情况 nnz 上限 |
| 输入点特征 | `x, y, z, intensity` 共 4 维 | `POINT_FEATURE_ENCODING` | 输入 feature 宽度为 4 |
| VFE | `MeanVFE` | [pcdet/models/backbones_3d/vfe/mean_vfe.py](../../pcdet/models/backbones_3d/vfe/mean_vfe.py) | 将每个 voxel 聚合为 4 维输出特征 |
| 3D backbone | `VoxelBackBone8x_INT8` | [pcdet/models/backbones_3d/spconv_backbone_qat.py](../../pcdet/models/backbones_3d/spconv_backbone_qat.py) | 当前静态拓扑来源 |

### 3.2 VFE 输出与 sparse tensor 定义

`MeanVFE` 的 forward 逻辑是：

1. 读取 `voxels`，形状为 `(num_voxels, max_points_per_voxel, C)`
2. 读取 `voxel_num_points`，形状为 `(num_voxels)`
3. 对每个 voxel 内部点特征做求和并除以点数
4. 输出 `batch_dict['voxel_features']`，形状为 `(num_voxels, 4)`

因此，进入 sparse backbone 之前的软件对象如下：

| 对象 | 形状 | 含义 |
|---|---|---|
| `voxels` | `(num_voxels, 5, 4)` | 体素内原始点特征缓存，`5` 来自当前 KITTI 配置上限 |
| `voxel_num_points` | `(num_voxels)` | 每个 voxel 内真实点数 |
| `voxel_features` | `(num_voxels, 4)` | MeanVFE 输出，作为 sparse backbone 的 feature 输入 |
| `voxel_coords` | `(num_voxels, 4)` | 坐标格式为 `[batch_idx, z, y, x]` |

随后 backbone 构造：

```python
input_sp_tensor = spconv.SparseConvTensor(
    features=voxel_features,
    indices=voxel_coords.int(),
    spatial_shape=self.sparse_shape,
    batch_size=batch_size
)
```

这意味着硬件若要严格对应软件接口，需要意识到 `SparseConvTensor` 至少维护以下四元组：

1. `features`：形状 `(nnz, C)`，只保存 active voxel 的特征
2. `indices`：形状 `(nnz, 4)`，格式 `[batch, z, y, x]`
3. `spatial_shape`：当前层稀疏张量的地址空间上界
4. `batch_size`：批次维度

### 3.3 backbone 输出到 `batch_dict` 的关键对象

在 [pcdet/models/backbones_3d/spconv_backbone.py](../../pcdet/models/backbones_3d/spconv_backbone.py) 与 [pcdet/models/backbones_3d/spconv_backbone_qat.py](../../pcdet/models/backbones_3d/spconv_backbone_qat.py) 中，backbone 会写出：

1. `multi_scale_3d_features['x_conv1']`
2. `multi_scale_3d_features['x_conv2']`
3. `multi_scale_3d_features['x_conv3']`
4. `multi_scale_3d_features['x_conv4']`
5. `encoded_spconv_tensor`
6. `multi_scale_3d_strides`
7. `encoded_spconv_tensor_stride = 8`

这里的 `encoded_spconv_tensor_stride = 8` 指的是 downstream 检测头关注的 xy 下采样倍数，而不是 z 方向的累计 stride。对硬件而言，`conv_out` 之后的真实累计 stride 已经是：

1. x 方向：8
2. y 方向：8
3. z 方向：16

---

## 4. backbone 逐层静态参数总表

### 4.1 层级总表

下表中的 `XYZ 输入/输出` 均指代码级 sparse tensor 的地址空间尺寸，而不是物理点云范围。

| 层 | 类型 | `indice_key` | kernel | stride | padding | `XYZ` 输入 | `XYZ` 输出 | `C_in -> C_out` | 逻辑权重形状 | BN/ReLU | active set | 累积 stride `(x,y,z)` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `conv_input` | `SubMConv3d` | `subm1` | `3x3x3` | `1x1x1` | `1` | `1408x1600x41` | `1408x1600x41` | `4 -> 16` | `3x3x3x4x16` | 是 | 保持 | `1,1,1` |
| `conv1[0]` | `SubMConv3d` | `subm1` | `3x3x3` | `1x1x1` | `1` | `1408x1600x41` | `1408x1600x41` | `16 -> 16` | `3x3x3x16x16` | 是 | 保持 | `1,1,1` |
| `conv2[0]` | `SparseConv3d` | `spconv2` | `3x3x3` | `2x2x2` | `1` | `1408x1600x41` | `704x800x21` | `16 -> 32` | `3x3x3x16x32` | 是 | 重建 | `2,2,2` |
| `conv2[1]` | `SubMConv3d` | `subm2` | `3x3x3` | `1x1x1` | `1` | `704x800x21` | `704x800x21` | `32 -> 32` | `3x3x3x32x32` | 是 | 保持 | `2,2,2` |
| `conv2[2]` | `SubMConv3d` | `subm2` | `3x3x3` | `1x1x1` | `1` | `704x800x21` | `704x800x21` | `32 -> 32` | `3x3x3x32x32` | 是 | 保持 | `2,2,2` |
| `conv3[0]` | `SparseConv3d` | `spconv3` | `3x3x3` | `2x2x2` | `1` | `704x800x21` | `352x400x11` | `32 -> 64` | `3x3x3x32x64` | 是 | 重建 | `4,4,4` |
| `conv3[1]` | `SubMConv3d` | `subm3` | `3x3x3` | `1x1x1` | `1` | `352x400x11` | `352x400x11` | `64 -> 64` | `3x3x3x64x64` | 是 | 保持 | `4,4,4` |
| `conv3[2]` | `SubMConv3d` | `subm3` | `3x3x3` | `1x1x1` | `1` | `352x400x11` | `352x400x11` | `64 -> 64` | `3x3x3x64x64` | 是 | 保持 | `4,4,4` |
| `conv4[0]` | `SparseConv3d` | `spconv4` | `3x3x3` | `2x2x2` | `(0,1,1)` | `352x400x11` | `176x200x5` | `64 -> 64` | `3x3x3x64x64` | 是 | 重建 | `8,8,8` |
| `conv4[1]` | `SubMConv3d` | `subm4` | `3x3x3` | `1x1x1` | `1` | `176x200x5` | `176x200x5` | `64 -> 64` | `3x3x3x64x64` | 是 | 保持 | `8,8,8` |
| `conv4[2]` | `SubMConv3d` | `subm4` | `3x3x3` | `1x1x1` | `1` | `176x200x5` | `176x200x5` | `64 -> 64` | `3x3x3x64x64` | 是 | 保持 | `8,8,8` |
| `conv_out` | `SparseConv3d` | `spconv_down2` | `3x1x1` | `2x1x1` | `0` | `176x200x5` | `176x200x2` | `64 -> 128` | `3x1x1x64x128` | 是 | 仅 z 压缩 | `8,8,16` |

### 4.2 这张表在硬件设计中的直接含义

1. `conv_input` 和 `conv1[0]` 共享完全相同的坐标集合和 `subm1` 语义，可视为同一个 stage 内的连续算子。
2. `conv2[0]`、`conv3[0]`、`conv4[0]` 是真正需要更新 output active coordinates 和新 stage metadata 的位置。
3. `conv_out` 将通道从 `64` 提高到 `128`，同时把 z 深度从 `5` 压到 `2`，这是 3D sparse backbone 到 BEV 压缩接口的前一跳。
4. 从 kernel volume 看，除 `conv_out` 外，其余算子都是 `27` 个逻辑核位置；`conv_out` 只有 `3` 个逻辑核位置。

### 4.3 关于尺寸解释的三个容易混淆点

1. `1408 x 1600 x 40` 是物理 voxel grid；`1408 x 1600 x 41` 是当前代码级 sparse tensor 地址空间。
2. 文档中所有 `XYZ` 尺寸均为便于硬件查看的 `x, y, z` 顺序，而代码中 `spatial_shape` 和 `indices` 实际按 `z, y, x` 组织。
3. 表中给出的权重形状是逻辑卷积核形状 `Kz x Ky x Kx x Cin x Cout`，用于硬件建模更直观；spconv 内部参数张量的实际内存排列可以随版本不同而不同。

---

## 5. 体素与稀疏张量在各层间的变化

### 5.1 从原始点云到 sparse backbone 输入

当前 SECOND 的输入链路为：

1. 原始点云经过范围裁剪和体素化
2. 形成 `voxels`、`voxel_num_points`、`voxel_coords`
3. `MeanVFE` 将每个 voxel 内的点特征做平均，得到 `voxel_features`
4. `voxel_features + voxel_coords + sparse_shape + batch_size` 被封装成 `SparseConvTensor`

因此，进入 `conv_input` 之前，硬件最少要对应两类元数据：

1. 稀疏特征流：`(nnz, 4)`
2. 稀疏坐标流：`(nnz, [batch, z, y, x])`

### 5.2 从 `input_sp_tensor` 到 `x_conv1`

这一段包括：

1. `conv_input`
2. `conv1[0]`

它们的共同特征是：

1. 都是 `SubMConv3d`
2. 输出 active coordinates 不变
3. 只改变 feature channel
4. 都带 `BatchNorm1d + ReLU`

因此，对 block-based 硬件而言，这一段最适合作为 stage-1：

1. 使用同一份 block ownership
2. 使用同一份 active coordinate metadata
3. 重点优化同 stage 内的特征复用和 halo 访问

### 5.3 从 `x_conv1` 到 `x_conv2`

这一段首先经过 `conv2[0]`：

1. 类型为 `SparseConv3d`
2. `stride=2`
3. 通道从 `16` 增加到 `32`
4. 空间地址尺寸从 `1408x1600x41` 变为 `704x800x21`

接下来的 `conv2[1]` 和 `conv2[2]` 都是 `SubMConv3d`，因此：

1. `conv2[0]` 负责生成 stage-2 的新 active set
2. `conv2[1]` 与 `conv2[2]` 只在 stage-2 内复用该 active set
3. 如果硬件要在 stride 边界更新 block metadata，第一次更新点就是 `conv2[0]`

### 5.4 从 `x_conv2` 到 `x_conv3`

这段与前一段同构：

1. `conv3[0]`：`SparseConv3d`，`32 -> 64`，`704x800x21 -> 352x400x11`
2. `conv3[1]`：`SubMConv3d`，stage-4 内复用
3. `conv3[2]`：`SubMConv3d`，stage-4 内复用

因此，`conv3[0]` 是第二个需要重建 stage metadata 的位置。

### 5.5 从 `x_conv3` 到 `x_conv4`

这一段的关键是 `conv4[0]`：

1. 类型为 `SparseConv3d`
2. `stride=2`
3. `padding=(0,1,1)`，顺序是 `z, y, x`
4. 尺寸从 `352x400x11` 变为 `176x200x5`

与前两个 stride boundary 的差别在于：

1. x/y 方向仍然是对称下采样
2. z 方向这里没有 padding
3. 对 z 维分块敏感的硬件，需要单独处理这一层的边界条件

`conv4[1]` 与 `conv4[2]` 继续在 stage-8 内复用 `conv4[0]` 产生的 active set。

### 5.6 从 `x_conv4` 到 `encoded_spconv_tensor`

`conv_out` 的作用是：

1. `kernel=(3,1,1)`
2. `stride=(2,1,1)`
3. `64 -> 128` 通道提升
4. `176x200x5 -> 176x200x2`

这说明：

1. xy 分辨率保持不变
2. z 深度进一步压缩
3. 对以 xy 平面为主做 block partition 的硬件，`conv_out` 通常不需要重建新的 xy block ownership
4. 对严格三维 `(bx, by, bz)` 分块的硬件，`conv_out` 至少意味着 z 方向 metadata 更新

另外，若将后续 [tools/cfgs/kitti_models/second.yaml](../../tools/cfgs/kitti_models/second.yaml) 中的 `HeightCompression` 作为接口边界来理解，则当前 `conv_out` 输出的 `z=2`、`c=128` 会在下一步压成 `256` 个 BEV 通道。但这属于 2D 后端的边界信息，不属于本文主体。

### 5.7 面向硬件的 metadata 更新点总结

| 位置 | 是否需要更新 active-set metadata | 原因 |
|---|---|---|
| 输入到 `conv_input` | 否 | 直接沿用 voxelization 结果 |
| `conv_input -> conv1[0]` | 否 | `SubMConv3d`，坐标不变 |
| `conv1[0] -> conv2[0]` | 是 | 第一次 stride-2 sparse conv |
| `conv2[0] -> conv2[1]/conv2[2]` | 否 | 复用 stage-2 active set |
| `conv2[2] -> conv3[0]` | 是 | 第二次 stride-2 sparse conv |
| `conv3[2] -> conv4[0]` | 是 | 第三次 stride-2 sparse conv，且 z padding 非对称 |
| `conv4[2] -> conv_out` | 取决于 z 分块策略 | xy 不变，z 压缩 |

---

## 6. 归一化、激活、残差与模块封装

### 6.1 当前 backbone 中归一化和激活的位置

当前选择的 `VoxelBackBone8x_INT8` 中，所有 sparse conv 后面都紧跟：

1. `BatchNorm1d`
2. `ReLU`

更具体地说：

1. `conv_input = SubMConv3d + BatchNorm1d + ReLU`
2. `conv1/conv2/conv3/conv4` 内使用的 `post_act_block = conv + BatchNorm1d + ReLU`
3. `conv_out = SparseConv3d + BatchNorm1d + ReLU`

### 6.2 `BatchNorm1d` 的软件语义

这里的 `BatchNorm1d(C)` 不是对稠密 `XYZ` 体进行归一化，而是对 `SparseConvTensor.features` 做归一化。也就是说：

1. 输入形状是 `(nnz, C)`
2. 归一化沿通道维进行
3. 参与归一化的是 active voxel 特征，而不是整个 dense volume

这会影响硬件实现方式：

1. 若采用训练态仿真，需要支持稀疏 feature matrix 上的 BN
2. 若只做推理加速，BN 可以在部署阶段离线折叠进卷积权重与偏置，但这属于部署优化，不改变本文的代码级拓扑事实

### 6.3 当前 backbone 没有残差旁路

需要明确区分两个 backbone：

1. 本文主角 `VoxelBackBone8x_INT8`：没有 `SparseBasicBlock` 残差结构
2. 对照实现 `VoxelResBackBone8x`：包含 residual addition

因此在当前 SECOND 配置下，硬件主数据通路不需要为残差分支预留：

1. identity feature buffer
2. 残差加法同步控制
3. residual path 带宽预算

### 6.4 `indice_key` 的工程意义

当前代码中同一 stage 的多个 `SubMConv3d` 使用相同的 `indice_key`：

1. stage-1：`subm1`
2. stage-2：`subm2`
3. stage-4：`subm3`
4. stage-8：`subm4`

它反映的工程事实是：

1. 同一 stage 内输出坐标集合不变
2. 该 stage 的稀疏邻接关系可以在软件层被视为同一个 active-set 语义空间
3. 硬件上可以据此把 stage 内多个 `SubMConv` 视为复用同一坐标域的连续算子

---

## 7. 权重、算子接口与运算语义

### 7.1 逻辑权重形状

为了便于数字电路与 AI 加速器建模，本文统一用逻辑形状：

$$
K_z \times K_y \times K_x \times C_{in} \times C_{out}
$$

来描述每层权重，而不依赖具体 spconv 版本的内部参数排布。

因此，当前 backbone 的权重可以简化理解为：

1. 除 `conv_out` 外，其余卷积核体积都等价于 `3 x 3 x 3`
2. `conv_out` 的卷积核体积等价于 `3 x 1 x 1`
3. 典型大权重张量包括：
   1. `conv3[0]`：`3x3x3x32x64`
   2. `conv3[1]`：`3x3x3x64x64`
   3. `conv4[0]`：`3x3x3x64x64`
   4. `conv_out`：`3x1x1x64x128`

### 7.2 静态权重形状不等于运行时有效 MAC 总量

对稠密卷积而言，输出每个 site 的乘加数通常直接由 kernel volume 和 channel 数决定；对稀疏卷积而言，仍要额外受 active neighbors 影响。

因此：

1. 静态上，一个 `3x3x3` sparse conv 的邻域上限是 `27`
2. 但运行时每个输出位置的真实邻居个数由稀疏分布和 rulebook 决定
3. 硬件估算算力上限时可先用静态 kernel volume 建模
4. 硬件估算平均负载时仍需要后续动态统计 `rulebook edge count`

### 7.3 `SubMConv3d` 与 `SparseConv3d` 的本质区别

| 算子 | 输出坐标来源 | 坐标集合是否变化 | 硬件主要难点 |
|---|---|---|---|
| `SubMConv3d` | 直接沿用输入 active coordinates | 否 | 同 stage 内 feature 复用与 halo 访问 |
| `SparseConv3d`，`stride=2` | 生成 coarse grid 上的新输出坐标 | 是 | 新 stage 坐标生成、metadata 更新、跨 block 映射 |
| `SparseConv3d`，`stride=(2,1,1)` | 主要只改变 z 方向分辨率 | 部分变化 | z 维压缩和与后续接口的对接 |

### 7.4 `conv_out` 的接口意义

`conv_out` 虽然还是 sparse conv，但它在系统接口上更像 3D sparse backbone 的出口：

1. 之后的 `HeightCompression` 会把 z 维压平到 channel 维
2. 因此 `conv_out` 前后的接口组织方式很可能不同
3. 若硬件架构希望把 3D sparse engine 与 2D BEV engine 分开，`conv_out` 是天然的边界位置

---

## 8. INT8/QAT 与当前硬件接口的关系

### 8.1 为什么本文要单独提 `VoxelBackBone8x_INT8`

当前配置不是普通的 `VoxelBackBone8x`，而是 `VoxelBackBone8x_INT8`。这说明当前代码库已经在 backbone 内部加入了量化训练和 INT8 推理相关辅助逻辑。

### 8.2 会改变什么，不会改变什么

不会改变的部分：

1. sparse backbone 的层级拓扑
2. 每层的 kernel、stride、padding、channel 宽度
3. 每个 stage 的 active-set 变化位置
4. `BatchNorm1d + ReLU` 的插入位置

会增加的部分：

1. INT8 权重缓存逻辑
2. 权重量化 scale 管理
3. 激活 fake-quant 逻辑
4. QAT / INT8 inference 控制状态

### 8.3 从代码可见的量化辅助接口

在 [pcdet/models/backbones_3d/spconv_backbone_qat.py](../../pcdet/models/backbones_3d/spconv_backbone_qat.py) 中，可以看到至少以下控制逻辑：

1. `convert_to_int8()`：为各卷积权重生成 INT8 表示与 scale
2. `enable_int8_inference()`：切换到 INT8 推理模式
3. `enable_qat()`：打开或关闭激活 fake-quant
4. `_apply_activation_fake_quant()`：对激活做量化仿真

### 8.4 对硬件论文主线的建议处理方式

如果你的论文主线是“SECOND 稀疏卷积加速器的数据流、分块和缓存设计”，那么更合理的写法是：

1. 把量化视作与稀疏数据流正交的第二设计维度
2. 先把不含量化细节的 sparse conv 数据流、坐标更新和 block metadata 设计清楚
3. 再在后续章节单独讨论 INT8 权重存储、scale 管理和激活量化

也就是说，当前 `VoxelBackBone8x_INT8` 应被理解为：

1. 稀疏 backbone 拓扑不变
2. 量化控制路径额外存在
3. 主数据流分析仍可沿用本文的静态层级参数表

---

## 9. Block Partition 问题应拆成哪三层

为了避免论文和硬件设计表述混乱，建议将 block partition 分成三个层次。

### 9.1 逻辑分块

定义当前 active voxel 属于哪个 block。

这对应：

1. block ID 的计算方式
2. zone LUT 的查找方式
3. 当前 layer 或 stage 的 block occupancy 统计

### 9.2 访问扩展

定义为计算卷积所需要额外访问哪些邻域 voxel。

这对应：

1. halo duplication
2. block 边界跨越时的邻块访问
3. 多层同 stage 融合时 halo 半径累积

### 9.3 物理重排

定义数据是否真的被搬运到新的片上 block buffer 组织下。

这对应：

1. SRAM layout 是否变化
2. DRAM stream 是否重排
3. NoC 目的地址是否重映射

论文建议强调：

1. stride 后通常必须更新逻辑分块
2. 不一定必须做完整物理重排
3. 可以采用“逻辑新 block + 物理旧存储 + 索引映射表”的折中方案

---

## 10. stride、padding 与 halo 的本质影响

### 10.1 `SubMConv3d` 的影响

`SubMConv3d` 不会扩展输出坐标集合，输出仅发生在原有 active site 上。

因此：

1. block ownership 可以保持不变
2. 如果逐层执行，每层只需要 1-hop halo
3. 如果在硬件中将多个 `SubMConv` 融合为一个 stage 连续执行，则 halo 半径会累积

若连续执行 $m$ 个 $3 \times 3 \times 3$ 的同分辨率卷积，halo 半径近似为：

$$
r = m
$$

若 block 大小为 $B_x \times B_y \times B_z$，则 halo duplication 的体积膨胀近似为：

$$
\rho \approx \frac{(B_x + 2r)(B_y + 2r)(B_z + 2r)}{B_x B_y B_z} - 1
$$

这说明小 block 虽然减小单块存储，但会显著增加 halo 冗余。

### 10.2 `SparseConv3d(stride=2)` 的影响

这是最关键的变化点。

它带来的不是简单的 halo 扩张，而是：

1. 输出坐标集合改变
2. 输出分辨率降低
3. 输出 block occupancy 分布与输入层显著不同
4. block 归属必须以输出活跃坐标为准重新计算

因此，面对 stride 层，更合理的做法不是继续沿用旧 block ID，而是：

1. 先生成当前 stride 层的输出坐标
2. 在新分辨率网格上重新做 block ownership
3. 再对下一 stage 的 `SubMConv` 复用该 partition

### 10.3 `conv4[0]` 的非对称 padding

当前 backbone 的 `conv4[0]` 使用：

1. `kernel=3x3x3`
2. `stride=2x2x2`
3. `padding=(0,1,1)`，按 `z,y,x`

它的硬件含义是：

1. x/y 仍然像常规 3x3 邻域那样处理
2. z 方向边界条件与前两次 stride layer 不同
3. 若硬件 block 主要以 xy 平面为主划分，可把 z 视为 block 内局部维度处理
4. 若硬件显式按 z 做 block，则这一层必须单独评估边界规则

### 10.4 `conv_out` 不是新的 xy stage

虽然 `conv_out` 是 `SparseConv3d`，但它和 `conv2[0]`、`conv3[0]`、`conv4[0]` 的地位不同：

1. xy 尺寸不再缩小
2. z 深度继续压缩
3. 更适合作为 3D sparse engine 与后续 BEV 接口的边界，而不是新的平面 stage 切换点

---

## 11. 面向当前 backbone 的硬件实施建议

### 11.1 stage 划分建议

对当前 SECOND backbone，最自然的硬件 stage 划分是：

1. stage-1：`conv_input + conv1[0]`
2. stage-2：`conv2[0] + conv2[1] + conv2[2]`
3. stage-4：`conv3[0] + conv3[1] + conv3[2]`
4. stage-8：`conv4[0] + conv4[1] + conv4[2]`
5. z-compress：`conv_out`

### 11.2 metadata 何时更新，何时复用

| 模块段 | metadata 动作 | 原因 |
|---|---|---|
| `conv_input + conv1[0]` | 复用初始 metadata | 都是 `SubMConv3d` |
| `conv2[0]` | 构建 stage-2 metadata | 第一次 stride-2 sparse conv |
| `conv2[1] + conv2[2]` | 复用 stage-2 metadata | 同一 active set |
| `conv3[0]` | 构建 stage-4 metadata | 第二次 stride-2 sparse conv |
| `conv3[1] + conv3[2]` | 复用 stage-4 metadata | 同一 active set |
| `conv4[0]` | 构建 stage-8 metadata | 第三次 stride-2 sparse conv |
| `conv4[1] + conv4[2]` | 复用 stage-8 metadata | 同一 active set |
| `conv_out` | 按 z 分块策略决定是否更新 | xy 不变，z 压缩 |

### 11.3 为什么不建议“全网只做一次 block partition”

因为这与当前代码的真实 active-set 演化不一致：

1. `conv2[0]` 之后坐标集合已经不是输入 voxel grid
2. `conv3[0]` 和 `conv4[0]` 会继续生成新的 coarse-grid 坐标集合
3. 因此只按输入 voxel 层统计 block occupancy，会低估后续 stage 的真实分布变化

### 11.4 为什么也不建议“每层都做完整物理重排”

因为这通常过于昂贵，且与当前 backbone 的层级结构也不匹配：

1. 同一 stage 内多个 `SubMConv` 共用 active set
2. 更合理的是在 stride boundary 更新逻辑 block ownership 和 metadata
3. 物理数据尽量通过地址映射和索引表来复用，而不是每层都大规模搬运

### 11.5 设计 block buffer 容量时应看的量

如果后续进入动态统计阶段，建议至少区分三类量：

1. `Unique occupancy per block`
2. `Request count per block`
3. `Rulebook edge count per block`

其中：

1. 第一类更接近存储压力
2. 第二类更接近带宽和入口队列压力
3. 第三类更接近实际稀疏邻接与 MAC 触发压力

---

## 12. 当前文档之外，后续仍值得补的动态分析

本文已经补齐了静态代码级参数，但如果论文要进一步从“静态结构说明”走向“硬件容量和调度参数定量优化”，还需要后续动态统计。

### 12.1 建议补充的动态指标

对每个 stage，建议最终统计：

1. `Unique occupancy per block`
2. `Request count per block with halo duplication`
3. `Rulebook edge count per block`
4. `Cross-block dependency ratio`
5. `Empty block ratio`
6. `P50/P90/P95/P99/P99.9/max`
7. `Overflow rate`

### 12.2 grid-locked 与 world-locked 的比较仍然值得做

#### 方案 A：grid-locked partition

各层沿用同样的索引空间阈值和 block 大小。

优点：

1. 硬件实现简单
2. LUT 不需要按层变化

缺点：

1. 深层每个 block 覆盖的物理世界范围越来越大
2. zone 的几何含义会漂移

#### 方案 B：world-locked partition

按物理空间覆盖范围保持不变，对每个 stage 的 block size 和 zone threshold 按累计 stride 缩放。

若第 $l$ 层累计 stride 为 $S_l$，则建议：

$$
T_l = \left\lceil \frac{T_0}{S_l} \right\rceil
$$

优点：

1. zone 的物理意义稳定
2. 不同 stage 间统计更可比

缺点：

1. LUT 需要按 stage 生成
2. 硬件控制更复杂

### 12.3 当前已有脚本的定位

现有分析脚本：

1. [mycode/voxel_analyze_with_boudary_rtl_unfixed.py](../voxel_analyze_with_boudary_rtl_unfixed.py)
2. [mycode/rtl_unfixed_block_partition.py](../rtl_unfixed_block_partition.py)
3. [mycode/markdown/online_block_partitioning_algorithm_summary.md](../online_block_partitioning_algorithm_summary.md)

对当前论文工作的价值主要在于：

1. 已经能分析输入 voxel 层的 block request 统计
2. 已经能支撑 zone-based unfixed partition 的早期验证
3. 还需要进一步扩展到 backbone 中间 stage 的 active coordinates

---

## 13. 硬件设计查阅摘要

### 13.1 一页摘要表

| 项目 | 当前 SECOND 静态事实 |
|---|---|
| 输入物理 voxel grid | `1408 x 1600 x 40`，按 `XYZ` |
| backbone 代码级输入空间 | `1408 x 1600 x 41`，等价于 `spatial_shape=[41,1600,1408]` |
| 输入特征宽度 | `4`，来自 `MeanVFE` 输出 |
| 稀疏坐标格式 | `[batch_idx, z, y, x]` |
| 当前 backbone | `VoxelBackBone8x_INT8` |
| 当前 backbone 是否残差版 | 否 |
| 每个 sparse conv 后是否有 BN/ReLU | 是 |
| 主要 stage 边界 | `conv2[0]`、`conv3[0]`、`conv4[0]` |
| `conv_out` 的角色 | z 压缩与 3D-to-BEV 接口边界 |
| `encoded_spconv_tensor_stride` | `8`，表示 xy stride |

### 13.2 对硬件设计最关键的五条规则

1. 不要把物理 voxel grid 的 `z=40` 和代码级 sparse spatial shape 的 `z=41` 混为一谈。
2. 不要假设所有层都沿用输入层 block partition；真正的 stage 边界在 `conv2[0]`、`conv3[0]`、`conv4[0]`。
3. 不要把当前 backbone 误认为残差结构；本文绑定的 `VoxelBackBone8x_INT8` 没有 residual addition。
4. 不要把这些层当成“卷积后直接接下一层”；软件拓扑上每个 sparse conv 后都有 `BatchNorm1d + ReLU`。
5. 若硬件主线是 block-based sparse conv engine，最合理的实现不是每层重排，而是在 stride boundary 更新逻辑 metadata，在同一 stage 内复用 active-set 视图。

### 13.3 本文可直接回答哪些硬件问题

这份文档现在可以直接回答以下问题，而不需要再反复回查源码：

1. 当前 SECOND backbone 一共有多少个 3D sparse conv 算子
2. 每一层的输入输出空间尺寸、通道数和权重逻辑形状是什么
3. 哪些层会改变 active coordinates，哪些层只复用上一层 active set
4. 稀疏卷积层之间是否存在 BN、ReLU、残差旁路
5. `conv_out` 与后续 BEV 接口的边界在哪里
6. stage-level block partition 在当前 backbone 中应该放在哪些边界上实施

如果后续进入容量建模与论文定量实验阶段，再在本文基础上补充逐 stage 动态统计即可。
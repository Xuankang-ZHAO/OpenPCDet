# 三种 Block Structuring 方案的 Feature Map 存储与 Hash Entry 对比

本表基于 KITTI `val/000216`、INT8 SECOND 3D backbone 的现有统计结果，对三种 Block Structuring 和分配方案进行统一比较：

- **固定块 + 固定容量（Fixed-capacity）**：全网络采用 `10x10x6` 固定块；每个物化 block 预留 600 个 voxel slot。每个 block 对应一个 hash entry，因此 **hash entry 数等于物化 block 数**。
- **固定块 + Page（Fixed-block paging）**：同样采用 `10x10x6` 固定块；每个 block 按实际存储记录数分配 64-voxel page。每个 page 对应一个 hash entry，因此 **hash entry 数等于 page 数**。
- **可变块 + Page（Proposed）**：各 stage 按空间 zone 采用不同 block size，并按实际存储记录数分配 64-voxel page。每个 page 对应一个 hash entry，因此 **hash entry 数等于 page 数**。

## 统计口径

- 上一层 OFM 与下一层 IFM 是同一个 feature map，表中只统计一次。
- 不同卷积层产生的 feature map 即使有效体素坐标相同，特征值和存储生命周期仍不同，因此分别列出。
- “有效体素数”不包含 halo 副本；DRAM 分配量包含坐标、特征、halo 副本和预留空槽。
- “物化 Blocks”包含核心区域有有效体素的 block，以及因 halo 副本而产生的 halo-only block。
- “Blocks > 128 voxels”统计实际存储记录数（有效体素及 halo 副本）超过 128 的物化 block 数；这些 block 超过两个 64-voxel page 的直接处理容量，需要进入 reshape/异常处理路径。
- 初始输入作为第一个待存储 feature map 纳入统计。
- `conv_out.0` 的逻辑输出包含 6612 个有效体素和 128 个通道，但现有三种统计均未为它分配 block/page，因此不纳入 DRAM 和 hash entry 对比。
- 表中 MiB 按 `1 MiB = 2^20 Byte` 计算。

## 按 Feature Map 去重后的对比

<table>
  <thead>
    <tr>
      <th rowspan="2">Feature map（产生它的层）</th>
      <th rowspan="2">有效体素数</th>
      <th rowspan="2">通道数</th>
      <th colspan="4">固定块 + 固定容量</th>
      <th colspan="4">固定块 + Page</th>
      <th colspan="4">Proposed 可变块 + Page</th>
    </tr>
    <tr>
      <th>DRAM / MiB</th>
      <th>Blocks</th>
      <th>Hash entries</th>
      <th>Blocks &gt; 128 voxels</th>
      <th>DRAM / MiB</th>
      <th>Blocks</th>
      <th>Hash entries</th>
      <th>Blocks &gt; 128 voxels</th>
      <th>DRAM / MiB</th>
      <th>Blocks</th>
      <th>Hash entries</th>
      <th>Blocks &gt; 128 voxels</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>初始输入</td><td>15000</td><td>4</td><td>29.7318</td><td>4330</td><td>4330</td><td>0</td><td>4.2402</td><td>4330</td><td>4342</td><td>0</td><td>1.1230</td><td>1112</td><td>1150</td><td>5</td></tr>
    <tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>60.5484</td><td>4409</td><td>4409</td><td>0</td><td>6.4761</td><td>4409</td><td>4421</td><td>0</td><td>1.6523</td><td>1094</td><td>1128</td><td>4</td></tr>
    <tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>59.4635</td><td>4330</td><td>4330</td><td>0</td><td>6.3604</td><td>4330</td><td>4342</td><td>0</td><td>1.6846</td><td>1112</td><td>1150</td><td>5</td></tr>
    <tr><td><code>conv2.0.0</code></td><td>25762</td><td>32</td><td>48.4314</td><td>2116</td><td>2116</td><td>43</td><td>5.8105</td><td>2116</td><td>2380</td><td>43</td><td>3.7988</td><td>1316</td><td>1556</td><td>43</td></tr>
    <tr><td><code>conv2.1.0</code></td><td>25762</td><td>32</td><td>48.4314</td><td>2116</td><td>2116</td><td>43</td><td>5.8105</td><td>2116</td><td>2380</td><td>43</td><td>3.7988</td><td>1316</td><td>1556</td><td>43</td></tr>
    <tr><td><code>conv2.2.0</code></td><td>25762</td><td>32</td><td>45.9824</td><td>2009</td><td>2009</td><td>38</td><td>5.5054</td><td>2009</td><td>2255</td><td>38</td><td>4.3481</td><td>1511</td><td>1781</td><td>42</td></tr>
    <tr><td><code>conv3.0.0</code></td><td>19089</td><td>64</td><td>23.5245</td><td>571</td><td>571</td><td>61</td><td>3.7266</td><td>571</td><td>848</td><td>61</td><td>5.9282</td><td>1138</td><td>1349</td><td>31</td></tr>
    <tr><td><code>conv3.1.0</code></td><td>19089</td><td>64</td><td>23.5245</td><td>571</td><td>571</td><td>61</td><td>3.7266</td><td>571</td><td>848</td><td>61</td><td>5.9282</td><td>1138</td><td>1349</td><td>31</td></tr>
    <tr><td><code>conv3.2.0</code></td><td>19089</td><td>64</td><td>22.7829</td><td>553</td><td>553</td><td>60</td><td>3.6650</td><td>553</td><td>834</td><td>60</td><td>5.8008</td><td>1105</td><td>1320</td><td>33</td></tr>
    <tr><td><code>conv4.0.0</code></td><td>8495</td><td>64</td><td>6.5506</td><td>159</td><td>159</td><td>31</td><td>1.2480</td><td>159</td><td>284</td><td>31</td><td>2.1445</td><td>394</td><td>488</td><td>4</td></tr>
    <tr><td><code>conv4.1.0</code></td><td>8495</td><td>64</td><td>6.5506</td><td>159</td><td>159</td><td>31</td><td>1.2480</td><td>159</td><td>284</td><td>31</td><td>2.1445</td><td>394</td><td>488</td><td>4</td></tr>
    <tr><td><code>conv4.2.0</code></td><td>8495</td><td>64</td><td>6.1798</td><td>150</td><td>150</td><td>14</td><td>0.9800</td><td>150</td><td>223</td><td>14</td><td>1.6040</td><td>351</td><td>365</td><td>1</td></tr>
    <tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>60.5484</strong></td><td><strong>4409</strong></td><td><strong>4409</strong></td><td><strong>61</strong></td><td><strong>6.4761</strong></td><td><strong>4409</strong></td><td><strong>4421</strong></td><td><strong>61</strong></td><td><strong>5.9282</strong></td><td><strong>1511</strong></td><td><strong>1781</strong></td><td><strong>43</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值，因此 DRAM、block 数、hash entry 数和超 128-voxel block 数的最大值不一定来自同一个 feature map。

相对于固定容量方案，Proposed 将单个 feature map 的最大 DRAM 分配量降低 **90.21%**，将最大 hash entry 数降低 **59.61%**。相对于固定块分页方案，Proposed 将单个 feature map 的最大 DRAM 分配量降低 **8.46%**，将最大 hash entry 数降低 **59.72%**。

## 执行时 IFM 与 OFM 同时驻留峰值

逐 feature map 表用于观察单个数据对象的存储和索引效率。实际执行一层卷积时，IFM 与 OFM 需要同时驻留，因此硬件所需的 DRAM 容量和 hash entry 容量应按相邻两个 feature map 之和计算。

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 120.0119 | 12.8364 | 11.8564 |
| Hash entries | 8739 | 8763 | 3337 |

相对于固定容量方案，Proposed 将执行时 DRAM 峰值降低 **90.12%**，将 hash entry 峰值降低 **61.81%**。相对于固定块分页方案，Proposed 将执行时 DRAM 峰值降低 **7.63%**，将 hash entry 峰值降低 **61.92%**。

需要注意，Proposed 报告中的 `2698 pages at conv3.1.0` 是 **DRAM Byte 峰值所在层**的 page 数，并非全网络最大的同时驻留 page 数。用于确定片上 hash 表容量的正确峰值是 `3337 entries`，出现在 `conv2.2.0`：IFM 为 1556 pages，OFM 为 1781 pages。

## 结果含义

固定块分页将固定容量 block 内的大量预留空槽替换成按需 page，因而显著降低 DRAM 占用；但每个非空 block 至少需要一个 page，跨页 block 还需要多个 page，所以其最大 hash entry 数没有相对固定容量方案下降。

Proposed 进一步改变空间分块粒度：稀疏区域使用较大 block，减少物化 block、halo 边界和小占用 page；高占用区域使用较小 block，限制单个 block 的体素数量及 reshape 压力。现有单帧结果显示，其最突出的收益是将执行时峰值 hash entry 需求从固定块分页方案的 8763 降至 3337，同时将 DRAM 峰值从 12.8364 MiB 降至 11.8564 MiB。

因此，论文中宜将逐 feature map 表用于解释 Block Structuring 如何影响存储与索引效率，再用同时驻留峰值表说明最终硬件容量需求。

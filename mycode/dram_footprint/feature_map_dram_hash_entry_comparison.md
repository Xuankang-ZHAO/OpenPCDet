# 三种 Block Structuring 方案的 Feature Map 存储与 Hash Entry 对比

本表基于 KITTI `val/000216`、INT8 SECOND 3D backbone，对三种 Block Structuring 方案做统一比较。
体素上限是 `kitti_dataset.yaml` 的 `MAX_NUMBER_OF_VOXELS.test=40000`。

- 加载：KITTI FOV（`FOV_POINTS_ONLY=True`）。
- 模型：hardware-reference INT8 SECOND 3D backbone，checkpoint `checkpoint_epoch_10.pth`。
- Halo：由下一层 kernel/padding 决定的窗口角点复制；`conv_out` 逻辑输出不分配 DRAM。
- 三种方案：固定块+固定容量（`10x10x6`、每块 600 slot）、固定块+Page、Proposed 可变块+Page。
- 上一层 OFM 与下一层 IFM 是同一 feature map，表中只统计一次。
- Hash entries：固定容量方案等于物化 block 数；两种 Page 方案等于 page 数。
- 执行时峰值按 IFM 与 OFM 同时驻留求和。
- Generated: `2026-09-27T16:20:13`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000216.bin`
- 输入体素: `15118`，坐标 SHA-256 `59cf47ea7347a1af112a588cf4ff5757d83cd151588cd196934025a5590709c6`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>15118</td><td>4</td><td>29.8004</td><td>4340</td><td>4340</td><td>0</td><td>4.2500</td><td>4340</td><td>4352</td><td>0</td><td>1.1279</td><td>1117</td><td>1155</td><td>5</td></tr>
<tr><td><code>conv_input.0</code></td><td>15118</td><td>16</td><td>60.7132</td><td>4421</td><td>4421</td><td>0</td><td>6.4937</td><td>4421</td><td>4433</td><td>0</td><td>1.6611</td><td>1100</td><td>1134</td><td>4</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15118</td><td>16</td><td>59.6008</td><td>4340</td><td>4340</td><td>0</td><td>6.3750</td><td>4340</td><td>4352</td><td>0</td><td>1.6919</td><td>1117</td><td>1155</td><td>5</td></tr>
<tr><td><code>conv2.0.0</code></td><td>25846</td><td>32</td><td>48.4314</td><td>2116</td><td>2116</td><td>45</td><td>5.8154</td><td>2116</td><td>2382</td><td>45</td><td>3.8062</td><td>1318</td><td>1559</td><td>43</td></tr>
<tr><td><code>conv2.1.0</code></td><td>25846</td><td>32</td><td>48.4314</td><td>2116</td><td>2116</td><td>45</td><td>5.8154</td><td>2116</td><td>2382</td><td>45</td><td>3.8062</td><td>1318</td><td>1559</td><td>43</td></tr>
<tr><td><code>conv2.2.0</code></td><td>25846</td><td>32</td><td>45.9824</td><td>2009</td><td>2009</td><td>39</td><td>5.5103</td><td>2009</td><td>2257</td><td>39</td><td>4.3506</td><td>1512</td><td>1782</td><td>42</td></tr>
<tr><td><code>conv3.0.0</code></td><td>19116</td><td>64</td><td>23.5245</td><td>571</td><td>571</td><td>61</td><td>3.7266</td><td>571</td><td>848</td><td>61</td><td>5.9370</td><td>1140</td><td>1351</td><td>31</td></tr>
<tr><td><code>conv3.1.0</code></td><td>19116</td><td>64</td><td>23.5245</td><td>571</td><td>571</td><td>61</td><td>3.7266</td><td>571</td><td>848</td><td>61</td><td>5.9370</td><td>1140</td><td>1351</td><td>31</td></tr>
<tr><td><code>conv3.2.0</code></td><td>19116</td><td>64</td><td>22.7829</td><td>553</td><td>553</td><td>60</td><td>3.6650</td><td>553</td><td>834</td><td>60</td><td>5.8096</td><td>1107</td><td>1322</td><td>33</td></tr>
<tr><td><code>conv4.0.0</code></td><td>8499</td><td>64</td><td>6.5506</td><td>159</td><td>159</td><td>31</td><td>1.2480</td><td>159</td><td>284</td><td>31</td><td>2.1445</td><td>394</td><td>488</td><td>4</td></tr>
<tr><td><code>conv4.1.0</code></td><td>8499</td><td>64</td><td>6.5506</td><td>159</td><td>159</td><td>31</td><td>1.2480</td><td>159</td><td>284</td><td>31</td><td>2.1445</td><td>394</td><td>488</td><td>4</td></tr>
<tr><td><code>conv4.2.0</code></td><td>8499</td><td>64</td><td>6.1798</td><td>150</td><td>150</td><td>14</td><td>0.9800</td><td>150</td><td>223</td><td>14</td><td>1.6040</td><td>351</td><td>365</td><td>1</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>60.7132</strong></td><td><strong>4421</strong></td><td><strong>4421</strong></td><td><strong>61</strong></td><td><strong>6.4937</strong></td><td><strong>4421</strong></td><td><strong>4433</strong></td><td><strong>61</strong></td><td><strong>5.9370</strong></td><td><strong>1512</strong></td><td><strong>1782</strong></td><td><strong>43</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **90.13%**，将 hash entry 峰值降低 **61.87%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **7.73%**，将 hash entry 峰值降低 **61.97%**。

## 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 120.3140 | 12.8687 | 11.8740 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 8761 | 8785 | 3341 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

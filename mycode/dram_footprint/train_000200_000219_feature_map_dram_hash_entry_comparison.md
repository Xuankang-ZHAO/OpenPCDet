# 三种 Block Structuring 方案在 filename 000200–000219 上的稳定性与泛化

本文件复现 `feature_map_dram_hash_entry_comparison.md` 在 KITTI `val/000216` 上的口径，
以及 `train_000000_000019_feature_map_dram_hash_entry_comparison.md` 的算法与实验设置。
000000–000019 属于前 200 帧 profiling 集合（filename `000000`–`000199`）；
本实验从其后再选连续 20 帧：filename `000200`–`000219`。
这些 ID 按 KITTI 官方 train/val 划分交错出现，加载时仍合并 train/val infos。

- 加载：KITTI FOV（`FOV_POINTS_ONLY=True`），与 golden 导出一致。
- 模型：hardware-reference INT8 SECOND 3D backbone，checkpoint `checkpoint_epoch_10.pth`。
- Halo：由下一层 kernel/padding 决定的窗口角点复制；`conv_out` 逻辑输出不分配 DRAM。
- 三种方案：固定块+固定容量（`10x10x6`、每块 600 slot）、固定块+Page、Proposed 可变块+Page。
- 上一层 OFM 与下一层 IFM 是同一 feature map，表中只统计一次。
- Hash entries：固定容量方案等于物化 block 数；两种 Page 方案等于 page 数。
- 执行时峰值按 IFM 与 OFM 同时驻留求和；DRAM 峰值层与 hash 峰值层可能不同。
- Generated: `2026-09-17T19:43:34`

## 20 帧执行时峰值总表

| Frame | Split | 输入体素 | 触达 15000 上限 | 固定容量 DRAM | 固定块 Page DRAM | Proposed DRAM | 固定容量 Hash | 固定块 Page Hash | Proposed Hash | DRAM vs 固定容量 | Hash vs 固定容量 | DRAM vs 固定块 Page | Hash vs 固定块 Page |
| --- | --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `000200` | train | 15000 | Y | 140.9683 | 15.0615 | 13.5791 | 10265 | 10282 | 3948 | 90.37% | 61.54% | 9.84% | 61.60% |
| `000201` | val | 14777 | N | 63.0341 | 7.3047 | 7.5146 | 4590 | 4712 | 2430 | 88.08% | 47.06% | -2.87% | 48.43% |
| `000202` | train | 15000 | Y | 170.9885 | 18.2417 | 17.0947 | 12451 | 12453 | 5324 | 90.00% | 57.24% | 6.29% | 57.25% |
| `000203` | val | 12241 | N | 112.6923 | 12.0498 | 9.0923 | 8206 | 8226 | 2549 | 91.93% | 68.94% | 24.54% | 69.01% |
| `000204` | val | 15000 | Y | 146.5988 | 15.6562 | 14.2778 | 10675 | 10688 | 4459 | 90.26% | 58.23% | 8.80% | 58.28% |
| `000205` | train | 15000 | Y | 112.1704 | 11.9707 | 12.1113 | 8168 | 8172 | 3670 | 89.20% | 55.07% | -1.17% | 55.09% |
| `000206` | train | 15000 | Y | 102.0905 | 10.8926 | 11.4653 | 7434 | 7436 | 3626 | 88.77% | 51.22% | -5.26% | 51.24% |
| `000207` | val | 14881 | N | 150.2518 | 16.0400 | 12.2212 | 10941 | 10950 | 3390 | 91.87% | 69.02% | 23.81% | 69.04% |
| `000208` | train | 15000 | Y | 80.1865 | 8.6572 | 10.1865 | 5839 | 5910 | 3140 | 87.30% | 46.22% | -17.66% | 46.87% |
| `000209` | train | 15000 | Y | 44.1376 | 5.0293 | 6.7148 | 3214 | 3369 | 2276 | 84.79% | 29.18% | -33.51% | 32.44% |
| `000210` | train | 15000 | Y | 112.7609 | 12.0381 | 11.3643 | 8211 | 8218 | 4037 | 89.92% | 50.83% | 5.60% | 50.88% |
| `000211` | val | 13390 | N | 72.1802 | 7.8613 | 9.0527 | 5256 | 5326 | 2849 | 87.46% | 45.80% | -15.16% | 46.51% |
| `000212` | val | 15000 | Y | 75.3937 | 8.2422 | 10.1250 | 5490 | 5622 | 3045 | 86.57% | 44.54% | -22.84% | 45.84% |
| `000213` | val | 14462 | N | 82.1777 | 9.2529 | 10.9688 | 5984 | 6058 | 3096 | 86.65% | 48.26% | -18.54% | 48.89% |
| `000214` | train | 15000 | Y | 109.1354 | 11.6572 | 12.4541 | 7947 | 7958 | 4304 | 88.59% | 45.84% | -6.84% | 45.92% |
| `000215` | train | 14457 | N | 66.8243 | 7.3008 | 6.8115 | 4866 | 4984 | 2076 | 89.81% | 57.34% | 6.70% | 58.35% |
| `000216` | val | 15000 | Y | 120.0119 | 12.8364 | 11.8564 | 8739 | 8763 | 3337 | 90.12% | 61.81% | 7.63% | 61.92% |
| `000217` | train | 15000 | Y | 89.2914 | 9.6680 | 11.3599 | 6502 | 6600 | 3292 | 87.28% | 49.37% | -17.50% | 50.12% |
| `000218` | val | 14636 | N | 77.8107 | 8.4595 | 8.6177 | 5666 | 5775 | 2658 | 88.92% | 53.09% | -1.87% | 53.97% |
| `000219` | train | 15000 | Y | 144.9097 | 15.4761 | 13.6758 | 10552 | 10565 | 4260 | 90.56% | 59.63% | 11.63% | 59.68% |

相对固定容量 / 固定块分页，Proposed 在 20 帧上的降低比例（正值表示 Proposed 更小）：

| 指标 | 最小 | 平均 | 最大 |
| --- | ---: | ---: | ---: |
| DRAM vs 固定容量 | 84.79% | 88.92% | 91.93% |
| Hash vs 固定容量 | 29.18% | 53.01% | 69.02% |
| DRAM vs 固定块 Page | -33.51% | -1.92% | 24.54% |
| Hash vs 固定块 Page | 32.44% | 53.57% | 69.04% |

## Frame `train/000200`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000200.bin`
- 输入体素: `15000`，坐标 SHA-256 `0bef8a5c45d4dd806d8c117f7e0f3ccf8107fe04102513a74a12651dde73dabd`
- 触达 15000 voxel 上限: `True`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>34.2293</td><td>4985</td><td>4985</td><td>0</td><td>4.8760</td><td>4985</td><td>4993</td><td>0</td><td>1.1729</td><td>1173</td><td>1201</td><td>5</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>72.5098</td><td>5280</td><td>5280</td><td>0</td><td>7.7476</td><td>5280</td><td>5289</td><td>0</td><td>1.6787</td><td>1117</td><td>1146</td><td>5</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>68.4586</td><td>4985</td><td>4985</td><td>0</td><td>7.3140</td><td>4985</td><td>4993</td><td>0</td><td>1.7593</td><td>1173</td><td>1201</td><td>5</td></tr>
<tr><td><code>conv2.0.0</code></td><td>30888</td><td>32</td><td>55.8014</td><td>2438</td><td>2438</td><td>40</td><td>6.6504</td><td>2438</td><td>2724</td><td>40</td><td>4.6338</td><td>1658</td><td>1898</td><td>39</td></tr>
<tr><td><code>conv2.1.0</code></td><td>30888</td><td>32</td><td>55.8014</td><td>2438</td><td>2438</td><td>40</td><td>6.6504</td><td>2438</td><td>2724</td><td>40</td><td>4.6338</td><td>1658</td><td>1898</td><td>39</td></tr>
<tr><td><code>conv2.2.0</code></td><td>30888</td><td>32</td><td>55.9616</td><td>2445</td><td>2445</td><td>35</td><td>6.5332</td><td>2445</td><td>2676</td><td>35</td><td>5.0049</td><td>1810</td><td>2050</td><td>34</td></tr>
<tr><td><code>conv3.0.0</code></td><td>23592</td><td>64</td><td>30.4459</td><td>739</td><td>739</td><td>70</td><td>4.5835</td><td>739</td><td>1043</td><td>70</td><td>6.7896</td><td>1348</td><td>1545</td><td>25</td></tr>
<tr><td><code>conv3.1.0</code></td><td>23592</td><td>64</td><td>30.4459</td><td>739</td><td>739</td><td>70</td><td>4.5835</td><td>739</td><td>1043</td><td>70</td><td>6.7896</td><td>1348</td><td>1545</td><td>25</td></tr>
<tr><td><code>conv3.2.0</code></td><td>23592</td><td>64</td><td>30.9814</td><td>752</td><td>752</td><td>71</td><td>4.5923</td><td>752</td><td>1045</td><td>71</td><td>6.6313</td><td>1319</td><td>1509</td><td>23</td></tr>
<tr><td><code>conv4.0.0</code></td><td>11817</td><td>64</td><td>7.4982</td><td>182</td><td>182</td><td>48</td><td>1.5864</td><td>182</td><td>361</td><td>48</td><td>2.7070</td><td>493</td><td>616</td><td>7</td></tr>
<tr><td><code>conv4.1.0</code></td><td>11817</td><td>64</td><td>7.4982</td><td>182</td><td>182</td><td>48</td><td>1.5864</td><td>182</td><td>361</td><td>48</td><td>2.7070</td><td>493</td><td>616</td><td>7</td></tr>
<tr><td><code>conv4.2.0</code></td><td>11817</td><td>64</td><td>7.0038</td><td>170</td><td>170</td><td>32</td><td>1.2261</td><td>170</td><td>279</td><td>32</td><td>2.0039</td><td>431</td><td>456</td><td>1</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>72.5098</strong></td><td><strong>5280</strong></td><td><strong>5280</strong></td><td><strong>71</strong></td><td><strong>7.7476</strong></td><td><strong>5280</strong></td><td><strong>5289</strong></td><td><strong>71</strong></td><td><strong>6.7896</strong></td><td><strong>1810</strong></td><td><strong>2050</strong></td><td><strong>39</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **90.37%**，将 hash entry 峰值降低 **61.54%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **9.84%**，将 hash entry 峰值降低 **61.60%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 140.9683 | 15.0615 | 13.5791 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 10265 | 10282 | 3948 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `val/000201`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000201.bin`
- 输入体素: `14777`，坐标 SHA-256 `fc6b2659caa0ad98415c0a31c942ac7bdc112ae313270dc8c33ab586daf49b32`
- 触达 15000 voxel 上限: `False`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>14777</td><td>4</td><td>14.3440</td><td>2089</td><td>2089</td><td>4</td><td>2.0986</td><td>2089</td><td>2149</td><td>4</td><td>0.8633</td><td>790</td><td>884</td><td>15</td></tr>
<tr><td><code>conv_input.0</code></td><td>14777</td><td>16</td><td>34.3460</td><td>2501</td><td>2501</td><td>3</td><td>3.7544</td><td>2501</td><td>2563</td><td>3</td><td>1.2715</td><td>777</td><td>868</td><td>18</td></tr>
<tr><td><code>conv1.0.0</code></td><td>14777</td><td>16</td><td>28.6880</td><td>2089</td><td>2089</td><td>4</td><td>3.1479</td><td>2089</td><td>2149</td><td>4</td><td>1.2949</td><td>790</td><td>884</td><td>15</td></tr>
<tr><td><code>conv2.0.0</code></td><td>20564</td><td>32</td><td>25.9094</td><td>1132</td><td>1132</td><td>86</td><td>3.6523</td><td>1132</td><td>1496</td><td>86</td><td>2.6221</td><td>829</td><td>1074</td><td>45</td></tr>
<tr><td><code>conv2.1.0</code></td><td>20564</td><td>32</td><td>25.9094</td><td>1132</td><td>1132</td><td>86</td><td>3.6523</td><td>1132</td><td>1496</td><td>86</td><td>2.6221</td><td>829</td><td>1074</td><td>45</td></tr>
<tr><td><code>conv2.2.0</code></td><td>20564</td><td>32</td><td>22.2473</td><td>972</td><td>972</td><td>59</td><td>3.0713</td><td>972</td><td>1258</td><td>59</td><td>3.3105</td><td>1089</td><td>1356</td><td>52</td></tr>
<tr><td><code>conv3.0.0</code></td><td>12119</td><td>64</td><td>14.3784</td><td>349</td><td>349</td><td>53</td><td>2.3774</td><td>349</td><td>541</td><td>53</td><td>3.7573</td><td>728</td><td>855</td><td>16</td></tr>
<tr><td><code>conv3.1.0</code></td><td>12119</td><td>64</td><td>14.3784</td><td>349</td><td>349</td><td>53</td><td>2.3774</td><td>349</td><td>541</td><td>53</td><td>3.7573</td><td>728</td><td>855</td><td>16</td></tr>
<tr><td><code>conv3.2.0</code></td><td>12119</td><td>64</td><td>14.6255</td><td>355</td><td>355</td><td>53</td><td>2.3818</td><td>355</td><td>542</td><td>53</td><td>3.6914</td><td>705</td><td>840</td><td>12</td></tr>
<tr><td><code>conv4.0.0</code></td><td>4933</td><td>64</td><td>3.5431</td><td>86</td><td>86</td><td>22</td><td>0.7119</td><td>86</td><td>162</td><td>22</td><td>1.2524</td><td>226</td><td>285</td><td>1</td></tr>
<tr><td><code>conv4.1.0</code></td><td>4933</td><td>64</td><td>3.5431</td><td>86</td><td>86</td><td>22</td><td>0.7119</td><td>86</td><td>162</td><td>22</td><td>1.2524</td><td>226</td><td>285</td><td>1</td></tr>
<tr><td><code>conv4.2.0</code></td><td>4933</td><td>64</td><td>3.0899</td><td>75</td><td>75</td><td>14</td><td>0.5273</td><td>75</td><td>120</td><td>14</td><td>0.8921</td><td>200</td><td>203</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>34.3460</strong></td><td><strong>2501</strong></td><td><strong>2501</strong></td><td><strong>86</strong></td><td><strong>3.7544</strong></td><td><strong>2501</strong></td><td><strong>2563</strong></td><td><strong>86</strong></td><td><strong>3.7573</strong></td><td><strong>1089</strong></td><td><strong>1356</strong></td><td><strong>52</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **88.08%**，将 hash entry 峰值降低 **47.06%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-2.87%**，将 hash entry 峰值降低 **48.43%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 63.0341 | 7.3047 | 7.5146 |
| DRAM 峰值层 | `conv1.0.0` | `conv2.1.0` | `conv3.1.0` |
| Hash entries | 4590 | 4712 | 2430 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000202`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000202.bin`
- 输入体素: `15000`，坐标 SHA-256 `beec452619619b9d1db9b8fc31f64470fd727ca4c7ce80d1037b52b4ffd8a4c3`
- 触达 15000 voxel 上限: `True`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>41.8854</td><td>6100</td><td>6100</td><td>0</td><td>5.9580</td><td>6100</td><td>6101</td><td>0</td><td>1.3457</td><td>1358</td><td>1378</td><td>1</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>87.2177</td><td>6351</td><td>6351</td><td>0</td><td>9.3047</td><td>6351</td><td>6352</td><td>0</td><td>1.9863</td><td>1338</td><td>1356</td><td>1</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>83.7708</td><td>6100</td><td>6100</td><td>0</td><td>8.9370</td><td>6100</td><td>6101</td><td>0</td><td>2.0186</td><td>1358</td><td>1378</td><td>1</td></tr>
<tr><td><code>conv2.0.0</code></td><td>35062</td><td>32</td><td>70.4041</td><td>3076</td><td>3076</td><td>75</td><td>8.5547</td><td>3076</td><td>3504</td><td>75</td><td>6.1841</td><td>2229</td><td>2533</td><td>36</td></tr>
<tr><td><code>conv2.1.0</code></td><td>35062</td><td>32</td><td>70.4041</td><td>3076</td><td>3076</td><td>75</td><td>8.5547</td><td>3076</td><td>3504</td><td>75</td><td>6.1841</td><td>2229</td><td>2533</td><td>36</td></tr>
<tr><td><code>conv2.2.0</code></td><td>35062</td><td>32</td><td>66.1240</td><td>2889</td><td>2889</td><td>72</td><td>8.0127</td><td>2889</td><td>3282</td><td>72</td><td>6.8140</td><td>2473</td><td>2791</td><td>31</td></tr>
<tr><td><code>conv3.0.0</code></td><td>28955</td><td>64</td><td>36.2961</td><td>881</td><td>881</td><td>110</td><td>5.8228</td><td>881</td><td>1325</td><td>110</td><td>8.5474</td><td>1611</td><td>1945</td><td>42</td></tr>
<tr><td><code>conv3.1.0</code></td><td>28955</td><td>64</td><td>36.2961</td><td>881</td><td>881</td><td>110</td><td>5.8228</td><td>881</td><td>1325</td><td>110</td><td>8.5474</td><td>1611</td><td>1945</td><td>42</td></tr>
<tr><td><code>conv3.2.0</code></td><td>28955</td><td>64</td><td>35.8429</td><td>870</td><td>870</td><td>110</td><td>5.7656</td><td>870</td><td>1312</td><td>110</td><td>8.5342</td><td>1607</td><td>1942</td><td>44</td></tr>
<tr><td><code>conv4.0.0</code></td><td>14543</td><td>64</td><td>8.1161</td><td>197</td><td>197</td><td>69</td><td>1.9292</td><td>197</td><td>439</td><td>69</td><td>3.2080</td><td>509</td><td>730</td><td>30</td></tr>
<tr><td><code>conv4.1.0</code></td><td>14543</td><td>64</td><td>8.1161</td><td>197</td><td>197</td><td>69</td><td>1.9292</td><td>197</td><td>439</td><td>69</td><td>3.2080</td><td>509</td><td>730</td><td>30</td></tr>
<tr><td><code>conv4.2.0</code></td><td>14543</td><td>64</td><td>7.5394</td><td>183</td><td>183</td><td>46</td><td>1.4810</td><td>183</td><td>337</td><td>46</td><td>2.2061</td><td>464</td><td>502</td><td>3</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>87.2177</strong></td><td><strong>6351</strong></td><td><strong>6351</strong></td><td><strong>110</strong></td><td><strong>9.3047</strong></td><td><strong>6351</strong></td><td><strong>6352</strong></td><td><strong>110</strong></td><td><strong>8.5474</strong></td><td><strong>2473</strong></td><td><strong>2791</strong></td><td><strong>44</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **90.00%**，将 hash entry 峰值降低 **57.24%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **6.29%**，将 hash entry 峰值降低 **57.25%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 170.9885 | 18.2417 | 17.0947 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 12451 | 12453 | 5324 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `val/000203`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000203.bin`
- 输入体素: `12241`，坐标 SHA-256 `f3eaeb15e16dafd7383bb01ed0ca1d6f7932748c3c08948eaeb1bbf27921698a`
- 触达 15000 voxel 上限: `False`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>12241</td><td>4</td><td>28.3241</td><td>4125</td><td>4125</td><td>0</td><td>4.0371</td><td>4125</td><td>4134</td><td>0</td><td>0.8594</td><td>866</td><td>880</td><td>0</td></tr>
<tr><td><code>conv_input.0</code></td><td>12241</td><td>16</td><td>56.0440</td><td>4081</td><td>4081</td><td>0</td><td>5.9941</td><td>4081</td><td>4092</td><td>0</td><td>1.3022</td><td>878</td><td>889</td><td>0</td></tr>
<tr><td><code>conv1.0.0</code></td><td>12241</td><td>16</td><td>56.6483</td><td>4125</td><td>4125</td><td>0</td><td>6.0557</td><td>4125</td><td>4134</td><td>0</td><td>1.2891</td><td>866</td><td>880</td><td>0</td></tr>
<tr><td><code>conv2.0.0</code></td><td>21894</td><td>32</td><td>40.9470</td><td>1789</td><td>1789</td><td>23</td><td>4.8291</td><td>1789</td><td>1978</td><td>23</td><td>2.9126</td><td>1050</td><td>1193</td><td>17</td></tr>
<tr><td><code>conv2.1.0</code></td><td>21894</td><td>32</td><td>40.9470</td><td>1789</td><td>1789</td><td>23</td><td>4.8291</td><td>1789</td><td>1978</td><td>23</td><td>2.9126</td><td>1050</td><td>1193</td><td>17</td></tr>
<tr><td><code>conv2.2.0</code></td><td>21894</td><td>32</td><td>46.6690</td><td>2039</td><td>2039</td><td>27</td><td>5.3906</td><td>2039</td><td>2208</td><td>27</td><td>3.3105</td><td>1221</td><td>1356</td><td>25</td></tr>
<tr><td><code>conv3.0.0</code></td><td>17167</td><td>64</td><td>20.1462</td><td>489</td><td>489</td><td>50</td><td>3.0630</td><td>489</td><td>697</td><td>50</td><td>4.5044</td><td>920</td><td>1025</td><td>11</td></tr>
<tr><td><code>conv3.1.0</code></td><td>17167</td><td>64</td><td>20.1462</td><td>489</td><td>489</td><td>50</td><td>3.0630</td><td>489</td><td>697</td><td>50</td><td>4.5044</td><td>920</td><td>1025</td><td>11</td></tr>
<tr><td><code>conv3.2.0</code></td><td>17167</td><td>64</td><td>20.0226</td><td>486</td><td>486</td><td>47</td><td>3.1025</td><td>486</td><td>706</td><td>47</td><td>4.5879</td><td>923</td><td>1044</td><td>14</td></tr>
<tr><td><code>conv4.0.0</code></td><td>9256</td><td>64</td><td>5.1910</td><td>126</td><td>126</td><td>39</td><td>1.1689</td><td>126</td><td>266</td><td>39</td><td>2.0083</td><td>356</td><td>457</td><td>8</td></tr>
<tr><td><code>conv4.1.0</code></td><td>9256</td><td>64</td><td>5.1910</td><td>126</td><td>126</td><td>39</td><td>1.1689</td><td>126</td><td>266</td><td>39</td><td>2.0083</td><td>356</td><td>457</td><td>8</td></tr>
<tr><td><code>conv4.2.0</code></td><td>9256</td><td>64</td><td>5.0674</td><td>123</td><td>123</td><td>19</td><td>0.9272</td><td>123</td><td>211</td><td>19</td><td>1.4766</td><td>318</td><td>336</td><td>4</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>56.6483</strong></td><td><strong>4125</strong></td><td><strong>4125</strong></td><td><strong>50</strong></td><td><strong>6.0557</strong></td><td><strong>4125</strong></td><td><strong>4134</strong></td><td><strong>50</strong></td><td><strong>4.5879</strong></td><td><strong>1221</strong></td><td><strong>1356</strong></td><td><strong>25</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **91.93%**，将 hash entry 峰值降低 **68.94%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **24.54%**，将 hash entry 峰值降低 **69.01%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 112.6923 | 12.0498 | 9.0923 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.2.0` |
| Hash entries | 8206 | 8226 | 2549 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `val/000204`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000204.bin`
- 输入体素: `15000`，坐标 SHA-256 `ed114309009f0eb03d701516b4cb668a093c1753510106488c6ec17a86a10d29`
- 触达 15000 voxel 上限: `True`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>35.0601</td><td>5106</td><td>5106</td><td>0</td><td>4.9932</td><td>5106</td><td>5113</td><td>0</td><td>1.2891</td><td>1287</td><td>1320</td><td>1</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>76.4786</td><td>5569</td><td>5569</td><td>0</td><td>8.1665</td><td>5569</td><td>5575</td><td>0</td><td>1.8428</td><td>1225</td><td>1258</td><td>3</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>70.1202</td><td>5106</td><td>5106</td><td>0</td><td>7.4897</td><td>5106</td><td>5113</td><td>0</td><td>1.9336</td><td>1287</td><td>1320</td><td>1</td></tr>
<tr><td><code>conv2.0.0</code></td><td>33912</td><td>32</td><td>61.7752</td><td>2699</td><td>2699</td><td>49</td><td>7.4927</td><td>2699</td><td>3069</td><td>49</td><td>4.9487</td><td>1695</td><td>2027</td><td>46</td></tr>
<tr><td><code>conv2.1.0</code></td><td>33912</td><td>32</td><td>61.7752</td><td>2699</td><td>2699</td><td>49</td><td>7.4927</td><td>2699</td><td>3069</td><td>49</td><td>4.9487</td><td>1695</td><td>2027</td><td>46</td></tr>
<tr><td><code>conv2.2.0</code></td><td>33912</td><td>32</td><td>60.6537</td><td>2650</td><td>2650</td><td>43</td><td>7.2241</td><td>2650</td><td>2959</td><td>43</td><td>5.9375</td><td>2097</td><td>2432</td><td>52</td></tr>
<tr><td><code>conv3.0.0</code></td><td>26811</td><td>64</td><td>28.9215</td><td>702</td><td>702</td><td>99</td><td>4.7900</td><td>702</td><td>1090</td><td>99</td><td>7.0884</td><td>1352</td><td>1613</td><td>39</td></tr>
<tr><td><code>conv3.1.0</code></td><td>26811</td><td>64</td><td>28.9215</td><td>702</td><td>702</td><td>99</td><td>4.7900</td><td>702</td><td>1090</td><td>99</td><td>7.0884</td><td>1352</td><td>1613</td><td>39</td></tr>
<tr><td><code>conv3.2.0</code></td><td>26811</td><td>64</td><td>29.0451</td><td>705</td><td>705</td><td>105</td><td>4.9175</td><td>705</td><td>1119</td><td>105</td><td>7.1895</td><td>1361</td><td>1636</td><td>42</td></tr>
<tr><td><code>conv4.0.0</code></td><td>12706</td><td>64</td><td>7.5806</td><td>184</td><td>184</td><td>59</td><td>1.6919</td><td>184</td><td>385</td><td>59</td><td>2.8564</td><td>491</td><td>650</td><td>13</td></tr>
<tr><td><code>conv4.1.0</code></td><td>12706</td><td>64</td><td>7.5806</td><td>184</td><td>184</td><td>59</td><td>1.6919</td><td>184</td><td>385</td><td>59</td><td>2.8564</td><td>491</td><td>650</td><td>13</td></tr>
<tr><td><code>conv4.2.0</code></td><td>12706</td><td>64</td><td>7.1686</td><td>174</td><td>174</td><td>35</td><td>1.3315</td><td>174</td><td>303</td><td>35</td><td>2.0830</td><td>445</td><td>474</td><td>3</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>76.4786</strong></td><td><strong>5569</strong></td><td><strong>5569</strong></td><td><strong>105</strong></td><td><strong>8.1665</strong></td><td><strong>5569</strong></td><td><strong>5575</strong></td><td><strong>105</strong></td><td><strong>7.1895</strong></td><td><strong>2097</strong></td><td><strong>2432</strong></td><td><strong>52</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **90.26%**，将 hash entry 峰值降低 **58.23%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **8.80%**，将 hash entry 峰值降低 **58.28%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 146.5988 | 15.6562 | 14.2778 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.2.0` |
| Hash entries | 10675 | 10688 | 4459 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000205`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000205.bin`
- 输入体素: `15000`，坐标 SHA-256 `a502b776b53c7e51be35cac1cb522fd898b49199ccccefd410bf61d33b131168`
- 触达 15000 voxel 上限: `True`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>27.2530</td><td>3969</td><td>3969</td><td>0</td><td>3.8779</td><td>3969</td><td>3971</td><td>0</td><td>1.0537</td><td>1036</td><td>1079</td><td>10</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>57.6645</td><td>4199</td><td>4199</td><td>0</td><td>6.1538</td><td>4199</td><td>4201</td><td>0</td><td>1.5703</td><td>1032</td><td>1072</td><td>10</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>54.5059</td><td>3969</td><td>3969</td><td>0</td><td>5.8169</td><td>3969</td><td>3971</td><td>0</td><td>1.5806</td><td>1036</td><td>1079</td><td>10</td></tr>
<tr><td><code>conv2.0.0</code></td><td>27963</td><td>32</td><td>44.1742</td><td>1930</td><td>1930</td><td>38</td><td>5.4688</td><td>1930</td><td>2240</td><td>38</td><td>4.1772</td><td>1468</td><td>1711</td><td>33</td></tr>
<tr><td><code>conv2.1.0</code></td><td>27963</td><td>32</td><td>44.1742</td><td>1930</td><td>1930</td><td>38</td><td>5.4688</td><td>1930</td><td>2240</td><td>38</td><td>4.1772</td><td>1468</td><td>1711</td><td>33</td></tr>
<tr><td><code>conv2.2.0</code></td><td>27963</td><td>32</td><td>41.6565</td><td>1820</td><td>1820</td><td>39</td><td>5.0928</td><td>1820</td><td>2086</td><td>39</td><td>4.7827</td><td>1710</td><td>1959</td><td>45</td></tr>
<tr><td><code>conv3.0.0</code></td><td>19120</td><td>64</td><td>22.9477</td><td>557</td><td>557</td><td>68</td><td>3.6343</td><td>557</td><td>827</td><td>68</td><td>6.0557</td><td>1209</td><td>1378</td><td>21</td></tr>
<tr><td><code>conv3.1.0</code></td><td>19120</td><td>64</td><td>22.9477</td><td>557</td><td>557</td><td>68</td><td>3.6343</td><td>557</td><td>827</td><td>68</td><td>6.0557</td><td>1209</td><td>1378</td><td>21</td></tr>
<tr><td><code>conv3.2.0</code></td><td>19120</td><td>64</td><td>22.8653</td><td>555</td><td>555</td><td>68</td><td>3.6123</td><td>555</td><td>822</td><td>68</td><td>5.9941</td><td>1210</td><td>1364</td><td>20</td></tr>
<tr><td><code>conv4.0.0</code></td><td>8957</td><td>64</td><td>6.0974</td><td>148</td><td>148</td><td>41</td><td>1.2832</td><td>148</td><td>292</td><td>41</td><td>2.2017</td><td>397</td><td>501</td><td>5</td></tr>
<tr><td><code>conv4.1.0</code></td><td>8957</td><td>64</td><td>6.0974</td><td>148</td><td>148</td><td>41</td><td>1.2832</td><td>148</td><td>292</td><td>41</td><td>2.2017</td><td>397</td><td>501</td><td>5</td></tr>
<tr><td><code>conv4.2.0</code></td><td>8957</td><td>64</td><td>5.3146</td><td>129</td><td>129</td><td>22</td><td>0.9536</td><td>129</td><td>217</td><td>22</td><td>1.6216</td><td>356</td><td>369</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>57.6645</strong></td><td><strong>4199</strong></td><td><strong>4199</strong></td><td><strong>68</strong></td><td><strong>6.1538</strong></td><td><strong>4199</strong></td><td><strong>4201</strong></td><td><strong>68</strong></td><td><strong>6.0557</strong></td><td><strong>1710</strong></td><td><strong>1959</strong></td><td><strong>45</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.20%**，将 hash entry 峰值降低 **55.07%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-1.17%**，将 hash entry 峰值降低 **55.09%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 112.1704 | 11.9707 | 12.1113 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 8168 | 8172 | 3670 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000206`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000206.bin`
- 输入体素: `15000`，坐标 SHA-256 `ceed73d1146a8c5300d373f3a4d57239872a4fcf0a7411fdf9bc72b9238ff2ae`
- 触达 15000 voxel 上限: `True`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>24.9596</td><td>3635</td><td>3635</td><td>0</td><td>3.5508</td><td>3635</td><td>3636</td><td>0</td><td>1.1025</td><td>1064</td><td>1129</td><td>16</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>52.1713</td><td>3799</td><td>3799</td><td>0</td><td>5.5664</td><td>3799</td><td>3800</td><td>0</td><td>1.5322</td><td>985</td><td>1046</td><td>15</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>49.9191</td><td>3635</td><td>3635</td><td>0</td><td>5.3262</td><td>3635</td><td>3636</td><td>0</td><td>1.6538</td><td>1064</td><td>1129</td><td>16</td></tr>
<tr><td><code>conv2.0.0</code></td><td>27640</td><td>32</td><td>41.1301</td><td>1797</td><td>1797</td><td>61</td><td>5.2271</td><td>1797</td><td>2141</td><td>61</td><td>4.1187</td><td>1401</td><td>1687</td><td>68</td></tr>
<tr><td><code>conv2.1.0</code></td><td>27640</td><td>32</td><td>41.1301</td><td>1797</td><td>1797</td><td>61</td><td>5.2271</td><td>1797</td><td>2141</td><td>61</td><td>4.1187</td><td>1401</td><td>1687</td><td>68</td></tr>
<tr><td><code>conv2.2.0</code></td><td>27640</td><td>32</td><td>34.9960</td><td>1529</td><td>1529</td><td>63</td><td>4.4653</td><td>1529</td><td>1829</td><td>63</td><td>4.7339</td><td>1637</td><td>1939</td><td>68</td></tr>
<tr><td><code>conv3.0.0</code></td><td>19029</td><td>64</td><td>17.5095</td><td>425</td><td>425</td><td>84</td><td>3.2344</td><td>425</td><td>736</td><td>84</td><td>5.7261</td><td>1090</td><td>1303</td><td>33</td></tr>
<tr><td><code>conv3.1.0</code></td><td>19029</td><td>64</td><td>17.5095</td><td>425</td><td>425</td><td>84</td><td>3.2344</td><td>425</td><td>736</td><td>84</td><td>5.7261</td><td>1090</td><td>1303</td><td>33</td></tr>
<tr><td><code>conv3.2.0</code></td><td>19029</td><td>64</td><td>17.4271</td><td>423</td><td>423</td><td>81</td><td>3.1597</td><td>423</td><td>719</td><td>81</td><td>5.7393</td><td>1097</td><td>1306</td><td>35</td></tr>
<tr><td><code>conv4.0.0</code></td><td>8258</td><td>64</td><td>4.3671</td><td>106</td><td>106</td><td>36</td><td>1.0723</td><td>106</td><td>244</td><td>36</td><td>2.0435</td><td>350</td><td>465</td><td>9</td></tr>
<tr><td><code>conv4.1.0</code></td><td>8258</td><td>64</td><td>4.3671</td><td>106</td><td>106</td><td>36</td><td>1.0723</td><td>106</td><td>244</td><td>36</td><td>2.0435</td><td>350</td><td>465</td><td>9</td></tr>
<tr><td><code>conv4.2.0</code></td><td>8258</td><td>64</td><td>4.2435</td><td>103</td><td>103</td><td>29</td><td>0.8306</td><td>103</td><td>189</td><td>29</td><td>1.4854</td><td>320</td><td>338</td><td>3</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>52.1713</strong></td><td><strong>3799</strong></td><td><strong>3799</strong></td><td><strong>84</strong></td><td><strong>5.5664</strong></td><td><strong>3799</strong></td><td><strong>3800</strong></td><td><strong>84</strong></td><td><strong>5.7393</strong></td><td><strong>1637</strong></td><td><strong>1939</strong></td><td><strong>68</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **88.77%**，将 hash entry 峰值降低 **51.22%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-5.26%**，将 hash entry 峰值降低 **51.24%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 102.0905 | 10.8926 | 11.4653 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.2.0` |
| Hash entries | 7434 | 7436 | 3626 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `val/000207`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000207.bin`
- 输入体素: `14881`，坐标 SHA-256 `2c75094c07c773714f6012c5939965e5b055f5d6cc73a7d770d448403e81cec1`
- 触达 15000 voxel 上限: `False`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>14881</td><td>4</td><td>37.4977</td><td>5461</td><td>5461</td><td>0</td><td>5.3369</td><td>5461</td><td>5465</td><td>0</td><td>1.0645</td><td>1072</td><td>1090</td><td>3</td></tr>
<tr><td><code>conv_input.0</code></td><td>14881</td><td>16</td><td>75.2563</td><td>5480</td><td>5480</td><td>0</td><td>8.0347</td><td>5480</td><td>5485</td><td>0</td><td>1.5967</td><td>1074</td><td>1090</td><td>4</td></tr>
<tr><td><code>conv1.0.0</code></td><td>14881</td><td>16</td><td>74.9954</td><td>5461</td><td>5461</td><td>0</td><td>8.0054</td><td>5461</td><td>5465</td><td>0</td><td>1.5967</td><td>1072</td><td>1090</td><td>3</td></tr>
<tr><td><code>conv2.0.0</code></td><td>27416</td><td>32</td><td>57.9071</td><td>2530</td><td>2530</td><td>27</td><td>6.8018</td><td>2530</td><td>2786</td><td>27</td><td>3.9307</td><td>1438</td><td>1610</td><td>20</td></tr>
<tr><td><code>conv2.1.0</code></td><td>27416</td><td>32</td><td>57.9071</td><td>2530</td><td>2530</td><td>27</td><td>6.8018</td><td>2530</td><td>2786</td><td>27</td><td>3.9307</td><td>1438</td><td>1610</td><td>20</td></tr>
<tr><td><code>conv2.2.0</code></td><td>27416</td><td>32</td><td>57.0374</td><td>2492</td><td>2492</td><td>16</td><td>6.6113</td><td>2492</td><td>2708</td><td>16</td><td>4.3457</td><td>1597</td><td>1780</td><td>25</td></tr>
<tr><td><code>conv3.0.0</code></td><td>20277</td><td>64</td><td>25.0900</td><td>609</td><td>609</td><td>56</td><td>3.8145</td><td>609</td><td>868</td><td>56</td><td>6.1084</td><td>1235</td><td>1390</td><td>19</td></tr>
<tr><td><code>conv3.1.0</code></td><td>20277</td><td>64</td><td>25.0900</td><td>609</td><td>609</td><td>56</td><td>3.8145</td><td>609</td><td>868</td><td>56</td><td>6.1084</td><td>1235</td><td>1390</td><td>19</td></tr>
<tr><td><code>conv3.2.0</code></td><td>20277</td><td>64</td><td>24.8428</td><td>603</td><td>603</td><td>51</td><td>3.8013</td><td>603</td><td>865</td><td>51</td><td>6.1128</td><td>1239</td><td>1391</td><td>22</td></tr>
<tr><td><code>conv4.0.0</code></td><td>10029</td><td>64</td><td>6.8390</td><td>166</td><td>166</td><td>43</td><td>1.4326</td><td>166</td><td>326</td><td>43</td><td>2.3862</td><td>439</td><td>543</td><td>8</td></tr>
<tr><td><code>conv4.1.0</code></td><td>10029</td><td>64</td><td>6.8390</td><td>166</td><td>166</td><td>43</td><td>1.4326</td><td>166</td><td>326</td><td>43</td><td>2.3862</td><td>439</td><td>543</td><td>8</td></tr>
<tr><td><code>conv4.2.0</code></td><td>10029</td><td>64</td><td>6.3034</td><td>153</td><td>153</td><td>19</td><td>1.0591</td><td>153</td><td>241</td><td>19</td><td>1.8413</td><td>394</td><td>419</td><td>2</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>75.2563</strong></td><td><strong>5480</strong></td><td><strong>5480</strong></td><td><strong>56</strong></td><td><strong>8.0347</strong></td><td><strong>5480</strong></td><td><strong>5485</strong></td><td><strong>56</strong></td><td><strong>6.1128</strong></td><td><strong>1597</strong></td><td><strong>1780</strong></td><td><strong>25</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **91.87%**，将 hash entry 峰值降低 **69.02%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **23.81%**，将 hash entry 峰值降低 **69.04%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 150.2518 | 16.0400 | 12.2212 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.2.0` |
| Hash entries | 10941 | 10950 | 3390 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000208`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000208.bin`
- 输入体素: `15000`，坐标 SHA-256 `b17e83a9054c4eaac4350984f172aacdef0d959d84107ca5e984b95f92df2b4e`
- 触达 15000 voxel 上限: `True`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>18.4090</td><td>2681</td><td>2681</td><td>5</td><td>2.6504</td><td>2681</td><td>2714</td><td>5</td><td>0.9980</td><td>920</td><td>1022</td><td>21</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>43.3685</td><td>3158</td><td>3158</td><td>4</td><td>4.6816</td><td>3158</td><td>3196</td><td>4</td><td>1.4941</td><td>923</td><td>1020</td><td>19</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>36.8179</td><td>2681</td><td>2681</td><td>5</td><td>3.9756</td><td>2681</td><td>2714</td><td>5</td><td>1.4971</td><td>920</td><td>1022</td><td>21</td></tr>
<tr><td><code>conv2.0.0</code></td><td>24875</td><td>32</td><td>32.0206</td><td>1399</td><td>1399</td><td>79</td><td>4.3115</td><td>1399</td><td>1766</td><td>79</td><td>3.5181</td><td>1122</td><td>1441</td><td>60</td></tr>
<tr><td><code>conv2.1.0</code></td><td>24875</td><td>32</td><td>32.0206</td><td>1399</td><td>1399</td><td>79</td><td>4.3115</td><td>1399</td><td>1766</td><td>79</td><td>3.5181</td><td>1122</td><td>1441</td><td>60</td></tr>
<tr><td><code>conv2.2.0</code></td><td>24875</td><td>32</td><td>29.2969</td><td>1280</td><td>1280</td><td>66</td><td>3.9331</td><td>1280</td><td>1611</td><td>66</td><td>4.1479</td><td>1352</td><td>1699</td><td>57</td></tr>
<tr><td><code>conv3.0.0</code></td><td>15757</td><td>64</td><td>16.5207</td><td>401</td><td>401</td><td>70</td><td>2.9048</td><td>401</td><td>661</td><td>70</td><td>5.0933</td><td>951</td><td>1159</td><td>31</td></tr>
<tr><td><code>conv3.1.0</code></td><td>15757</td><td>64</td><td>16.5207</td><td>401</td><td>401</td><td>70</td><td>2.9048</td><td>401</td><td>661</td><td>70</td><td>5.0933</td><td>951</td><td>1159</td><td>31</td></tr>
<tr><td><code>conv3.2.0</code></td><td>15757</td><td>64</td><td>17.1387</td><td>416</td><td>416</td><td>68</td><td>3.0059</td><td>416</td><td>684</td><td>68</td><td>5.0713</td><td>948</td><td>1154</td><td>33</td></tr>
<tr><td><code>conv4.0.0</code></td><td>6380</td><td>64</td><td>3.8727</td><td>94</td><td>94</td><td>30</td><td>0.8701</td><td>94</td><td>198</td><td>30</td><td>1.6216</td><td>285</td><td>369</td><td>6</td></tr>
<tr><td><code>conv4.1.0</code></td><td>6380</td><td>64</td><td>3.8727</td><td>94</td><td>94</td><td>30</td><td>0.8701</td><td>94</td><td>198</td><td>30</td><td>1.6216</td><td>285</td><td>369</td><td>6</td></tr>
<tr><td><code>conv4.2.0</code></td><td>6380</td><td>64</td><td>3.5019</td><td>85</td><td>85</td><td>18</td><td>0.6504</td><td>85</td><td>148</td><td>18</td><td>1.1426</td><td>250</td><td>260</td><td>1</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>43.3685</strong></td><td><strong>3158</strong></td><td><strong>3158</strong></td><td><strong>79</strong></td><td><strong>4.6816</strong></td><td><strong>3158</strong></td><td><strong>3196</strong></td><td><strong>79</strong></td><td><strong>5.0933</strong></td><td><strong>1352</strong></td><td><strong>1699</strong></td><td><strong>60</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **87.30%**，将 hash entry 峰值降低 **46.22%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-17.66%**，将 hash entry 峰值降低 **46.87%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 80.1865 | 8.6572 | 10.1865 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 5839 | 5910 | 3140 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000209`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000209.bin`
- 输入体素: `15000`，坐标 SHA-256 `4c7841032dc9f04261791f82de990a4569d68b4aebacae47d136392507e0a178`
- 触达 15000 voxel 上限: `True`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>10.8078</td><td>1574</td><td>1574</td><td>1</td><td>1.6133</td><td>1574</td><td>1652</td><td>1</td><td>0.8857</td><td>761</td><td>907</td><td>34</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>22.5220</td><td>1640</td><td>1640</td><td>0</td><td>2.5151</td><td>1640</td><td>1717</td><td>0</td><td>1.2876</td><td>740</td><td>879</td><td>30</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>21.6156</td><td>1574</td><td>1574</td><td>1</td><td>2.4199</td><td>1574</td><td>1652</td><td>1</td><td>1.3286</td><td>761</td><td>907</td><td>34</td></tr>
<tr><td><code>conv2.0.0</code></td><td>19115</td><td>32</td><td>15.4724</td><td>676</td><td>676</td><td>77</td><td>2.5146</td><td>676</td><td>1030</td><td>77</td><td>2.6855</td><td>761</td><td>1100</td><td>67</td></tr>
<tr><td><code>conv2.1.0</code></td><td>19115</td><td>32</td><td>15.4724</td><td>676</td><td>676</td><td>77</td><td>2.5146</td><td>676</td><td>1030</td><td>77</td><td>2.6855</td><td>761</td><td>1100</td><td>67</td></tr>
<tr><td><code>conv2.2.0</code></td><td>19115</td><td>32</td><td>13.5727</td><td>593</td><td>593</td><td>78</td><td>2.2754</td><td>593</td><td>932</td><td>78</td><td>2.8711</td><td>825</td><td>1176</td><td>68</td></tr>
<tr><td><code>conv3.0.0</code></td><td>9830</td><td>64</td><td>8.0750</td><td>196</td><td>196</td><td>56</td><td>1.7227</td><td>196</td><td>392</td><td>56</td><td>3.3442</td><td>598</td><td>761</td><td>28</td></tr>
<tr><td><code>conv3.1.0</code></td><td>9830</td><td>64</td><td>8.0750</td><td>196</td><td>196</td><td>56</td><td>1.7227</td><td>196</td><td>392</td><td>56</td><td>3.3442</td><td>598</td><td>761</td><td>28</td></tr>
<tr><td><code>conv3.2.0</code></td><td>9830</td><td>64</td><td>8.4045</td><td>204</td><td>204</td><td>56</td><td>1.7578</td><td>204</td><td>400</td><td>56</td><td>3.3706</td><td>587</td><td>767</td><td>38</td></tr>
<tr><td><code>conv4.0.0</code></td><td>3286</td><td>64</td><td>2.3071</td><td>56</td><td>56</td><td>13</td><td>0.4790</td><td>56</td><td>109</td><td>13</td><td>1.0854</td><td>185</td><td>247</td><td>2</td></tr>
<tr><td><code>conv4.1.0</code></td><td>3286</td><td>64</td><td>2.3071</td><td>56</td><td>56</td><td>13</td><td>0.4790</td><td>56</td><td>109</td><td>13</td><td>1.0854</td><td>185</td><td>247</td><td>2</td></tr>
<tr><td><code>conv4.2.0</code></td><td>3286</td><td>64</td><td>1.9363</td><td>47</td><td>47</td><td>12</td><td>0.3647</td><td>47</td><td>83</td><td>12</td><td>0.6240</td><td>142</td><td>142</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>22.5220</strong></td><td><strong>1640</strong></td><td><strong>1640</strong></td><td><strong>78</strong></td><td><strong>2.5151</strong></td><td><strong>1640</strong></td><td><strong>1717</strong></td><td><strong>78</strong></td><td><strong>3.3706</strong></td><td><strong>825</strong></td><td><strong>1176</strong></td><td><strong>68</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **84.79%**，将 hash entry 峰值降低 **29.18%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-33.51%**，将 hash entry 峰值降低 **32.44%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 44.1376 | 5.0293 | 6.7148 |
| DRAM 峰值层 | `conv1.0.0` | `conv2.1.0` | `conv3.2.0` |
| Hash entries | 3214 | 3369 | 2276 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000210`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000210.bin`
- 输入体素: `15000`，坐标 SHA-256 `2f6c0c7b659eb629eda24f58abaa1da6a25e66063a1627b7a6a593411ee81239`
- 触达 15000 voxel 上限: `True`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>27.3216</td><td>3979</td><td>3979</td><td>0</td><td>3.8887</td><td>3979</td><td>3982</td><td>0</td><td>1.1270</td><td>1120</td><td>1154</td><td>3</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>58.1177</td><td>4232</td><td>4232</td><td>0</td><td>6.2051</td><td>4232</td><td>4236</td><td>0</td><td>1.6069</td><td>1072</td><td>1097</td><td>2</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>54.6432</td><td>3979</td><td>3979</td><td>0</td><td>5.8330</td><td>3979</td><td>3982</td><td>0</td><td>1.6904</td><td>1120</td><td>1154</td><td>3</td></tr>
<tr><td><code>conv2.0.0</code></td><td>28462</td><td>32</td><td>43.9453</td><td>1920</td><td>1920</td><td>64</td><td>5.5518</td><td>1920</td><td>2274</td><td>64</td><td>4.7388</td><td>1685</td><td>1941</td><td>34</td></tr>
<tr><td><code>conv2.1.0</code></td><td>28462</td><td>32</td><td>43.9453</td><td>1920</td><td>1920</td><td>64</td><td>5.5518</td><td>1920</td><td>2274</td><td>64</td><td>4.7388</td><td>1685</td><td>1941</td><td>34</td></tr>
<tr><td><code>conv2.2.0</code></td><td>28462</td><td>32</td><td>38.5437</td><td>1684</td><td>1684</td><td>43</td><td>4.7827</td><td>1684</td><td>1959</td><td>43</td><td>5.1172</td><td>1812</td><td>2096</td><td>47</td></tr>
<tr><td><code>conv3.0.0</code></td><td>20661</td><td>64</td><td>24.0189</td><td>583</td><td>583</td><td>73</td><td>3.9155</td><td>583</td><td>891</td><td>73</td><td>5.6821</td><td>1143</td><td>1293</td><td>7</td></tr>
<tr><td><code>conv3.1.0</code></td><td>20661</td><td>64</td><td>24.0189</td><td>583</td><td>583</td><td>73</td><td>3.9155</td><td>583</td><td>891</td><td>73</td><td>5.6821</td><td>1143</td><td>1293</td><td>7</td></tr>
<tr><td><code>conv3.2.0</code></td><td>20661</td><td>64</td><td>24.8016</td><td>602</td><td>602</td><td>71</td><td>3.9067</td><td>602</td><td>889</td><td>71</td><td>5.6821</td><td>1144</td><td>1293</td><td>9</td></tr>
<tr><td><code>conv4.0.0</code></td><td>9963</td><td>64</td><td>5.7266</td><td>139</td><td>139</td><td>42</td><td>1.3140</td><td>139</td><td>299</td><td>42</td><td>2.2939</td><td>388</td><td>522</td><td>0</td></tr>
<tr><td><code>conv4.1.0</code></td><td>9963</td><td>64</td><td>5.7266</td><td>139</td><td>139</td><td>42</td><td>1.3140</td><td>139</td><td>299</td><td>42</td><td>2.2939</td><td>388</td><td>522</td><td>0</td></tr>
<tr><td><code>conv4.2.0</code></td><td>9963</td><td>64</td><td>5.4794</td><td>133</td><td>133</td><td>23</td><td>0.9976</td><td>133</td><td>227</td><td>23</td><td>1.6567</td><td>366</td><td>377</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>58.1177</strong></td><td><strong>4232</strong></td><td><strong>4232</strong></td><td><strong>73</strong></td><td><strong>6.2051</strong></td><td><strong>4232</strong></td><td><strong>4236</strong></td><td><strong>73</strong></td><td><strong>5.6821</strong></td><td><strong>1812</strong></td><td><strong>2096</strong></td><td><strong>47</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.92%**，将 hash entry 峰值降低 **50.83%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **5.60%**，将 hash entry 峰值降低 **50.88%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 112.7609 | 12.0381 | 11.3643 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 8211 | 8218 | 4037 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `val/000211`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000211.bin`
- 输入体素: `13390`，坐标 SHA-256 `6b028d7b32a0cf534ab9783b681c1aa9a68e3d2a7b33ea9096c4f0c7111d00f3`
- 触达 15000 voxel 上限: `False`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>13390</td><td>4</td><td>16.9533</td><td>2469</td><td>2469</td><td>1</td><td>2.4463</td><td>2469</td><td>2505</td><td>1</td><td>0.8965</td><td>854</td><td>918</td><td>8</td></tr>
<tr><td><code>conv_input.0</code></td><td>13390</td><td>16</td><td>38.2736</td><td>2787</td><td>2787</td><td>1</td><td>4.1323</td><td>2787</td><td>2821</td><td>1</td><td>1.3271</td><td>839</td><td>906</td><td>9</td></tr>
<tr><td><code>conv1.0.0</code></td><td>13390</td><td>16</td><td>33.9066</td><td>2469</td><td>2469</td><td>1</td><td>3.6694</td><td>2469</td><td>2505</td><td>1</td><td>1.3447</td><td>854</td><td>918</td><td>8</td></tr>
<tr><td><code>conv2.0.0</code></td><td>21586</td><td>32</td><td>29.4800</td><td>1288</td><td>1288</td><td>62</td><td>3.9307</td><td>1288</td><td>1610</td><td>62</td><td>3.1812</td><td>1057</td><td>1303</td><td>43</td></tr>
<tr><td><code>conv2.1.0</code></td><td>21586</td><td>32</td><td>29.4800</td><td>1288</td><td>1288</td><td>62</td><td>3.9307</td><td>1288</td><td>1610</td><td>62</td><td>3.1812</td><td>1057</td><td>1303</td><td>43</td></tr>
<tr><td><code>conv2.2.0</code></td><td>21586</td><td>32</td><td>25.1770</td><td>1100</td><td>1100</td><td>52</td><td>3.3667</td><td>1100</td><td>1379</td><td>52</td><td>3.7744</td><td>1299</td><td>1546</td><td>41</td></tr>
<tr><td><code>conv3.0.0</code></td><td>13347</td><td>64</td><td>16.0675</td><td>390</td><td>390</td><td>57</td><td>2.6719</td><td>390</td><td>608</td><td>57</td><td>4.5264</td><td>907</td><td>1030</td><td>9</td></tr>
<tr><td><code>conv3.1.0</code></td><td>13347</td><td>64</td><td>16.0675</td><td>390</td><td>390</td><td>57</td><td>2.6719</td><td>390</td><td>608</td><td>57</td><td>4.5264</td><td>907</td><td>1030</td><td>9</td></tr>
<tr><td><code>conv3.2.0</code></td><td>13347</td><td>64</td><td>15.8203</td><td>384</td><td>384</td><td>56</td><td>2.6279</td><td>384</td><td>598</td><td>56</td><td>4.4517</td><td>884</td><td>1013</td><td>14</td></tr>
<tr><td><code>conv4.0.0</code></td><td>5984</td><td>64</td><td>3.6255</td><td>88</td><td>88</td><td>27</td><td>0.8174</td><td>88</td><td>186</td><td>27</td><td>1.4854</td><td>257</td><td>338</td><td>2</td></tr>
<tr><td><code>conv4.1.0</code></td><td>5984</td><td>64</td><td>3.6255</td><td>88</td><td>88</td><td>27</td><td>0.8174</td><td>88</td><td>186</td><td>27</td><td>1.4854</td><td>257</td><td>338</td><td>2</td></tr>
<tr><td><code>conv4.2.0</code></td><td>5984</td><td>64</td><td>3.3783</td><td>82</td><td>82</td><td>16</td><td>0.6021</td><td>82</td><td>137</td><td>16</td><td>1.0371</td><td>233</td><td>236</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>38.2736</strong></td><td><strong>2787</strong></td><td><strong>2787</strong></td><td><strong>62</strong></td><td><strong>4.1323</strong></td><td><strong>2787</strong></td><td><strong>2821</strong></td><td><strong>62</strong></td><td><strong>4.5264</strong></td><td><strong>1299</strong></td><td><strong>1546</strong></td><td><strong>43</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **87.46%**，将 hash entry 峰值降低 **45.80%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-15.16%**，将 hash entry 峰值降低 **46.51%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 72.1802 | 7.8613 | 9.0527 |
| DRAM 峰值层 | `conv1.0.0` | `conv2.1.0` | `conv3.1.0` |
| Hash entries | 5256 | 5326 | 2849 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `val/000212`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000212.bin`
- 输入体素: `15000`，坐标 SHA-256 `dee387e94b86bc221b7c082c4d1aab1ff45a9dd2bfe4de025da6abfdee0a7bbb`
- 触达 15000 voxel 上限: `True`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>18.0244</td><td>2625</td><td>2625</td><td>9</td><td>2.6270</td><td>2625</td><td>2690</td><td>9</td><td>0.9658</td><td>888</td><td>989</td><td>23</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>39.3448</td><td>2865</td><td>2865</td><td>8</td><td>4.2949</td><td>2865</td><td>2932</td><td>8</td><td>1.4648</td><td>898</td><td>1000</td><td>20</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>36.0489</td><td>2625</td><td>2625</td><td>9</td><td>3.9404</td><td>2625</td><td>2690</td><td>9</td><td>1.4487</td><td>888</td><td>989</td><td>23</td></tr>
<tr><td><code>conv2.0.0</code></td><td>24256</td><td>32</td><td>28.9993</td><td>1267</td><td>1267</td><td>77</td><td>4.1211</td><td>1267</td><td>1688</td><td>77</td><td>3.4814</td><td>1096</td><td>1426</td><td>68</td></tr>
<tr><td><code>conv2.1.0</code></td><td>24256</td><td>32</td><td>28.9993</td><td>1267</td><td>1267</td><td>77</td><td>4.1211</td><td>1267</td><td>1688</td><td>77</td><td>3.4814</td><td>1096</td><td>1426</td><td>68</td></tr>
<tr><td><code>conv2.2.0</code></td><td>24256</td><td>32</td><td>26.4130</td><td>1154</td><td>1154</td><td>74</td><td>3.6938</td><td>1154</td><td>1513</td><td>74</td><td>3.9526</td><td>1268</td><td>1619</td><td>72</td></tr>
<tr><td><code>conv3.0.0</code></td><td>14688</td><td>64</td><td>18.2510</td><td>443</td><td>443</td><td>63</td><td>3.0190</td><td>443</td><td>687</td><td>63</td><td>5.0098</td><td>971</td><td>1140</td><td>28</td></tr>
<tr><td><code>conv3.1.0</code></td><td>14688</td><td>64</td><td>18.2510</td><td>443</td><td>443</td><td>63</td><td>3.0190</td><td>443</td><td>687</td><td>63</td><td>5.0098</td><td>971</td><td>1140</td><td>28</td></tr>
<tr><td><code>conv3.2.0</code></td><td>14688</td><td>64</td><td>18.4570</td><td>448</td><td>448</td><td>65</td><td>3.0146</td><td>448</td><td>686</td><td>65</td><td>5.1152</td><td>977</td><td>1164</td><td>29</td></tr>
<tr><td><code>conv4.0.0</code></td><td>6169</td><td>64</td><td>4.9026</td><td>119</td><td>119</td><td>22</td><td>0.9272</td><td>119</td><td>211</td><td>22</td><td>1.7402</td><td>314</td><td>396</td><td>2</td></tr>
<tr><td><code>conv4.1.0</code></td><td>6169</td><td>64</td><td>4.9026</td><td>119</td><td>119</td><td>22</td><td>0.9272</td><td>119</td><td>211</td><td>22</td><td>1.7402</td><td>314</td><td>396</td><td>2</td></tr>
<tr><td><code>conv4.2.0</code></td><td>6169</td><td>64</td><td>4.3671</td><td>106</td><td>106</td><td>17</td><td>0.7119</td><td>106</td><td>162</td><td>17</td><td>1.1997</td><td>268</td><td>273</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>39.3448</strong></td><td><strong>2865</strong></td><td><strong>2865</strong></td><td><strong>77</strong></td><td><strong>4.2949</strong></td><td><strong>2865</strong></td><td><strong>2932</strong></td><td><strong>77</strong></td><td><strong>5.1152</strong></td><td><strong>1268</strong></td><td><strong>1619</strong></td><td><strong>72</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **86.57%**，将 hash entry 峰值降低 **44.54%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-22.84%**，将 hash entry 峰值降低 **45.84%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 75.3937 | 8.2422 | 10.1250 |
| DRAM 峰值层 | `conv1.0.0` | `conv2.1.0` | `conv3.2.0` |
| Hash entries | 5490 | 5622 | 3045 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `val/000213`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000213.bin`
- 输入体素: `14462`，坐标 SHA-256 `1c3a1801c486f43c2cfd82f7e4dfa448f60409799be6a9dda55f6bf752c19496`
- 触达 15000 voxel 上限: `False`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>14462</td><td>4</td><td>19.6724</td><td>2865</td><td>2865</td><td>2</td><td>2.8330</td><td>2865</td><td>2901</td><td>2</td><td>0.9668</td><td>922</td><td>990</td><td>8</td></tr>
<tr><td><code>conv_input.0</code></td><td>14462</td><td>16</td><td>42.8329</td><td>3119</td><td>3119</td><td>2</td><td>4.6245</td><td>3119</td><td>3157</td><td>2</td><td>1.4092</td><td>900</td><td>962</td><td>9</td></tr>
<tr><td><code>conv1.0.0</code></td><td>14462</td><td>16</td><td>39.3448</td><td>2865</td><td>2865</td><td>2</td><td>4.2495</td><td>2865</td><td>2901</td><td>2</td><td>1.4502</td><td>922</td><td>990</td><td>8</td></tr>
<tr><td><code>conv2.0.0</code></td><td>24190</td><td>32</td><td>34.5840</td><td>1511</td><td>1511</td><td>77</td><td>4.6265</td><td>1511</td><td>1895</td><td>77</td><td>3.5083</td><td>1159</td><td>1437</td><td>39</td></tr>
<tr><td><code>conv2.1.0</code></td><td>24190</td><td>32</td><td>34.5840</td><td>1511</td><td>1511</td><td>77</td><td>4.6265</td><td>1511</td><td>1895</td><td>77</td><td>3.5083</td><td>1159</td><td>1437</td><td>39</td></tr>
<tr><td><code>conv2.2.0</code></td><td>24190</td><td>32</td><td>31.4713</td><td>1375</td><td>1375</td><td>61</td><td>4.1284</td><td>1375</td><td>1691</td><td>61</td><td>4.0503</td><td>1356</td><td>1659</td><td>53</td></tr>
<tr><td><code>conv3.0.0</code></td><td>15463</td><td>64</td><td>22.3709</td><td>543</td><td>543</td><td>57</td><td>3.4365</td><td>543</td><td>782</td><td>57</td><td>5.4844</td><td>1087</td><td>1248</td><td>8</td></tr>
<tr><td><code>conv3.1.0</code></td><td>15463</td><td>64</td><td>22.3709</td><td>543</td><td>543</td><td>57</td><td>3.4365</td><td>543</td><td>782</td><td>57</td><td>5.4844</td><td>1087</td><td>1248</td><td>8</td></tr>
<tr><td><code>conv3.2.0</code></td><td>15463</td><td>64</td><td>22.7005</td><td>551</td><td>551</td><td>54</td><td>3.4673</td><td>551</td><td>789</td><td>54</td><td>5.3657</td><td>1065</td><td>1221</td><td>6</td></tr>
<tr><td><code>conv4.0.0</code></td><td>6870</td><td>64</td><td>6.0150</td><td>146</td><td>146</td><td>26</td><td>1.0942</td><td>146</td><td>249</td><td>26</td><td>1.9248</td><td>355</td><td>438</td><td>0</td></tr>
<tr><td><code>conv4.1.0</code></td><td>6870</td><td>64</td><td>6.0150</td><td>146</td><td>146</td><td>26</td><td>1.0942</td><td>146</td><td>249</td><td>26</td><td>1.9248</td><td>355</td><td>438</td><td>0</td></tr>
<tr><td><code>conv4.2.0</code></td><td>6870</td><td>64</td><td>5.2734</td><td>128</td><td>128</td><td>14</td><td>0.8306</td><td>128</td><td>189</td><td>14</td><td>1.3491</td><td>300</td><td>307</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>42.8329</strong></td><td><strong>3119</strong></td><td><strong>3119</strong></td><td><strong>77</strong></td><td><strong>4.6265</strong></td><td><strong>3119</strong></td><td><strong>3157</strong></td><td><strong>77</strong></td><td><strong>5.4844</strong></td><td><strong>1356</strong></td><td><strong>1659</strong></td><td><strong>53</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **86.65%**，将 hash entry 峰值降低 **48.26%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-18.54%**，将 hash entry 峰值降低 **48.89%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 82.1777 | 9.2529 | 10.9688 |
| DRAM 峰值层 | `conv1.0.0` | `conv2.1.0` | `conv3.1.0` |
| Hash entries | 5984 | 6058 | 3096 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000214`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000214.bin`
- 输入体素: `15000`，坐标 SHA-256 `4acf010ff1b5f8e9346f9cf96d5ac498bf7b3a07824571071e42870fd10021c1`
- 触达 15000 voxel 上限: `True`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>26.0719</td><td>3797</td><td>3797</td><td>0</td><td>3.7139</td><td>3797</td><td>3803</td><td>0</td><td>1.2588</td><td>1270</td><td>1289</td><td>3</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>56.9916</td><td>4150</td><td>4150</td><td>0</td><td>6.0864</td><td>4150</td><td>4155</td><td>0</td><td>1.8838</td><td>1264</td><td>1286</td><td>5</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>52.1439</td><td>3797</td><td>3797</td><td>0</td><td>5.5708</td><td>3797</td><td>3803</td><td>0</td><td>1.8882</td><td>1270</td><td>1289</td><td>3</td></tr>
<tr><td><code>conv2.0.0</code></td><td>29823</td><td>32</td><td>42.8009</td><td>1870</td><td>1870</td><td>63</td><td>5.4785</td><td>1870</td><td>2244</td><td>63</td><td>5.0098</td><td>1725</td><td>2052</td><td>38</td></tr>
<tr><td><code>conv2.1.0</code></td><td>29823</td><td>32</td><td>42.8009</td><td>1870</td><td>1870</td><td>63</td><td>5.4785</td><td>1870</td><td>2244</td><td>63</td><td>5.0098</td><td>1725</td><td>2052</td><td>38</td></tr>
<tr><td><code>conv2.2.0</code></td><td>29823</td><td>32</td><td>39.0015</td><td>1704</td><td>1704</td><td>48</td><td>4.8682</td><td>1704</td><td>1994</td><td>48</td><td>5.4980</td><td>1903</td><td>2252</td><td>61</td></tr>
<tr><td><code>conv3.0.0</code></td><td>20906</td><td>64</td><td>28.3447</td><td>688</td><td>688</td><td>70</td><td>4.2363</td><td>688</td><td>964</td><td>70</td><td>6.1436</td><td>1254</td><td>1398</td><td>8</td></tr>
<tr><td><code>conv3.1.0</code></td><td>20906</td><td>64</td><td>28.3447</td><td>688</td><td>688</td><td>70</td><td>4.2363</td><td>688</td><td>964</td><td>70</td><td>6.1436</td><td>1254</td><td>1398</td><td>8</td></tr>
<tr><td><code>conv3.2.0</code></td><td>20906</td><td>64</td><td>28.4271</td><td>690</td><td>690</td><td>73</td><td>4.2583</td><td>690</td><td>969</td><td>73</td><td>6.3105</td><td>1286</td><td>1436</td><td>8</td></tr>
<tr><td><code>conv4.0.0</code></td><td>9880</td><td>64</td><td>7.1274</td><td>173</td><td>173</td><td>37</td><td>1.3799</td><td>173</td><td>314</td><td>37</td><td>2.5137</td><td>466</td><td>572</td><td>1</td></tr>
<tr><td><code>conv4.1.0</code></td><td>9880</td><td>64</td><td>7.1274</td><td>173</td><td>173</td><td>37</td><td>1.3799</td><td>173</td><td>314</td><td>37</td><td>2.5137</td><td>466</td><td>572</td><td>1</td></tr>
<tr><td><code>conv4.2.0</code></td><td>9880</td><td>64</td><td>6.5506</td><td>159</td><td>159</td><td>23</td><td>1.0767</td><td>159</td><td>245</td><td>23</td><td>1.8589</td><td>418</td><td>423</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>56.9916</strong></td><td><strong>4150</strong></td><td><strong>4150</strong></td><td><strong>73</strong></td><td><strong>6.0864</strong></td><td><strong>4150</strong></td><td><strong>4155</strong></td><td><strong>73</strong></td><td><strong>6.3105</strong></td><td><strong>1903</strong></td><td><strong>2252</strong></td><td><strong>61</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **88.59%**，将 hash entry 峰值降低 **45.84%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-6.84%**，将 hash entry 峰值降低 **45.92%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 109.1354 | 11.6572 | 12.4541 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.2.0` |
| Hash entries | 7947 | 7958 | 4304 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000215`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000215.bin`
- 输入体素: `14457`，坐标 SHA-256 `0455c094b39f2e6ac1b60f1919063d93eea6d3e5bda048aa2315d7add7b5977f`
- 触达 15000 voxel 上限: `False`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>14457</td><td>4</td><td>16.4932</td><td>2402</td><td>2402</td><td>7</td><td>2.3965</td><td>2402</td><td>2454</td><td>7</td><td>0.8477</td><td>742</td><td>868</td><td>31</td></tr>
<tr><td><code>conv_input.0</code></td><td>14457</td><td>16</td><td>33.8379</td><td>2464</td><td>2464</td><td>8</td><td>3.7061</td><td>2464</td><td>2530</td><td>8</td><td>1.2378</td><td>725</td><td>845</td><td>27</td></tr>
<tr><td><code>conv1.0.0</code></td><td>14457</td><td>16</td><td>32.9865</td><td>2402</td><td>2402</td><td>7</td><td>3.5947</td><td>2402</td><td>2454</td><td>7</td><td>1.2715</td><td>742</td><td>868</td><td>31</td></tr>
<tr><td><code>conv2.0.0</code></td><td>19653</td><td>32</td><td>23.6435</td><td>1033</td><td>1033</td><td>85</td><td>3.3130</td><td>1033</td><td>1357</td><td>85</td><td>2.4292</td><td>730</td><td>995</td><td>50</td></tr>
<tr><td><code>conv2.1.0</code></td><td>19653</td><td>32</td><td>23.6435</td><td>1033</td><td>1033</td><td>85</td><td>3.3130</td><td>1033</td><td>1357</td><td>85</td><td>2.4292</td><td>730</td><td>995</td><td>50</td></tr>
<tr><td><code>conv2.2.0</code></td><td>19653</td><td>32</td><td>23.1628</td><td>1012</td><td>1012</td><td>72</td><td>3.2080</td><td>1012</td><td>1314</td><td>72</td><td>2.6392</td><td>807</td><td>1081</td><td>60</td></tr>
<tr><td><code>conv3.0.0</code></td><td>10735</td><td>64</td><td>11.6180</td><td>282</td><td>282</td><td>48</td><td>2.0127</td><td>282</td><td>458</td><td>48</td><td>3.4058</td><td>642</td><td>775</td><td>23</td></tr>
<tr><td><code>conv3.1.0</code></td><td>10735</td><td>64</td><td>11.6180</td><td>282</td><td>282</td><td>48</td><td>2.0127</td><td>282</td><td>458</td><td>48</td><td>3.4058</td><td>642</td><td>775</td><td>23</td></tr>
<tr><td><code>conv3.2.0</code></td><td>10735</td><td>64</td><td>11.9064</td><td>289</td><td>289</td><td>50</td><td>2.0479</td><td>289</td><td>466</td><td>50</td><td>3.4058</td><td>643</td><td>775</td><td>23</td></tr>
<tr><td><code>conv4.0.0</code></td><td>4219</td><td>64</td><td>3.1311</td><td>76</td><td>76</td><td>17</td><td>0.5933</td><td>76</td><td>135</td><td>17</td><td>1.1733</td><td>212</td><td>267</td><td>0</td></tr>
<tr><td><code>conv4.1.0</code></td><td>4219</td><td>64</td><td>3.1311</td><td>76</td><td>76</td><td>17</td><td>0.5933</td><td>76</td><td>135</td><td>17</td><td>1.1733</td><td>212</td><td>267</td><td>0</td></tr>
<tr><td><code>conv4.2.0</code></td><td>4219</td><td>64</td><td>2.8015</td><td>68</td><td>68</td><td>10</td><td>0.4658</td><td>68</td><td>106</td><td>10</td><td>0.7778</td><td>170</td><td>177</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>33.8379</strong></td><td><strong>2464</strong></td><td><strong>2464</strong></td><td><strong>85</strong></td><td><strong>3.7061</strong></td><td><strong>2464</strong></td><td><strong>2530</strong></td><td><strong>85</strong></td><td><strong>3.4058</strong></td><td><strong>807</strong></td><td><strong>1081</strong></td><td><strong>60</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.81%**，将 hash entry 峰值降低 **57.34%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **6.70%**，将 hash entry 峰值降低 **58.35%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 66.8243 | 7.3008 | 6.8115 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 4866 | 4984 | 2076 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `val/000216`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000216.bin`
- 输入体素: `15000`，坐标 SHA-256 `b8c429c5bbf27db0e6416a7a26eff2971b631a2783a74da99c761c822ad02443`
- 触达 15000 voxel 上限: `True`

### 按 Feature Map 去重后的对比

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

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **90.12%**，将 hash entry 峰值降低 **61.81%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **7.63%**，将 hash entry 峰值降低 **61.92%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 120.0119 | 12.8364 | 11.8564 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 8739 | 8763 | 3337 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000217`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000217.bin`
- 输入体素: `15000`，坐标 SHA-256 `d13a56f5438ec21f6291fa0b2138df138041225d656b59a2cca5d51b6e5ef4d0`
- 触达 15000 voxel 上限: `True`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>21.5126</td><td>3133</td><td>3133</td><td>6</td><td>3.1035</td><td>3133</td><td>3178</td><td>6</td><td>1.0723</td><td>1016</td><td>1098</td><td>10</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>46.2662</td><td>3369</td><td>3369</td><td>5</td><td>5.0127</td><td>3369</td><td>3422</td><td>5</td><td>1.5996</td><td>1014</td><td>1092</td><td>9</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>43.0252</td><td>3133</td><td>3133</td><td>6</td><td>4.6553</td><td>3133</td><td>3178</td><td>6</td><td>1.6084</td><td>1016</td><td>1098</td><td>10</td></tr>
<tr><td><code>conv2.0.0</code></td><td>26213</td><td>32</td><td>33.4167</td><td>1460</td><td>1460</td><td>94</td><td>4.5532</td><td>1460</td><td>1865</td><td>94</td><td>3.9087</td><td>1239</td><td>1601</td><td>74</td></tr>
<tr><td><code>conv2.1.0</code></td><td>26213</td><td>32</td><td>33.4167</td><td>1460</td><td>1460</td><td>94</td><td>4.5532</td><td>1460</td><td>1865</td><td>94</td><td>3.9087</td><td>1239</td><td>1601</td><td>74</td></tr>
<tr><td><code>conv2.2.0</code></td><td>26213</td><td>32</td><td>31.1050</td><td>1359</td><td>1359</td><td>81</td><td>4.2407</td><td>1359</td><td>1737</td><td>81</td><td>4.1284</td><td>1312</td><td>1691</td><td>83</td></tr>
<tr><td><code>conv3.0.0</code></td><td>16946</td><td>64</td><td>19.3634</td><td>470</td><td>470</td><td>64</td><td>3.3750</td><td>470</td><td>768</td><td>64</td><td>5.6733</td><td>1044</td><td>1291</td><td>33</td></tr>
<tr><td><code>conv3.1.0</code></td><td>16946</td><td>64</td><td>19.3634</td><td>470</td><td>470</td><td>64</td><td>3.3750</td><td>470</td><td>768</td><td>64</td><td>5.6733</td><td>1044</td><td>1291</td><td>33</td></tr>
<tr><td><code>conv3.2.0</code></td><td>16946</td><td>64</td><td>19.0338</td><td>462</td><td>462</td><td>69</td><td>3.3311</td><td>462</td><td>758</td><td>69</td><td>5.6865</td><td>1057</td><td>1294</td><td>28</td></tr>
<tr><td><code>conv4.0.0</code></td><td>7220</td><td>64</td><td>3.7903</td><td>92</td><td>92</td><td>39</td><td>0.9404</td><td>92</td><td>214</td><td>39</td><td>1.7842</td><td>298</td><td>406</td><td>2</td></tr>
<tr><td><code>conv4.1.0</code></td><td>7220</td><td>64</td><td>3.7903</td><td>92</td><td>92</td><td>39</td><td>0.9404</td><td>92</td><td>214</td><td>39</td><td>1.7842</td><td>298</td><td>406</td><td>2</td></tr>
<tr><td><code>conv4.2.0</code></td><td>7220</td><td>64</td><td>3.4607</td><td>84</td><td>84</td><td>19</td><td>0.7163</td><td>84</td><td>163</td><td>19</td><td>1.2217</td><td>267</td><td>278</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>46.2662</strong></td><td><strong>3369</strong></td><td><strong>3369</strong></td><td><strong>94</strong></td><td><strong>5.0127</strong></td><td><strong>3369</strong></td><td><strong>3422</strong></td><td><strong>94</strong></td><td><strong>5.6865</strong></td><td><strong>1312</strong></td><td><strong>1691</strong></td><td><strong>83</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **87.28%**，将 hash entry 峰值降低 **49.37%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-17.50%**，将 hash entry 峰值降低 **50.12%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 89.2914 | 9.6680 | 11.3599 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.2.0` |
| Hash entries | 6502 | 6600 | 3292 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `val/000218`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000218.bin`
- 输入体素: `14636`，坐标 SHA-256 `7fb5295ad4a5dfc7066fb219a96d9354483a840e7450c9471567139147f427d9`
- 触达 15000 voxel 上限: `False`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>14636</td><td>4</td><td>18.8553</td><td>2746</td><td>2746</td><td>2</td><td>2.7295</td><td>2746</td><td>2795</td><td>2</td><td>0.9287</td><td>877</td><td>951</td><td>15</td></tr>
<tr><td><code>conv_input.0</code></td><td>14636</td><td>16</td><td>40.1001</td><td>2920</td><td>2920</td><td>1</td><td>4.3652</td><td>2920</td><td>2980</td><td>1</td><td>1.4121</td><td>890</td><td>964</td><td>18</td></tr>
<tr><td><code>conv1.0.0</code></td><td>14636</td><td>16</td><td>37.7106</td><td>2746</td><td>2746</td><td>2</td><td>4.0942</td><td>2746</td><td>2795</td><td>2</td><td>1.3931</td><td>877</td><td>951</td><td>15</td></tr>
<tr><td><code>conv2.0.0</code></td><td>21655</td><td>32</td><td>32.5470</td><td>1422</td><td>1422</td><td>72</td><td>4.1772</td><td>1422</td><td>1711</td><td>72</td><td>3.0713</td><td>1053</td><td>1258</td><td>35</td></tr>
<tr><td><code>conv2.1.0</code></td><td>21655</td><td>32</td><td>32.5470</td><td>1422</td><td>1422</td><td>72</td><td>4.1772</td><td>1422</td><td>1711</td><td>72</td><td>3.0713</td><td>1053</td><td>1258</td><td>35</td></tr>
<tr><td><code>conv2.2.0</code></td><td>21655</td><td>32</td><td>30.8075</td><td>1346</td><td>1346</td><td>55</td><td>3.9307</td><td>1346</td><td>1610</td><td>55</td><td>3.4180</td><td>1182</td><td>1400</td><td>36</td></tr>
<tr><td><code>conv3.0.0</code></td><td>14397</td><td>64</td><td>20.2286</td><td>491</td><td>491</td><td>44</td><td>2.9531</td><td>491</td><td>672</td><td>44</td><td>4.2979</td><td>867</td><td>978</td><td>5</td></tr>
<tr><td><code>conv3.1.0</code></td><td>14397</td><td>64</td><td>20.2286</td><td>491</td><td>491</td><td>44</td><td>2.9531</td><td>491</td><td>672</td><td>44</td><td>4.2979</td><td>867</td><td>978</td><td>5</td></tr>
<tr><td><code>conv3.2.0</code></td><td>14397</td><td>64</td><td>20.2286</td><td>491</td><td>491</td><td>45</td><td>2.9927</td><td>491</td><td>681</td><td>45</td><td>4.3198</td><td>876</td><td>983</td><td>5</td></tr>
<tr><td><code>conv4.0.0</code></td><td>6793</td><td>64</td><td>4.7791</td><td>116</td><td>116</td><td>24</td><td>0.9404</td><td>116</td><td>214</td><td>24</td><td>1.6831</td><td>310</td><td>383</td><td>0</td></tr>
<tr><td><code>conv4.1.0</code></td><td>6793</td><td>64</td><td>4.7791</td><td>116</td><td>116</td><td>24</td><td>0.9404</td><td>116</td><td>214</td><td>24</td><td>1.6831</td><td>310</td><td>383</td><td>0</td></tr>
<tr><td><code>conv4.2.0</code></td><td>6793</td><td>64</td><td>4.4495</td><td>108</td><td>108</td><td>15</td><td>0.7295</td><td>108</td><td>166</td><td>15</td><td>1.2261</td><td>274</td><td>279</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>40.1001</strong></td><td><strong>2920</strong></td><td><strong>2920</strong></td><td><strong>72</strong></td><td><strong>4.3652</strong></td><td><strong>2920</strong></td><td><strong>2980</strong></td><td><strong>72</strong></td><td><strong>4.3198</strong></td><td><strong>1182</strong></td><td><strong>1400</strong></td><td><strong>36</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **88.92%**，将 hash entry 峰值降低 **53.09%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-1.87%**，将 hash entry 峰值降低 **53.97%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 77.8107 | 8.4595 | 8.6177 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.2.0` |
| Hash entries | 5666 | 5775 | 2658 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000219`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000219.bin`
- 输入体素: `15000`，坐标 SHA-256 `c86036bde94c607343cfb0e338e1989f955a53d111db587f9a25647ed92b80fd`
- 触达 15000 voxel 上限: `True`

### 按 Feature Map 去重后的对比

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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>36.1176</td><td>5260</td><td>5260</td><td>1</td><td>5.1426</td><td>5260</td><td>5266</td><td>1</td><td>1.1982</td><td>1197</td><td>1227</td><td>3</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>72.6746</td><td>5292</td><td>5292</td><td>2</td><td>7.7622</td><td>5292</td><td>5299</td><td>2</td><td>1.7402</td><td>1160</td><td>1188</td><td>2</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>72.2351</td><td>5260</td><td>5260</td><td>1</td><td>7.7139</td><td>5260</td><td>5266</td><td>1</td><td>1.7974</td><td>1197</td><td>1227</td><td>3</td></tr>
<tr><td><code>conv2.0.0</code></td><td>32478</td><td>32</td><td>54.2221</td><td>2369</td><td>2369</td><td>50</td><td>6.6553</td><td>2369</td><td>2726</td><td>50</td><td>4.9512</td><td>1752</td><td>2028</td><td>45</td></tr>
<tr><td><code>conv2.1.0</code></td><td>32478</td><td>32</td><td>54.2221</td><td>2369</td><td>2369</td><td>50</td><td>6.6553</td><td>2369</td><td>2726</td><td>50</td><td>4.9512</td><td>1752</td><td>2028</td><td>45</td></tr>
<tr><td><code>conv2.2.0</code></td><td>32478</td><td>32</td><td>52.6199</td><td>2299</td><td>2299</td><td>46</td><td>6.3647</td><td>2299</td><td>2607</td><td>46</td><td>5.4492</td><td>1923</td><td>2232</td><td>50</td></tr>
<tr><td><code>conv3.0.0</code></td><td>25133</td><td>64</td><td>27.0264</td><td>656</td><td>656</td><td>99</td><td>4.5439</td><td>656</td><td>1034</td><td>99</td><td>6.8379</td><td>1315</td><td>1556</td><td>40</td></tr>
<tr><td><code>conv3.1.0</code></td><td>25133</td><td>64</td><td>27.0264</td><td>656</td><td>656</td><td>99</td><td>4.5439</td><td>656</td><td>1034</td><td>99</td><td>6.8379</td><td>1315</td><td>1556</td><td>40</td></tr>
<tr><td><code>conv3.2.0</code></td><td>25133</td><td>64</td><td>27.3560</td><td>664</td><td>664</td><td>97</td><td>4.5615</td><td>664</td><td>1038</td><td>97</td><td>6.7808</td><td>1295</td><td>1543</td><td>36</td></tr>
<tr><td><code>conv4.0.0</code></td><td>12065</td><td>64</td><td>6.5506</td><td>159</td><td>159</td><td>55</td><td>1.5952</td><td>159</td><td>363</td><td>55</td><td>2.6104</td><td>443</td><td>594</td><td>17</td></tr>
<tr><td><code>conv4.1.0</code></td><td>12065</td><td>64</td><td>6.5506</td><td>159</td><td>159</td><td>55</td><td>1.5952</td><td>159</td><td>363</td><td>55</td><td>2.6104</td><td>443</td><td>594</td><td>17</td></tr>
<tr><td><code>conv4.2.0</code></td><td>12065</td><td>64</td><td>6.3446</td><td>154</td><td>154</td><td>36</td><td>1.1909</td><td>154</td><td>271</td><td>36</td><td>1.9072</td><td>413</td><td>434</td><td>2</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>72.6746</strong></td><td><strong>5292</strong></td><td><strong>5292</strong></td><td><strong>99</strong></td><td><strong>7.7622</strong></td><td><strong>5292</strong></td><td><strong>5299</strong></td><td><strong>99</strong></td><td><strong>6.8379</strong></td><td><strong>1923</strong></td><td><strong>2232</strong></td><td><strong>50</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **90.56%**，将 hash entry 峰值降低 **59.63%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **11.63%**，将 hash entry 峰值降低 **59.68%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 144.9097 | 15.4761 | 13.6758 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 10552 | 10565 | 4260 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

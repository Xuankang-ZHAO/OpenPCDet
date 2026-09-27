# 三种 Block Structuring 方案在 train 000000–000019 上的稳定性与泛化

本文件复现 `feature_map_dram_hash_entry_comparison.md` 在 KITTI `val/000216` 上的口径，
对 filename `000000`–`000019` 做相同实验。
体素上限是 `kitti_dataset.yaml` 的 `MAX_NUMBER_OF_VOXELS.test=40000`。

- 加载：KITTI FOV（`FOV_POINTS_ONLY=True`），与 golden 导出一致。
- 模型：hardware-reference INT8 SECOND 3D backbone，checkpoint `checkpoint_epoch_10.pth`。
- Halo：由下一层 kernel/padding 决定的窗口角点复制；`conv_out` 逻辑输出不分配 DRAM。
- 三种方案：固定块+固定容量（`10x10x6`、每块 600 slot）、固定块+Page、Proposed 可变块+Page。
- 上一层 OFM 与下一层 IFM 是同一 feature map，表中只统计一次。
- Hash entries：固定容量方案等于物化 block 数；两种 Page 方案等于 page 数。
- 执行时峰值按 IFM 与 OFM 同时驻留求和；DRAM 峰值层与 hash 峰值层可能不同。
- Generated: `2026-09-27T16:17:18`

## 20 帧执行时峰值总表

| Frame | 输入体素 | 触达 40000 上限 | 固定容量 DRAM | 固定块 Page DRAM | Proposed DRAM | 固定容量 Hash | 固定块 Page Hash | Proposed Hash | DRAM vs 固定容量 | Hash vs 固定容量 | DRAM vs 固定块 Page | Hash vs 固定块 Page |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `000000` | 16811 | N | 48.9166 | 5.7617 | 7.4048 | 3562 | 3720 | 2502 | 84.86% | 29.76% | -28.52% | 32.74% |
| `000001` | 15463 | N | 119.0094 | 12.7222 | 12.9463 | 8666 | 8685 | 4318 | 89.12% | 50.17% | -1.76% | 50.28% |
| `000002` | 14809 | N | 65.9592 | 7.3682 | 6.1787 | 4803 | 5030 | 1820 | 90.63% | 62.11% | 16.14% | 63.82% |
| `000003` | 14584 | N | 60.7819 | 6.7485 | 6.6138 | 4426 | 4607 | 1931 | 89.12% | 56.37% | 2.00% | 58.09% |
| `000004` | 15365 | N | 132.7835 | 14.9170 | 13.9438 | 9669 | 9669 | 4531 | 89.50% | 53.14% | 6.52% | 53.14% |
| `000005` | 16827 | N | 133.5114 | 14.2412 | 14.1812 | 9722 | 9722 | 4521 | 89.38% | 53.50% | 0.42% | 53.50% |
| `000006` | 15023 | N | 89.5660 | 9.6211 | 12.0674 | 6522 | 6568 | 3960 | 86.53% | 39.28% | -25.43% | 39.71% |
| `000007` | 15891 | N | 145.9946 | 15.5918 | 14.7393 | 10631 | 10644 | 4852 | 89.90% | 54.36% | 5.47% | 54.42% |
| `000008` | 13081 | N | 68.1427 | 7.3887 | 9.0439 | 4962 | 5044 | 2600 | 86.73% | 47.60% | -22.40% | 48.45% |
| `000009` | 15688 | N | 120.9869 | 13.1641 | 13.2275 | 8810 | 8810 | 4554 | 89.07% | 48.31% | -0.48% | 48.31% |
| `000010` | 13094 | N | 93.0817 | 9.9551 | 10.3184 | 6778 | 6796 | 3648 | 88.91% | 46.18% | -3.65% | 46.32% |
| `000011` | 16158 | N | 95.0317 | 10.1748 | 10.6084 | 6920 | 6946 | 3649 | 88.84% | 47.27% | -4.26% | 47.47% |
| `000012` | 14839 | N | 144.9921 | 15.4863 | 13.5791 | 10558 | 10572 | 3332 | 90.63% | 68.44% | 12.32% | 68.48% |
| `000013` | 17054 | N | 150.4715 | 16.0854 | 15.7412 | 10957 | 10981 | 4798 | 89.54% | 56.21% | 2.14% | 56.31% |
| `000014` | 17045 | N | 149.2905 | 15.9375 | 16.3828 | 10871 | 10880 | 4927 | 89.03% | 54.68% | -2.79% | 54.72% |
| `000015` | 14241 | N | 73.0728 | 8.3691 | 10.7051 | 5321 | 5409 | 3016 | 85.35% | 43.32% | -27.91% | 44.24% |
| `000016` | 14000 | N | 95.5948 | 10.2524 | 10.7051 | 6961 | 6999 | 3646 | 88.80% | 47.62% | -4.41% | 47.91% |
| `000017` | 14853 | N | 116.7023 | 13.0469 | 13.3374 | 8498 | 8505 | 4476 | 88.57% | 47.33% | -2.23% | 47.37% |
| `000018` | 14889 | N | 137.7411 | 14.6997 | 14.1328 | 10030 | 10035 | 3609 | 89.74% | 64.02% | 3.86% | 64.04% |
| `000019` | 13435 | N | 67.5659 | 7.4209 | 6.9565 | 4920 | 5066 | 2061 | 89.70% | 58.11% | 6.26% | 59.32% |

相对固定容量 / 固定块分页，Proposed 在 20 帧上的降低比例（正值表示 Proposed 更小）：

| 指标 | 最小 | 平均 | 最大 |
| --- | ---: | ---: | ---: |
| DRAM vs 固定容量 | 84.86% | 88.70% | 90.63% |
| Hash vs 固定容量 | 29.76% | 51.39% | 68.44% |
| DRAM vs 固定块 Page | -28.52% | -3.44% | 16.14% |
| Hash vs 固定块 Page | 32.74% | 51.93% | 68.48% |

## Frame `train/000000`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000000.bin`
- 输入体素: `16811`，坐标 SHA-256 `9da7a8888976ef73913079df66c2171f09b08adc43a97da1ed77f6134f6e5618`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>16811</td><td>4</td><td>11.5700</td><td>1685</td><td>1685</td><td>2</td><td>1.7188</td><td>1685</td><td>1760</td><td>2</td><td>1.0361</td><td>911</td><td>1061</td><td>34</td></tr>
<tr><td><code>conv_input.0</code></td><td>16811</td><td>16</td><td>25.7767</td><td>1877</td><td>1877</td><td>0</td><td>2.8711</td><td>1877</td><td>1960</td><td>0</td><td>1.5205</td><td>898</td><td>1038</td><td>29</td></tr>
<tr><td><code>conv1.0.0</code></td><td>16811</td><td>16</td><td>23.1400</td><td>1685</td><td>1685</td><td>2</td><td>2.5781</td><td>1685</td><td>1760</td><td>2</td><td>1.5542</td><td>911</td><td>1061</td><td>34</td></tr>
<tr><td><code>conv2.0.0</code></td><td>22003</td><td>32</td><td>16.6168</td><td>726</td><td>726</td><td>120</td><td>2.8809</td><td>726</td><td>1180</td><td>120</td><td>2.9004</td><td>799</td><td>1188</td><td>75</td></tr>
<tr><td><code>conv2.1.0</code></td><td>22003</td><td>32</td><td>16.6168</td><td>726</td><td>726</td><td>120</td><td>2.8809</td><td>726</td><td>1180</td><td>120</td><td>2.9004</td><td>799</td><td>1188</td><td>75</td></tr>
<tr><td><code>conv2.2.0</code></td><td>22003</td><td>32</td><td>15.0375</td><td>657</td><td>657</td><td>98</td><td>2.6099</td><td>657</td><td>1069</td><td>98</td><td>3.2080</td><td>892</td><td>1314</td><td>77</td></tr>
<tr><td><code>conv3.0.0</code></td><td>11047</td><td>64</td><td>8.5281</td><td>207</td><td>207</td><td>60</td><td>1.8940</td><td>207</td><td>431</td><td>60</td><td>3.6958</td><td>644</td><td>841</td><td>26</td></tr>
<tr><td><code>conv3.1.0</code></td><td>11047</td><td>64</td><td>8.5281</td><td>207</td><td>207</td><td>60</td><td>1.8940</td><td>207</td><td>431</td><td>60</td><td>3.6958</td><td>644</td><td>841</td><td>26</td></tr>
<tr><td><code>conv3.2.0</code></td><td>11047</td><td>64</td><td>8.8989</td><td>216</td><td>216</td><td>60</td><td>1.9072</td><td>216</td><td>434</td><td>60</td><td>3.7090</td><td>629</td><td>844</td><td>35</td></tr>
<tr><td><code>conv4.0.0</code></td><td>3609</td><td>64</td><td>2.3483</td><td>57</td><td>57</td><td>16</td><td>0.5186</td><td>57</td><td>118</td><td>16</td><td>1.1118</td><td>184</td><td>253</td><td>2</td></tr>
<tr><td><code>conv4.1.0</code></td><td>3609</td><td>64</td><td>2.3483</td><td>57</td><td>57</td><td>16</td><td>0.5186</td><td>57</td><td>118</td><td>16</td><td>1.1118</td><td>184</td><td>253</td><td>2</td></tr>
<tr><td><code>conv4.2.0</code></td><td>3609</td><td>64</td><td>2.0599</td><td>50</td><td>50</td><td>13</td><td>0.3999</td><td>50</td><td>91</td><td>13</td><td>0.6592</td><td>150</td><td>150</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>25.7767</strong></td><td><strong>1877</strong></td><td><strong>1877</strong></td><td><strong>120</strong></td><td><strong>2.8809</strong></td><td><strong>1877</strong></td><td><strong>1960</strong></td><td><strong>120</strong></td><td><strong>3.7090</strong></td><td><strong>911</strong></td><td><strong>1314</strong></td><td><strong>77</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **84.86%**，将 hash entry 峰值降低 **29.76%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-28.52%**，将 hash entry 峰值降低 **32.74%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 48.9166 | 5.7617 | 7.4048 |
| DRAM 峰值层 | `conv1.0.0` | `conv2.1.0` | `conv3.2.0` |
| Hash entries | 3562 | 3720 | 2502 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000001`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000001.bin`
- 输入体素: `15463`，坐标 SHA-256 `78aa1f37c9232cbb82917306338f7312d135b814393c932042e77823abfe36b4`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>15463</td><td>4</td><td>28.8048</td><td>4195</td><td>4195</td><td>0</td><td>4.1074</td><td>4195</td><td>4206</td><td>0</td><td>1.2178</td><td>1207</td><td>1247</td><td>5</td></tr>
<tr><td><code>conv_input.0</code></td><td>15463</td><td>16</td><td>61.3998</td><td>4471</td><td>4471</td><td>0</td><td>6.5610</td><td>4471</td><td>4479</td><td>0</td><td>1.7783</td><td>1181</td><td>1214</td><td>5</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15463</td><td>16</td><td>57.6096</td><td>4195</td><td>4195</td><td>0</td><td>6.1611</td><td>4195</td><td>4206</td><td>0</td><td>1.8267</td><td>1207</td><td>1247</td><td>5</td></tr>
<tr><td><code>conv2.0.0</code></td><td>30502</td><td>32</td><td>45.5704</td><td>1991</td><td>1991</td><td>96</td><td>5.9155</td><td>1991</td><td>2423</td><td>96</td><td>5.0171</td><td>1731</td><td>2055</td><td>55</td></tr>
<tr><td><code>conv2.1.0</code></td><td>30502</td><td>32</td><td>45.5704</td><td>1991</td><td>1991</td><td>96</td><td>5.9155</td><td>1991</td><td>2423</td><td>96</td><td>5.0171</td><td>1731</td><td>2055</td><td>55</td></tr>
<tr><td><code>conv2.2.0</code></td><td>30502</td><td>32</td><td>42.6865</td><td>1865</td><td>1865</td><td>66</td><td>5.3613</td><td>1865</td><td>2196</td><td>66</td><td>5.5249</td><td>1919</td><td>2263</td><td>58</td></tr>
<tr><td><code>conv3.0.0</code></td><td>21973</td><td>64</td><td>22.8653</td><td>555</td><td>555</td><td>83</td><td>3.8760</td><td>555</td><td>882</td><td>83</td><td>6.4731</td><td>1258</td><td>1473</td><td>31</td></tr>
<tr><td><code>conv3.1.0</code></td><td>21973</td><td>64</td><td>22.8653</td><td>555</td><td>555</td><td>83</td><td>3.8760</td><td>555</td><td>882</td><td>83</td><td>6.4731</td><td>1258</td><td>1473</td><td>31</td></tr>
<tr><td><code>conv3.2.0</code></td><td>21973</td><td>64</td><td>23.1125</td><td>561</td><td>561</td><td>82</td><td>3.8848</td><td>561</td><td>884</td><td>82</td><td>6.4248</td><td>1251</td><td>1462</td><td>27</td></tr>
<tr><td><code>conv4.0.0</code></td><td>10632</td><td>64</td><td>5.6442</td><td>137</td><td>137</td><td>46</td><td>1.3359</td><td>137</td><td>304</td><td>46</td><td>2.4873</td><td>414</td><td>566</td><td>10</td></tr>
<tr><td><code>conv4.1.0</code></td><td>10632</td><td>64</td><td>5.6442</td><td>137</td><td>137</td><td>46</td><td>1.3359</td><td>137</td><td>304</td><td>46</td><td>2.4873</td><td>414</td><td>566</td><td>10</td></tr>
<tr><td><code>conv4.2.0</code></td><td>10632</td><td>64</td><td>5.3970</td><td>131</td><td>131</td><td>31</td><td>1.0415</td><td>131</td><td>237</td><td>31</td><td>1.7710</td><td>387</td><td>403</td><td>3</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>61.3998</strong></td><td><strong>4471</strong></td><td><strong>4471</strong></td><td><strong>96</strong></td><td><strong>6.5610</strong></td><td><strong>4471</strong></td><td><strong>4479</strong></td><td><strong>96</strong></td><td><strong>6.4731</strong></td><td><strong>1919</strong></td><td><strong>2263</strong></td><td><strong>58</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.12%**，将 hash entry 峰值降低 **50.17%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-1.76%**，将 hash entry 峰值降低 **50.28%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 119.0094 | 12.7222 | 12.9463 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 8666 | 8685 | 4318 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000002`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000002.bin`
- 输入体素: `14809`，坐标 SHA-256 `0650175ebc7f42bcd8a3cfdf0f183ce2f54b52935e0b60011e4658624fa606db`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>14809</td><td>4</td><td>16.9189</td><td>2464</td><td>2464</td><td>14</td><td>2.5127</td><td>2464</td><td>2573</td><td>14</td><td>0.8096</td><td>689</td><td>829</td><td>36</td></tr>
<tr><td><code>conv_input.0</code></td><td>14809</td><td>16</td><td>32.1213</td><td>2339</td><td>2339</td><td>11</td><td>3.5991</td><td>2339</td><td>2457</td><td>11</td><td>1.1982</td><td>685</td><td>818</td><td>34</td></tr>
<tr><td><code>conv1.0.0</code></td><td>14809</td><td>16</td><td>33.8379</td><td>2464</td><td>2464</td><td>14</td><td>3.7690</td><td>2464</td><td>2573</td><td>14</td><td>1.2144</td><td>689</td><td>829</td><td>36</td></tr>
<tr><td><code>conv2.0.0</code></td><td>17310</td><td>32</td><td>21.2631</td><td>929</td><td>929</td><td>60</td><td>2.8540</td><td>929</td><td>1169</td><td>60</td><td>2.1582</td><td>671</td><td>884</td><td>43</td></tr>
<tr><td><code>conv2.1.0</code></td><td>17310</td><td>32</td><td>21.2631</td><td>929</td><td>929</td><td>60</td><td>2.8540</td><td>929</td><td>1169</td><td>60</td><td>2.1582</td><td>671</td><td>884</td><td>43</td></tr>
<tr><td><code>conv2.2.0</code></td><td>17310</td><td>32</td><td>22.5906</td><td>987</td><td>987</td><td>58</td><td>3.0518</td><td>987</td><td>1250</td><td>58</td><td>2.2852</td><td>725</td><td>936</td><td>40</td></tr>
<tr><td><code>conv3.0.0</code></td><td>10581</td><td>64</td><td>9.8053</td><td>238</td><td>238</td><td>37</td><td>1.7710</td><td>238</td><td>403</td><td>37</td><td>3.0894</td><td>578</td><td>703</td><td>27</td></tr>
<tr><td><code>conv3.1.0</code></td><td>10581</td><td>64</td><td>9.8053</td><td>238</td><td>238</td><td>37</td><td>1.7710</td><td>238</td><td>403</td><td>37</td><td>3.0894</td><td>578</td><td>703</td><td>27</td></tr>
<tr><td><code>conv3.2.0</code></td><td>10581</td><td>64</td><td>10.2997</td><td>250</td><td>250</td><td>39</td><td>1.8765</td><td>250</td><td>427</td><td>39</td><td>3.0146</td><td>560</td><td>686</td><td>23</td></tr>
<tr><td><code>conv4.0.0</code></td><td>4695</td><td>64</td><td>2.5131</td><td>61</td><td>61</td><td>23</td><td>0.6064</td><td>61</td><td>138</td><td>23</td><td>1.0063</td><td>163</td><td>229</td><td>8</td></tr>
<tr><td><code>conv4.1.0</code></td><td>4695</td><td>64</td><td>2.5131</td><td>61</td><td>61</td><td>23</td><td>0.6064</td><td>61</td><td>138</td><td>23</td><td>1.0063</td><td>163</td><td>229</td><td>8</td></tr>
<tr><td><code>conv4.2.0</code></td><td>4695</td><td>64</td><td>2.3071</td><td>56</td><td>56</td><td>14</td><td>0.4482</td><td>56</td><td>102</td><td>14</td><td>0.6812</td><td>145</td><td>155</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>33.8379</strong></td><td><strong>2464</strong></td><td><strong>2464</strong></td><td><strong>60</strong></td><td><strong>3.7690</strong></td><td><strong>2464</strong></td><td><strong>2573</strong></td><td><strong>60</strong></td><td><strong>3.0894</strong></td><td><strong>725</strong></td><td><strong>936</strong></td><td><strong>43</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **90.63%**，将 hash entry 峰值降低 **62.11%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **16.14%**，将 hash entry 峰值降低 **63.82%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 65.9592 | 7.3682 | 6.1787 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 4803 | 5030 | 1820 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000003`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000003.bin`
- 输入体素: `14584`，坐标 SHA-256 `f3515e4c2371a077d4a68f414310b38c95fc33f1e630b74c997f50d909fd8a4c`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>14584</td><td>4</td><td>15.2847</td><td>2226</td><td>2226</td><td>12</td><td>2.2598</td><td>2226</td><td>2314</td><td>12</td><td>0.8604</td><td>752</td><td>881</td><td>33</td></tr>
<tr><td><code>conv_input.0</code></td><td>14584</td><td>16</td><td>30.2124</td><td>2200</td><td>2200</td><td>8</td><td>3.3589</td><td>2200</td><td>2293</td><td>8</td><td>1.2803</td><td>750</td><td>874</td><td>33</td></tr>
<tr><td><code>conv1.0.0</code></td><td>14584</td><td>16</td><td>30.5695</td><td>2226</td><td>2226</td><td>12</td><td>3.3896</td><td>2226</td><td>2314</td><td>12</td><td>1.2905</td><td>752</td><td>881</td><td>33</td></tr>
<tr><td><code>conv2.0.0</code></td><td>19764</td><td>32</td><td>19.2947</td><td>843</td><td>843</td><td>71</td><td>2.8076</td><td>843</td><td>1150</td><td>71</td><td>2.2876</td><td>663</td><td>937</td><td>62</td></tr>
<tr><td><code>conv2.1.0</code></td><td>19764</td><td>32</td><td>19.2947</td><td>843</td><td>843</td><td>71</td><td>2.8076</td><td>843</td><td>1150</td><td>71</td><td>2.2876</td><td>663</td><td>937</td><td>62</td></tr>
<tr><td><code>conv2.2.0</code></td><td>19764</td><td>32</td><td>20.6680</td><td>903</td><td>903</td><td>70</td><td>2.9761</td><td>903</td><td>1219</td><td>70</td><td>2.4268</td><td>694</td><td>994</td><td>67</td></tr>
<tr><td><code>conv3.0.0</code></td><td>11187</td><td>64</td><td>9.7229</td><td>236</td><td>236</td><td>55</td><td>1.8325</td><td>236</td><td>417</td><td>55</td><td>3.3003</td><td>608</td><td>751</td><td>32</td></tr>
<tr><td><code>conv3.1.0</code></td><td>11187</td><td>64</td><td>9.7229</td><td>236</td><td>236</td><td>55</td><td>1.8325</td><td>236</td><td>417</td><td>55</td><td>3.3003</td><td>608</td><td>751</td><td>32</td></tr>
<tr><td><code>conv3.2.0</code></td><td>11187</td><td>64</td><td>10.1761</td><td>247</td><td>247</td><td>53</td><td>1.8896</td><td>247</td><td>430</td><td>53</td><td>3.3135</td><td>608</td><td>754</td><td>27</td></tr>
<tr><td><code>conv4.0.0</code></td><td>4219</td><td>64</td><td>2.5543</td><td>62</td><td>62</td><td>19</td><td>0.5801</td><td>62</td><td>132</td><td>19</td><td>0.9932</td><td>162</td><td>226</td><td>3</td></tr>
<tr><td><code>conv4.1.0</code></td><td>4219</td><td>64</td><td>2.5543</td><td>62</td><td>62</td><td>19</td><td>0.5801</td><td>62</td><td>132</td><td>19</td><td>0.9932</td><td>162</td><td>226</td><td>3</td></tr>
<tr><td><code>conv4.2.0</code></td><td>4219</td><td>64</td><td>2.1835</td><td>53</td><td>53</td><td>12</td><td>0.4219</td><td>53</td><td>96</td><td>12</td><td>0.6724</td><td>148</td><td>153</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>30.5695</strong></td><td><strong>2226</strong></td><td><strong>2226</strong></td><td><strong>71</strong></td><td><strong>3.3896</strong></td><td><strong>2226</strong></td><td><strong>2314</strong></td><td><strong>71</strong></td><td><strong>3.3135</strong></td><td><strong>752</strong></td><td><strong>994</strong></td><td><strong>67</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.12%**，将 hash entry 峰值降低 **56.37%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **2.00%**，将 hash entry 峰值降低 **58.09%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 60.7819 | 6.7485 | 6.6138 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.2.0` |
| Hash entries | 4426 | 4607 | 1931 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000004`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000004.bin`
- 输入体素: `15365`，坐标 SHA-256 `72cdab0ecc5756247d1dda65a6c0bda1fb1d63a1421a1bb08682f5f74146722c`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>15365</td><td>4</td><td>30.2261</td><td>4402</td><td>4402</td><td>0</td><td>4.2988</td><td>4402</td><td>4402</td><td>0</td><td>1.1650</td><td>1171</td><td>1193</td><td>1</td></tr>
<tr><td><code>conv_input.0</code></td><td>15365</td><td>16</td><td>72.3312</td><td>5267</td><td>5267</td><td>0</td><td>7.7153</td><td>5267</td><td>5267</td><td>0</td><td>1.6099</td><td>1070</td><td>1099</td><td>2</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15365</td><td>16</td><td>60.4523</td><td>4402</td><td>4402</td><td>0</td><td>6.4482</td><td>4402</td><td>4402</td><td>0</td><td>1.7476</td><td>1171</td><td>1193</td><td>1</td></tr>
<tr><td><code>conv2.0.0</code></td><td>31398</td><td>32</td><td>60.6079</td><td>2648</td><td>2648</td><td>70</td><td>7.4585</td><td>2648</td><td>3055</td><td>70</td><td>4.9438</td><td>1744</td><td>2025</td><td>35</td></tr>
<tr><td><code>conv2.1.0</code></td><td>31398</td><td>32</td><td>60.6079</td><td>2648</td><td>2648</td><td>70</td><td>7.4585</td><td>2648</td><td>3055</td><td>70</td><td>4.9438</td><td>1744</td><td>2025</td><td>35</td></tr>
<tr><td><code>conv2.2.0</code></td><td>31398</td><td>32</td><td>49.8505</td><td>2178</td><td>2178</td><td>42</td><td>6.0034</td><td>2178</td><td>2459</td><td>42</td><td>6.1182</td><td>2223</td><td>2506</td><td>36</td></tr>
<tr><td><code>conv3.0.0</code></td><td>24073</td><td>64</td><td>26.4496</td><td>642</td><td>642</td><td>91</td><td>4.3594</td><td>642</td><td>992</td><td>91</td><td>6.9653</td><td>1379</td><td>1585</td><td>23</td></tr>
<tr><td><code>conv3.1.0</code></td><td>24073</td><td>64</td><td>26.4496</td><td>642</td><td>642</td><td>91</td><td>4.3594</td><td>642</td><td>992</td><td>91</td><td>6.9653</td><td>1379</td><td>1585</td><td>23</td></tr>
<tr><td><code>conv3.2.0</code></td><td>24073</td><td>64</td><td>26.4908</td><td>643</td><td>643</td><td>89</td><td>4.3813</td><td>643</td><td>997</td><td>89</td><td>6.9785</td><td>1385</td><td>1588</td><td>24</td></tr>
<tr><td><code>conv4.0.0</code></td><td>11841</td><td>64</td><td>6.7154</td><td>163</td><td>163</td><td>53</td><td>1.5205</td><td>163</td><td>346</td><td>53</td><td>2.6191</td><td>460</td><td>596</td><td>9</td></tr>
<tr><td><code>conv4.1.0</code></td><td>11841</td><td>64</td><td>6.7154</td><td>163</td><td>163</td><td>53</td><td>1.5205</td><td>163</td><td>346</td><td>53</td><td>2.6191</td><td>460</td><td>596</td><td>9</td></tr>
<tr><td><code>conv4.2.0</code></td><td>11841</td><td>64</td><td>6.3446</td><td>154</td><td>154</td><td>33</td><td>1.1646</td><td>154</td><td>265</td><td>33</td><td>1.9688</td><td>421</td><td>448</td><td>3</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>72.3312</strong></td><td><strong>5267</strong></td><td><strong>5267</strong></td><td><strong>91</strong></td><td><strong>7.7153</strong></td><td><strong>5267</strong></td><td><strong>5267</strong></td><td><strong>91</strong></td><td><strong>6.9785</strong></td><td><strong>2223</strong></td><td><strong>2506</strong></td><td><strong>36</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.50%**，将 hash entry 峰值降低 **53.14%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **6.52%**，将 hash entry 峰值降低 **53.14%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 132.7835 | 14.9170 | 13.9438 |
| DRAM 峰值层 | `conv1.0.0` | `conv2.1.0` | `conv3.2.0` |
| Hash entries | 9669 | 9669 | 4531 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000005`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000005.bin`
- 输入体素: `16827`，坐标 SHA-256 `dd839a8ee7cbe8bf91ea650ad9c9696df7840f299efc659e02b2930d8325f691`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>16827</td><td>4</td><td>31.8535</td><td>4639</td><td>4639</td><td>0</td><td>4.5303</td><td>4639</td><td>4639</td><td>0</td><td>1.1973</td><td>1186</td><td>1226</td><td>6</td></tr>
<tr><td><code>conv_input.0</code></td><td>16827</td><td>16</td><td>69.8044</td><td>5083</td><td>5083</td><td>0</td><td>7.4458</td><td>5083</td><td>5083</td><td>0</td><td>1.7300</td><td>1136</td><td>1181</td><td>7</td></tr>
<tr><td><code>conv1.0.0</code></td><td>16827</td><td>16</td><td>63.7070</td><td>4639</td><td>4639</td><td>0</td><td>6.7954</td><td>4639</td><td>4639</td><td>0</td><td>1.7959</td><td>1186</td><td>1226</td><td>6</td></tr>
<tr><td><code>conv2.0.0</code></td><td>36597</td><td>32</td><td>51.5900</td><td>2254</td><td>2254</td><td>100</td><td>6.7871</td><td>2254</td><td>2780</td><td>100</td><td>5.1294</td><td>1731</td><td>2101</td><td>48</td></tr>
<tr><td><code>conv2.1.0</code></td><td>36597</td><td>32</td><td>51.5900</td><td>2254</td><td>2254</td><td>100</td><td>6.7871</td><td>2254</td><td>2780</td><td>100</td><td>5.1294</td><td>1731</td><td>2101</td><td>48</td></tr>
<tr><td><code>conv2.2.0</code></td><td>36597</td><td>32</td><td>48.0652</td><td>2100</td><td>2100</td><td>73</td><td>6.1597</td><td>2100</td><td>2523</td><td>73</td><td>5.9082</td><td>1993</td><td>2420</td><td>51</td></tr>
<tr><td><code>conv3.0.0</code></td><td>26079</td><td>64</td><td>26.2436</td><td>637</td><td>637</td><td>121</td><td>4.6934</td><td>637</td><td>1068</td><td>121</td><td>7.0093</td><td>1301</td><td>1595</td><td>30</td></tr>
<tr><td><code>conv3.1.0</code></td><td>26079</td><td>64</td><td>26.2436</td><td>637</td><td>637</td><td>121</td><td>4.6934</td><td>637</td><td>1068</td><td>121</td><td>7.0093</td><td>1301</td><td>1595</td><td>30</td></tr>
<tr><td><code>conv3.2.0</code></td><td>26079</td><td>64</td><td>26.2848</td><td>638</td><td>638</td><td>121</td><td>4.7065</td><td>638</td><td>1071</td><td>121</td><td>7.1719</td><td>1321</td><td>1632</td><td>34</td></tr>
<tr><td><code>conv4.0.0</code></td><td>11517</td><td>64</td><td>6.1798</td><td>150</td><td>150</td><td>51</td><td>1.4897</td><td>150</td><td>339</td><td>51</td><td>2.6982</td><td>431</td><td>614</td><td>7</td></tr>
<tr><td><code>conv4.1.0</code></td><td>11517</td><td>64</td><td>6.1798</td><td>150</td><td>150</td><td>51</td><td>1.4897</td><td>150</td><td>339</td><td>51</td><td>2.6982</td><td>431</td><td>614</td><td>7</td></tr>
<tr><td><code>conv4.2.0</code></td><td>11517</td><td>64</td><td>5.8090</td><td>141</td><td>141</td><td>34</td><td>1.1514</td><td>141</td><td>262</td><td>34</td><td>1.8237</td><td>398</td><td>415</td><td>1</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>69.8044</strong></td><td><strong>5083</strong></td><td><strong>5083</strong></td><td><strong>121</strong></td><td><strong>7.4458</strong></td><td><strong>5083</strong></td><td><strong>5083</strong></td><td><strong>121</strong></td><td><strong>7.1719</strong></td><td><strong>1993</strong></td><td><strong>2420</strong></td><td><strong>51</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.38%**，将 hash entry 峰值降低 **53.50%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **0.42%**，将 hash entry 峰值降低 **53.50%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 133.5114 | 14.2412 | 14.1812 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.2.0` |
| Hash entries | 9722 | 9722 | 4521 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000006`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000006.bin`
- 输入体素: `15023`，坐标 SHA-256 `c2c882c22fa59bc15d97909043ee58d88605ad6c3db51472734b99b52cf5419d`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>15023</td><td>4</td><td>22.1718</td><td>3229</td><td>3229</td><td>0</td><td>3.1758</td><td>3229</td><td>3252</td><td>0</td><td>1.2461</td><td>1215</td><td>1276</td><td>16</td></tr>
<tr><td><code>conv_input.0</code></td><td>15023</td><td>16</td><td>45.2225</td><td>3293</td><td>3293</td><td>0</td><td>4.8574</td><td>3293</td><td>3316</td><td>0</td><td>1.7593</td><td>1145</td><td>1201</td><td>16</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15023</td><td>16</td><td>44.3436</td><td>3229</td><td>3229</td><td>0</td><td>4.7637</td><td>3229</td><td>3252</td><td>0</td><td>1.8691</td><td>1215</td><td>1276</td><td>16</td></tr>
<tr><td><code>conv2.0.0</code></td><td>24877</td><td>32</td><td>38.7497</td><td>1693</td><td>1693</td><td>64</td><td>4.7559</td><td>1693</td><td>1948</td><td>64</td><td>4.7144</td><td>1699</td><td>1931</td><td>50</td></tr>
<tr><td><code>conv2.1.0</code></td><td>24877</td><td>32</td><td>38.7497</td><td>1693</td><td>1693</td><td>64</td><td>4.7559</td><td>1693</td><td>1948</td><td>64</td><td>4.7144</td><td>1699</td><td>1931</td><td>50</td></tr>
<tr><td><code>conv2.2.0</code></td><td>24877</td><td>32</td><td>32.7072</td><td>1429</td><td>1429</td><td>47</td><td>4.0088</td><td>1429</td><td>1642</td><td>47</td><td>4.9536</td><td>1777</td><td>2029</td><td>52</td></tr>
<tr><td><code>conv3.0.0</code></td><td>17770</td><td>64</td><td>24.1425</td><td>586</td><td>586</td><td>61</td><td>3.6255</td><td>586</td><td>825</td><td>61</td><td>6.0337</td><td>1232</td><td>1373</td><td>9</td></tr>
<tr><td><code>conv3.1.0</code></td><td>17770</td><td>64</td><td>24.1425</td><td>586</td><td>586</td><td>61</td><td>3.6255</td><td>586</td><td>825</td><td>61</td><td>6.0337</td><td>1232</td><td>1373</td><td>9</td></tr>
<tr><td><code>conv3.2.0</code></td><td>17770</td><td>64</td><td>24.0189</td><td>583</td><td>583</td><td>59</td><td>3.6035</td><td>583</td><td>820</td><td>59</td><td>5.9766</td><td>1228</td><td>1360</td><td>9</td></tr>
<tr><td><code>conv4.0.0</code></td><td>8533</td><td>64</td><td>6.3446</td><td>154</td><td>154</td><td>37</td><td>1.2437</td><td>154</td><td>283</td><td>37</td><td>2.2896</td><td>419</td><td>521</td><td>1</td></tr>
<tr><td><code>conv4.1.0</code></td><td>8533</td><td>64</td><td>6.3446</td><td>154</td><td>154</td><td>37</td><td>1.2437</td><td>154</td><td>283</td><td>37</td><td>2.2896</td><td>419</td><td>521</td><td>1</td></tr>
<tr><td><code>conv4.2.0</code></td><td>8533</td><td>64</td><td>5.8090</td><td>141</td><td>141</td><td>21</td><td>0.9800</td><td>141</td><td>223</td><td>21</td><td>1.6436</td><td>367</td><td>374</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>45.2225</strong></td><td><strong>3293</strong></td><td><strong>3293</strong></td><td><strong>64</strong></td><td><strong>4.8574</strong></td><td><strong>3293</strong></td><td><strong>3316</strong></td><td><strong>64</strong></td><td><strong>6.0337</strong></td><td><strong>1777</strong></td><td><strong>2029</strong></td><td><strong>52</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **86.53%**，将 hash entry 峰值降低 **39.28%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-25.43%**，将 hash entry 峰值降低 **39.71%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 89.5660 | 9.6211 | 12.0674 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 6522 | 6568 | 3960 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000007`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000007.bin`
- 输入体素: `15891`，坐标 SHA-256 `0cb15a4f0a7447d45081a07c3d2836edd24fbeb57f55764aa46e5bc850524baf`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>15891</td><td>4</td><td>34.0851</td><td>4964</td><td>4964</td><td>0</td><td>4.8535</td><td>4964</td><td>4970</td><td>0</td><td>1.3223</td><td>1329</td><td>1354</td><td>2</td></tr>
<tr><td><code>conv_input.0</code></td><td>15891</td><td>16</td><td>77.8244</td><td>5667</td><td>5667</td><td>1</td><td>8.3115</td><td>5667</td><td>5674</td><td>1</td><td>1.9102</td><td>1281</td><td>1304</td><td>2</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15891</td><td>16</td><td>68.1702</td><td>4964</td><td>4964</td><td>0</td><td>7.2803</td><td>4964</td><td>4970</td><td>0</td><td>1.9834</td><td>1329</td><td>1354</td><td>2</td></tr>
<tr><td><code>conv2.0.0</code></td><td>33225</td><td>32</td><td>61.5234</td><td>2688</td><td>2688</td><td>59</td><td>7.4463</td><td>2688</td><td>3050</td><td>59</td><td>5.2808</td><td>1897</td><td>2163</td><td>38</td></tr>
<tr><td><code>conv2.1.0</code></td><td>33225</td><td>32</td><td>61.5234</td><td>2688</td><td>2688</td><td>59</td><td>7.4463</td><td>2688</td><td>3050</td><td>59</td><td>5.2808</td><td>1897</td><td>2163</td><td>38</td></tr>
<tr><td><code>conv2.2.0</code></td><td>33225</td><td>32</td><td>52.0020</td><td>2272</td><td>2272</td><td>36</td><td>6.2109</td><td>2272</td><td>2544</td><td>36</td><td>6.5649</td><td>2403</td><td>2689</td><td>35</td></tr>
<tr><td><code>conv3.0.0</code></td><td>26459</td><td>64</td><td>28.7567</td><td>698</td><td>698</td><td>88</td><td>4.6934</td><td>698</td><td>1068</td><td>88</td><td>7.3696</td><td>1434</td><td>1677</td><td>33</td></tr>
<tr><td><code>conv3.1.0</code></td><td>26459</td><td>64</td><td>28.7567</td><td>698</td><td>698</td><td>88</td><td>4.6934</td><td>698</td><td>1068</td><td>88</td><td>7.3696</td><td>1434</td><td>1677</td><td>33</td></tr>
<tr><td><code>conv3.2.0</code></td><td>26459</td><td>64</td><td>28.7979</td><td>699</td><td>699</td><td>82</td><td>4.7197</td><td>699</td><td>1074</td><td>82</td><td>7.2949</td><td>1422</td><td>1660</td><td>31</td></tr>
<tr><td><code>conv4.0.0</code></td><td>12723</td><td>64</td><td>7.6218</td><td>185</td><td>185</td><td>58</td><td>1.7139</td><td>185</td><td>390</td><td>58</td><td>2.8301</td><td>479</td><td>644</td><td>12</td></tr>
<tr><td><code>conv4.1.0</code></td><td>12723</td><td>64</td><td>7.6218</td><td>185</td><td>185</td><td>58</td><td>1.7139</td><td>185</td><td>390</td><td>58</td><td>2.8301</td><td>479</td><td>644</td><td>12</td></tr>
<tr><td><code>conv4.2.0</code></td><td>12723</td><td>64</td><td>7.3746</td><td>179</td><td>179</td><td>33</td><td>1.3140</td><td>179</td><td>299</td><td>33</td><td>2.0566</td><td>443</td><td>468</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>77.8244</strong></td><td><strong>5667</strong></td><td><strong>5667</strong></td><td><strong>88</strong></td><td><strong>8.3115</strong></td><td><strong>5667</strong></td><td><strong>5674</strong></td><td><strong>88</strong></td><td><strong>7.3696</strong></td><td><strong>2403</strong></td><td><strong>2689</strong></td><td><strong>38</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.90%**，将 hash entry 峰值降低 **54.36%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **5.47%**，将 hash entry 峰值降低 **54.42%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 145.9946 | 15.5918 | 14.7393 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 10631 | 10644 | 4852 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000008`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000008.bin`
- 输入体素: `13081`，坐标 SHA-256 `e32d392d967bdaf3a1307b331e9ba3c178033daa4b0533798195fa4aa48a9e04`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>13081</td><td>4</td><td>16.3971</td><td>2388</td><td>2388</td><td>2</td><td>2.3711</td><td>2388</td><td>2428</td><td>2</td><td>0.9043</td><td>848</td><td>926</td><td>10</td></tr>
<tr><td><code>conv_input.0</code></td><td>13081</td><td>16</td><td>35.3485</td><td>2574</td><td>2574</td><td>5</td><td>3.8320</td><td>2574</td><td>2616</td><td>5</td><td>1.3521</td><td>852</td><td>923</td><td>9</td></tr>
<tr><td><code>conv1.0.0</code></td><td>13081</td><td>16</td><td>32.7942</td><td>2388</td><td>2388</td><td>2</td><td>3.5566</td><td>2388</td><td>2428</td><td>2</td><td>1.3564</td><td>848</td><td>926</td><td>10</td></tr>
<tr><td><code>conv2.0.0</code></td><td>20294</td><td>32</td><td>27.1683</td><td>1187</td><td>1187</td><td>66</td><td>3.6426</td><td>1187</td><td>1492</td><td>66</td><td>3.0469</td><td>990</td><td>1248</td><td>49</td></tr>
<tr><td><code>conv2.1.0</code></td><td>20294</td><td>32</td><td>27.1683</td><td>1187</td><td>1187</td><td>66</td><td>3.6426</td><td>1187</td><td>1492</td><td>66</td><td>3.0469</td><td>990</td><td>1248</td><td>49</td></tr>
<tr><td><code>conv2.2.0</code></td><td>20294</td><td>32</td><td>24.7879</td><td>1083</td><td>1083</td><td>61</td><td>3.3594</td><td>1083</td><td>1376</td><td>61</td><td>3.3008</td><td>1083</td><td>1352</td><td>56</td></tr>
<tr><td><code>conv3.0.0</code></td><td>12359</td><td>64</td><td>15.5731</td><td>378</td><td>378</td><td>56</td><td>2.5576</td><td>378</td><td>582</td><td>56</td><td>4.5220</td><td>891</td><td>1029</td><td>17</td></tr>
<tr><td><code>conv3.1.0</code></td><td>12359</td><td>64</td><td>15.5731</td><td>378</td><td>378</td><td>56</td><td>2.5576</td><td>378</td><td>582</td><td>56</td><td>4.5220</td><td>891</td><td>1029</td><td>17</td></tr>
<tr><td><code>conv3.2.0</code></td><td>12359</td><td>64</td><td>15.6967</td><td>381</td><td>381</td><td>60</td><td>2.6104</td><td>381</td><td>594</td><td>60</td><td>4.4165</td><td>859</td><td>1005</td><td>12</td></tr>
<tr><td><code>conv4.0.0</code></td><td>5297</td><td>64</td><td>3.2547</td><td>79</td><td>79</td><td>25</td><td>0.7207</td><td>79</td><td>164</td><td>25</td><td>1.3184</td><td>241</td><td>300</td><td>1</td></tr>
<tr><td><code>conv4.1.0</code></td><td>5297</td><td>64</td><td>3.2547</td><td>79</td><td>79</td><td>25</td><td>0.7207</td><td>79</td><td>164</td><td>25</td><td>1.3184</td><td>241</td><td>300</td><td>1</td></tr>
<tr><td><code>conv4.2.0</code></td><td>5297</td><td>64</td><td>3.0899</td><td>75</td><td>75</td><td>16</td><td>0.5405</td><td>75</td><td>123</td><td>16</td><td>0.9800</td><td>220</td><td>223</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>35.3485</strong></td><td><strong>2574</strong></td><td><strong>2574</strong></td><td><strong>66</strong></td><td><strong>3.8320</strong></td><td><strong>2574</strong></td><td><strong>2616</strong></td><td><strong>66</strong></td><td><strong>4.5220</strong></td><td><strong>1083</strong></td><td><strong>1352</strong></td><td><strong>56</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **86.73%**，将 hash entry 峰值降低 **47.60%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-22.40%**，将 hash entry 峰值降低 **48.45%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 68.1427 | 7.3887 | 9.0439 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 4962 | 5044 | 2600 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000009`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000009.bin`
- 输入体素: `15688`，坐标 SHA-256 `f0786a3573c850417d67ad9a55c7fb1b6c793ac5be6bed630106ab2d1d3872d1`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>15688</td><td>4</td><td>27.5139</td><td>4007</td><td>4007</td><td>0</td><td>3.9131</td><td>4007</td><td>4007</td><td>0</td><td>1.1865</td><td>1194</td><td>1215</td><td>2</td></tr>
<tr><td><code>conv_input.0</code></td><td>15688</td><td>16</td><td>65.9592</td><td>4803</td><td>4803</td><td>0</td><td>7.0356</td><td>4803</td><td>4803</td><td>0</td><td>1.7021</td><td>1141</td><td>1162</td><td>3</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15688</td><td>16</td><td>55.0278</td><td>4007</td><td>4007</td><td>0</td><td>5.8696</td><td>4007</td><td>4007</td><td>0</td><td>1.7798</td><td>1194</td><td>1215</td><td>2</td></tr>
<tr><td><code>conv2.0.0</code></td><td>32100</td><td>32</td><td>51.9333</td><td>2269</td><td>2269</td><td>70</td><td>6.5820</td><td>2269</td><td>2696</td><td>70</td><td>5.0073</td><td>1788</td><td>2051</td><td>35</td></tr>
<tr><td><code>conv2.1.0</code></td><td>32100</td><td>32</td><td>51.9333</td><td>2269</td><td>2269</td><td>70</td><td>6.5820</td><td>2269</td><td>2696</td><td>70</td><td>5.0073</td><td>1788</td><td>2051</td><td>35</td></tr>
<tr><td><code>conv2.2.0</code></td><td>32100</td><td>32</td><td>43.9453</td><td>1920</td><td>1920</td><td>42</td><td>5.4175</td><td>1920</td><td>2219</td><td>42</td><td>6.1108</td><td>2253</td><td>2503</td><td>33</td></tr>
<tr><td><code>conv3.0.0</code></td><td>23356</td><td>64</td><td>27.5208</td><td>668</td><td>668</td><td>82</td><td>4.4121</td><td>668</td><td>1004</td><td>82</td><td>6.6138</td><td>1353</td><td>1505</td><td>14</td></tr>
<tr><td><code>conv3.1.0</code></td><td>23356</td><td>64</td><td>27.5208</td><td>668</td><td>668</td><td>82</td><td>4.4121</td><td>668</td><td>1004</td><td>82</td><td>6.6138</td><td>1353</td><td>1505</td><td>14</td></tr>
<tr><td><code>conv3.2.0</code></td><td>23356</td><td>64</td><td>26.9028</td><td>653</td><td>653</td><td>76</td><td>4.2715</td><td>653</td><td>972</td><td>76</td><td>6.5566</td><td>1333</td><td>1492</td><td>12</td></tr>
<tr><td><code>conv4.0.0</code></td><td>11346</td><td>64</td><td>7.3746</td><td>179</td><td>179</td><td>51</td><td>1.5908</td><td>179</td><td>362</td><td>51</td><td>2.7246</td><td>481</td><td>620</td><td>2</td></tr>
<tr><td><code>conv4.1.0</code></td><td>11346</td><td>64</td><td>7.3746</td><td>179</td><td>179</td><td>51</td><td>1.5908</td><td>179</td><td>362</td><td>51</td><td>2.7246</td><td>481</td><td>620</td><td>2</td></tr>
<tr><td><code>conv4.2.0</code></td><td>11346</td><td>64</td><td>6.7566</td><td>164</td><td>164</td><td>27</td><td>1.1953</td><td>164</td><td>272</td><td>27</td><td>1.9600</td><td>435</td><td>446</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>65.9592</strong></td><td><strong>4803</strong></td><td><strong>4803</strong></td><td><strong>82</strong></td><td><strong>7.0356</strong></td><td><strong>4803</strong></td><td><strong>4803</strong></td><td><strong>82</strong></td><td><strong>6.6138</strong></td><td><strong>2253</strong></td><td><strong>2503</strong></td><td><strong>35</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.07%**，将 hash entry 峰值降低 **48.31%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-0.48%**，将 hash entry 峰值降低 **48.31%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 120.9869 | 13.1641 | 13.2275 |
| DRAM 峰值层 | `conv1.0.0` | `conv2.1.0` | `conv3.1.0` |
| Hash entries | 8810 | 8810 | 4554 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000010`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000010.bin`
- 输入体素: `13094`，坐标 SHA-256 `4acce1cd485e7759ee11a4e397e7966dc1afcd28fd637c4d7a4f3cbeac7635f0`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>13094</td><td>4</td><td>21.8971</td><td>3189</td><td>3189</td><td>0</td><td>3.1221</td><td>3189</td><td>3197</td><td>0</td><td>1.0732</td><td>1065</td><td>1099</td><td>7</td></tr>
<tr><td><code>conv_input.0</code></td><td>13094</td><td>16</td><td>49.2874</td><td>3589</td><td>3589</td><td>0</td><td>5.2720</td><td>3589</td><td>3599</td><td>0</td><td>1.6406</td><td>1092</td><td>1120</td><td>5</td></tr>
<tr><td><code>conv1.0.0</code></td><td>13094</td><td>16</td><td>43.7943</td><td>3189</td><td>3189</td><td>0</td><td>4.6831</td><td>3189</td><td>3197</td><td>0</td><td>1.6099</td><td>1065</td><td>1099</td><td>7</td></tr>
<tr><td><code>conv2.0.0</code></td><td>24190</td><td>32</td><td>35.3394</td><td>1544</td><td>1544</td><td>68</td><td>4.6313</td><td>1544</td><td>1897</td><td>68</td><td>4.2725</td><td>1507</td><td>1750</td><td>26</td></tr>
<tr><td><code>conv2.1.0</code></td><td>24190</td><td>32</td><td>35.3394</td><td>1544</td><td>1544</td><td>68</td><td>4.6313</td><td>1544</td><td>1897</td><td>68</td><td>4.2725</td><td>1507</td><td>1750</td><td>26</td></tr>
<tr><td><code>conv2.2.0</code></td><td>24190</td><td>32</td><td>31.2424</td><td>1365</td><td>1365</td><td>46</td><td>3.9453</td><td>1365</td><td>1616</td><td>46</td><td>4.6338</td><td>1634</td><td>1898</td><td>32</td></tr>
<tr><td><code>conv3.0.0</code></td><td>17257</td><td>64</td><td>22.8241</td><td>554</td><td>554</td><td>61</td><td>3.4717</td><td>554</td><td>790</td><td>61</td><td>5.1592</td><td>1048</td><td>1174</td><td>1</td></tr>
<tr><td><code>conv3.1.0</code></td><td>17257</td><td>64</td><td>22.8241</td><td>554</td><td>554</td><td>61</td><td>3.4717</td><td>554</td><td>790</td><td>61</td><td>5.1592</td><td>1048</td><td>1174</td><td>1</td></tr>
<tr><td><code>conv3.2.0</code></td><td>17257</td><td>64</td><td>22.7417</td><td>552</td><td>552</td><td>61</td><td>3.4849</td><td>552</td><td>793</td><td>61</td><td>5.1548</td><td>1051</td><td>1173</td><td>1</td></tr>
<tr><td><code>conv4.0.0</code></td><td>8204</td><td>64</td><td>7.0450</td><td>171</td><td>171</td><td>29</td><td>1.2437</td><td>171</td><td>283</td><td>29</td><td>2.3467</td><td>447</td><td>534</td><td>0</td></tr>
<tr><td><code>conv4.1.0</code></td><td>8204</td><td>64</td><td>7.0450</td><td>171</td><td>171</td><td>29</td><td>1.2437</td><td>171</td><td>283</td><td>29</td><td>2.3467</td><td>447</td><td>534</td><td>0</td></tr>
<tr><td><code>conv4.2.0</code></td><td>8204</td><td>64</td><td>6.4682</td><td>157</td><td>157</td><td>20</td><td>0.9888</td><td>157</td><td>225</td><td>20</td><td>1.7139</td><td>390</td><td>390</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>49.2874</strong></td><td><strong>3589</strong></td><td><strong>3589</strong></td><td><strong>68</strong></td><td><strong>5.2720</strong></td><td><strong>3589</strong></td><td><strong>3599</strong></td><td><strong>68</strong></td><td><strong>5.1592</strong></td><td><strong>1634</strong></td><td><strong>1898</strong></td><td><strong>32</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **88.91%**，将 hash entry 峰值降低 **46.18%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-3.65%**，将 hash entry 峰值降低 **46.32%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 93.0817 | 9.9551 | 10.3184 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 6778 | 6796 | 3648 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000011`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000011.bin`
- 输入体素: `16158`，坐标 SHA-256 `185caa889f5567a85c652cc6f4ef0992266c2232c4e1154ab2c476ec2d249bf6`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>16158</td><td>4</td><td>23.1812</td><td>3376</td><td>3376</td><td>0</td><td>3.3086</td><td>3376</td><td>3388</td><td>0</td><td>1.2285</td><td>1203</td><td>1258</td><td>12</td></tr>
<tr><td><code>conv_input.0</code></td><td>16158</td><td>16</td><td>48.6694</td><td>3544</td><td>3544</td><td>0</td><td>5.2119</td><td>3544</td><td>3558</td><td>0</td><td>1.7710</td><td>1158</td><td>1209</td><td>11</td></tr>
<tr><td><code>conv1.0.0</code></td><td>16158</td><td>16</td><td>46.3623</td><td>3376</td><td>3376</td><td>0</td><td>4.9629</td><td>3376</td><td>3388</td><td>0</td><td>1.8428</td><td>1203</td><td>1258</td><td>12</td></tr>
<tr><td><code>conv2.0.0</code></td><td>27314</td><td>32</td><td>36.0031</td><td>1573</td><td>1573</td><td>67</td><td>4.6826</td><td>1573</td><td>1918</td><td>67</td><td>4.3677</td><td>1529</td><td>1789</td><td>44</td></tr>
<tr><td><code>conv2.1.0</code></td><td>27314</td><td>32</td><td>36.0031</td><td>1573</td><td>1573</td><td>67</td><td>4.6826</td><td>1573</td><td>1918</td><td>67</td><td>4.3677</td><td>1529</td><td>1789</td><td>44</td></tr>
<tr><td><code>conv2.2.0</code></td><td>27314</td><td>32</td><td>32.0892</td><td>1402</td><td>1402</td><td>54</td><td>4.1235</td><td>1402</td><td>1689</td><td>54</td><td>4.5410</td><td>1594</td><td>1860</td><td>46</td></tr>
<tr><td><code>conv3.0.0</code></td><td>17536</td><td>64</td><td>19.2398</td><td>467</td><td>467</td><td>72</td><td>3.1992</td><td>467</td><td>728</td><td>72</td><td>5.3042</td><td>1040</td><td>1207</td><td>13</td></tr>
<tr><td><code>conv3.1.0</code></td><td>17536</td><td>64</td><td>19.2398</td><td>467</td><td>467</td><td>72</td><td>3.1992</td><td>467</td><td>728</td><td>72</td><td>5.3042</td><td>1040</td><td>1207</td><td>13</td></tr>
<tr><td><code>conv3.2.0</code></td><td>17536</td><td>64</td><td>20.1050</td><td>488</td><td>488</td><td>68</td><td>3.2871</td><td>488</td><td>748</td><td>68</td><td>5.2646</td><td>1046</td><td>1198</td><td>13</td></tr>
<tr><td><code>conv4.0.0</code></td><td>8080</td><td>64</td><td>4.4907</td><td>109</td><td>109</td><td>40</td><td>1.0635</td><td>109</td><td>242</td><td>40</td><td>1.9336</td><td>335</td><td>440</td><td>4</td></tr>
<tr><td><code>conv4.1.0</code></td><td>8080</td><td>64</td><td>4.4907</td><td>109</td><td>109</td><td>40</td><td>1.0635</td><td>109</td><td>242</td><td>40</td><td>1.9336</td><td>335</td><td>440</td><td>4</td></tr>
<tr><td><code>conv4.2.0</code></td><td>8080</td><td>64</td><td>4.2023</td><td>102</td><td>102</td><td>25</td><td>0.8218</td><td>102</td><td>187</td><td>25</td><td>1.3843</td><td>304</td><td>315</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>48.6694</strong></td><td><strong>3544</strong></td><td><strong>3544</strong></td><td><strong>72</strong></td><td><strong>5.2119</strong></td><td><strong>3544</strong></td><td><strong>3558</strong></td><td><strong>72</strong></td><td><strong>5.3042</strong></td><td><strong>1594</strong></td><td><strong>1860</strong></td><td><strong>46</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **88.84%**，将 hash entry 峰值降低 **47.27%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-4.26%**，将 hash entry 峰值降低 **47.47%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 95.0317 | 10.1748 | 10.6084 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 6920 | 6946 | 3649 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000012`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000012.bin`
- 输入体素: `14839`，坐标 SHA-256 `21bcbb8e68fba77fadb0d23e30fd6f2a3978d37ee2a5a4e7ce941629451ab71d`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>14839</td><td>4</td><td>38.3148</td><td>5580</td><td>5580</td><td>1</td><td>5.4551</td><td>5580</td><td>5586</td><td>1</td><td>1.0840</td><td>1085</td><td>1110</td><td>4</td></tr>
<tr><td><code>conv_input.0</code></td><td>14839</td><td>16</td><td>68.3624</td><td>4978</td><td>4978</td><td>1</td><td>7.3037</td><td>4978</td><td>4986</td><td>1</td><td>1.6304</td><td>1089</td><td>1113</td><td>5</td></tr>
<tr><td><code>conv1.0.0</code></td><td>14839</td><td>16</td><td>76.6296</td><td>5580</td><td>5580</td><td>1</td><td>8.1826</td><td>5580</td><td>5586</td><td>1</td><td>1.6260</td><td>1085</td><td>1110</td><td>4</td></tr>
<tr><td><code>conv2.0.0</code></td><td>29088</td><td>32</td><td>57.9987</td><td>2534</td><td>2534</td><td>13</td><td>6.6162</td><td>2534</td><td>2710</td><td>13</td><td>4.0039</td><td>1481</td><td>1640</td><td>25</td></tr>
<tr><td><code>conv2.1.0</code></td><td>29088</td><td>32</td><td>57.9987</td><td>2534</td><td>2534</td><td>13</td><td>6.6162</td><td>2534</td><td>2710</td><td>13</td><td>4.0039</td><td>1481</td><td>1640</td><td>25</td></tr>
<tr><td><code>conv2.2.0</code></td><td>29088</td><td>32</td><td>64.4531</td><td>2816</td><td>2816</td><td>9</td><td>7.4609</td><td>2816</td><td>3056</td><td>9</td><td>4.1309</td><td>1529</td><td>1692</td><td>25</td></tr>
<tr><td><code>conv3.0.0</code></td><td>23444</td><td>64</td><td>27.8503</td><td>676</td><td>676</td><td>66</td><td>4.1528</td><td>676</td><td>945</td><td>66</td><td>6.7896</td><td>1375</td><td>1545</td><td>35</td></tr>
<tr><td><code>conv3.1.0</code></td><td>23444</td><td>64</td><td>27.8503</td><td>676</td><td>676</td><td>66</td><td>4.1528</td><td>676</td><td>945</td><td>66</td><td>6.7896</td><td>1375</td><td>1545</td><td>35</td></tr>
<tr><td><code>conv3.2.0</code></td><td>23444</td><td>64</td><td>27.3148</td><td>663</td><td>663</td><td>68</td><td>4.1221</td><td>663</td><td>938</td><td>68</td><td>6.6709</td><td>1342</td><td>1518</td><td>39</td></tr>
<tr><td><code>conv4.0.0</code></td><td>11761</td><td>64</td><td>7.8690</td><td>191</td><td>191</td><td>46</td><td>1.6084</td><td>191</td><td>366</td><td>46</td><td>2.7949</td><td>523</td><td>636</td><td>18</td></tr>
<tr><td><code>conv4.1.0</code></td><td>11761</td><td>64</td><td>7.8690</td><td>191</td><td>191</td><td>46</td><td>1.6084</td><td>191</td><td>366</td><td>46</td><td>2.7949</td><td>523</td><td>636</td><td>18</td></tr>
<tr><td><code>conv4.2.0</code></td><td>11761</td><td>64</td><td>7.2922</td><td>177</td><td>177</td><td>21</td><td>1.2129</td><td>177</td><td>276</td><td>21</td><td>2.1841</td><td>467</td><td>497</td><td>5</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>76.6296</strong></td><td><strong>5580</strong></td><td><strong>5580</strong></td><td><strong>68</strong></td><td><strong>8.1826</strong></td><td><strong>5580</strong></td><td><strong>5586</strong></td><td><strong>68</strong></td><td><strong>6.7896</strong></td><td><strong>1529</strong></td><td><strong>1692</strong></td><td><strong>39</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **90.63%**，将 hash entry 峰值降低 **68.44%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **12.32%**，将 hash entry 峰值降低 **68.48%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 144.9921 | 15.4863 | 13.5791 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 10558 | 10572 | 3332 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000013`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000013.bin`
- 输入体素: `17054`，坐标 SHA-256 `22ae83bd6c749e0183174d2fcbfb88e9fc623564d066821379d3cbb0dca6ef33`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>17054</td><td>4</td><td>36.6394</td><td>5336</td><td>5336</td><td>0</td><td>5.2236</td><td>5336</td><td>5349</td><td>0</td><td>1.4502</td><td>1441</td><td>1485</td><td>4</td></tr>
<tr><td><code>conv_input.0</code></td><td>17054</td><td>16</td><td>77.1927</td><td>5621</td><td>5621</td><td>0</td><td>8.2500</td><td>5621</td><td>5632</td><td>0</td><td>2.1108</td><td>1395</td><td>1441</td><td>5</td></tr>
<tr><td><code>conv1.0.0</code></td><td>17054</td><td>16</td><td>73.2788</td><td>5336</td><td>5336</td><td>0</td><td>7.8354</td><td>5336</td><td>5349</td><td>0</td><td>2.1753</td><td>1441</td><td>1485</td><td>4</td></tr>
<tr><td><code>conv2.0.0</code></td><td>35690</td><td>32</td><td>58.9828</td><td>2577</td><td>2577</td><td>115</td><td>7.5928</td><td>2577</td><td>3110</td><td>115</td><td>5.5371</td><td>1856</td><td>2268</td><td>74</td></tr>
<tr><td><code>conv2.1.0</code></td><td>35690</td><td>32</td><td>58.9828</td><td>2577</td><td>2577</td><td>115</td><td>7.5928</td><td>2577</td><td>3110</td><td>115</td><td>5.5371</td><td>1856</td><td>2268</td><td>74</td></tr>
<tr><td><code>conv2.2.0</code></td><td>35690</td><td>32</td><td>55.9387</td><td>2444</td><td>2444</td><td>109</td><td>7.2192</td><td>2444</td><td>2957</td><td>109</td><td>6.1768</td><td>2115</td><td>2530</td><td>80</td></tr>
<tr><td><code>conv3.0.0</code></td><td>25945</td><td>64</td><td>28.0975</td><td>682</td><td>682</td><td>110</td><td>4.9087</td><td>682</td><td>1117</td><td>110</td><td>7.8706</td><td>1421</td><td>1791</td><td>69</td></tr>
<tr><td><code>conv3.1.0</code></td><td>25945</td><td>64</td><td>28.0975</td><td>682</td><td>682</td><td>110</td><td>4.9087</td><td>682</td><td>1117</td><td>110</td><td>7.8706</td><td>1421</td><td>1791</td><td>69</td></tr>
<tr><td><code>conv3.2.0</code></td><td>25945</td><td>64</td><td>27.7267</td><td>673</td><td>673</td><td>108</td><td>4.8428</td><td>673</td><td>1102</td><td>108</td><td>7.7212</td><td>1393</td><td>1757</td><td>63</td></tr>
<tr><td><code>conv4.0.0</code></td><td>11907</td><td>64</td><td>5.8502</td><td>142</td><td>142</td><td>49</td><td>1.5117</td><td>142</td><td>344</td><td>49</td><td>2.6543</td><td>410</td><td>604</td><td>30</td></tr>
<tr><td><code>conv4.1.0</code></td><td>11907</td><td>64</td><td>5.8502</td><td>142</td><td>142</td><td>49</td><td>1.5117</td><td>142</td><td>344</td><td>49</td><td>2.6543</td><td>410</td><td>604</td><td>30</td></tr>
<tr><td><code>conv4.2.0</code></td><td>11907</td><td>64</td><td>5.4794</td><td>133</td><td>133</td><td>32</td><td>1.1294</td><td>133</td><td>257</td><td>32</td><td>1.7666</td><td>374</td><td>402</td><td>1</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>77.1927</strong></td><td><strong>5621</strong></td><td><strong>5621</strong></td><td><strong>115</strong></td><td><strong>8.2500</strong></td><td><strong>5621</strong></td><td><strong>5632</strong></td><td><strong>115</strong></td><td><strong>7.8706</strong></td><td><strong>2115</strong></td><td><strong>2530</strong></td><td><strong>80</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.54%**，将 hash entry 峰值降低 **56.21%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **2.14%**，将 hash entry 峰值降低 **56.31%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 150.4715 | 16.0854 | 15.7412 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 10957 | 10981 | 4798 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000014`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000014.bin`
- 输入体素: `17045`，坐标 SHA-256 `4a79eedf805c4f2d1880ad4da0a75d4cc90469c8094d06d2a4c45d6c6c8a8a40`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>17045</td><td>4</td><td>35.7674</td><td>5209</td><td>5209</td><td>0</td><td>5.0898</td><td>5209</td><td>5212</td><td>0</td><td>1.3477</td><td>1344</td><td>1380</td><td>4</td></tr>
<tr><td><code>conv_input.0</code></td><td>17045</td><td>16</td><td>77.7557</td><td>5662</td><td>5662</td><td>0</td><td>8.3027</td><td>5662</td><td>5668</td><td>0</td><td>2.0039</td><td>1332</td><td>1368</td><td>5</td></tr>
<tr><td><code>conv1.0.0</code></td><td>17045</td><td>16</td><td>71.5347</td><td>5209</td><td>5209</td><td>0</td><td>7.6348</td><td>5209</td><td>5212</td><td>0</td><td>2.0215</td><td>1344</td><td>1380</td><td>4</td></tr>
<tr><td><code>conv2.0.0</code></td><td>36802</td><td>32</td><td>54.5654</td><td>2384</td><td>2384</td><td>130</td><td>7.2705</td><td>2384</td><td>2978</td><td>130</td><td>5.6958</td><td>1928</td><td>2333</td><td>63</td></tr>
<tr><td><code>conv2.1.0</code></td><td>36802</td><td>32</td><td>54.5654</td><td>2384</td><td>2384</td><td>130</td><td>7.2705</td><td>2384</td><td>2978</td><td>130</td><td>5.6958</td><td>1928</td><td>2333</td><td>63</td></tr>
<tr><td><code>conv2.2.0</code></td><td>36802</td><td>32</td><td>50.8118</td><td>2220</td><td>2220</td><td>120</td><td>6.6528</td><td>2220</td><td>2725</td><td>120</td><td>6.3330</td><td>2161</td><td>2594</td><td>65</td></tr>
<tr><td><code>conv3.0.0</code></td><td>27033</td><td>64</td><td>29.1275</td><td>707</td><td>707</td><td>124</td><td>5.1504</td><td>707</td><td>1172</td><td>124</td><td>8.1914</td><td>1490</td><td>1864</td><td>55</td></tr>
<tr><td><code>conv3.1.0</code></td><td>27033</td><td>64</td><td>29.1275</td><td>707</td><td>707</td><td>124</td><td>5.1504</td><td>707</td><td>1172</td><td>124</td><td>8.1914</td><td>1490</td><td>1864</td><td>55</td></tr>
<tr><td><code>conv3.2.0</code></td><td>27033</td><td>64</td><td>29.2099</td><td>709</td><td>709</td><td>122</td><td>5.1592</td><td>709</td><td>1174</td><td>122</td><td>8.0420</td><td>1456</td><td>1830</td><td>49</td></tr>
<tr><td><code>conv4.0.0</code></td><td>12099</td><td>64</td><td>6.0150</td><td>146</td><td>146</td><td>57</td><td>1.5645</td><td>146</td><td>356</td><td>57</td><td>2.7598</td><td>429</td><td>628</td><td>13</td></tr>
<tr><td><code>conv4.1.0</code></td><td>12099</td><td>64</td><td>6.0150</td><td>146</td><td>146</td><td>57</td><td>1.5645</td><td>146</td><td>356</td><td>57</td><td>2.7598</td><td>429</td><td>628</td><td>13</td></tr>
<tr><td><code>conv4.2.0</code></td><td>12099</td><td>64</td><td>5.8502</td><td>142</td><td>142</td><td>36</td><td>1.1865</td><td>142</td><td>270</td><td>36</td><td>1.8149</td><td>397</td><td>413</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>77.7557</strong></td><td><strong>5662</strong></td><td><strong>5662</strong></td><td><strong>130</strong></td><td><strong>8.3027</strong></td><td><strong>5662</strong></td><td><strong>5668</strong></td><td><strong>130</strong></td><td><strong>8.1914</strong></td><td><strong>2161</strong></td><td><strong>2594</strong></td><td><strong>65</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.03%**，将 hash entry 峰值降低 **54.68%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-2.79%**，将 hash entry 峰值降低 **54.72%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 149.2905 | 15.9375 | 16.3828 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 10871 | 10880 | 4927 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000015`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000015.bin`
- 输入体素: `14241`，坐标 SHA-256 `82949cbe297835536b61590e42bcb13d5d878f02dac28dd4999963f0796960c8`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>14241</td><td>4</td><td>17.2760</td><td>2516</td><td>2516</td><td>0</td><td>2.5020</td><td>2516</td><td>2562</td><td>0</td><td>1.0732</td><td>1028</td><td>1099</td><td>15</td></tr>
<tr><td><code>conv_input.0</code></td><td>14241</td><td>16</td><td>38.5208</td><td>2805</td><td>2805</td><td>1</td><td>4.1704</td><td>2805</td><td>2847</td><td>1</td><td>1.6201</td><td>1033</td><td>1106</td><td>15</td></tr>
<tr><td><code>conv1.0.0</code></td><td>14241</td><td>16</td><td>34.5520</td><td>2516</td><td>2516</td><td>0</td><td>3.7529</td><td>2516</td><td>2562</td><td>0</td><td>1.6099</td><td>1028</td><td>1099</td><td>15</td></tr>
<tr><td><code>conv2.0.0</code></td><td>22176</td><td>32</td><td>30.9219</td><td>1351</td><td>1351</td><td>84</td><td>4.1846</td><td>1351</td><td>1714</td><td>84</td><td>3.4497</td><td>1136</td><td>1413</td><td>53</td></tr>
<tr><td><code>conv2.1.0</code></td><td>22176</td><td>32</td><td>30.9219</td><td>1351</td><td>1351</td><td>84</td><td>4.1846</td><td>1351</td><td>1714</td><td>84</td><td>3.4497</td><td>1136</td><td>1413</td><td>53</td></tr>
<tr><td><code>conv2.2.0</code></td><td>22176</td><td>32</td><td>28.1067</td><td>1228</td><td>1228</td><td>68</td><td>3.7671</td><td>1228</td><td>1543</td><td>68</td><td>3.9136</td><td>1304</td><td>1603</td><td>66</td></tr>
<tr><td><code>conv3.0.0</code></td><td>13613</td><td>64</td><td>19.4870</td><td>473</td><td>473</td><td>54</td><td>3.0542</td><td>473</td><td>695</td><td>54</td><td>5.3525</td><td>1060</td><td>1218</td><td>24</td></tr>
<tr><td><code>conv3.1.0</code></td><td>13613</td><td>64</td><td>19.4870</td><td>473</td><td>473</td><td>54</td><td>3.0542</td><td>473</td><td>695</td><td>54</td><td>5.3525</td><td>1060</td><td>1218</td><td>24</td></tr>
<tr><td><code>conv3.2.0</code></td><td>13613</td><td>64</td><td>18.8690</td><td>458</td><td>458</td><td>54</td><td>3.0103</td><td>458</td><td>685</td><td>54</td><td>5.2778</td><td>1043</td><td>1201</td><td>26</td></tr>
<tr><td><code>conv4.0.0</code></td><td>5931</td><td>64</td><td>4.4495</td><td>108</td><td>108</td><td>22</td><td>0.8701</td><td>108</td><td>198</td><td>22</td><td>1.7314</td><td>314</td><td>394</td><td>1</td></tr>
<tr><td><code>conv4.1.0</code></td><td>5931</td><td>64</td><td>4.4495</td><td>108</td><td>108</td><td>22</td><td>0.8701</td><td>108</td><td>198</td><td>22</td><td>1.7314</td><td>314</td><td>394</td><td>1</td></tr>
<tr><td><code>conv4.2.0</code></td><td>5931</td><td>64</td><td>4.0375</td><td>98</td><td>98</td><td>18</td><td>0.6724</td><td>98</td><td>153</td><td>18</td><td>1.1514</td><td>259</td><td>262</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>38.5208</strong></td><td><strong>2805</strong></td><td><strong>2805</strong></td><td><strong>84</strong></td><td><strong>4.1846</strong></td><td><strong>2805</strong></td><td><strong>2847</strong></td><td><strong>84</strong></td><td><strong>5.3525</strong></td><td><strong>1304</strong></td><td><strong>1603</strong></td><td><strong>66</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **85.35%**，将 hash entry 峰值降低 **43.32%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-27.91%**，将 hash entry 峰值降低 **44.24%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 73.0728 | 8.3691 | 10.7051 |
| DRAM 峰值层 | `conv1.0.0` | `conv2.1.0` | `conv3.1.0` |
| Hash entries | 5321 | 5409 | 3016 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000016`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000016.bin`
- 输入体素: `14000`，坐标 SHA-256 `71ae4975910284b27b6a2019f5186a5340dbdf0acc0bf0e1f8b930fe0cde2837`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>14000</td><td>4</td><td>23.5039</td><td>3423</td><td>3423</td><td>1</td><td>3.3613</td><td>3423</td><td>3442</td><td>1</td><td>1.1201</td><td>1105</td><td>1147</td><td>7</td></tr>
<tr><td><code>conv_input.0</code></td><td>14000</td><td>16</td><td>48.5870</td><td>3538</td><td>3538</td><td>1</td><td>5.2104</td><td>3538</td><td>3557</td><td>1</td><td>1.7124</td><td>1130</td><td>1169</td><td>7</td></tr>
<tr><td><code>conv1.0.0</code></td><td>14000</td><td>16</td><td>47.0078</td><td>3423</td><td>3423</td><td>1</td><td>5.0420</td><td>3423</td><td>3442</td><td>1</td><td>1.6802</td><td>1105</td><td>1147</td><td>7</td></tr>
<tr><td><code>conv2.0.0</code></td><td>24716</td><td>32</td><td>37.1475</td><td>1623</td><td>1623</td><td>67</td><td>4.6240</td><td>1623</td><td>1894</td><td>67</td><td>4.4507</td><td>1568</td><td>1823</td><td>47</td></tr>
<tr><td><code>conv2.1.0</code></td><td>24716</td><td>32</td><td>37.1475</td><td>1623</td><td>1623</td><td>67</td><td>4.6240</td><td>1623</td><td>1894</td><td>67</td><td>4.4507</td><td>1568</td><td>1823</td><td>47</td></tr>
<tr><td><code>conv2.2.0</code></td><td>24716</td><td>32</td><td>36.5753</td><td>1598</td><td>1598</td><td>49</td><td>4.4629</td><td>1598</td><td>1828</td><td>49</td><td>4.2896</td><td>1515</td><td>1757</td><td>45</td></tr>
<tr><td><code>conv3.0.0</code></td><td>17597</td><td>64</td><td>27.6855</td><td>672</td><td>672</td><td>60</td><td>3.9287</td><td>672</td><td>894</td><td>60</td><td>5.3525</td><td>1114</td><td>1218</td><td>9</td></tr>
<tr><td><code>conv3.1.0</code></td><td>17597</td><td>64</td><td>27.6855</td><td>672</td><td>672</td><td>60</td><td>3.9287</td><td>672</td><td>894</td><td>60</td><td>5.3525</td><td>1114</td><td>1218</td><td>9</td></tr>
<tr><td><code>conv3.2.0</code></td><td>17597</td><td>64</td><td>27.5208</td><td>668</td><td>668</td><td>60</td><td>3.9023</td><td>668</td><td>888</td><td>60</td><td>5.3130</td><td>1103</td><td>1209</td><td>9</td></tr>
<tr><td><code>conv4.0.0</code></td><td>8755</td><td>64</td><td>7.5394</td><td>183</td><td>183</td><td>33</td><td>1.3403</td><td>183</td><td>305</td><td>33</td><td>2.3687</td><td>449</td><td>539</td><td>1</td></tr>
<tr><td><code>conv4.1.0</code></td><td>8755</td><td>64</td><td>7.5394</td><td>183</td><td>183</td><td>33</td><td>1.3403</td><td>183</td><td>305</td><td>33</td><td>2.3687</td><td>449</td><td>539</td><td>1</td></tr>
<tr><td><code>conv4.2.0</code></td><td>8755</td><td>64</td><td>6.8390</td><td>166</td><td>166</td><td>23</td><td>1.0723</td><td>166</td><td>244</td><td>23</td><td>1.7446</td><td>394</td><td>397</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>48.5870</strong></td><td><strong>3538</strong></td><td><strong>3538</strong></td><td><strong>67</strong></td><td><strong>5.2104</strong></td><td><strong>3538</strong></td><td><strong>3557</strong></td><td><strong>67</strong></td><td><strong>5.3525</strong></td><td><strong>1568</strong></td><td><strong>1823</strong></td><td><strong>47</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **88.80%**，将 hash entry 峰值降低 **47.62%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-4.41%**，将 hash entry 峰值降低 **47.91%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 95.5948 | 10.2524 | 10.7051 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 6961 | 6999 | 3646 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.1.0` |

## Frame `train/000017`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000017.bin`
- 输入体素: `14853`，坐标 SHA-256 `0bea72489a88d54dd5866b8b824bc9c721fd691dabd3a374f91bf84aebc3525d`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>14853</td><td>4</td><td>26.9714</td><td>3928</td><td>3928</td><td>1</td><td>3.8398</td><td>3928</td><td>3932</td><td>1</td><td>1.1846</td><td>1195</td><td>1213</td><td>2</td></tr>
<tr><td><code>conv_input.0</code></td><td>14853</td><td>16</td><td>62.7594</td><td>4570</td><td>4570</td><td>0</td><td>6.6987</td><td>4570</td><td>4573</td><td>0</td><td>1.6875</td><td>1138</td><td>1152</td><td>3</td></tr>
<tr><td><code>conv1.0.0</code></td><td>14853</td><td>16</td><td>53.9429</td><td>3928</td><td>3928</td><td>1</td><td>5.7598</td><td>3928</td><td>3932</td><td>1</td><td>1.7769</td><td>1195</td><td>1213</td><td>2</td></tr>
<tr><td><code>conv2.0.0</code></td><td>28471</td><td>32</td><td>54.1763</td><td>2367</td><td>2367</td><td>60</td><td>6.5234</td><td>2367</td><td>2672</td><td>60</td><td>4.8560</td><td>1798</td><td>1989</td><td>29</td></tr>
<tr><td><code>conv2.1.0</code></td><td>28471</td><td>32</td><td>54.1763</td><td>2367</td><td>2367</td><td>60</td><td>6.5234</td><td>2367</td><td>2672</td><td>60</td><td>4.8560</td><td>1798</td><td>1989</td><td>29</td></tr>
<tr><td><code>conv2.2.0</code></td><td>28471</td><td>32</td><td>44.8151</td><td>1958</td><td>1958</td><td>33</td><td>5.2637</td><td>1958</td><td>2156</td><td>33</td><td>6.0718</td><td>2285</td><td>2487</td><td>30</td></tr>
<tr><td><code>conv3.0.0</code></td><td>22775</td><td>64</td><td>28.1387</td><td>683</td><td>683</td><td>70</td><td>4.2056</td><td>683</td><td>957</td><td>70</td><td>6.6445</td><td>1357</td><td>1512</td><td>26</td></tr>
<tr><td><code>conv3.1.0</code></td><td>22775</td><td>64</td><td>28.1387</td><td>683</td><td>683</td><td>70</td><td>4.2056</td><td>683</td><td>957</td><td>70</td><td>6.6445</td><td>1357</td><td>1512</td><td>26</td></tr>
<tr><td><code>conv3.2.0</code></td><td>22775</td><td>64</td><td>28.3035</td><td>687</td><td>687</td><td>74</td><td>4.2671</td><td>687</td><td>971</td><td>74</td><td>6.6929</td><td>1359</td><td>1523</td><td>25</td></tr>
<tr><td><code>conv4.0.0</code></td><td>11354</td><td>64</td><td>8.0750</td><td>196</td><td>196</td><td>44</td><td>1.5996</td><td>196</td><td>364</td><td>44</td><td>2.7729</td><td>516</td><td>631</td><td>10</td></tr>
<tr><td><code>conv4.1.0</code></td><td>11354</td><td>64</td><td>8.0750</td><td>196</td><td>196</td><td>44</td><td>1.5996</td><td>196</td><td>364</td><td>44</td><td>2.7729</td><td>516</td><td>631</td><td>10</td></tr>
<tr><td><code>conv4.2.0</code></td><td>11354</td><td>64</td><td>7.4570</td><td>181</td><td>181</td><td>23</td><td>1.2173</td><td>181</td><td>277</td><td>23</td><td>2.1226</td><td>466</td><td>483</td><td>2</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>62.7594</strong></td><td><strong>4570</strong></td><td><strong>4570</strong></td><td><strong>74</strong></td><td><strong>6.6987</strong></td><td><strong>4570</strong></td><td><strong>4573</strong></td><td><strong>74</strong></td><td><strong>6.6929</strong></td><td><strong>2285</strong></td><td><strong>2487</strong></td><td><strong>30</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **88.57%**，将 hash entry 峰值降低 **47.33%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-2.23%**，将 hash entry 峰值降低 **47.37%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 116.7023 | 13.0469 | 13.3374 |
| DRAM 峰值层 | `conv1.0.0` | `conv2.1.0` | `conv3.2.0` |
| Hash entries | 8498 | 8505 | 4476 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000018`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000018.bin`
- 输入体素: `14889`，坐标 SHA-256 `946a904b81ca403867446bd9d3fb87cf37185bc908e3215837b47fefa39c5383`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>14889</td><td>4</td><td>33.9615</td><td>4946</td><td>4946</td><td>0</td><td>4.8320</td><td>4946</td><td>4948</td><td>0</td><td>1.1348</td><td>1142</td><td>1162</td><td>2</td></tr>
<tr><td><code>conv_input.0</code></td><td>14889</td><td>16</td><td>69.8181</td><td>5084</td><td>5084</td><td>0</td><td>7.4517</td><td>5084</td><td>5087</td><td>0</td><td>1.6699</td><td>1124</td><td>1140</td><td>2</td></tr>
<tr><td><code>conv1.0.0</code></td><td>14889</td><td>16</td><td>67.9230</td><td>4946</td><td>4946</td><td>0</td><td>7.2480</td><td>4946</td><td>4948</td><td>0</td><td>1.7021</td><td>1142</td><td>1162</td><td>2</td></tr>
<tr><td><code>conv2.0.0</code></td><td>30596</td><td>32</td><td>49.0265</td><td>2142</td><td>2142</td><td>66</td><td>5.9985</td><td>2142</td><td>2457</td><td>66</td><td>4.2188</td><td>1520</td><td>1728</td><td>26</td></tr>
<tr><td><code>conv2.1.0</code></td><td>30596</td><td>32</td><td>49.0265</td><td>2142</td><td>2142</td><td>66</td><td>5.9985</td><td>2142</td><td>2457</td><td>66</td><td>4.2188</td><td>1520</td><td>1728</td><td>26</td></tr>
<tr><td><code>conv2.2.0</code></td><td>30596</td><td>32</td><td>54.6570</td><td>2388</td><td>2388</td><td>46</td><td>6.5332</td><td>2388</td><td>2676</td><td>46</td><td>4.5923</td><td>1654</td><td>1881</td><td>27</td></tr>
<tr><td><code>conv3.0.0</code></td><td>23904</td><td>64</td><td>29.7043</td><td>721</td><td>721</td><td>71</td><td>4.4692</td><td>721</td><td>1017</td><td>71</td><td>7.0664</td><td>1420</td><td>1608</td><td>25</td></tr>
<tr><td><code>conv3.1.0</code></td><td>23904</td><td>64</td><td>29.7043</td><td>721</td><td>721</td><td>71</td><td>4.4692</td><td>721</td><td>1017</td><td>71</td><td>7.0664</td><td>1420</td><td>1608</td><td>25</td></tr>
<tr><td><code>conv3.2.0</code></td><td>23904</td><td>64</td><td>29.3747</td><td>713</td><td>713</td><td>72</td><td>4.4385</td><td>713</td><td>1010</td><td>72</td><td>6.9873</td><td>1398</td><td>1590</td><td>26</td></tr>
<tr><td><code>conv4.0.0</code></td><td>11936</td><td>64</td><td>9.4345</td><td>229</td><td>229</td><td>44</td><td>1.7358</td><td>229</td><td>395</td><td>44</td><td>2.9795</td><td>553</td><td>678</td><td>12</td></tr>
<tr><td><code>conv4.1.0</code></td><td>11936</td><td>64</td><td>9.4345</td><td>229</td><td>229</td><td>44</td><td>1.7358</td><td>229</td><td>395</td><td>44</td><td>2.9795</td><td>553</td><td>678</td><td>12</td></tr>
<tr><td><code>conv4.2.0</code></td><td>11936</td><td>64</td><td>8.5693</td><td>208</td><td>208</td><td>24</td><td>1.3403</td><td>208</td><td>305</td><td>24</td><td>2.2852</td><td>497</td><td>520</td><td>2</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>69.8181</strong></td><td><strong>5084</strong></td><td><strong>5084</strong></td><td><strong>72</strong></td><td><strong>7.4517</strong></td><td><strong>5084</strong></td><td><strong>5087</strong></td><td><strong>72</strong></td><td><strong>7.0664</strong></td><td><strong>1654</strong></td><td><strong>1881</strong></td><td><strong>27</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.74%**，将 hash entry 峰值降低 **64.02%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **3.86%**，将 hash entry 峰值降低 **64.04%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 137.7411 | 14.6997 | 14.1328 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 10030 | 10035 | 3609 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000019`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000019.bin`
- 输入体素: `13435`，坐标 SHA-256 `6ac4b0174cdbef63775b7146144b5470143e0adfe65f5609a3effaa97ea868f0`
- 触达 40000 voxel 上限: `False`

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
<tr><td>初始输入</td><td>13435</td><td>4</td><td>16.8503</td><td>2454</td><td>2454</td><td>9</td><td>2.4678</td><td>2454</td><td>2527</td><td>9</td><td>0.8408</td><td>769</td><td>861</td><td>23</td></tr>
<tr><td><code>conv_input.0</code></td><td>13435</td><td>16</td><td>33.8654</td><td>2466</td><td>2466</td><td>11</td><td>3.7192</td><td>2466</td><td>2539</td><td>11</td><td>1.2173</td><td>743</td><td>831</td><td>24</td></tr>
<tr><td><code>conv1.0.0</code></td><td>13435</td><td>16</td><td>33.7006</td><td>2454</td><td>2454</td><td>9</td><td>3.7017</td><td>2454</td><td>2527</td><td>9</td><td>1.2612</td><td>769</td><td>861</td><td>23</td></tr>
<tr><td><code>conv2.0.0</code></td><td>18418</td><td>32</td><td>26.0925</td><td>1140</td><td>1140</td><td>50</td><td>3.3496</td><td>1140</td><td>1372</td><td>50</td><td>2.3999</td><td>777</td><td>983</td><td>34</td></tr>
<tr><td><code>conv2.1.0</code></td><td>18418</td><td>32</td><td>26.0925</td><td>1140</td><td>1140</td><td>50</td><td>3.3496</td><td>1140</td><td>1372</td><td>50</td><td>2.3999</td><td>777</td><td>983</td><td>34</td></tr>
<tr><td><code>conv2.2.0</code></td><td>18418</td><td>32</td><td>27.4200</td><td>1198</td><td>1198</td><td>52</td><td>3.4912</td><td>1198</td><td>1430</td><td>52</td><td>2.6318</td><td>847</td><td>1078</td><td>37</td></tr>
<tr><td><code>conv3.0.0</code></td><td>11482</td><td>64</td><td>12.8540</td><td>312</td><td>312</td><td>34</td><td>2.1577</td><td>312</td><td>491</td><td>34</td><td>3.4497</td><td>681</td><td>785</td><td>18</td></tr>
<tr><td><code>conv3.1.0</code></td><td>11482</td><td>64</td><td>12.8540</td><td>312</td><td>312</td><td>34</td><td>2.1577</td><td>312</td><td>491</td><td>34</td><td>3.4497</td><td>681</td><td>785</td><td>18</td></tr>
<tr><td><code>conv3.2.0</code></td><td>11482</td><td>64</td><td>12.8540</td><td>312</td><td>312</td><td>35</td><td>2.0962</td><td>312</td><td>477</td><td>35</td><td>3.5068</td><td>685</td><td>798</td><td>16</td></tr>
<tr><td><code>conv4.0.0</code></td><td>5258</td><td>64</td><td>3.2135</td><td>78</td><td>78</td><td>21</td><td>0.6724</td><td>78</td><td>153</td><td>21</td><td>1.1777</td><td>222</td><td>268</td><td>1</td></tr>
<tr><td><code>conv4.1.0</code></td><td>5258</td><td>64</td><td>3.2135</td><td>78</td><td>78</td><td>21</td><td>0.6724</td><td>78</td><td>153</td><td>21</td><td>1.1777</td><td>222</td><td>268</td><td>1</td></tr>
<tr><td><code>conv4.2.0</code></td><td>5258</td><td>64</td><td>3.1311</td><td>76</td><td>76</td><td>10</td><td>0.5581</td><td>76</td><td>127</td><td>10</td><td>0.8833</td><td>191</td><td>201</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>33.8654</strong></td><td><strong>2466</strong></td><td><strong>2466</strong></td><td><strong>52</strong></td><td><strong>3.7192</strong></td><td><strong>2466</strong></td><td><strong>2539</strong></td><td><strong>52</strong></td><td><strong>3.5068</strong></td><td><strong>847</strong></td><td><strong>1078</strong></td><td><strong>37</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.70%**，将 hash entry 峰值降低 **58.11%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **6.26%**，将 hash entry 峰值降低 **59.32%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 67.5659 | 7.4209 | 6.9565 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.2.0` |
| Hash entries | 4920 | 5066 | 2061 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

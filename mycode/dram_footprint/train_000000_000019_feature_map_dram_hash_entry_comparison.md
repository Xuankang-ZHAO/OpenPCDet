# 三种 Block Structuring 方案在 train 000000–000019 上的稳定性与泛化

本文件复现 `feature_map_dram_hash_entry_comparison.md` 在 KITTI `val/000216` 上的口径，
对 `accdesign/second_rtl_golden_packages/frames` 中的 20 帧补充 golden（`train/000000`–`train/000019`）做相同实验。

- 加载：KITTI FOV（`FOV_POINTS_ONLY=True`），与 golden 导出一致。
- 模型：hardware-reference INT8 SECOND 3D backbone，checkpoint `checkpoint_epoch_10.pth`。
- Halo：由下一层 kernel/padding 决定的窗口角点复制；`conv_out` 逻辑输出不分配 DRAM。
- 三种方案：固定块+固定容量（`10x10x6`、每块 600 slot）、固定块+Page、Proposed 可变块+Page。
- 上一层 OFM 与下一层 IFM 是同一 feature map，表中只统计一次。
- Hash entries：固定容量方案等于物化 block 数；两种 Page 方案等于 page 数。
- 执行时峰值按 IFM 与 OFM 同时驻留求和；DRAM 峰值层与 hash 峰值层可能不同。
- Generated: `2026-09-17T19:16:34`

## 20 帧执行时峰值总表

| Frame | 输入体素 | 触达 15000 上限 | 固定容量 DRAM | 固定块 Page DRAM | Proposed DRAM | 固定容量 Hash | 固定块 Page Hash | Proposed Hash | DRAM vs 固定容量 | Hash vs 固定容量 | DRAM vs 固定块 Page | Hash vs 固定块 Page |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `000000` | 15000 | Y | 45.1126 | 5.0781 | 6.8818 | 3285 | 3441 | 2298 | 84.75% | 30.05% | -35.52% | 33.22% |
| `000001` | 15000 | Y | 118.2678 | 12.6416 | 12.8232 | 8612 | 8630 | 4267 | 89.16% | 50.45% | -1.44% | 50.56% |
| `000002` | 14809 | N | 65.9592 | 7.3682 | 6.1787 | 4803 | 5030 | 1820 | 90.63% | 62.11% | 16.14% | 63.82% |
| `000003` | 14584 | N | 60.7819 | 6.7485 | 6.6138 | 4426 | 4607 | 1931 | 89.12% | 56.37% | 2.00% | 58.09% |
| `000004` | 15000 | Y | 132.1930 | 14.7998 | 13.8340 | 9626 | 9626 | 4490 | 89.54% | 53.36% | 6.53% | 53.36% |
| `000005` | 15000 | Y | 130.1193 | 13.8794 | 13.6978 | 9475 | 9475 | 4330 | 89.47% | 54.30% | 1.31% | 54.30% |
| `000006` | 15000 | Y | 89.5660 | 9.6211 | 12.0674 | 6522 | 6568 | 3958 | 86.53% | 39.31% | -25.43% | 39.74% |
| `000007` | 15000 | Y | 144.1269 | 15.3882 | 14.5283 | 10495 | 10505 | 4763 | 89.92% | 54.62% | 5.59% | 54.66% |
| `000008` | 13081 | N | 68.1427 | 7.3887 | 9.0439 | 4962 | 5044 | 2600 | 86.73% | 47.60% | -22.40% | 48.45% |
| `000009` | 15000 | Y | 119.7510 | 12.9297 | 13.0254 | 8720 | 8720 | 4471 | 89.12% | 48.73% | -0.74% | 48.73% |
| `000010` | 13094 | N | 93.0817 | 9.9551 | 10.3184 | 6778 | 6796 | 3648 | 88.91% | 46.18% | -3.65% | 46.32% |
| `000011` | 15000 | Y | 92.8894 | 9.9463 | 10.3623 | 6764 | 6790 | 3546 | 88.84% | 47.58% | -4.18% | 47.78% |
| `000012` | 14839 | N | 144.9921 | 15.4863 | 13.5791 | 10558 | 10572 | 3332 | 90.63% | 68.44% | 12.32% | 68.48% |
| `000013` | 15000 | Y | 145.5276 | 15.5420 | 15.3281 | 10597 | 10610 | 4639 | 89.47% | 56.22% | 1.38% | 56.28% |
| `000014` | 15000 | Y | 145.1569 | 15.4966 | 15.8730 | 10570 | 10579 | 4736 | 89.06% | 55.19% | -2.43% | 55.23% |
| `000015` | 14241 | N | 73.0728 | 8.3691 | 10.7051 | 5321 | 5409 | 3016 | 85.35% | 43.32% | -27.91% | 44.24% |
| `000016` | 14000 | N | 95.5948 | 10.2524 | 10.7051 | 6961 | 6999 | 3646 | 88.80% | 47.62% | -4.41% | 47.91% |
| `000017` | 14853 | N | 116.7023 | 13.0469 | 13.3374 | 8498 | 8505 | 4476 | 88.57% | 47.33% | -2.23% | 47.37% |
| `000018` | 14889 | N | 137.7411 | 14.6997 | 14.1328 | 10030 | 10035 | 3609 | 89.74% | 64.02% | 3.86% | 64.04% |
| `000019` | 13435 | N | 67.5659 | 7.4209 | 6.9565 | 4920 | 5066 | 2061 | 89.70% | 58.11% | 6.26% | 59.32% |

相对固定容量 / 固定块分页，Proposed 在 20 帧上的降低比例（正值表示 Proposed 更小）：

| 指标 | 最小 | 平均 | 最大 |
| --- | ---: | ---: | ---: |
| DRAM vs 固定容量 | 84.75% | 88.70% | 90.63% |
| Hash vs 固定容量 | 30.05% | 51.55% | 68.44% |
| DRAM vs 固定块 Page | -35.52% | -3.75% | 16.14% |
| Hash vs 固定块 Page | 33.22% | 52.09% | 68.48% |

## Frame `train/000000`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000000.bin`
- 输入体素: `15000`，坐标 SHA-256 `3c16a43fa65b3c311f61e432e66915b5f60b3a4d65e8723c96015c7971b69bb6`
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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>10.9245</td><td>1591</td><td>1591</td><td>2</td><td>1.6260</td><td>1591</td><td>1665</td><td>2</td><td>0.8848</td><td>757</td><td>906</td><td>34</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>23.2635</td><td>1694</td><td>1694</td><td>0</td><td>2.6016</td><td>1694</td><td>1776</td><td>0</td><td>1.2935</td><td>744</td><td>883</td><td>29</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>21.8491</td><td>1591</td><td>1591</td><td>2</td><td>2.4390</td><td>1591</td><td>1665</td><td>2</td><td>1.3271</td><td>757</td><td>906</td><td>34</td></tr>
<tr><td><code>conv2.0.0</code></td><td>19290</td><td>32</td><td>15.5869</td><td>681</td><td>681</td><td>77</td><td>2.5391</td><td>681</td><td>1040</td><td>77</td><td>2.7051</td><td>765</td><td>1108</td><td>66</td></tr>
<tr><td><code>conv2.1.0</code></td><td>19290</td><td>32</td><td>15.5869</td><td>681</td><td>681</td><td>77</td><td>2.5391</td><td>681</td><td>1040</td><td>77</td><td>2.7051</td><td>765</td><td>1108</td><td>66</td></tr>
<tr><td><code>conv2.2.0</code></td><td>19290</td><td>32</td><td>14.1220</td><td>617</td><td>617</td><td>78</td><td>2.3560</td><td>617</td><td>965</td><td>78</td><td>2.9053</td><td>826</td><td>1190</td><td>69</td></tr>
<tr><td><code>conv3.0.0</code></td><td>9910</td><td>64</td><td>8.3221</td><td>202</td><td>202</td><td>56</td><td>1.7710</td><td>202</td><td>403</td><td>56</td><td>3.4409</td><td>617</td><td>783</td><td>26</td></tr>
<tr><td><code>conv3.1.0</code></td><td>9910</td><td>64</td><td>8.3221</td><td>202</td><td>202</td><td>56</td><td>1.7710</td><td>202</td><td>403</td><td>56</td><td>3.4409</td><td>617</td><td>783</td><td>26</td></tr>
<tr><td><code>conv3.2.0</code></td><td>9910</td><td>64</td><td>8.6929</td><td>211</td><td>211</td><td>55</td><td>1.7886</td><td>211</td><td>407</td><td>55</td><td>3.4189</td><td>596</td><td>778</td><td>35</td></tr>
<tr><td><code>conv4.0.0</code></td><td>3337</td><td>64</td><td>2.2659</td><td>55</td><td>55</td><td>13</td><td>0.4746</td><td>55</td><td>108</td><td>13</td><td>1.0679</td><td>183</td><td>243</td><td>2</td></tr>
<tr><td><code>conv4.1.0</code></td><td>3337</td><td>64</td><td>2.2659</td><td>55</td><td>55</td><td>13</td><td>0.4746</td><td>55</td><td>108</td><td>13</td><td>1.0679</td><td>183</td><td>243</td><td>2</td></tr>
<tr><td><code>conv4.2.0</code></td><td>3337</td><td>64</td><td>2.0187</td><td>49</td><td>49</td><td>12</td><td>0.3779</td><td>49</td><td>86</td><td>12</td><td>0.6328</td><td>144</td><td>144</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>23.2635</strong></td><td><strong>1694</strong></td><td><strong>1694</strong></td><td><strong>78</strong></td><td><strong>2.6016</strong></td><td><strong>1694</strong></td><td><strong>1776</strong></td><td><strong>78</strong></td><td><strong>3.4409</strong></td><td><strong>826</strong></td><td><strong>1190</strong></td><td><strong>69</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **84.75%**，将 hash entry 峰值降低 **30.05%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-35.52%**，将 hash entry 峰值降低 **33.22%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 45.1126 | 5.0781 | 6.8818 |
| DRAM 峰值层 | `conv1.0.0` | `conv2.1.0` | `conv3.1.0` |
| Hash entries | 3285 | 3441 | 2298 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000001`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000001.bin`
- 输入体素: `15000`，坐标 SHA-256 `f78d40a815f1f6366582c8214be119dc668b37f573b44391b5d72fb75ead8189`
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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>28.6743</td><td>4176</td><td>4176</td><td>0</td><td>4.0879</td><td>4176</td><td>4186</td><td>0</td><td>1.1865</td><td>1175</td><td>1215</td><td>5</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>60.9192</td><td>4436</td><td>4436</td><td>0</td><td>6.5098</td><td>4436</td><td>4444</td><td>0</td><td>1.7300</td><td>1148</td><td>1181</td><td>5</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>57.3486</td><td>4176</td><td>4176</td><td>0</td><td>6.1318</td><td>4176</td><td>4186</td><td>0</td><td>1.7798</td><td>1175</td><td>1215</td><td>5</td></tr>
<tr><td><code>conv2.0.0</code></td><td>29804</td><td>32</td><td>45.4330</td><td>1985</td><td>1985</td><td>83</td><td>5.8423</td><td>1985</td><td>2393</td><td>83</td><td>4.9707</td><td>1723</td><td>2036</td><td>53</td></tr>
<tr><td><code>conv2.1.0</code></td><td>29804</td><td>32</td><td>45.4330</td><td>1985</td><td>1985</td><td>83</td><td>5.8423</td><td>1985</td><td>2393</td><td>83</td><td>4.9707</td><td>1723</td><td>2036</td><td>53</td></tr>
<tr><td><code>conv2.2.0</code></td><td>29804</td><td>32</td><td>42.6178</td><td>1862</td><td>1862</td><td>61</td><td>5.3125</td><td>1862</td><td>2176</td><td>61</td><td>5.4468</td><td>1903</td><td>2231</td><td>52</td></tr>
<tr><td><code>conv3.0.0</code></td><td>21705</td><td>64</td><td>22.7417</td><td>552</td><td>552</td><td>83</td><td>3.8364</td><td>552</td><td>873</td><td>83</td><td>6.4116</td><td>1254</td><td>1459</td><td>31</td></tr>
<tr><td><code>conv3.1.0</code></td><td>21705</td><td>64</td><td>22.7417</td><td>552</td><td>552</td><td>83</td><td>3.8364</td><td>552</td><td>873</td><td>83</td><td>6.4116</td><td>1254</td><td>1459</td><td>31</td></tr>
<tr><td><code>conv3.2.0</code></td><td>21705</td><td>64</td><td>23.0301</td><td>559</td><td>559</td><td>81</td><td>3.8496</td><td>559</td><td>876</td><td>81</td><td>6.3457</td><td>1243</td><td>1444</td><td>27</td></tr>
<tr><td><code>conv4.0.0</code></td><td>10564</td><td>64</td><td>5.6442</td><td>137</td><td>137</td><td>46</td><td>1.3359</td><td>137</td><td>304</td><td>46</td><td>2.4741</td><td>414</td><td>563</td><td>10</td></tr>
<tr><td><code>conv4.1.0</code></td><td>10564</td><td>64</td><td>5.6442</td><td>137</td><td>137</td><td>46</td><td>1.3359</td><td>137</td><td>304</td><td>46</td><td>2.4741</td><td>414</td><td>563</td><td>10</td></tr>
<tr><td><code>conv4.2.0</code></td><td>10564</td><td>64</td><td>5.3970</td><td>131</td><td>131</td><td>31</td><td>1.0415</td><td>131</td><td>237</td><td>31</td><td>1.7666</td><td>386</td><td>402</td><td>3</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>60.9192</strong></td><td><strong>4436</strong></td><td><strong>4436</strong></td><td><strong>83</strong></td><td><strong>6.5098</strong></td><td><strong>4436</strong></td><td><strong>4444</strong></td><td><strong>83</strong></td><td><strong>6.4116</strong></td><td><strong>1903</strong></td><td><strong>2231</strong></td><td><strong>53</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.16%**，将 hash entry 峰值降低 **50.45%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-1.44%**，将 hash entry 峰值降低 **50.56%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 118.2678 | 12.6416 | 12.8232 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 8612 | 8630 | 4267 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000002`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000002.bin`
- 输入体素: `14809`，坐标 SHA-256 `0650175ebc7f42bcd8a3cfdf0f183ce2f54b52935e0b60011e4658624fa606db`
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
- 输入体素: `15000`，坐标 SHA-256 `3060267669ef71b0d26d10faf2e617e5fd7ecc9d30122d3a61d4d4946b8c3228`
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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>30.1094</td><td>4385</td><td>4385</td><td>0</td><td>4.2822</td><td>4385</td><td>4385</td><td>0</td><td>1.1406</td><td>1146</td><td>1168</td><td>1</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>71.9742</td><td>5241</td><td>5241</td><td>0</td><td>7.6772</td><td>5241</td><td>5241</td><td>0</td><td>1.5747</td><td>1046</td><td>1075</td><td>2</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>60.2188</td><td>4385</td><td>4385</td><td>0</td><td>6.4233</td><td>4385</td><td>4385</td><td>0</td><td>1.7109</td><td>1146</td><td>1168</td><td>1</td></tr>
<tr><td><code>conv2.0.0</code></td><td>30845</td><td>32</td><td>60.5164</td><td>2644</td><td>2644</td><td>58</td><td>7.3999</td><td>2644</td><td>3031</td><td>58</td><td>4.9146</td><td>1738</td><td>2013</td><td>33</td></tr>
<tr><td><code>conv2.1.0</code></td><td>30845</td><td>32</td><td>60.5164</td><td>2644</td><td>2644</td><td>58</td><td>7.3999</td><td>2644</td><td>3031</td><td>58</td><td>4.9146</td><td>1738</td><td>2013</td><td>33</td></tr>
<tr><td><code>conv2.2.0</code></td><td>30845</td><td>32</td><td>49.8505</td><td>2178</td><td>2178</td><td>37</td><td>5.9619</td><td>2178</td><td>2442</td><td>37</td><td>6.0474</td><td>2210</td><td>2477</td><td>31</td></tr>
<tr><td><code>conv3.0.0</code></td><td>23870</td><td>64</td><td>26.3672</td><td>640</td><td>640</td><td>91</td><td>4.3286</td><td>640</td><td>985</td><td>91</td><td>6.9170</td><td>1375</td><td>1574</td><td>23</td></tr>
<tr><td><code>conv3.1.0</code></td><td>23870</td><td>64</td><td>26.3672</td><td>640</td><td>640</td><td>91</td><td>4.3286</td><td>640</td><td>985</td><td>91</td><td>6.9170</td><td>1375</td><td>1574</td><td>23</td></tr>
<tr><td><code>conv3.2.0</code></td><td>23870</td><td>64</td><td>26.4084</td><td>641</td><td>641</td><td>89</td><td>4.3638</td><td>641</td><td>993</td><td>89</td><td>6.9126</td><td>1379</td><td>1573</td><td>24</td></tr>
<tr><td><code>conv4.0.0</code></td><td>11789</td><td>64</td><td>6.7154</td><td>163</td><td>163</td><td>52</td><td>1.5161</td><td>163</td><td>345</td><td>52</td><td>2.6104</td><td>460</td><td>594</td><td>9</td></tr>
<tr><td><code>conv4.1.0</code></td><td>11789</td><td>64</td><td>6.7154</td><td>163</td><td>163</td><td>52</td><td>1.5161</td><td>163</td><td>345</td><td>52</td><td>2.6104</td><td>460</td><td>594</td><td>9</td></tr>
<tr><td><code>conv4.2.0</code></td><td>11789</td><td>64</td><td>6.3446</td><td>154</td><td>154</td><td>33</td><td>1.1646</td><td>154</td><td>265</td><td>33</td><td>1.9688</td><td>421</td><td>448</td><td>3</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>71.9742</strong></td><td><strong>5241</strong></td><td><strong>5241</strong></td><td><strong>91</strong></td><td><strong>7.6772</strong></td><td><strong>5241</strong></td><td><strong>5241</strong></td><td><strong>91</strong></td><td><strong>6.9170</strong></td><td><strong>2210</strong></td><td><strong>2477</strong></td><td><strong>33</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.54%**，将 hash entry 峰值降低 **53.36%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **6.53%**，将 hash entry 峰值降低 **53.36%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 132.1930 | 14.7998 | 13.8340 |
| DRAM 峰值层 | `conv1.0.0` | `conv2.1.0` | `conv3.1.0` |
| Hash entries | 9626 | 9626 | 4490 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000005`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000005.bin`
- 输入体素: `15000`，坐标 SHA-256 `f1676047e0f763270d21f61428a6702422fe4e660b146d368c10d215c891cbdc`
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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>31.2080</td><td>4545</td><td>4545</td><td>0</td><td>4.4385</td><td>4545</td><td>4545</td><td>0</td><td>1.0713</td><td>1057</td><td>1097</td><td>6</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>67.7032</td><td>4930</td><td>4930</td><td>0</td><td>7.2217</td><td>4930</td><td>4930</td><td>0</td><td>1.5439</td><td>1009</td><td>1054</td><td>7</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>62.4161</td><td>4545</td><td>4545</td><td>0</td><td>6.6577</td><td>4545</td><td>4545</td><td>0</td><td>1.6069</td><td>1057</td><td>1097</td><td>6</td></tr>
<tr><td><code>conv2.0.0</code></td><td>33997</td><td>32</td><td>50.8804</td><td>2223</td><td>2223</td><td>56</td><td>6.4868</td><td>2223</td><td>2657</td><td>56</td><td>4.9463</td><td>1701</td><td>2026</td><td>39</td></tr>
<tr><td><code>conv2.1.0</code></td><td>33997</td><td>32</td><td>50.8804</td><td>2223</td><td>2223</td><td>56</td><td>6.4868</td><td>2223</td><td>2657</td><td>56</td><td>4.9463</td><td>1701</td><td>2026</td><td>39</td></tr>
<tr><td><code>conv2.2.0</code></td><td>33997</td><td>32</td><td>47.3328</td><td>2068</td><td>2068</td><td>50</td><td>5.9131</td><td>2068</td><td>2422</td><td>50</td><td>5.6250</td><td>1935</td><td>2304</td><td>40</td></tr>
<tr><td><code>conv3.0.0</code></td><td>25085</td><td>64</td><td>25.9964</td><td>631</td><td>631</td><td>116</td><td>4.5659</td><td>631</td><td>1039</td><td>116</td><td>6.7720</td><td>1276</td><td>1541</td><td>30</td></tr>
<tr><td><code>conv3.1.0</code></td><td>25085</td><td>64</td><td>25.9964</td><td>631</td><td>631</td><td>116</td><td>4.5659</td><td>631</td><td>1039</td><td>116</td><td>6.7720</td><td>1276</td><td>1541</td><td>30</td></tr>
<tr><td><code>conv3.2.0</code></td><td>25085</td><td>64</td><td>26.2024</td><td>636</td><td>636</td><td>115</td><td>4.6099</td><td>636</td><td>1049</td><td>115</td><td>6.9258</td><td>1292</td><td>1576</td><td>34</td></tr>
<tr><td><code>conv4.0.0</code></td><td>11267</td><td>64</td><td>6.1798</td><td>150</td><td>150</td><td>49</td><td>1.4722</td><td>150</td><td>335</td><td>49</td><td>2.6411</td><td>429</td><td>601</td><td>7</td></tr>
<tr><td><code>conv4.1.0</code></td><td>11267</td><td>64</td><td>6.1798</td><td>150</td><td>150</td><td>49</td><td>1.4722</td><td>150</td><td>335</td><td>49</td><td>2.6411</td><td>429</td><td>601</td><td>7</td></tr>
<tr><td><code>conv4.2.0</code></td><td>11267</td><td>64</td><td>5.8090</td><td>141</td><td>141</td><td>34</td><td>1.1338</td><td>141</td><td>258</td><td>34</td><td>1.7974</td><td>392</td><td>409</td><td>1</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>67.7032</strong></td><td><strong>4930</strong></td><td><strong>4930</strong></td><td><strong>116</strong></td><td><strong>7.2217</strong></td><td><strong>4930</strong></td><td><strong>4930</strong></td><td><strong>116</strong></td><td><strong>6.9258</strong></td><td><strong>1935</strong></td><td><strong>2304</strong></td><td><strong>40</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.47%**，将 hash entry 峰值降低 **54.30%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **1.31%**，将 hash entry 峰值降低 **54.30%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 130.1193 | 13.8794 | 13.6978 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.2.0` |
| Hash entries | 9475 | 9475 | 4330 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000006`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000006.bin`
- 输入体素: `15000`，坐标 SHA-256 `1ab6c458d412224b36b4bab68090ed6ec15d7ccc2ab3360514fea4e10b496917`
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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>22.1718</td><td>3229</td><td>3229</td><td>0</td><td>3.1758</td><td>3229</td><td>3252</td><td>0</td><td>1.2451</td><td>1214</td><td>1275</td><td>16</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>45.2225</td><td>3293</td><td>3293</td><td>0</td><td>4.8574</td><td>3293</td><td>3316</td><td>0</td><td>1.7593</td><td>1145</td><td>1201</td><td>16</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>44.3436</td><td>3229</td><td>3229</td><td>0</td><td>4.7637</td><td>3229</td><td>3252</td><td>0</td><td>1.8677</td><td>1214</td><td>1275</td><td>16</td></tr>
<tr><td><code>conv2.0.0</code></td><td>24839</td><td>32</td><td>38.7497</td><td>1693</td><td>1693</td><td>62</td><td>4.7461</td><td>1693</td><td>1944</td><td>62</td><td>4.7144</td><td>1699</td><td>1931</td><td>50</td></tr>
<tr><td><code>conv2.1.0</code></td><td>24839</td><td>32</td><td>38.7497</td><td>1693</td><td>1693</td><td>62</td><td>4.7461</td><td>1693</td><td>1944</td><td>62</td><td>4.7144</td><td>1699</td><td>1931</td><td>50</td></tr>
<tr><td><code>conv2.2.0</code></td><td>24839</td><td>32</td><td>32.7072</td><td>1429</td><td>1429</td><td>46</td><td>4.0039</td><td>1429</td><td>1640</td><td>46</td><td>4.9487</td><td>1777</td><td>2027</td><td>51</td></tr>
<tr><td><code>conv3.0.0</code></td><td>17762</td><td>64</td><td>24.1425</td><td>586</td><td>586</td><td>61</td><td>3.6255</td><td>586</td><td>825</td><td>61</td><td>6.0337</td><td>1232</td><td>1373</td><td>9</td></tr>
<tr><td><code>conv3.1.0</code></td><td>17762</td><td>64</td><td>24.1425</td><td>586</td><td>586</td><td>61</td><td>3.6255</td><td>586</td><td>825</td><td>61</td><td>6.0337</td><td>1232</td><td>1373</td><td>9</td></tr>
<tr><td><code>conv3.2.0</code></td><td>17762</td><td>64</td><td>24.0189</td><td>583</td><td>583</td><td>59</td><td>3.6035</td><td>583</td><td>820</td><td>59</td><td>5.9722</td><td>1228</td><td>1359</td><td>9</td></tr>
<tr><td><code>conv4.0.0</code></td><td>8531</td><td>64</td><td>6.3446</td><td>154</td><td>154</td><td>37</td><td>1.2437</td><td>154</td><td>283</td><td>37</td><td>2.2896</td><td>419</td><td>521</td><td>1</td></tr>
<tr><td><code>conv4.1.0</code></td><td>8531</td><td>64</td><td>6.3446</td><td>154</td><td>154</td><td>37</td><td>1.2437</td><td>154</td><td>283</td><td>37</td><td>2.2896</td><td>419</td><td>521</td><td>1</td></tr>
<tr><td><code>conv4.2.0</code></td><td>8531</td><td>64</td><td>5.8090</td><td>141</td><td>141</td><td>21</td><td>0.9800</td><td>141</td><td>223</td><td>21</td><td>1.6436</td><td>367</td><td>374</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>45.2225</strong></td><td><strong>3293</strong></td><td><strong>3293</strong></td><td><strong>62</strong></td><td><strong>4.8574</strong></td><td><strong>3293</strong></td><td><strong>3316</strong></td><td><strong>62</strong></td><td><strong>6.0337</strong></td><td><strong>1777</strong></td><td><strong>2027</strong></td><td><strong>51</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **86.53%**，将 hash entry 峰值降低 **39.31%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-25.43%**，将 hash entry 峰值降低 **39.74%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 89.5660 | 9.6211 | 12.0674 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 6522 | 6568 | 3958 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000007`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000007.bin`
- 输入体素: `15000`，坐标 SHA-256 `b3e2b2d7a6aa77595bf0022143c202224d8087e84aeb6c70529c84b8d2a04b8a`
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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>33.6594</td><td>4902</td><td>4902</td><td>0</td><td>4.7920</td><td>4902</td><td>4907</td><td>0</td><td>1.2598</td><td>1267</td><td>1290</td><td>2</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>76.8082</td><td>5593</td><td>5593</td><td>1</td><td>8.2002</td><td>5593</td><td>5598</td><td>1</td><td>1.8208</td><td>1222</td><td>1243</td><td>2</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>67.3187</td><td>4902</td><td>4902</td><td>0</td><td>7.1880</td><td>4902</td><td>4907</td><td>0</td><td>1.8896</td><td>1267</td><td>1290</td><td>2</td></tr>
<tr><td><code>conv2.0.0</code></td><td>32168</td><td>32</td><td>61.2946</td><td>2678</td><td>2678</td><td>44</td><td>7.3340</td><td>2678</td><td>3004</td><td>44</td><td>5.1953</td><td>1880</td><td>2128</td><td>35</td></tr>
<tr><td><code>conv2.1.0</code></td><td>32168</td><td>32</td><td>61.2946</td><td>2678</td><td>2678</td><td>44</td><td>7.3340</td><td>2678</td><td>3004</td><td>44</td><td>5.1953</td><td>1880</td><td>2128</td><td>35</td></tr>
<tr><td><code>conv2.2.0</code></td><td>32168</td><td>32</td><td>51.7044</td><td>2259</td><td>2259</td><td>30</td><td>6.1108</td><td>2259</td><td>2503</td><td>30</td><td>6.4331</td><td>2371</td><td>2635</td><td>31</td></tr>
<tr><td><code>conv3.0.0</code></td><td>26027</td><td>64</td><td>28.6331</td><td>695</td><td>695</td><td>86</td><td>4.6450</td><td>695</td><td>1057</td><td>86</td><td>7.2642</td><td>1422</td><td>1653</td><td>33</td></tr>
<tr><td><code>conv3.1.0</code></td><td>26027</td><td>64</td><td>28.6331</td><td>695</td><td>695</td><td>86</td><td>4.6450</td><td>695</td><td>1057</td><td>86</td><td>7.2642</td><td>1422</td><td>1653</td><td>33</td></tr>
<tr><td><code>conv3.2.0</code></td><td>26027</td><td>64</td><td>28.7979</td><td>699</td><td>699</td><td>80</td><td>4.6846</td><td>699</td><td>1066</td><td>80</td><td>7.1675</td><td>1406</td><td>1631</td><td>31</td></tr>
<tr><td><code>conv4.0.0</code></td><td>12602</td><td>64</td><td>7.6218</td><td>185</td><td>185</td><td>56</td><td>1.7051</td><td>185</td><td>388</td><td>56</td><td>2.8037</td><td>478</td><td>638</td><td>12</td></tr>
<tr><td><code>conv4.1.0</code></td><td>12602</td><td>64</td><td>7.6218</td><td>185</td><td>185</td><td>56</td><td>1.7051</td><td>185</td><td>388</td><td>56</td><td>2.8037</td><td>478</td><td>638</td><td>12</td></tr>
<tr><td><code>conv4.2.0</code></td><td>12602</td><td>64</td><td>7.3746</td><td>179</td><td>179</td><td>33</td><td>1.3052</td><td>179</td><td>297</td><td>33</td><td>2.0479</td><td>441</td><td>466</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>76.8082</strong></td><td><strong>5593</strong></td><td><strong>5593</strong></td><td><strong>86</strong></td><td><strong>8.2002</strong></td><td><strong>5593</strong></td><td><strong>5598</strong></td><td><strong>86</strong></td><td><strong>7.2642</strong></td><td><strong>2371</strong></td><td><strong>2635</strong></td><td><strong>35</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.92%**，将 hash entry 峰值降低 **54.62%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **5.59%**，将 hash entry 峰值降低 **54.66%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 144.1269 | 15.3882 | 14.5283 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 10495 | 10505 | 4763 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000008`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000008.bin`
- 输入体素: `13081`，坐标 SHA-256 `e32d392d967bdaf3a1307b331e9ba3c178033daa4b0533798195fa4aa48a9e04`
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
- 输入体素: `15000`，坐标 SHA-256 `25d5851280af78d514a84f314cc16a6f3e50d1ebbb5e048f07e47df8f6946425`
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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>27.2804</td><td>3973</td><td>3973</td><td>0</td><td>3.8799</td><td>3973</td><td>3973</td><td>0</td><td>1.1367</td><td>1143</td><td>1164</td><td>2</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>65.1901</td><td>4747</td><td>4747</td><td>0</td><td>6.9536</td><td>4747</td><td>4747</td><td>0</td><td>1.6304</td><td>1092</td><td>1113</td><td>3</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>54.5609</td><td>3973</td><td>3973</td><td>0</td><td>5.8198</td><td>3973</td><td>3973</td><td>0</td><td>1.7051</td><td>1143</td><td>1164</td><td>2</td></tr>
<tr><td><code>conv2.0.0</code></td><td>31060</td><td>32</td><td>51.6129</td><td>2255</td><td>2255</td><td>48</td><td>6.4648</td><td>2255</td><td>2648</td><td>48</td><td>4.9243</td><td>1774</td><td>2017</td><td>27</td></tr>
<tr><td><code>conv2.1.0</code></td><td>31060</td><td>32</td><td>51.6129</td><td>2255</td><td>2255</td><td>48</td><td>6.4648</td><td>2255</td><td>2648</td><td>48</td><td>4.9243</td><td>1774</td><td>2017</td><td>27</td></tr>
<tr><td><code>conv2.2.0</code></td><td>31060</td><td>32</td><td>43.6707</td><td>1908</td><td>1908</td><td>32</td><td>5.3198</td><td>1908</td><td>2179</td><td>32</td><td>5.9912</td><td>2223</td><td>2454</td><td>29</td></tr>
<tr><td><code>conv3.0.0</code></td><td>22954</td><td>64</td><td>27.3972</td><td>665</td><td>665</td><td>81</td><td>4.3638</td><td>665</td><td>993</td><td>81</td><td>6.5127</td><td>1344</td><td>1482</td><td>14</td></tr>
<tr><td><code>conv3.1.0</code></td><td>22954</td><td>64</td><td>27.3972</td><td>665</td><td>665</td><td>81</td><td>4.3638</td><td>665</td><td>993</td><td>81</td><td>6.5127</td><td>1344</td><td>1482</td><td>14</td></tr>
<tr><td><code>conv3.2.0</code></td><td>22954</td><td>64</td><td>26.8204</td><td>651</td><td>651</td><td>74</td><td>4.2363</td><td>651</td><td>964</td><td>74</td><td>6.4380</td><td>1321</td><td>1465</td><td>12</td></tr>
<tr><td><code>conv4.0.0</code></td><td>11246</td><td>64</td><td>7.3746</td><td>179</td><td>179</td><td>49</td><td>1.5820</td><td>179</td><td>360</td><td>49</td><td>2.7070</td><td>481</td><td>616</td><td>2</td></tr>
<tr><td><code>conv4.1.0</code></td><td>11246</td><td>64</td><td>7.3746</td><td>179</td><td>179</td><td>49</td><td>1.5820</td><td>179</td><td>360</td><td>49</td><td>2.7070</td><td>481</td><td>616</td><td>2</td></tr>
<tr><td><code>conv4.2.0</code></td><td>11246</td><td>64</td><td>6.7566</td><td>164</td><td>164</td><td>27</td><td>1.1865</td><td>164</td><td>270</td><td>27</td><td>1.9468</td><td>432</td><td>443</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>65.1901</strong></td><td><strong>4747</strong></td><td><strong>4747</strong></td><td><strong>81</strong></td><td><strong>6.9536</strong></td><td><strong>4747</strong></td><td><strong>4747</strong></td><td><strong>81</strong></td><td><strong>6.5127</strong></td><td><strong>2223</strong></td><td><strong>2454</strong></td><td><strong>29</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.12%**，将 hash entry 峰值降低 **48.73%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-0.74%**，将 hash entry 峰值降低 **48.73%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 119.7510 | 12.9297 | 13.0254 |
| DRAM 峰值层 | `conv1.0.0` | `conv2.1.0` | `conv3.1.0` |
| Hash entries | 8720 | 8720 | 4471 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000010`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000010.bin`
- 输入体素: `13094`，坐标 SHA-256 `4acce1cd485e7759ee11a4e397e7966dc1afcd28fd637c4d7a4f3cbeac7635f0`
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
- 输入体素: `15000`，坐标 SHA-256 `5ac10c0005153a6457d72091d3c3d9450081cc05b02aaa3ec3a52ee3fca037a5`
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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>22.7417</td><td>3312</td><td>3312</td><td>0</td><td>3.2461</td><td>3312</td><td>3324</td><td>0</td><td>1.1387</td><td>1111</td><td>1166</td><td>12</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>47.4060</td><td>3452</td><td>3452</td><td>0</td><td>5.0771</td><td>3452</td><td>3466</td><td>0</td><td>1.6553</td><td>1079</td><td>1130</td><td>11</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>45.4834</td><td>3312</td><td>3312</td><td>0</td><td>4.8691</td><td>3312</td><td>3324</td><td>0</td><td>1.7080</td><td>1111</td><td>1166</td><td>12</td></tr>
<tr><td><code>conv2.0.0</code></td><td>25873</td><td>32</td><td>35.6598</td><td>1558</td><td>1558</td><td>48</td><td>4.5312</td><td>1558</td><td>1856</td><td>48</td><td>4.2554</td><td>1511</td><td>1743</td><td>37</td></tr>
<tr><td><code>conv2.1.0</code></td><td>25873</td><td>32</td><td>35.6598</td><td>1558</td><td>1558</td><td>48</td><td>4.5312</td><td>1558</td><td>1856</td><td>48</td><td>4.2554</td><td>1511</td><td>1743</td><td>37</td></tr>
<tr><td><code>conv2.2.0</code></td><td>25873</td><td>32</td><td>31.6544</td><td>1383</td><td>1383</td><td>42</td><td>3.9746</td><td>1383</td><td>1628</td><td>42</td><td>4.4019</td><td>1566</td><td>1803</td><td>39</td></tr>
<tr><td><code>conv3.0.0</code></td><td>16977</td><td>64</td><td>19.1574</td><td>465</td><td>465</td><td>68</td><td>3.1377</td><td>465</td><td>714</td><td>68</td><td>5.1812</td><td>1027</td><td>1179</td><td>13</td></tr>
<tr><td><code>conv3.1.0</code></td><td>16977</td><td>64</td><td>19.1574</td><td>465</td><td>465</td><td>68</td><td>3.1377</td><td>465</td><td>714</td><td>68</td><td>5.1812</td><td>1027</td><td>1179</td><td>13</td></tr>
<tr><td><code>conv3.2.0</code></td><td>16977</td><td>64</td><td>20.0638</td><td>487</td><td>487</td><td>66</td><td>3.2388</td><td>487</td><td>737</td><td>66</td><td>5.1328</td><td>1032</td><td>1168</td><td>13</td></tr>
<tr><td><code>conv4.0.0</code></td><td>7946</td><td>64</td><td>4.4907</td><td>109</td><td>109</td><td>38</td><td>1.0503</td><td>109</td><td>239</td><td>38</td><td>1.9116</td><td>334</td><td>435</td><td>4</td></tr>
<tr><td><code>conv4.1.0</code></td><td>7946</td><td>64</td><td>4.4907</td><td>109</td><td>109</td><td>38</td><td>1.0503</td><td>109</td><td>239</td><td>38</td><td>1.9116</td><td>334</td><td>435</td><td>4</td></tr>
<tr><td><code>conv4.2.0</code></td><td>7946</td><td>64</td><td>4.2023</td><td>102</td><td>102</td><td>24</td><td>0.8130</td><td>102</td><td>185</td><td>24</td><td>1.3755</td><td>302</td><td>313</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>47.4060</strong></td><td><strong>3452</strong></td><td><strong>3452</strong></td><td><strong>68</strong></td><td><strong>5.0771</strong></td><td><strong>3452</strong></td><td><strong>3466</strong></td><td><strong>68</strong></td><td><strong>5.1812</strong></td><td><strong>1566</strong></td><td><strong>1803</strong></td><td><strong>39</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **88.84%**，将 hash entry 峰值降低 **47.58%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-4.18%**，将 hash entry 峰值降低 **47.78%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 92.8894 | 9.9463 | 10.3623 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 6764 | 6790 | 3546 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000012`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000012.bin`
- 输入体素: `14839`，坐标 SHA-256 `21bcbb8e68fba77fadb0d23e30fd6f2a3978d37ee2a5a4e7ce941629451ab71d`
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
- 输入体素: `15000`，坐标 SHA-256 `1aa8ff64cec274da8031a4dfb4e3c8501149d897fe2aaa20ccd0e8b47fcf83e3`
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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>35.3622</td><td>5150</td><td>5150</td><td>0</td><td>5.0361</td><td>5150</td><td>5157</td><td>0</td><td>1.3057</td><td>1298</td><td>1337</td><td>4</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>74.8032</td><td>5447</td><td>5447</td><td>0</td><td>7.9878</td><td>5447</td><td>5453</td><td>0</td><td>1.8955</td><td>1253</td><td>1294</td><td>5</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>70.7245</td><td>5150</td><td>5150</td><td>0</td><td>7.5542</td><td>5150</td><td>5157</td><td>0</td><td>1.9585</td><td>1298</td><td>1337</td><td>4</td></tr>
<tr><td><code>conv2.0.0</code></td><td>33478</td><td>32</td><td>58.1131</td><td>2539</td><td>2539</td><td>95</td><td>7.3389</td><td>2539</td><td>3006</td><td>95</td><td>5.3638</td><td>1825</td><td>2197</td><td>63</td></tr>
<tr><td><code>conv2.1.0</code></td><td>33478</td><td>32</td><td>58.1131</td><td>2539</td><td>2539</td><td>95</td><td>7.3389</td><td>2539</td><td>3006</td><td>95</td><td>5.3638</td><td>1825</td><td>2197</td><td>63</td></tr>
<tr><td><code>conv2.2.0</code></td><td>33478</td><td>32</td><td>54.9088</td><td>2399</td><td>2399</td><td>89</td><td>6.9360</td><td>2399</td><td>2841</td><td>89</td><td>5.9619</td><td>2065</td><td>2442</td><td>71</td></tr>
<tr><td><code>conv3.0.0</code></td><td>25069</td><td>64</td><td>27.8915</td><td>677</td><td>677</td><td>103</td><td>4.8076</td><td>677</td><td>1094</td><td>103</td><td>7.6641</td><td>1387</td><td>1744</td><td>69</td></tr>
<tr><td><code>conv3.1.0</code></td><td>25069</td><td>64</td><td>27.8915</td><td>677</td><td>677</td><td>103</td><td>4.8076</td><td>677</td><td>1094</td><td>103</td><td>7.6641</td><td>1387</td><td>1744</td><td>69</td></tr>
<tr><td><code>conv3.2.0</code></td><td>25069</td><td>64</td><td>27.5208</td><td>668</td><td>668</td><td>102</td><td>4.7417</td><td>668</td><td>1079</td><td>102</td><td>7.5322</td><td>1363</td><td>1714</td><td>63</td></tr>
<tr><td><code>conv4.0.0</code></td><td>11646</td><td>64</td><td>5.8502</td><td>142</td><td>142</td><td>47</td><td>1.4810</td><td>142</td><td>337</td><td>47</td><td>2.6235</td><td>406</td><td>597</td><td>30</td></tr>
<tr><td><code>conv4.1.0</code></td><td>11646</td><td>64</td><td>5.8502</td><td>142</td><td>142</td><td>47</td><td>1.4810</td><td>142</td><td>337</td><td>47</td><td>2.6235</td><td>406</td><td>597</td><td>30</td></tr>
<tr><td><code>conv4.2.0</code></td><td>11646</td><td>64</td><td>5.4382</td><td>132</td><td>132</td><td>31</td><td>1.1074</td><td>132</td><td>252</td><td>31</td><td>1.7402</td><td>368</td><td>396</td><td>1</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>74.8032</strong></td><td><strong>5447</strong></td><td><strong>5447</strong></td><td><strong>103</strong></td><td><strong>7.9878</strong></td><td><strong>5447</strong></td><td><strong>5453</strong></td><td><strong>103</strong></td><td><strong>7.6641</strong></td><td><strong>2065</strong></td><td><strong>2442</strong></td><td><strong>71</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.47%**，将 hash entry 峰值降低 **56.22%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **1.38%**，将 hash entry 峰值降低 **56.28%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 145.5276 | 15.5420 | 15.3281 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 10597 | 10610 | 4639 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000014`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000014.bin`
- 输入体素: `15000`，坐标 SHA-256 `8435842187bacbaa475fe0bc06a5878128eacfe14fd3a40a2b14dca8e849e406`
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
<tr><td>初始输入</td><td>15000</td><td>4</td><td>34.7786</td><td>5065</td><td>5065</td><td>0</td><td>4.9492</td><td>5065</td><td>5068</td><td>0</td><td>1.2119</td><td>1205</td><td>1241</td><td>4</td></tr>
<tr><td><code>conv_input.0</code></td><td>15000</td><td>16</td><td>75.5997</td><td>5505</td><td>5505</td><td>0</td><td>8.0728</td><td>5505</td><td>5511</td><td>0</td><td>1.7959</td><td>1190</td><td>1226</td><td>5</td></tr>
<tr><td><code>conv1.0.0</code></td><td>15000</td><td>16</td><td>69.5572</td><td>5065</td><td>5065</td><td>0</td><td>7.4238</td><td>5065</td><td>5068</td><td>0</td><td>1.8179</td><td>1205</td><td>1241</td><td>4</td></tr>
<tr><td><code>conv2.0.0</code></td><td>34137</td><td>32</td><td>53.7643</td><td>2349</td><td>2349</td><td>94</td><td>6.9800</td><td>2349</td><td>2859</td><td>94</td><td>5.5054</td><td>1895</td><td>2255</td><td>55</td></tr>
<tr><td><code>conv2.1.0</code></td><td>34137</td><td>32</td><td>53.7643</td><td>2349</td><td>2349</td><td>94</td><td>6.9800</td><td>2349</td><td>2859</td><td>94</td><td>5.5054</td><td>1895</td><td>2255</td><td>55</td></tr>
<tr><td><code>conv2.2.0</code></td><td>34137</td><td>32</td><td>50.0565</td><td>2187</td><td>2187</td><td>94</td><td>6.3745</td><td>2187</td><td>2611</td><td>94</td><td>6.0571</td><td>2098</td><td>2481</td><td>56</td></tr>
<tr><td><code>conv3.0.0</code></td><td>25946</td><td>64</td><td>28.9627</td><td>703</td><td>703</td><td>119</td><td>5.0317</td><td>703</td><td>1145</td><td>119</td><td>7.9365</td><td>1463</td><td>1806</td><td>55</td></tr>
<tr><td><code>conv3.1.0</code></td><td>25946</td><td>64</td><td>28.9627</td><td>703</td><td>703</td><td>119</td><td>5.0317</td><td>703</td><td>1145</td><td>119</td><td>7.9365</td><td>1463</td><td>1806</td><td>55</td></tr>
<tr><td><code>conv3.2.0</code></td><td>25946</td><td>64</td><td>29.1275</td><td>707</td><td>707</td><td>116</td><td>5.0537</td><td>707</td><td>1150</td><td>116</td><td>7.7871</td><td>1427</td><td>1772</td><td>49</td></tr>
<tr><td><code>conv4.0.0</code></td><td>11819</td><td>64</td><td>6.0150</td><td>146</td><td>146</td><td>54</td><td>1.5381</td><td>146</td><td>350</td><td>54</td><td>2.7026</td><td>427</td><td>615</td><td>13</td></tr>
<tr><td><code>conv4.1.0</code></td><td>11819</td><td>64</td><td>6.0150</td><td>146</td><td>146</td><td>54</td><td>1.5381</td><td>146</td><td>350</td><td>54</td><td>2.7026</td><td>427</td><td>615</td><td>13</td></tr>
<tr><td><code>conv4.2.0</code></td><td>11819</td><td>64</td><td>5.8502</td><td>142</td><td>142</td><td>36</td><td>1.1646</td><td>142</td><td>265</td><td>36</td><td>1.7842</td><td>390</td><td>406</td><td>0</td></tr>
<tr><td><strong>单个 feature map 最大值</strong></td><td>—</td><td>—</td><td><strong>75.5997</strong></td><td><strong>5505</strong></td><td><strong>5505</strong></td><td><strong>119</strong></td><td><strong>8.0728</strong></td><td><strong>5505</strong></td><td><strong>5511</strong></td><td><strong>119</strong></td><td><strong>7.9365</strong></td><td><strong>2098</strong></td><td><strong>2481</strong></td><td><strong>56</strong></td></tr>
  </tbody>
</table>

“单个 feature map 最大值”一行对每个指标独立取最大值。

- 相对固定容量，Proposed 将执行时 DRAM 峰值降低 **89.06%**，将 hash entry 峰值降低 **55.19%**。
- 相对固定块分页，Proposed 将执行时 DRAM 峰值降低 **-2.43%**，将 hash entry 峰值降低 **55.23%**。

### 执行时 IFM 与 OFM 同时驻留峰值

| 执行时同时驻留峰值 | 固定块 + 固定容量 | 固定块 + Page | Proposed 可变块 + Page |
| --- | ---: | ---: | ---: |
| DRAM / MiB | 145.1569 | 15.4966 | 15.8730 |
| DRAM 峰值层 | `conv1.0.0` | `conv1.0.0` | `conv3.1.0` |
| Hash entries | 10570 | 10579 | 4736 |
| Hash 峰值层 | `conv_input.0` | `conv_input.0` | `conv2.2.0` |

## Frame `train/000015`

- Point cloud: `/home/vipuser/桌面/OpenPCDet/data/kitti/training/velodyne/000015.bin`
- 输入体素: `14241`，坐标 SHA-256 `82949cbe297835536b61590e42bcb13d5d878f02dac28dd4999963f0796960c8`
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

# Online Block Partition 算法对接说明（基于当前RTL实现）

本文面向算法/软件仿真同事，整理 `online_block_partitioning.sv` 当前版本的实际行为，目标是做出与硬件一致的输入输出序列。

## 1. 模块做什么

对每个输入体素 `(x, y, z)`，模块会输出 1 到 8 个 block request：

- 第 1 个一定是主块（primary）
- 后续最多 7 个是 halo 邻块（由边界触发）

每个输出 beat 包含：

- `req_block_id`：18 bit 紧凑块ID
- `halo_dir`：6 bit halo 方向编码（主块时全 0）

## 2. 接口语义（对接最关键）

### 输入握手

- 只有当 `voxel_ready=1` 时才能送入新体素
- 当前设计中 `voxel_ready` 仅在 `IDLE` 态为 1
- 因此是“逐体素串行处理”，一个体素全部 primary+halo 输出完，才接下一个

### 输出握手

- `req_valid` 高电平表示当前拍输出了一个有效 block request
- 同拍可读取：`req_block_id` 和 `halo_dir`

### 错误信号

- `partitioning_err` 为保护性异常脉冲（正常流程不应出现）

## 3. 块ID定义（当前为18 bit）

`req_block_id` 打包格式：

`{zone[1:0], x_local[6:0], y_local[6:0], z_local[1:0]}`

说明：

- `zone`：2 bit，来自 `zone_lut`
- `x_local`、`y_local`：7 bit 二补码，表示相对 LiDAR 原点的有符号 block index
- `z_local`：2 bit，来自 `z >> log2_bz` 后的低2位

## 4. 区域划分与块尺寸（来自当前zone_lut）

先计算二维Chebyshev距离：

- `dx = abs(x - LIDAR_X)`
- `dy = abs(y - LIDAR_Y)`
- `dist = max(dx, dy)`

当前 `zone_lut.sv` 规则：

- `dist < 64`：`zone=0`, `log2_bx=4`, `log2_by=4`, `log2_bz=4`（块尺寸 16/16/16）
- `64 <= dist < 512`：`zone=1`, `log2_bx=3`, `log2_by=3`, `log2_bz=4`（8/8/16）
- `512 <= dist < 768`：`zone=2`, `log2_bx=4`, `log2_by=4`, `log2_bz=4`（16/16/16）
- `dist >= 768`：`zone=3`, `log2_bx=6`, `log2_by=6`, `log2_bz=4`（64/64/16）

## 5. 主块索引计算

给定每轴 `log2_b*`，块索引为：

- `blk_x = (x - LIDAR_X) >>> log2_bx`（算术右移，负数向下取整语义）
- `blk_y = (y - LIDAR_Y) >>> log2_by`
- `blk_z = z >> log2_bz`（无符号逻辑右移）

然后映射到 `req_block_id` 的字段。

## 6. 边界判定（决定是否产生halo）

对主块坐标判定 6 个边界标志：

- x低边界：`(x & ((1<<log2_bx)-1)) == 0` 且 `x != 0`
- x高边界：`(x & mask_x) == mask_x` 且 `x != MAX_X`
- y低边界：`(y & mask_y) == 0` 且 `y != 0`
- y高边界：`(y & mask_y) == mask_y` 且 `y != MAX_Y`
- z低边界：`(z & mask_z) == 0` 且 `z != 0`
- z高边界：`(z & mask_z) == mask_z` 且 `z != MAX_Z`

其中 `MAX_X/MAX_Y/MAX_Z` 分别是对应位宽全1。

注意：

- 边界标志只在主块阶段计算一次
- 后续所有halo都复用这组标志，不会重新判定

## 7. halo组合规则与输出顺序

### 组合编号

halo组合编号 `1..7` 对应三轴bit掩码 `{z,y,x}`：

- `1 (001)`：x
- `2 (010)`：y
- `3 (011)`：x+y
- `4 (100)`：z
- `5 (101)`：x+z
- `6 (110)`：y+z
- `7 (111)`：x+y+z

### 组合是否有效

仅当涉及的轴“确实在边界上”时组合有效。例如：

- `x+y` 需要 x轴边界有效 且 y轴边界有效
- `x+y+z` 需要三轴都有效

### 实际输出顺序

顺序固定为：

1. 主块（`halo_dir=0`）
2. 按编号从小到大输出有效halo：1,2,3,4,5,6,7 中的子集

即：不是几何距离优先，而是固定优先级编码顺序。

## 8. halo_dir编码（6 bit）

按 `{z[1:0], y[1:0], x[1:0]}` 打包：

- `00`: 该轴不偏移
- `01`: 该轴走 `-1` 邻块（NEG）
- `10`: 该轴走 `+1` 邻块（POS）

主块固定 `halo_dir = 6'b00_00_00`。

对某一轴，如果该halo组合使能该轴：

- 若主块命中“低边界”，该轴编码为 `01`
- 否则（主块命中高边界），该轴编码为 `10`

## 9. 状态机时序（单体素）

对一个体素，主块路径为：

- `SAMP_DIS -> ZONE_LUT -> CAL_ID -> EMIT`

若还有halo：

- `EMIT -> NEXT_HALO -> ZONE_LUT -> CAL_ID -> EMIT -> ...`

直到无后续有效halo，回到 `IDLE`，`voxel_ready` 拉高。

## 10. 与软件仿真对齐的参考伪代码

```python
from dataclasses import dataclass

DIR_NONE = 0b00
DIR_NEG  = 0b01
DIR_POS  = 0b10

@dataclass
class ReqBeat:
    block_id: int
    halo_dir: int


def zone_lut(dist: int):
    if dist < 64:
        return 0, 4, 4, 4
    elif dist < 512:
        return 1, 3, 3, 4
    elif dist < 768:
        return 2, 4, 4, 4
    else:
        return 3, 6, 6, 4


def arshift_floor(v: int, s: int) -> int:
    # Python右移对int天然是算术右移
    return v >> s


def pack_block_id(zone, blk_x, blk_y, blk_z):
    x7 = blk_x & 0x7F
    y7 = blk_y & 0x7F
    z2 = blk_z & 0x03
    return (zone << 16) | (x7 << 9) | (y7 << 2) | z2


def axis_boundary_flags(coord, log2_b, coord_max):
    mask = (1 << log2_b) - 1
    lo = ((coord & mask) == 0) and (coord != 0)
    hi = ((coord & mask) == mask) and (coord != coord_max)
    return lo, hi


def emit_for_one_voxel(x, y, z, lidar_x=0, lidar_y=800,
                       max_x=0xFFFF, max_y=0xFFFF, max_z=0xFFFF):
    dx = abs(x - lidar_x)
    dy = abs(y - lidar_y)
    dist = max(dx, dy)
    zone, log2_bx, log2_by, log2_bz = zone_lut(dist)

    # 主块索引
    blk_x = arshift_floor(x - lidar_x, log2_bx)
    blk_y = arshift_floor(y - lidar_y, log2_by)
    blk_z = z >> log2_bz

    beats = [ReqBeat(pack_block_id(zone, blk_x, blk_y, blk_z), 0)]

    # 主块边界（只算一次）
    x_lo, x_hi = axis_boundary_flags(x, log2_bx, max_x)
    y_lo, y_hi = axis_boundary_flags(y, log2_by, max_y)
    z_lo, z_hi = axis_boundary_flags(z, log2_bz, max_z)

    bx_on = x_lo or x_hi
    by_on = y_lo or y_hi
    bz_on = z_lo or z_hi

    valid = {
        1: bx_on,
        2: by_on,
        3: bx_on and by_on,
        4: bz_on,
        5: bx_on and bz_on,
        6: by_on and bz_on,
        7: bx_on and by_on and bz_on,
    }

    for h in range(1, 8):
        if not valid[h]:
            continue

        nx, ny, nz = x, y, z
        hdir = 0

        # x轴
        if h & 0b001:
            if x_lo:
                nx -= 1
                hdir |= (DIR_NEG << 0)
            else:
                nx += 1
                hdir |= (DIR_POS << 0)

        # y轴
        if h & 0b010:
            if y_lo:
                ny -= 1
                hdir |= (DIR_NEG << 2)
            else:
                ny += 1
                hdir |= (DIR_POS << 2)

        # z轴
        if h & 0b100:
            if z_lo:
                nz -= 1
                hdir |= (DIR_NEG << 4)
            else:
                nz += 1
                hdir |= (DIR_POS << 4)

        # 每个halo坐标都要重新查zone并算block_id（与RTL一致）
        ndx = abs(nx - lidar_x)
        ndy = abs(ny - lidar_y)
        ndist = max(ndx, ndy)
        nzone, nbx, nby, nbz = zone_lut(ndist)

        n_blk_x = arshift_floor(nx - lidar_x, nbx)
        n_blk_y = arshift_floor(ny - lidar_y, nby)
        n_blk_z = nz >> nbz

        beats.append(ReqBeat(pack_block_id(nzone, n_blk_x, n_blk_y, n_blk_z), hdir))

    return beats
```

## 11. 对接时建议统一的核对项

建议算法仿真和RTL联调时，逐体素逐beat核对：

- beat总数是否一致（1~8）
- halo输出顺序是否一致（固定优先级）
- 每个beat的 `halo_dir` 是否一致
- 每个beat的 `req_block_id` 是否一致（含有符号 `x_local/y_local`）

如果以上四项都一致，通常可判定软件模型已对齐当前硬件实现。

## 12. 版本边界说明

- 本文仅对应当前仓库中的 `online_block_partitioning.sv` + `zone_lut.sv` 行为
- 若后续修改 `zone_lut` 分段或 `block_id` 打包位宽，需要同步更新软件模型

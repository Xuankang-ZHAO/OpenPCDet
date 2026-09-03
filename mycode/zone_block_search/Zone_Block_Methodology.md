# Zone-Aware Variable-Size Block Partitioning：方法论与实验参考

> 本文档用于指导 zone 边界和 block size 的候选生成、少量配置比较与论文表述。
> 核心方法链：**Fig. 5(a) 确立 Chebyshev 距离轴 → 按硬件对齐粒度构造左闭右开的细方形环 → 逐尺寸统计两页覆盖率 → 每个细方形环选择最大通过尺寸 → 合并连续标签 → Golden model 最终取舍。**
>
> 该流程不是全局优化器，也不要求大规模配置扫描。Table I 中各 stage 的 LiDAR reference 是固定输入；现有 zone thresholds 和 block sizes 仅是参考候选，算法的目标是为它们寻找更合适的替代值。

**全文约定：**block 内体素数 $N_b$ 指该 block 实际存储的全部体素记录，即自身 active 体素和从邻块复制来的 halo 体素。页数 $P_b$ 与 reshape 判据使用同一个 $N_b$。当 $N_b>128$、即占用超过两个 64-voxel page 时，block 进入 reshape；这是一条存储侧处理规则，而不是不可违反的配置约束。

---

## 1. 问题、目标与方法边界

Fig. 5(a) 已经说明 voxel density 随距离明显变化，因此一个全局固定 block size 很难同时兼顾密集区与稀疏区。但仅凭曲线形状，还不能为各 stage 产生数据驱动的 zone boundaries 和 block sizes。审稿人可能追问：这些参数是否纯手调，是否比较过邻近配置和固定尺寸基线？

Table I 中不同类型的参数必须明确区分：

| 参数 | 在本方法中的角色 |
|---|---|
| LiDAR reference：stage 0–3 分别为 $(0,800)$、$(0,400)$、$(0,200)$、$(0,100)$ | **固定输入**；用于定义各 stage 的 block-grid anchor 和 zone 中心，不参与搜索 |
| 当前 $T_0/T_1/T_2$ | **参考值**；不是目标答案，由本算法重新生成候选边界 |
| 当前各 zone block size | **参考值**；不是目标答案，由本算法在硬件尺寸菜单中重新选择 |

算法完成后，最终 Table I 保留原 LiDAR-reference 列，并用 Golden model 选出的 thresholds 和 block sizes 更新其余参数列。

本文档要建立的是一条**简洁、可复现的候选生成方法**：

1. 用 Fig. 5(a) 说明 Chebyshev 距离是有效的一维 profiling 轴；
2. 按硬件对齐粒度将距离划成细区间，并对每个合法 block size 直接统计包含 halo 的逐块占用；
3. 在每个细方形环中选择满足两页覆盖率要求的最大 block size；
4. 去除短暂标签抖动并合并连续相同标签，得到原始 zone 边界和尺寸；
5. 最后根据 zone 数量限制生成少量合并/邻近候选，用 Golden model 选择用于更新 Table I 的最终配置。

上述流程先在一个 stage 上建立并验证，随后对其余 stage 独立重复。每个 stage 使用自己的 occupancy profile、block-size 菜单和 zone thresholds；一个 stage 的具体阈值和尺寸不能简单按比例缩放为另一个 stage 的参数。

因此，本文不声称解析推导能唯一确定最优配置，也不做所有阈值、尺寸和代价因素的完整扫描。设计口径是：

> **profile-guided candidate generation, followed by trace-driven selection.**

---

## 2. 符号与口径

| 符号 | 含义 | 备注 |
|---|---|---|
| $s$ | SECOND backbone 的 stage 编号（0–3） | 每 stage 的 $xy$ 网格尺寸减半 |
| $d$ | BEV 平面到该 stage 的 LiDAR reference $(x_{0,s},y_{0,s})$ 的 Chebyshev 距离 | $d=\max(\lvert x-x_{0,s}\rvert,\lvert y-y_{0,s}\rvert)$ |
| $N_s(d)$ | 距离 $d$ 的可见环上、沿 $z$ 求和的非空体素数 | 统计空间中的唯一体素，不重复计算 halo |
| $L_s(d)$ | 该 Chebyshev 环上位于有效 FOV 内的 $xy$ 格位数 | 前向 FOV 中通常对应三条可见边 |
| $\rho_s(d)$ | $N_s(d)/L_s(d)$ | 每个 $xy$ 格位上整列非空体素数的径向平均估计，可大于 1 |
| $b$ | 一个 block | 核心尺寸为 $B_x\times B_y\times B_z$ |
| $b^{+}$ | $b$ 的 halo 扩展 footprint | kernel=3 时，每个需要 halo 的方向向外扩 1 格 |
| $B_k$ | 第 $k$ 个候选 block shape | $B_{x,k}\times B_{y,k}\times B_{z,k}$；按体积从小到大排序 |
| $G_{z,s}$ | stage $s$ 的有效 voxel-space $z$ 尺寸 | $B_z<G_{z,s}$ 时存在多个 $z$-blocks，必须统计 $z$-halo |
| $\mathcal B_s^{x<y}$、$\mathcal B_s^{x>y}$ | stage $s$ 的两个方向固定 block-size 菜单 | 非正方形尺寸分别满足 $B_x<B_y$ 和 $B_x>B_y$；正方形尺寸由两者共享 |
| $N_b$ | block $b$ 实际存储的体素记录数 | 包含自身 active 体素和 halo 复制 |
| $C_P$ | 一个 DRAM page 的体素容量 | $C_P=64$ |
| $C_{2P}$ | 两页直接处理阈值 | $C_{2P}=2C_P=128$；$N_b>C_{2P}$ 时 reshape |
| $P_b$ | block 页数 | $P_b=\lceil N_b/C_P\rceil$ |
| $\Delta d_s$ | stage $s$ 的 profiling 距离粒度和边界量化单位 | stage 0 取 64；宜由候选平面边长的最小公倍数确定 |
| $Z_{j,s}$ | stage $s$ 的第 $j$ 个细方形环 | 两个左闭右开嵌套矩形之差，详见2.1 |
| $R_{j,k,s}$ | stage $s$ 的细方形环 $j$ 使用尺寸 $k$ 时的两页覆盖率 | 200帧样本中满足 $N_b\le128$ 的 block 比例 |
| $q$ | 候选筛选所用分位数 | 工作值为 P95；不是正确性条件 |
| $T_{i,s}$ | stage $s$ 独立确定的第 $i$ 个 zone 边界 | 原始 zones 可有多个边界；最终为4个 zone 时对应 $T_{0,s},T_{1,s},T_{2,s}$ |

profiling 中的统计对象是**实际被物化并分配页面的 block**。除核心区域含有 active voxel 的 block 外，自身核心为空、但因相邻 block 的 halo 复制而含有存储记录的 halo-only block 也会被物化并分配页面。因此，halo-only block 同样纳入 $N_b$、物化 block 数、页面数和两页覆盖率统计；该口径在所有候选尺寸下保持一致。

### 2.1 Zone 的左闭右开边界

Fig. 5(a) 可以继续使用对称的 Chebyshev 距离 $d$ 描述总体密度趋势，但实际 zone 归属采用以 LiDAR reference 为中心的**左闭右开矩形**。对于 stage $s$ 和阈值 $T$，一个坐标位于该矩形内，当且仅当以下两个条件同时成立：

$$
-T\le x-x_{0,s}<T,
$$

$$
-T\le y-y_{0,s}<T.
$$

记上述判断为 `inside(T)`，则最终四个 zones 定义为：

- Zone 0：`inside(T0)`；
- Zone 1：`inside(T1)`，但不属于 Zone 0；
- Zone 2：`inside(T2)`，但不属于 Zone 0/1；
- Zone 3：其余有效 FOV。

下面只借用当前 Table I 的 stage-0 数值解释半开边界语义。LiDAR reference $(x_{0,0},y_{0,0})=(0,800)$ 是固定输入，而64/512/768是待搜索的参考 thresholds，不表示算法必须保留这些边界：

- Zone 0：$x\in[-64,64)$，且 $y-800\in[-64,64)$；
- Zone 1：$x\in[-512,512)$ 且 $y-800\in[-512,512)$，但不属于 Zone 0；
- Zone 2：$x\in[-768,768)$ 且 $y-800\in[-768,768)$，但不属于 Zone 0/1；
- Zone 3：其余区域。

因此，$x-x_{0,0}=-64$ 属于 Zone 0，而 $x-x_{0,0}=+64$ 已属于 Zone 1。该约定保证每个坐标只属于一个 zone，并使正边界成为外层 zone 中第一个 block 的起点。

算法得到新的 thresholds 后，仍按完全相同的左闭右开规则解释新边界。例如新的 $T_{0,0}$ 若不是64，上述区间中的64应整体替换为新值，而 LiDAR reference $(0,800)$ 保持不变。

同一规则也用于 profiling 的细方形环。第0个细区间为 `inside(delta_d)`；对于 $j\ge1$，$Z_{j,s}$ 位于外层矩形 `inside((j+1)delta_d)` 内，但不属于内层矩形 `inside(j*delta_d)`。实现时必须分别检查 $x$ 和 $y$ 的上下界，不能直接用对称的 $d<T$ 判断，因为后者会把 $-T$ 与 $+T$ 放到同一侧。

---

## 3. Fig. 5(a) 能说明什么

### 3.1 径向密度是一阶统计

$\rho_s(d)$ 是同一 Chebyshev 环上所有有效 $xy$ 格位的平均。对于中心位于距离 $d_b$、平面尺寸为 $B_x\times B_y$ 的 block，其带 halo 的矩形平面范围为 $(B_x+2)\times(B_y+2)$。当 block 覆盖完整 $z$ 列且内部密度变化不大时，可以用

$$
\overline N_b\approx \rho_s(d_b)(B_x+2)(B_y+2)
$$

粗略估计 block 的平均占用。这个公式只表达“密度越高，相同 block 中的体素通常越多”，不用于直接决定最终尺寸。由于 $\rho_s(d)$ 平均了同一距离上的不同方位，而且候选 block 可能只覆盖一部分 $z$ 范围，实际尺寸选择仍使用后续逐 block 统计。

Halo 统计本身是三维规则，不因 stage 编号而改变。对于任一 stage，只要 $B_z<G_{z,s}$，voxel space 就会被分成多个 $z$-blocks，每个 block 都应像 $x/y$ 方向一样从相邻 $z$-block 复制所需 halo；位于整个 voxel space 上下边缘的 block 只复制实际存在的一侧。若 $B_z\ge G_{z,s}$、一个 block 已覆盖完整 $z$ 维度，则不存在内部 $z$ 边界，也不产生 $z$ 向复制。最终 $N_b$ 始终由 $x/y/z$ 三个方向的真实 halo 扩展结果共同决定。

因此，Fig. 5(a) 适合回答：

- 哪些距离段密度较高，初始 block 应较小；
- 哪些距离段密度较低，可以尝试较大的 block；
- 哪些位置附近值得放置 zone 边界候选。

它不单独决定最终尺寸，也不需要承担尾部建模任务。

### 3.2 为什么还需要块级 P95

径向平均会抹掉方位角差异、帧间变化和局部聚簇。相同 $\rho_s(d)$ 下，不同 block 的 $N_b$ 仍可能相差很大。因此，用 Fig. 5(a) 确定距离轴和大致结构后，再以直接测量的两页覆盖率 $R_{j,k,s}$ 判断某个具体尺寸是否经常超过两页。

Fig. 5(a) 的 P10–P90 带与块级 P95 不需要一致：前者描述逐帧径向平均密度的波动，后者描述逐块存储占用的上尾。

---

## 4. 可操作的分箱搜索算法

### 4.1 算法性质

该方法是一个**离线、profile-guided、基于经验分位数约束的离散网格搜索**。更具体地说，它对每个左闭右开的细方形环独立选择满足两页覆盖率要求的最大 block size，再通过标签去抖和连续区间合并提取 zone 边界。

它只遍历单个 stage 的有限 block-size 菜单，不扫描 zone 阈值与尺寸的全部笛卡尔积。zone 边界是逐环尺寸标签变化后的输出，而不是预先参与联合搜索的参数。

### 4.2 Stage 0 的固定输入与搜索空间

以 stage 0 为例，先固定以下输入：

- $B_z=16$。本轮搜索只调整 $xy$ 尺寸，$z$ 尺寸作为预先选定的设计参数；
- $C_{2P}=128$。恰好128个体素仍占两页，因此通过条件使用 $N_b\le128$；
- $q=0.95$。即初始尺寸要求至少约95%的 profiling block 不触发 reshape；
- $\Delta d_0=64$，所有原始 zone 边界均写成 $T_{i,0}=64m$，其中 $m$ 是非负整数；
- 一个对所有候选尺寸统一的 block-grid anchor，固定为 Table I 中 stage 0 的 LiDAR reference $(x_{0,0},y_{0,0})=(0,800)$。

这里“以 LiDAR reference 为 anchor”是指：LiDAR reference 是所有 block 边界网格的共同基准交点，而不是某个 block 的几何中心。对于候选尺寸 $B_x\times B_y\times16$，平面中的 blocks 按以下区间向四周铺设：

$$
x\in[x_{0,0}+uB_x,\ x_{0,0}+(u+1)B_x),
$$

$$
y\in[y_{0,0}+vB_y,\ y_{0,0}+(v+1)B_y),
$$

其中 $u$ 和 $v$ 为整数。对任一 voxel，block 坐标应先减去 LiDAR reference 再做向下整除：

$$
u=\left\lfloor\frac{x-x_{0,0}}{B_x}\right\rfloor,\qquad
v=\left\lfloor\frac{y-y_{0,0}}{B_y}\right\rfloor.
$$

这与直接计算 $\lfloor x/B_x\rfloor$、$\lfloor y/B_y\rfloor$ 不同，后者以全局坐标零点为分块起点，不能保证 LiDAR-centered zone 边界与 block grid 对齐。软件实现对负偏移量必须采用数学意义上的向下整除，而不是向零截断。铺设时只保留与 stage-0 有效 FOV 相交的 blocks，并在最外侧按 FOV 裁剪。

stage 0 将候选尺寸分为两个方向固定的菜单。第一套菜单中的非正方形尺寸均满足 $B_x<B_y$：

$$
\mathcal B_0^{x<y}=\{
8{\times}8{\times}16,
8{\times}16{\times}16,
16{\times}16{\times}16,
16{\times}32{\times}16,
32{\times}32{\times}16,
32{\times}64{\times}16,
64{\times}64{\times}16
\}.
$$

第二套菜单中的非正方形尺寸均满足 $B_x>B_y$：

$$
\mathcal B_0^{x>y}=\{
8{\times}8{\times}16,
16{\times}8{\times}16,
16{\times}16{\times}16,
32{\times}16{\times}16,
32{\times}32{\times}16,
64{\times}32{\times}16,
64{\times}64{\times}16
\}.
$$

两套菜单都按体积从小到大排列，相邻档体积相差2倍，最大尺寸由硬件支持范围决定。$8\times8$、$16\times16$、$32\times32$ 和 $64\times64$ 等正方形尺寸在两套菜单中相同，其 profiling 结果可以直接复用；只有三档矩形尺寸需要分别统计两个方向。

算法分别对 $\mathcal B_0^{x<y}$ 和 $\mathcal B_0^{x>y}$ 完整运行一次。每次运行中，所有 zones 只能从当前方向菜单选择尺寸，不在同一初始配置中混合 $B_x<B_y$ 与 $B_x>B_y$ 的矩形 blocks。这样可以比较前向 FOV 下两种方向的整体差异，同时避免把“矩形方向”扩展成逐 zone 的组合搜索维度。

$\Delta d_0=64$ 不只是任意采样步长：它也是所有候选 $x/y$ 边长的公倍数。阈值 $T=64m$ 对应相对坐标范围 $[-64m,64m)$，四条 block-grid 边界分别位于 $x=x_{0,0}-64m$、$x=x_{0,0}+64m$、$y=y_{0,0}-64m$ 和 $y=y_{0,0}+64m$。这些位置同时是所有候选尺寸的 block 边界，所以每个细方形环都能完整切分 blocks，不会在内部 zone 边界上截断 block。左侧/下侧边界上的坐标属于内层矩形，右侧/上侧边界上的坐标属于外层矩形。

对其他 stage，同样以 Table I 中该 stage 的 $(x_{0,s},y_{0,s})$ 为 anchor，并把两套方向菜单中所有候选 $B_x$ 和 $B_y$ 的最小公倍数作为 $\Delta d_s$；当候选尺寸均为2的幂次时，它通常就是最大的平面边长。也可以取该数值的整数倍作为更粗的距离粒度。

### 4.3 逐尺寸全局 profiling

先选择一个方向菜单，再对该菜单中的每个 $B_k$ 独立执行：

1. 从 LiDAR reference $(x_{0,0},y_{0,0})$ 定义的共同 block-boundary anchor 出发，用 $B_k$ 向有效 FOV 均匀铺设整个 stage-0 网格；
2. 按真实三维边界复制规则构造每个 block 的 halo：$x/y$ 方向始终按相邻 blocks 处理；当 $B_z<G_{z,0}$ 时，$z$ 方向也按相同规则处理，确保同一 voxel 在同一 block 内只计一次；
3. 对200个 profiling frames，收集每个实际物化 block 的 $N_b$，其中包括核心区域为空但因 halo 复制而被物化的 halo-only block；
4. 按2.1的左闭右开矩形判断，将完整 block 核心放入对应的细方形环 $Z_{j,0}$；
5. 对每个“细方形环 × block size”组合，汇总200帧中所有实际物化 block 的 $N_b$。

由于 block grid 和 zones 使用同一个 LiDAR reference，且所有候选平面边长均整除64，任何内部 block 核心都不会跨越细方形环边界。因此，block 可按其完整核心区域直接归入对应区间；只有有效 FOV 最外缘允许出现裁剪 block。若脚本在内部64倍数边界发现跨界 block，说明 anchor、半开区间或整数除法实现不一致，应修正分块，而不是按 block 中心距离近似归箱。

200帧合并后的统计单位是 **block-frame instance**。这种 pooled 统计会让 block 较多的帧贡献更多样本，与“所有实际 block 中有多少需要 reshape”的目标一致。统计200帧内各细方形环的全部物化 block，并同时记录样本数；若极远端区间几乎没有有效样本，则不依据该区间单独确定尺寸。

### 4.4 两页覆盖率与逐环尺寸选择

对每个细方形环 $j$ 和尺寸 $k$，计算两页覆盖率

$$
R_{j,k,0}
=\frac{N_b\le128\text{ 的 block 数}}{\text{该细方形环的全部物化 block 数}}.
$$

$R_{j,k,0}\ge0.95$ 就表示至少约95%的样本不超过两页，也就是该细方形环的经验 P95 不超过128。

随后，在当前方向菜单内，对每个宽度为64的细方形环选择满足覆盖率要求的最大尺寸：

$$
B^*(j)=\text{满足 }R_{j,k,0}\ge q\text{ 的最大 }B_k.
$$

当前方向菜单中的所有候选尺寸都应完成统计后再取最大通过者，不依赖覆盖率随尺寸严格单调的假设。若所有尺寸都通过，则选菜单中的最大尺寸；若所有尺寸都不通过，则以最小尺寸作为基础标签，并允许剩余异常 block reshape。两个方向菜单分别得到自己的标签序列和原始 zones。

$B^*(j)$ 是该细方形环的 **P95-guided preferred size**，不是全局最优尺寸。尺寸大于它仍然是合法配置，只是更可能触发 reshape，可在后续少量候选中保留。

### 4.5 标签去抖与原始 zone 提取

每个方向菜单都会形成一条离散标签序列。以下以 $B_x<B_y$ 菜单为示意；$B_x>B_y$ 菜单按相同步骤生成另一条序列：

```text
细方形环： Z_0      Z_1       Z_2       Z_3      ...
尺寸标签： 32x64    16x32     16x16     16x16   ...
```

先执行一次简单去抖：只有当新尺寸标签连续出现至少两个细方形环时才确认换档；等价地，也可以对标签序列使用宽度为3的中值滤波。该规则只用于消除单环统计抖动，不改变持续存在的密度结构。

之后对连续相同标签执行 run-length merging。例如，$Z_2$ 和 $Z_3$ 连续采用 $16\times16\times16$ 时，可合并为同一个 zone，其内外边界分别为128和256。标签发生稳定变化的位置就是原始 zone boundaries。Fig. 5(a) 局部可能存在次峰，因此原始 zone 数可以多于4，也不要求不同 stage 得到相同的 zone 数。

### 4.6 硬件收敛与少量 Golden-model 候选

两个方向菜单得到的原始 zones 分别作为 profiling 输出保留，不立即强制压缩成4个。需要映射到最终硬件时，对每个方向族分别执行：

1. 合并对主要密度结构影响最小的短区间；
2. 对有歧义的合并，分别保留相邻尺寸中较小档和较大档作为两个候选；
3. 在稳定换档点附近，只考虑边界移动一个 $\Delta d_s$ 的邻近方案；
4. 加入当前 Table I 的对应 stage 参数作为参考候选；
5. 每个方向菜单只保留一个基础方案及必要的邻近方案，再将两套方向候选合并交给 Golden model；总候选数量控制在约 4–10 组。

合并到硬件允许的 zone 数以后，必须按最终可变尺寸布局重新执行 halo、页面和 reshape 统计。原始逐环覆盖率用于生成候选，不能代替对最终配置的验证。

Golden model 首先比较 $B_x<B_y$ 与 $B_x>B_y$ 两个方向族，再在较优方向附近比较少量边界或尺寸变体。建议以总 DRAM traffic 或总页面访问量作为主要选择指标；reshape 比例、halo 复制比和 active block 数用于解释结果，不建立复杂的多目标加权函数。

### 4.7 分位数敏感性

$q=0.95$ 是候选筛选的 profiling 参数，不是正确性条件。它表示在200帧合并样本中，目标尺寸约有不超过5%的 block-frame instances 超过两页；它不保证 held-out 数据上严格不超过5%，也不代表受影响的体素或流量比例只有5%。

若需要敏感性分析，只需将 $q$ 改为0.90或0.98，重新执行逐环标签选择和合并，观察哪些稳定换档点发生变化。$q=0.98$ 比0.95更保守，$q=0.90$更宽松。对于较高分位数，必须同时报告各细方形环的有效样本数。

### 4.8 各 stage 独立重复

stage 0 可以作为论文和实验记录中的完整示例。完成相同脚本后，对 stage 1–3 分别输入自己的 $B_z$、两个方向菜单、$\Delta d_s$、occupancy traces 和 Table I 中的 LiDAR reference，并以各自的 LiDAR reference 重新锚定 block grid，独立执行4.3–4.7。

不同 stage 的 zone boundaries、block sizes 和原始 zone 数都可以不同；一个 stage 的结果不能简单按比例缩放为另一个 stage 的结果。最终只需将各 stage 经 Golden model 选出的配置汇总到 Table I。

---

## 5. 建议报告的统计量

这些统计量用于解释 Golden model 的选择，不需要全部进入候选生成规则：

1. **两页覆盖率**：满足 $N_b\le128$ 的 block 比例，以及 $P_b=1/2/>2$ 的 block 比例；
2. **reshape 影响**：reshape block 比例及其额外字节/页面比例，优先按流量报告；
3. **页利用率**：
   $$U=\frac{\sum_bN_b}{C_P\sum_bP_b};$$
4. **halo 复制比**：
   $$R_h=\frac{\sum_bN_b-N_{\mathrm{unique}}}{N_{\mathrm{unique}}};$$
5. **active block 数与总页面数**：用于说明小 block 和大 block 的主要取舍。

固定尺寸基线至少包含多个全局固定尺寸和 per-stage 最佳固定尺寸。固定尺寸与 zone-aware 必须使用相同的 halo、页分配和 reshape 规则。

---

## 6. 精简实验流程

```text
Step 0  数据切分
  → Step 1  固定该 stage 的 B_z、两个方向菜单、距离粒度和 grid anchor
  → Step 2  分别遍历两个菜单中的候选尺寸，收集各细方形环的 N_b 样本
  → Step 3  计算 R_j,k,s，并给每个区间标记最大通过尺寸
  → Step 4  标签去抖 + run-length merging，得到原始 zones
  → Step 5  按硬件数量合并并用 Golden model 比较少量候选
  → Step 6  held-out 数据上报告结果并比较固定尺寸基线
```

**Step 0 — 数据切分。** 从 KITTI training split 中选取约 200 帧作为 profiling 集；held-out validation 帧只用于最终报告。所有参数选择和 Golden-model 候选取舍均在 profiling 集上完成。

**Step 1 — 固定搜索输入。** Fig. 5(a) 的 $\rho_s(d)$ 负责说明采用 Chebyshev 距离分区的动机。对当前 stage 固定 $B_z$、$B_x<B_y$ 与 $B_x>B_y$ 两套菜单、$q$ 和 $\Delta d_s$，并把 Table I 中该 stage 的 LiDAR reference 设为所有候选尺寸共同的 block-boundary anchor；stage 0 使用4.2给出的具体参数。

**Step 2 — 逐尺寸全局 profiling。** 分别对两个方向菜单中的每个 block shape 均匀铺满整个网格，并在所有发生分块的方向执行真实 halo 复制；特别地，只要 $B_z<G_{z,s}$，就必须包含 $z$ 向 halo。随后按2.1定义的细方形环汇总200帧中每个物化 block 的 $N_b$，并记录各组合的样本数；正方形尺寸的结果由两个菜单复用。

**Step 3 — 逐环标记。** 计算 $R_{j,k,s}$，并在每个 $Z_{j,s}$ 中选择满足 $R_{j,k,s}\ge q$ 的最大尺寸。统计时同时记录各细方形环的样本数；若极远端区间几乎没有有效样本，则不依据该区间单独确定尺寸。

**Step 4 — 提取原始 zones。** 两个方向菜单分别对尺寸标签使用“两区间持续”规则或3-bin中值滤波去抖，再通过 run-length merging 合并连续相同标签。此时允许两套结果的原始 zone 数不同且多于4。

**Step 5 — 硬件收敛与 Golden model 取舍。** 分别根据该 stage 的硬件 zone 数量限制合并两个方向族的原始 zones，在稳定边界附近生成少量邻近方案，并加入当前 Table I 的对应配置。按最终可变尺寸布局重新统计后，用 Golden model 先比较方向族，再选择该 stage 的最终参数。

**Step 6 — held-out 验证。** 在 validation traces 上一次性报告最终 zone-aware 配置、多个固定尺寸和 per-stage 最佳固定尺寸的 block 数、页面数、halo、reshape 和 DRAM 指标。

---

## 7. 论文写作落点

### 7.1 建议正文段落

> Both the block sizes and the zone thresholds are generated by a profile-guided procedure rather
> than being manually fixed. Only the stage-wise LiDAR references listed in Table I are retained as
> fixed inputs; the listed thresholds and block sizes are reference candidates to be reselected by
> this procedure. Fig. 5(a) first establishes Chebyshev distance as the profiling axis.
> Rectangular block sizes are organized into two orientation-specific menus, one with $B_x<B_y$ and
> the other with $B_x>B_y$; square sizes are shared, and the two menus are profiled independently.
> For each supported block size, we uniformly partition the grid from a common block-boundary anchor
> located at the stage-wise LiDAR reference rather than at the global coordinate origin, replicate the
> halo in every partitioned dimension (including $z$ whenever $B_z$ is smaller than the stage's voxel-space
> depth), and pool the stored voxel counts of materialized blocks over 200 profiling frames. Each zone
> uses half-open bounds, $-T\le x-x_0<T$ and $-T\le y-y_0<T$, so the negative boundary remains in
> the inner zone while the positive boundary starts the outer zone. The profiling axis is divided into
> correspondingly aligned square rings (64 voxels wide for Stage 0), and each ring selects the largest
> block size for which at least 95% of the samples contain no more than 128 voxels, corresponding to
> two 64-voxel pages; larger blocks remain valid and are handled by reshape. Short label fluctuations
> are suppressed and consecutive bins with the same selected size are merged, so their transition
> points directly form the initial zone boundaries. If the resulting number of zones exceeds the
> hardware limit, we evaluate only a few adjacent merge and boundary variants from both orientation
> families with the trace-driven Golden model and revalidate the final variable-size layout. The same profiling procedure is
> independently repeated for the remaining stages, since each stage has its own occupancy profile,
> block-size menu, distance granularity, and zone thresholds. This procedure narrows the design space
> without claiming a closed-form global optimum, while outlier reshape preserves correctness.

### 7.2 防御口径

- 本方法不是逐帧自适应划分，而是低成本的离线 profiling 加静态配置；
- Fig. 5(a) 负责说明距离轴，逐环两页覆盖率负责产生尺寸标签，Golden model 负责最终选择；
- Fig. 5(a) 的对称 Chebyshev 距离只用于统计趋势；实际 zone 和 profiling bins 均使用 $x/y$ 同时判断的左闭右开矩形；
- 不声称 optimal，贡献点是避免纯经验手调并显著缩小候选空间；
- profiling 偏差只会使 reshape 或页面开销增加，不会造成存储溢出错误；
- novelty 表述限定为在 block-based SCONV accelerator 中使用 zone-aware variable-size partitioning，不与通用八叉树或算法层自适应分区混为一谈。

---

## 8. 明确不做的事

- 不扫描全部 zone 阈值与 block-size 笛卡尔积；
- 不建立多目标加权优化器、动态规划或逐帧 oracle；
- 不把计算模式、搜索能耗、搜索阵列深度或分段方式纳入 zone 选择规则；
- 不做逐帧八叉树、自适应建树或块内体素均衡；
- 不声称经验覆盖率规则给出全局最优解；
- 不引入第二个 occupancy 定义，$N_b$ 始终表示实际存储记录数并包含 halo；
- 正文方法控制在一个短段落，逐环覆盖率和标签序列可放补充材料或实验记录。

---

## 9. 实施前仍需确定的输入

1. **合法 block 菜单。** 确认 stage 1–3 的两套完整 $B_x\times B_y\times B_z$ 菜单；非正方形尺寸分别保持 $B_x<B_y$ 或 $B_x>B_y$，正方形尺寸共享。

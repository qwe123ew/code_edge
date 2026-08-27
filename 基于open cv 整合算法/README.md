# 整合 partition.py + colorize.py  任务 2/3/4 实现

本目录把 `C:\Users\18607\Desktop\task\code\partition.py` (边缘划分) 和
`C:\Users\18607\Desktop\task\code\colorize.py` (染色) 两个算法整合到统一流水线,
完成 docx 任务 2 (划分) + 3 (染色) + 4 (四色定理染色), 并附带三个**亮点**:

1. **闭合轮廓 = 物体边界** (`partition.is_closed`): 体现"边缘和物体的关系"
2. **区域级四色染色** (`colorize.region_four_color`): 闭运算 → 填充 → 区域邻接图 → DSATUR, 这才是四色定理的本义场景(地图染色)
3. **边缘闭合完整性分析** (`edge_closure.py`): 缺口检测 + 阶梯式形态学闭合, 衡量"边缘-物体"对齐度

## 文件

- [partition.py](partition.py) — 从 `C:\Users\18607\Desktop\task\code\partition.py` **原样搬入**
  - `zhang_suen_thin(binary)` — Zhang-Suen 细化
  - `neighbor_count(mask)` — 8-邻域计数
  - `partition_edges(edge_mask, min_len, endpoint_adj_r)` — 主划分函数
    - **关键**: 端点数 == 0 的段标记为 `is_closed` (闭合轮廓 = 物体边界)
    - **关键**: 用 `cKDTree` 给"端点距离 ≤ 2.5px"的段加邻接(缝合小断裂)
- [colorize.py](colorize.py) — 从 `C:\Users\18607\Desktop\task\code\colorize.py` **搬入并修正一处 off-by-one bug**
  - 修正: `region_four_color` 原来 adj 1-based 但 dsatur 0-based, 导致运行时 KeyError
  - `distinct_palette(K)` / `PALETTE4`
  - `render_segments(lab, palette, jmask, junction_color)`
  - `dsatur(adj, n_nodes, max_colors)` / `_backtrack_color(adj, order, n_colors)`
  - `four_color(adj, n_nodes, n_colors)` — DSATUR + 回溯兜底
  - `region_four_color(edge_mask)` — 区域级四色 (闭运算+填充+区域邻接+染色)
- [edge_closure.py](edge_closure.py) — **新增**: 边缘闭合完整性分析
  - `analyze_endpoint_gaps(lab, edge, max_gap)` — 端点 + cKDTree 找可缝合缺口对
  - `close_edge_gaps(edge, kernel_size, iterations)` — `cv2.morphologyEx(MORPH_CLOSE)`
  - `compute_closure_metrics(lab, edge)` — 5 项闭合性指标
  - `render_gap_visualization` / `render_closed_edge` — 可视化
- [run_integrated.py](run_integrated.py) — 5 张图端到端驱动 (~310 行)
  - 复用 `open cv 实现/canny_edge_detection.py` (模块默认阈值 10/100)
  - 调用 `partition.partition_edges` → `colorize.render_segments` → `colorize.four_color` → `colorize.region_four_color` → `edge_closure.*`
  - 写 **10 个 PNG + 1 个 JSON** 每张图
- [outputs/](outputs/) — 5 张测试图的全部产物

## 算法流水线

```
原图
  │
  ▼  (复用 open cv 实现)
Canny 10/100
  │
  ▼
partition.partition_edges(edge, min_len=4, endpoint_adj_r=2.5)
  ├─ zhang_suen_thin        ─► 1px 骨架
  ├─ neighbor_count          ─► 8-邻域数
  ├─ nc >= 3 → 交叉点; 移除 → 8-CC 标段
  ├─ cKDTree 端点距离邻接    ─► 缝合小断裂
  └─ ep_count == 0           ─► is_closed (闭合轮廓 = 物体边界)
  │
  ├─► colorize.render_segments        → 任务 3 黄金角逐段染色
  │
  ├─► colorize.four_color (DSATUR+回溯) → 任务 4 段级四色染色
  │
  ├─► colorize.region_four_color      → 亮点: 区域级四色 (闭运算+填充+区域邻接)
  │
  └─► 统计: n_segments / n_adjacent_pairs / n_closed_contours / n_regions / 4color stats
  │
  ▼  (新增) 边缘闭合完整性
edge_closure.analyze_endpoint_gaps  ─► 缺口对 (gap_list)
edge_closure.close_edge_gaps 3→5→7   ─► 阶梯式 MORPH_CLOSE
edge_closure.compute_closure_metrics ─► 闭合率 / 端点数 / 段数 (前后)
```

## 输出产物

每张图 **10 个 PNG + 1 个 JSON**:

| 文件 | 内容 |
|------|------|
| `<图名>_1_边缘图.png` | 输入的二值边缘图 (白底黑边) |
| `<图名>_2_结点拆分.png` | 骨架 (黑) + 交叉点 (红) |
| `<图名>_3_逐段染色.png` | 任务 3: 黄金角逐段染色 (结点白) |
| `<图名>_4_四色染色.png` | 任务 4: DSATUR 4-色染色 (结点黑) |
| `<图名>_5_原图叠加.png` | 任务 3 彩色边叠回原图 (alpha=0.6) |
| `<图名>_6_区域四色.png` | **亮点**: 区域级四色染色 (地图染色) |
| `<图名>_7_闭合轮廓.png` | **亮点**: 闭合边缘 (候选物体边界) 红色高亮 |
| `<图名>_8_缺口分析.png` | **亮点**: 端点(蓝点) + 缺口对(红线) + 缺口数 |
| `<图名>_9_闭合边缘.png` | **亮点**: 阶梯式 MORPH_CLOSE (kernel 3→5→7) 后边缘 |
| `<图名>_10_闭合后划分.png` | **亮点**: 闭合后再划分的逐段染色 (段会合并) |
| `<图名>_统计.json` | 全套统计 (含闭合性 4 项指标) |
| `summary.json` | 5 张图汇总 |

## 5 张测试图运行结果 (Canny 10/100, min_len=4, endpoint_r=2.5)

### 主指标

| 图片 | 边缘 | 骨架 | 结点 | 段数 | 邻接对 | 闭合轮廓 | 区域数 | 区域4色 | 段4色 | 段冲突 | 耗时 |
|------|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|---:|---:|
| 图片物体边缘的提取分割 | 7559 | 7290 | 2029 | 263 | 29 | 0 | 42 | 2 | 2 | **0** | 203 ms |
| bsds_368037 | 15887 | 14941 | 7531 | 488 | 46 | 0 | 69 | 3 | 3 | **0** | 725 ms |
| bsds_97010 | 26486 | 24765 | 13405 | 864 | 135 | 0 | 87 | 3 | 2 | **0** | 1115 ms |
| nyud_5017 | 4924 | 4748 | 1503 | 167 | 19 | 0 | 42 | 3 | 3 | **0** | 98 ms |
| nyud_6233 | 3953 | 3732 | 1472 | 119 | 7 | 0 | 27 | 3 | 3 | **0** | 79 ms |

**5 张图段级四色冲突 = 0, 区域级四色 = DSATUR 在 2-3 色内解出 (平面图必 ≤4 色)**。

### 边缘闭合完整性 (新增, 亮点 3)

阶梯式 MORPH_CLOSE 桥接策略: `kernel_size=3 → 5 → 7`, 各迭代 1 次。
单次 `MORPH_CLOSE` 可桥接 ≤ kernel/2 像素的缺口; 阶梯式叠加可处理 1-3px 缺口。

| 图片 | 可缝合缺口对 | 平均缺口 | 段数 (前) | 段数 (后) | 段减少 | 端点 (前) | 端点 (后) | 端点减少 | 完美闭合段 (前→后) |
|------|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| 图片物体边缘的提取分割 | 464 | 3.79 px | 263 | 121 | **54.0%** | 526 | 242 | 54.0% | 0 → 0 |
| bsds_368037 | 614 | 3.74 px | 488 | 417 | **14.5%** | 976 | 834 | 14.5% | 0 → 0 |
| bsds_97010 | 1649 | 3.69 px | 864 | 214 | **75.2%** | 1728 | 428 | 75.2% | 0 → 0 |
| nyud_5017 | 254 | 3.72 px | 167 | 110 | **34.1%** | 334 | 220 | 34.1% | 0 → 0 |
| nyud_6233 | 122 | 3.84 px | 119 | 105 | **11.8%** | 238 | 210 | 11.8% | 0 → 0 |

#### 闭合性结论

- **缺口规模**: 5 张图平均缺口 3.7-3.8 px, 单步 kernel=3 已能处理大部分; 阶梯式 3→5→7 可稳定覆盖 1-3px 缺口
- **段数大幅减少**: 闭合后段数减少 11.8%-75.2%, 说明很多原本的"开放段"被桥接合并为更长的段
- **closure_ratio 仍为 0**: 5 张图闭合前后完美闭合段数都 = 0, 因为 Canny 10/100 边带过于碎片化, 大量"应该闭合的环"在多处断裂, 即便 kernel=7 也无法把所有缺口拼回去
- **mean_endpoints_per_segment = 2.0**: 全部段都是带 2 个端点的开放线段, 没有端点数 = 0 的真正闭合环
- **本质原因**: 这是 Canny 低阈值的"过检"代价 — 阈值越低边缘越密但碎片越严重, 阈值越高碎片越少但漏检越多。**完全闭合需要更高级的边缘追踪 (edge contour following) 或图割后处理**, 单纯的形态学闭合并不能恢复全局闭合性

#### 闭合性指标定义

| 指标 | 公式 | 含义 |
|------|------|------|
| `n_segments` | `lab.max()` | 当前划分出的总段数 |
| `n_closed` | `端点数 == 0 的段` | 端点数 = 0 的段, 几何上为闭合环 |
| `n_endpoint_total` | `所有段端点之和` | 开放度总量 (越多越碎) |
| `closure_ratio` | `n_closed / n_segments` | 闭合率 ∈ [0, 1], 越大越闭合 |
| `mean_endpoints_per_segment` | `n_endpoint_total / n_segments` | 平均每段端点数, 2.0 表示全为开放线段 |

## 算法亮点详解

### 亮点 1: 闭合轮廓 = 物体边界

`partition.partition_edges` 不仅返回 `lab/jmask/adj`, 还在 `meta['is_closed']`
中标记每段是否为闭合轮廓 (端点数 == 0)。一條段无端点说明它在空间上形成
一个闭合环, 即可作为该物体的边界候选。统计中 `n_closed_contours` 即为
候选物体数 (本次 5 张图均为 0, 因为 Canny 10/100 边带经常有 1-2px 断裂)。

### 亮点 2: 区域级四色染色

`colorize.region_four_color` 把边缘图当成"地图边界":
1. 闭运算 (3×3 kernel, 2 次) 缝合小缺口
2. `binary_fill_holes` 填充闭合轮廓
3. 挖掉边缘线 → 内部连通区域
4. 各区域膨胀 2px → 检测"隔边缘相邻" → 建区域邻接图
5. DSATUR 4-色染色

这才是四色定理的经典场景 (国家地图着色), 比"边缘段邻接图"更直观。

### 亮点 3: 边缘闭合完整性 (新增)

`edge_closure.py` 把"边缘和物体能否对齐"作为可量化的指标:
1. **缺口检测**: 端点 (8-邻域数 == 1) + `cKDTree.query_pairs` 找 ≤5px 的可缝合缺口对
2. **阶梯式形态学闭合**: `MORPH_CLOSE` kernel 3→5→7, 单次可桥接 ≤kernel/2 px 缺口
3. **指标量化**: 闭合率、段数、端点数 (前后对比)
4. **可视化**: 蓝点 = 端点, 红线 = 缺口对, 红圈 = 缺口中心

## 用法

```bash
cd "C:\Users\18607\Desktop\边缘识别模型方法\整合算法"
python run_integrated.py
```

模块 API:

```python
import partition as pt
import colorize as cz
import edge_closure as ec
import numpy as np

edge = np.load("edge_mask.npy").astype(bool)
lab, jmask, adj, meta, skel = pt.partition_edges(edge, min_len=4, endpoint_adj_r=2.5)
# task 3
seg_rgb = cz.render_segments(lab, palette=cz.distinct_palette(meta["K"]), jmask=jmask)
# task 4
adj_0 = {i: {v-1 for v in adj[i+1]} for i in range(meta["K"])}
colors, used, method = cz.four_color(adj_0, meta["K"], 4)
# 区域四色
regions, n, region_colors, fill = cz.region_four_color(edge)
# 闭合完整性 (新增)
gaps, ep_mask = ec.analyze_endpoint_gaps(lab, edge, max_gap=5)
closed_edge = ec.close_edge_gaps(edge, kernel_size=3)
closed_edge = ec.close_edge_gaps(closed_edge, kernel_size=5)
closed_edge = ec.close_edge_gaps(closed_edge, kernel_size=7)
metrics = ec.compute_closure_metrics(lab, edge)
```

## 与其他目录对比

| 目录 | 划分算法 | 染色 | 任务 2/3/4 | 区域级 4 色 | 闭合轮廓 | 闭合完整性 |
|------|---------|------|:-:|:-:|:-:|:-:|
| **基于open cv 整合算法/ (本)** | **Zhang-Suen + 端点 KD-Tree** | **DSATUR+回溯** | ✓ | **✓** | **✓** | **✓** |

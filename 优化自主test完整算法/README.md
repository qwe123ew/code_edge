# edge_pipeline 算法 — 替代自研 MCT-Edge, 实现 docx 任务 2/3/4

本目录用 `C:\Users\18607\Desktop\test\edge_pipeline.py` , 完成 docx 任务 2 (划分) + 3 (染色) + 4 (四色定理染色), 并附带两个**亮点**:

1. **折线有序化** (`order_segments` + `render_polylines`): 把无序像素集排成有序折线, 起点绿点/终点红点, 闭合段两端不打点
2. **物体区域四色** (`extract_objects` + `region_adjacency` + `dsatur_coloring`): 由闭合边缘恢复物体面 + 含外部背景的完整地图四色染色

## 文件

- [edge_pipeline.py](edge_pipeline.py) — 从 `C:\Users\18607\Desktop\test\edge_pipeline.py` **原样搬入**
  - `canny(img, ...)` — 从零实现 Canny: DoG 梯度 + 亚像素 NMS + 局部自适应双阈值 + 滞后连接 + 方向性断口填补
  - `segment_edges(edge)` — 边缘划分: Zhang-Suen 骨架化 + `branch_points` (8-连通臂数 ≥ 3) + 8-CC 标段
  - `order_segments(seg)` — 折线有序化: 度数 1 端点起笔 → 优先延续来向 → 闭合环主路径 + 子链
  - `extract_objects(edge)` — 由闭合边缘恢复物体面: 闭运算缝合 → 自由空间 8-CC → 挖掉外部背景
  - `region_adjacency(regions, n)` — 区域邻接表: 各区域膨胀 2px 与其他区域相交 ⇒ 共享边缘 ⇒ 邻接
  - `render_colored(labels, palette)` / `render_polylines(...)` / `render_regions(...)` — 可视化
  - `dsatur_coloring(adjacency, max_colors=4)` / `four_color_palette()` — DSATUR 贪心 4-色染色
  - `run_pipeline(input, output_dir, ...)` — 端到端主入口
- [run_tasks_234.py](run_tasks_234.py) — 5 张图端到端驱动 (~120 行)
  - 调用 `edge_pipeline.run_pipeline` 处理每张图
  - 聚合 `summary.json`
  - 每张图写 6 个 PNG + 1 个 折线.json + 1 个 统计.json
- [outputs/](outputs/) — 5 张测试图的全部产物

## 与自研 MCT-Edge 算法的对比

| 维度 | 自研 MCT-Edge (v6) | edge_pipeline |
|------|-------------------|---------------|
| 边缘检测 | 自研 (DoG + DiZenzo 颜色结构张量 + 多尺度 NMS) | 自研 Canny (DoG + 亚像素 NMS + 自适应阈值) |
| 任务 2 划分 | 无 | branch_points + 8-CC |
| 任务 3 染色 | 无 | 黄金角 HSV |
| 任务 4 四色 | 无 | DSATUR 贪心 |
| 折线有序化 | 无 | 有 (主路径 + 子链) |
| 区域四色 | 无 | 有 (闭运算 + 8-CC + 邻接) |
| 依赖 | numpy + OpenCV + skimage + scipy | numpy + Pillow (零 OpenCV) |

**核心定位**: edge_pipeline 是一个**完整的"边缘检测 → 划分 → 染色 → 四色"全流程**, 而自研 MCT-Edge 只产出"边缘二值图", 下游任务需要另外实现.

## 算法流水线

```
彩色图
  │
  ▼  (edge_pipeline.canny, 从零实现)
DoG 梯度 → 亚像素 NMS → 局部自适应双阈值 → 滞后连接 → 方向性断口填补
  │ → edge[H,W] bool
  ▼
segment_edges(edge)
  ├─ skeletonize  (Zhang-Suen 细化)
  ├─ branch_points (8-连通臂数 ≥ 3)
  ├─ _connected_components (并查集 8-CC)
  ├─ 结点就近归属段
  └─ 邻接表: (a) 共享结点 (b) 像素 8-邻接
  │
  ├─► render_colored(labels, distinct_palette)  → 任务 3 逐段染色
  │
  ├─► dsatur_coloring(adjacency)  → 任务 4 段级四色染色
  │
  ├─► order_segments  → 折线有序化 (亮点)
  │
  └─► extract_objects + region_adjacency + dsatur  → 区域级四色 (亮点)
```

## 输出产物

每张图 **6 个 PNG + 1 个 JSON (折线) + 1 个 JSON (统计)**:

| 文件 | 内容 |
|------|------|
| `<图名>_1_边缘图.png` | Canny 二值边缘 (黑底白边) |
| `<图名>_2_结点标注.png` | 边缘 + 红色分叉点 |
| `<图名>_3_逐段染色.png` | **任务 3**: 黄金角逐段染色 |
| `<图名>_4_四色染色.png` | **任务 4**: DSATUR 4-色染色 |
| `<图名>_5_折线有序化.png` | **亮点**: 折线 (起点绿/终点红, 闭合段不打点) |
| `<图名>_6_物体区域四色.png` | **亮点**: 闭合边缘围成的物体面 + 外部背景的完整地图四色 |
| `<图名>_折线.json` | 每段有序折线 (主路径 + 子链 + 长度) |
| `<图名>_统计.json` | 全套统计 |
| `summary.json` | 5 张图汇总 |

## 5 张测试图运行结果

| 图片 | 边缘 | 骨架 | 结点 | 段数 | 邻接对 | 闭合段 | 段4色 | 段冲突 | 物体 | 区域4色 | 耗时 |
|------|---:|---:|---:|---:|---:|---:|:-:|---:|---:|:-:|---:|
| 图片物体边缘的提取分割 | 6325 | 5899 | 80 | 88 | 57 | 3 | 3 | **0** | 12 | 2 | 179 ms |
| bsds_368037 | 9458 | 8771 | 147 | 159 | 65 | 1 | 3 | **0** | 6 | 2 | 418 ms |
| bsds_97010 | 13258 | 10972 | 449 | 292 | 175 | 7 | 3 | **0** | 31 | 3 | 448 ms |
| nyud_5017 | 2918 | 2753 | 56 | 44 | 21 | 1 | 3 | **0** | 3 | 3 | 79 ms |
| nyud_6233 | 2625 | 2409 | 22 | 56 | 14 | 1 | 3 | **0** | 1 | 2 | 90 ms |

**段级四色冲突 = 0 (5/5), 区域级四色 = DSATUR 在 2-3 色内解出, 平面图必 ≤4 色**.

### 段长统计

| 图片 | 段平均长 | 最长段 | 总长 |
|------|---:|---:|---:|
| 图片物体边缘的提取分割 | 66.39 | 755.79 | 5842.75 |
| bsds_368037 | 55.65 | 1136.59 | 8847.94 |
| bsds_97010 | 36.20 | 600.66 | 10571.22 |
| nyud_5017 | 61.39 | 650.37 | 2701.19 |
| nyud_6233 | 44.02 | 194.60 | 2465.04 |

## 算法亮点详解

### 亮点 1: 折线有序化

`order_segments` 把"无序的段像素集"转成"有序折线":

- 段内像素是 8-连通图, 大多数是 1px 链, 但骨架局部 2px 宽处会有小分支
- **起笔**: 优先选度数 = 1 的端点 (开放线段); 闭合环选任意像素
- **走向**: 优先延续来向 (避免 2px 宽处来回锯齿)
- **分支**: 不"就近乱接" (会产生跨图跳跃), 而作为独立子链返回 — 长度统计不丢、渲染不打结
- **闭合检测**: 主路径首尾 8-邻接 → 标记为闭合段, 渲染时两端不打点 (避免起/终点绿/红覆盖)

### 亮点 2: 物体区域四色 (地图四色)

`extract_objects` + `region_adjacency` + `dsatur_coloring`:

1. 闭运算 (3x3 kernel, 2 次) 缝合 1-2px 边缘断口
2. 自由空间 (非边缘) 做 8-CC
3. 与图像边界连通的分量是外部背景, 挖掉
4. 剩余分量即"被闭合轮廓包围的面" — 物体 (或物体内部的孔洞, 如圆环会得到内外两个面, 恰对应平面图的两个 face)
5. 各区域膨胀 2px 检测"隔边缘相邻" → 区域邻接图
6. DSATUR 4-色染色 (含外部背景作第 n+1 个面)

这才是四色定理的经典场景 (国家地图着色), 比"边缘段邻接图"更直观.

## 用法

```bash
cd "C:\Users\18607\Desktop\边缘识别模型方法\优化自主test完整算法"
python run_tasks_234.py
```

或直接调用单图:

```python
import edge_pipeline as ep
stats = ep.run_pipeline(
    r"D:\图片\photo.png",
    r"D:\输出",
    sigma=1.4,
    operator="dog",     # DoG (高斯解析导数, 定位最准)
    adaptive=True,      # 局部自适应双阈值 (对比度归一化)
    use_color=True,     # 彩色: DiZenzo 结构张量
    color_mode="tensor",
    do_four_color=True,
)
```

模块 API (任务 2/3/4 核心):

```python
import edge_pipeline as ep
import numpy as np
from PIL import Image

rgb = np.asarray(Image.open("photo.png").convert("RGB"), dtype=np.float32)
edge = ep.canny(rgb, sigma=1.4, adaptive=True, use_color=True)
seg = ep.segment_edges(edge)  # 任务 2: 划分
ordered = ep.order_segments(seg)  # 折线有序化

# 任务 3
palette = ep.distinct_palette(seg.n_segments)
colored = ep.render_colored(seg.labels, palette)

# 任务 4
coloring = ep.dsatur_coloring(seg.adjacency, max_colors=4)
# 区域四色 (亮点)
obj_lab, n_obj, bg_mask, sealed = ep.extract_objects(seg.skeleton)
regions = np.where(bg_mask, n_obj + 1, obj_lab)
reg_adj = ep.region_adjacency(regions, n_obj + 1)
reg_colors = ep.dsatur_coloring(reg_adj, max_colors=4)
```

## 与其他目录对比

| 目录 | 划分算法 | 染色 | 任务 2/3/4 | 区域 4 色 | 折线有序 | 闭合检测 |
|------|---------|------|:-:|:-:|:-:|:-:|
| 基于open cv 整合算法/ | Zhang-Suen + 端点 KD-Tree | DSATUR+回溯 | ✓ | ✓ | ✗ | ✓ |
| **优化自主test完整算法/ (本)** | **Zhang-Suen + branch_points** | **DSATUR** | **✓** | **✓** | **✓** | **✓** |

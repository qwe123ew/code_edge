# -*- coding: utf-8 -*-
"""
步骤3: 边缘分段染色展示(每段不同颜色)。
步骤4: 四色定理染色 -- 段间邻接图上用 <=4 种颜色使相邻段颜色不同。

附: region_four_color -- 由闭合边缘恢复物体区域并做"地图四色染色",
    这正是四色定理的经典场景(平面地图着色)。

算法:
  - DSATUR(饱和度优先贪婪): 每次选"相邻已用颜色种数最多"的节点着最小可用色;
  - 若贪婪超过 4 色(理论上平面图不会, 但像素邻接图可能有误差),
    退化为按度数排序 + 回溯搜索, 保证找到 4 色解(四色定理保证存在)。
"""
import numpy as np
from scipy import ndimage as ndi
from matplotlib.colors import hsv_to_rgb


# ---------------------------------------------------------------- 色板
def distinct_palette(K):
    """K 个高区分度颜色: 黄金角均匀采样 HSV 色环"""
    idx = np.arange(K)[:, None]
    h = (idx * 0.6180339887) % 1.0
    s = np.full((K, 1), 0.85)
    v = np.full((K, 1), 1.0)
    rgb = hsv_to_rgb(np.hstack([h, s, v]))
    return (rgb * 255).astype(np.uint8)


PALETTE4 = np.array([[235, 25, 60],      # 红
                     [40, 180, 70],      # 绿
                     [30, 110, 230],     # 蓝
                     [250, 210, 30]],    # 黄
                    dtype=np.uint8)


# ---------------------------------------------------------------- 渲染
def render_segments(lab, palette=None, jmask=None, junction_color=(255, 255, 255)):
    """按段 id 渲染彩色边缘图"""
    H, W = lab.shape
    K = int(lab.max())
    if K == 0:
        return np.zeros((H, W, 3), np.uint8)
    if palette is None:
        palette = distinct_palette(K)
    out = np.zeros((H, W, 3), np.uint8)
    for sid in range(1, K + 1):
        out[lab == sid] = palette[sid - 1]
    if jmask is not None and jmask.any():
        out[jmask] = junction_color
    return out


def overlay(orig_rgb, seg_rgb, alpha=0.45):
    """彩色边缘叠加回原图"""
    o = orig_rgb.astype(np.float64)
    m = (seg_rgb.sum(axis=2, keepdims=True) > 0).astype(np.float64)
    return (o * (1 - alpha * m) + seg_rgb.astype(np.float64) * alpha * m
            ).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 图染色
def dsatur(adj, n_nodes, max_colors=None):
    """DSATUR 贪婪染色, 返回颜色数组(0-based); n_nodes 为节点数, adj 为 {i:set}"""
    color = [-1] * n_nodes
    uncolored = set(range(n_nodes))
    while uncolored:
        # 饱和度 = 相邻已染色数; 平手取度数大者
        best = max(uncolored, key=lambda v: (len({color[u] for u in adj[v] if color[u] >= 0}),
                                             len(adj[v])))
        used = {color[u] for u in adj[best] if color[u] >= 0}
        c = 0
        while c in used:
            c += 1
            if max_colors is not None and c >= max_colors:
                c = -1
                break
        if c == -1:
            return None  # 在 max_colors 限制下失败
        color[best] = c
        uncolored.discard(best)
    return color


def _backtrack_color(adj, order, n_colors):
    """回溯搜索 n_colors 色解"""
    color = [-1] * len(order)

    def dfs(k):
        if k == len(order):
            return True
        v = order[k]
        forbidden = {color[u] for u in adj[v] if color[u] >= 0}
        for c in range(n_colors):
            if c not in forbidden:
                color[v] = c
                if dfs(k + 1):
                    return True
                color[v] = -1
        return False

    return color if dfs(0) else None


def four_color(adj, n_nodes, n_colors=4):
    """四色染色: 先 DSATUR, 超出 n_colors 则回溯。返回 (colors, n_used, method)"""
    # DSATUR(不设上限)看用了几色
    greedy = dsatur(adj, n_nodes)
    if greedy is not None and max(greedy) + 1 <= n_colors:
        return greedy, max(greedy) + 1, 'DSATUR'
    # 限制上限重试
    limited = dsatur(adj, n_nodes, max_colors=n_colors)
    if limited is not None:
        return limited, n_colors, 'DSATUR(限4色)'
    # 回溯保证找到解(四色定理: 平面图必存在 4 色解)
    order = sorted(range(n_nodes), key=lambda v: -len(adj[v]))
    sol = _backtrack_color(adj, order, n_colors)
    if sol is not None:
        return sol, max(sol) + 1, '回溯搜索'
    return greedy, (max(greedy) + 1) if greedy else 0, '贪婪(回溯失败)'


def check_coloring(adj, colors):
    """验证: 任意相邻节点颜色不同"""
    bad = [(u, v) for u in adj for v in adj[u] if colors[u] == colors[v]]
    return len(bad) == 0, bad[:10]


# ---------------------------------------------------------------- 边缘 -> 物体
def region_four_color(edge_mask):
    """由闭合边缘恢复物体区域, 做区域级四色地图染色(四色定理的本义场景)。

    步骤: 闭运算缝合边缘小缺口 -> 填充闭合轮廓 -> 挖掉边缘线得到若干区域
          (含外部背景) -> 区域邻接图(小膨胀检测"隔边缘相邻") -> DSATUR 四色。
    返回 (区域标签图[0=边缘线], 内部区域数 n, 区域颜色数组[含背景], 填充掩码)
    """
    edge_m = ndi.binary_closing(edge_mask, structure=np.ones((3, 3), int), iterations=2)
    closed = ndi.binary_fill_holes(edge_m)
    inner = closed & ~edge_m          # 只挖掉边缘线本身(保留 2px 邻接缝隙)
    lab, n = ndi.label(inner, structure=np.ones((3, 3), np.int32))
    if n == 0:
        return lab, n, None, closed

    bg_id = n + 1                     # 背景区域(闭合区域之外)
    regions = np.where(~closed, bg_id, lab)   # 0 仍为边缘线

    # 区域邻接: 各区域膨胀 2px 后与其他区域相交 => 共享边缘(相邻)
    # 注意: dsatur / four_color 用 0-based 索引 (keys 0..n), 故 regions id - 1
    adj = {i: set() for i in range(0, n + 1)}
    masks = {i: (regions == i + 1) for i in range(0, n + 1)}  # region 1 → key 0
    for i in range(0, n + 1):
        di = ndi.binary_dilation(masks[i], iterations=2)
        for j in range(i + 1, n + 1):
            if (di & masks[j]).any():
                adj[i].add(j)
                adj[j].add(i)

    colors, used, method = four_color(adj, n + 1, 4)
    return regions, n, colors, closed

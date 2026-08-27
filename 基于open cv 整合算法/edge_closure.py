# -*- coding: utf-8 -*-
"""
边缘闭合完整性分析
==================

Canny 边带常常有 1-3px 的小缺口, 让原本"应该闭合"的物体轮廓变成带端点的
开放段, 丢失"边缘-物体"对应关系。本模块做三件事:

  1. analyze_endpoint_gaps(lab, jmask, edge, max_gap)
     - 找每段的端点 (8-邻域数 == 1 的段像素)
     - 用 cKDTree 找距离 ≤ max_gap 的端点对 (这些就是"缺口中能缝合的")
     - 返回 (gap_list, endpoint_mask)

  2. close_edge_gaps(edge, kernel_size, iterations)
     - cv2.morphologyEx(MORPH_CLOSE) 桥接 ≤ kernel/2 像素的缺口
     - 可多次迭代 (iterations)

  3. compute_closure_metrics(lab, jmask)
     - closure_ratio = 端点数==0 的段 / 总段数
     - 反映"边缘闭合完整性"

可视化:
  - render_gap_visualization(edge, gaps, endpoint_mask, out_shape)
    灰底 + 蓝点 (端点) + 红线 (gap pair) + 红圈 (gap 中点)
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


# 8-邻域核 (中心为 0)
_KERNEL_8N = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.int32)


# ============================================================================
# 1) 端点缺口分析
# ============================================================================

def analyze_endpoint_gaps(lab: np.ndarray, edge: np.ndarray,
                           max_gap: float = 5.0
                           ) -> tuple[list[tuple], np.ndarray]:
    """找每段端点 + 距离 ≤ max_gap 的"可缝合"端点对。

    Args:
        lab: 段标签图, 0=背景, ≥1=段号
        edge: 二值边缘图 (用于过滤, 端点必须在 edge 上)
        max_gap: 端点对最大距离 (像素), 超过此值的 gap 不算"可缝合"

    Returns:
        gaps: list of (y1, x1, y2, x2, distance) — 可缝合的端点对
        endpoint_mask: (H, W) bool, 端点位置
    """
    # 段像素的 8-邻域计数
    nc = ndi.convolve((lab > 0).astype(np.int32), _KERNEL_8N,
                      mode='constant', cval=0)
    endpoint_mask = (lab > 0) & (nc == 1) & edge

    endpoints = np.argwhere(endpoint_mask)
    if len(endpoints) < 2:
        return [], endpoint_mask

    tree = cKDTree(endpoints)
    pairs = tree.query_pairs(r=max_gap)
    gaps = []
    for i, j in pairs:
        y1, x1 = endpoints[i]
        y2, x2 = endpoints[j]
        d = float(np.hypot(y1 - y2, x1 - x2))
        gaps.append((int(y1), int(x1), int(y2), int(x2), d))
    return gaps, endpoint_mask


# ============================================================================
# 2) 形态学闭合
# ============================================================================

def close_edge_gaps(edge: np.ndarray, kernel_size: int = 3,
                    iterations: int = 1) -> np.ndarray:
    """用 cv2.morphologyEx(MORPH_CLOSE) 桥接边缘缺口。

    桥接能力: 单次迭代可桥接 ≤ kernel_size//2 像素的缺口。
    iterations 控制迭代次数 (多次叠加, 缺口更宽的也能填上, 但也可能糊掉细节)。

    Args:
        edge: (H, W) bool 二值边缘图
        kernel_size: 椭圆 kernel 大小 (奇数)
        iterations: 迭代次数

    Returns:
        (H, W) bool 闭合后的二值边缘图
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(edge.astype(np.uint8), cv2.MORPH_CLOSE, kernel,
                              iterations=iterations)
    return closed.astype(bool)


# ============================================================================
# 3) 闭合性指标
# ============================================================================

def compute_near_closed_segments(lab: np.ndarray, edge: np.ndarray,
                                 gap_threshold: float = 5.0
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """识别"近闭合"段: 端点数 == 2 且两端距离 ≤ gap_threshold 的段。

    Canny 边带几乎都是带 2 个端点的开轮廓, 不会自然产生"零端点的闭合段",
    所以 `n_closed` 恒为 0。本函数放宽判定: 只要两端的像素距离在 gap_threshold
    以内 (中间隔了 1-3 个像素的缺口), 就算"近闭合"——这些缺口用
    `cv2.MORPH_CLOSE` 是可以缝合的。

    Args:
        lab: 段标签图, 0=背景, ≥1=段号
        edge: 二值边缘图
        gap_threshold: 端点对最大像素距离, 默认 5

    Returns:
        near_closed:   (n_seg+1,) bool, 仅段 id 1..n_seg 有效
        gap_distances: (n_seg+1,) float32, 近闭合段的两端距离 (其它段为 0)
    """
    n_seg = int(lab.max())
    if n_seg == 0:
        return np.zeros(1, dtype=bool), np.zeros(1, dtype=np.float32)

    nc = ndi.convolve((lab > 0).astype(np.int32), _KERNEL_8N,
                      mode='constant', cval=0)
    ep_mask = (lab > 0) & (nc == 1) & edge

    near_closed = np.zeros(n_seg + 1, dtype=bool)
    gap_distances = np.zeros(n_seg + 1, dtype=np.float32)

    # 先把"端点数==0 且段足够长"的完美闭合段标记为近闭合 (gap=0)
    for sid in range(1, n_seg + 1):
        if (lab == sid).sum() > 4 and ((lab == sid) & ep_mask).sum() == 0:
            near_closed[sid] = True
            gap_distances[sid] = 0.0

    # 跨段端点配对 (cKDTree): 不能只看"同一段内的 2 端点距离",
    # 因为断成多段时每段两端相距是弧长; 正确做法是跨段配对。
    endpoints = np.argwhere(ep_mask)
    if len(endpoints) >= 2:
        seg_of_ep = lab[ep_mask]  # 每个端点所属的段 id
        tree = cKDTree(endpoints)
        pairs = tree.query_pairs(r=gap_threshold)
        for i, j in pairs:
            si, sj = int(seg_of_ep[i]), int(seg_of_ep[j])
            if si == sj or si < 1 or sj < 1:
                continue
            d = float(np.hypot(endpoints[i, 0] - endpoints[j, 0],
                                endpoints[i, 1] - endpoints[j, 1]))
            if d < gap_distances[si] or gap_distances[si] == 0:
                gap_distances[si] = d
            if d < gap_distances[sj] or gap_distances[sj] == 0:
                gap_distances[sj] = d
            if d <= gap_threshold:
                near_closed[si] = True
                near_closed[sj] = True
    return near_closed, gap_distances


def compute_closure_metrics(lab: np.ndarray, edge: np.ndarray,
                            gap_threshold: float = 5.0) -> dict:
    """计算"边缘闭合完整性"指标。

    指标:
        n_segments             总段数
        n_closed               端点数 == 0 的段 (完美闭合段, Canny 几乎为 0)
        n_near_closed          端点对距离 ≤ gap_threshold 的段 (近闭合, 可缝合)
        n_endpoint_total       所有段的端点总数
        closure_ratio          n_closed / n_segments
        near_closed_ratio      n_near_closed / n_segments
        mean_gap_px            近闭合段的平均端点距离
        mean_endpoints_per_segment  平均每段端点数
    """
    n_seg = int(lab.max())
    if n_seg == 0:
        return dict(n_segments=0, n_closed=0, n_near_closed=0,
                    n_endpoint_total=0, closure_ratio=0.0,
                    near_closed_ratio=0.0, mean_gap_px=0.0,
                    mean_endpoints_per_segment=0.0)

    nc = ndi.convolve((lab > 0).astype(np.int32), _KERNEL_8N,
                      mode='constant', cval=0)
    ep_mask = (lab > 0) & (nc == 1) & edge

    ep_count = np.zeros(n_seg + 1, dtype=np.int32)
    for sid in np.unique(lab[ep_mask]):
        if sid >= 1:
            ep_count[sid] = int(((lab == sid) & ep_mask).sum())

    n_closed = int((ep_count[1:] == 0).sum())
    n_endpoint_total = int(ep_count[1:].sum())

    near_closed, gap_distances = compute_near_closed_segments(
        lab, edge, gap_threshold=gap_threshold)
    n_near_closed = int(near_closed.sum())
    if n_near_closed > 0:
        mean_gap = float(gap_distances[near_closed].mean())
    else:
        mean_gap = 0.0

    return dict(
        n_segments=n_seg,
        n_closed=n_closed,
        n_near_closed=n_near_closed,
        n_endpoint_total=n_endpoint_total,
        closure_ratio=n_closed / n_seg if n_seg > 0 else 0.0,
        near_closed_ratio=n_near_closed / n_seg if n_seg > 0 else 0.0,
        mean_gap_px=round(mean_gap, 2),
        mean_endpoints_per_segment=n_endpoint_total / n_seg if n_seg > 0 else 0.0,
    )


# ============================================================================
# 4) 可视化
# ============================================================================

def render_gap_visualization(edge: np.ndarray, gaps: list[tuple],
                              endpoint_mask: np.ndarray,
                              out_shape: tuple[int, int]
                              ) -> np.ndarray:
    """缺口可视化: 灰底 + 蓝点 (端点) + 红线 (gap pair) + 红圈 (gap 中点)。"""
    out = np.full((*out_shape, 3), 255, dtype=np.uint8)
    out[edge] = (180, 180, 180)
    out[endpoint_mask] = (60, 60, 200)   # 端点: 蓝
    for y1, x1, y2, x2, d in gaps:
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 235), 1)
        my, mx = (y1 + y2) // 2, (x1 + x2) // 2
        cv2.circle(out, (int(mx), int(my)), 2, (0, 0, 235), -1)
    return out


def render_closed_edge(closed_edge: np.ndarray, out_shape: tuple[int, int]
                       ) -> np.ndarray:
    """渲染闭合后的边缘 (白底黑边)。"""
    out = np.full(out_shape, 255, dtype=np.uint8)
    out[closed_edge] = 0
    return out

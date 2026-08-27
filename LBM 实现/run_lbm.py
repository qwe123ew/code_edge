# -*- coding: utf-8 -*-
"""
LBM 去噪 + 传统边缘/划分/染色管线
==================================

LBM 在这里只做 **Perona-Malik 各向异性扩散去噪** (lbm_pipeline.lbm_denoise),
作为下游 Canny 边缘检测的预处理。

后续所有环节使用经典算法:
  - 边缘检测:  OpenCV Canny (auto-threshold 基于中位数)
  - 边缘划分:  Zhang-Suen 骨架 + branch_points (8-臂数 ≥ 3) + 8-CC 标段
  - 染色:      黄金角 137.5° HSV 逐段
  - 4 色:      DSATUR 贪心, 段级 + 区域级
  - 区域提取:  闭运算缝合 + 8-CC + 抠掉外部背景 (传统做法, 非 LBM)

输入: 5 张桌面固定测试图
输出: outputs/<图名>_* 共 8 个 PNG + 1 个统计 JSON + 1 个 summary
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import label as cc_label

from lbm_pipeline import lbm_denoise


HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_IMAGES = [
    r"C:\Users\18607\Desktop\边缘识别模型方法\图片物体边缘的提取分割.png",
    r"C:\Users\18607\Desktop\bsds_368037_原图.jpg",
    r"C:\Users\18607\Desktop\bsds_97010_原图.jpg",
    r"C:\Users\18607\Desktop\nyud_5017_原图.png",
    r"C:\Users\18607\Desktop\nyud_6233_原图.png",
]


# ===================== 工具 =====================
def imread_unicode(path: str) -> np.ndarray | None:
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    cv2.imencode(os.path.splitext(path)[1], img)[1].tofile(path)


# ===================== Zhang-Suen 骨架化 =====================
def _zs_neighbours(img):
    P = np.zeros(img.shape + (8,), dtype=np.uint8)
    P[..., 0] = np.roll(img, 1, axis=0)
    P[..., 1] = np.roll(np.roll(img, 1, axis=0), 1, axis=1)
    P[..., 2] = np.roll(img, 1, axis=1)
    P[..., 3] = np.roll(np.roll(img, -1, axis=0), 1, axis=1)
    P[..., 4] = np.roll(img, -1, axis=0)
    P[..., 5] = np.roll(np.roll(img, -1, axis=0), -1, axis=1)
    P[..., 6] = np.roll(img, -1, axis=1)
    P[..., 7] = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
    return P


def _zs_sub(img, sub):
    P = _zs_neighbours(img)
    P2, P3, P4, P5, P6, P7, P8, P9 = (P[..., i] for i in range(8))
    B = (P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9)
    A = (((P3 & ~P4) + (P4 & ~P5) + (P5 & ~P6) + (P6 & ~P7)
          + (P7 & ~P8) + (P8 & ~P9) + (P9 & ~P2) + (P2 & ~P3)))
    if sub == 0:
        cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) \
               & (P2 & P4 & P6 == 0) & (P4 & P6 & P8 == 0)
    else:
        cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) \
               & (P2 & P4 & P8 == 0) & (P2 & P6 & P8 == 0)
    return cond


def zhang_suen(img: np.ndarray) -> np.ndarray:
    """Zhang-Suen 骨架化. 输入 0/1 矩阵, 输出 0/1."""
    img = img.copy().astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for sub in (0, 1):
            m = _zs_sub(img, sub)
            if m.any():
                img[m] = 0
                changed = True
    return img


def _count_8n(img: np.ndarray) -> np.ndarray:
    nbr = np.zeros_like(img, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nbr += np.roll(np.roll(img, dy, axis=0), dx, axis=1)
    return nbr


# ===================== 边缘划分 =====================
@dataclass
class SegmentResult:
    labels: np.ndarray
    skeleton: np.ndarray
    branch_pts: np.ndarray
    n_segments: int
    adjacency: list
    endpoints: np.ndarray
    near_closed: np.ndarray = None      # (n_seg+1,) bool
    gap_distances: np.ndarray = None    # (n_seg+1,) float32


def _near_closed_segments(labels: np.ndarray, edge_bin: np.ndarray,
                          gap_threshold: float = 5.0):
    """跨段端点配对, 找 ≤ gap_threshold px 的可缝合段对。

    与 基于open cv 整合算法/edge_closure.compute_near_closed_segments 等价,
    这里独立实现以避免跨目录依赖。
    """
    from scipy.ndimage import convolve as ndi_convolve
    from scipy.spatial import cKDTree
    H, Wd = labels.shape
    n_seg = int(labels.max())
    if n_seg == 0:
        return np.zeros(1, dtype=bool), np.zeros(1, dtype=np.float32)

    kernel_8n = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
    nc = ndi_convolve((labels > 0).astype(np.int32), kernel_8n,
                      mode='constant', cval=0)
    ep_mask = (labels > 0) & (nc == 1)
    # 限制端点必须在 edge 上 (与 edge_closure 一致)
    ep_mask = ep_mask & (edge_bin > 0)

    near_closed = np.zeros(n_seg + 1, dtype=bool)
    gap_distances = np.zeros(n_seg + 1, dtype=np.float32)

    # 完美闭合 (端点数==0 且段足够长)
    for sid in range(1, n_seg + 1):
        if (labels == sid).sum() > 4 and ((labels == sid) & ep_mask).sum() == 0:
            near_closed[sid] = True
            gap_distances[sid] = 0.0

    # 跨段配对
    endpoints = np.argwhere(ep_mask)
    if len(endpoints) >= 2:
        seg_of_ep = labels[ep_mask]
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


def segment_edges(edge: np.ndarray) -> SegmentResult:
    """Canny 边缘 → Zhang-Suen 骨架 + branch_points + 8-CC + 邻接表 + 近闭合检测."""
    H, Wd = edge.shape

    kernel = np.ones((3, 3), np.uint8)
    edge_clean = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=1)
    edge_bin = (edge_clean > 0).astype(np.uint8)

    skel = zhang_suen(edge_bin)
    nbr = _count_8n(skel)
    branch = (skel == 1) & (nbr >= 3)

    seg_input = skel & (~branch)
    seg_input = seg_input.astype(np.uint8)
    labels, n_seg = cc_label(seg_input, structure=np.ones((3, 3), np.uint8))

    ep = (seg_input == 1) & (nbr == 1)

    adjacency: list[set[int]] = [set() for _ in range(n_seg + 1)]

    # (a) branch point 邻接
    for y, x in zip(*np.where(branch)):
        neighbors = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = int(y) + dy, int(x) + dx
                if 0 <= ny < H and 0 <= nx < Wd:
                    sid = int(labels[ny, nx])
                    if sid > 0:
                        neighbors.add(sid)
        nb_list = list(neighbors)
        for a in nb_list:
            for b in nb_list:
                if a != b:
                    adjacency[a].add(b)

    # (b) 像素 8-邻接
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]:
        shifted = np.roll(np.roll(labels, dy, axis=0), dx, axis=1)
        mask = (labels > 0) & (shifted > 0) & (shifted != labels)
        for a, b in zip(labels[mask].tolist(), shifted[mask].tolist()):
            adjacency[a].add(b)
            adjacency[b].add(a)

    # (c) 近闭合检测 (跨段端点配对)
    near_closed, gap_distances = _near_closed_segments(labels, edge_bin,
                                                      gap_threshold=5.0)

    return SegmentResult(
        labels=labels.astype(np.int32),
        skeleton=skel.astype(np.uint8) * 255,
        branch_pts=branch,
        n_segments=n_seg,
        adjacency=adjacency,
        endpoints=ep,
        near_closed=near_closed,
        gap_distances=gap_distances,
    )


# ===================== 黄金角 HSV 染色 =====================
def golden_palette(n: int) -> np.ndarray:
    palette = np.zeros((max(n, 1), 3), dtype=np.uint8)
    golden = 137.508
    for i in range(n):
        h = (i * golden) % 360.0
        s = 0.78
        v = 0.92 if (i % 3) != 0 else 0.80
        hsv = np.array([[[h / 2.0, s, v]]], dtype=np.float32)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        palette[i] = (bgr * 255).astype(np.uint8)
    return palette


def render_colored(labels: np.ndarray, palette: np.ndarray, bg=(255, 255, 255)) -> np.ndarray:
    out = np.full((*labels.shape, 3), bg, dtype=np.uint8)
    for i in range(1, len(palette)):
        mask = labels == i
        if mask.any():
            out[mask] = palette[i]
    return out


# ===================== DSATUR 4 色 =====================
def dsatur_color(adjacency: list[set[int]], n: int, max_colors: int = 4) -> np.ndarray:
    color = np.zeros(n + 1, dtype=np.int32)
    if n == 0:
        return color
    deg = np.array([len(adjacency[i]) for i in range(n + 1)])
    for _ in range(n):
        uncolored = np.where(color[1:] == 0)[0] + 1
        if len(uncolored) == 0:
            break
        sat = np.zeros(n + 1, dtype=np.int32)
        for i in range(1, n + 1):
            used = set()
            for j in adjacency[i]:
                if color[j] > 0:
                    used.add(color[j])
            sat[i] = len(used)
        score = sat[uncolored] * 1000 + deg[uncolored]
        u = uncolored[np.argmax(score)]
        used = {color[j] for j in adjacency[u] if color[j] > 0}
        for c in range(1, max_colors + 1):
            if c not in used:
                color[u] = c
                break
        else:
            color[u] = 0
    return color


# ===================== Auto-Canny (基于中位数) =====================
def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """经典 auto-Canny 阈值: 基于图像中位数."""
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


# ===================== 区域提取 (传统: 闭运算 + 8-CC) =====================
def extract_objects(edge: np.ndarray, kernel_iter: int = 2) -> tuple:
    """由 Canny 边缘提取物体区域: 闭运算缝合小缺口 → 8-CC → 抠掉外部背景.

    Returns:
        obj_labels: H×W int32, 0=背景/边缘, 1..N=物体区域
        n_objects:  int
    """
    H, Wd = edge.shape
    kernel = np.ones((3, 3), np.uint8)
    edge_sealed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel,
                                   iterations=kernel_iter)

    fillable = (edge_sealed == 0).astype(np.uint8)
    # 8-CC
    cc, n_cc = cc_label(fillable, structure=np.ones((3, 3), np.uint8))

    # 找外部背景: 与图像边界 8-连通的分量
    boundary_mask = np.zeros((H, Wd), dtype=bool)
    boundary_mask[0, :] = boundary_mask[-1, :] = True
    boundary_mask[:, 0] = boundary_mask[:, -1] = True
    bg_ids = set(cc[boundary_mask & (cc > 0)].tolist())

    # 物体 = 非边缘 ∩ 非背景
    obj_mask = (cc > 0) & (~np.isin(cc, list(bg_ids)))
    obj_labels = np.zeros((H, Wd), dtype=np.int32)
    obj_labels[obj_mask] = cc[obj_mask]
    # 重新编号 1..N
    unique_ids = np.unique(obj_labels)
    unique_ids = unique_ids[unique_ids > 0]
    remap = {old: new for new, old in enumerate(unique_ids, start=1)}
    for old, new in remap.items():
        obj_labels[obj_labels == old] = new
    n_objects = len(remap)

    return obj_labels, n_objects


# ===================== 单图主流程 =====================
def process_one(img_path: str, verbose: bool = True) -> dict:
    name = os.path.splitext(os.path.basename(img_path))[0]
    bgr = imread_unicode(img_path)
    assert bgr is not None, f"无法读取 {img_path}"
    H, Wd = bgr.shape[:2]

    if verbose:
        print(f"\n>>> {name} ({Wd}x{H})")

    stats = {"name": name, "size": [Wd, H]}

    # 0. 原图
    imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_0_原图.png"), bgr)

    # === 1. LBM 去噪 (Perona-Malik, auto 模式) ===
    t0 = time.time()
    denoised, lbm_info = lbm_denoise(bgr, n_steps=10, tau_max=0.7,
                                     k_edge=0.12, scheme="perona_malik_2",
                                     auto=True)
    t_lbm = time.time() - t0
    imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_1_LBM去噪.png"), denoised)

    # 1b. before / after 对比
    side = np.hstack([bgr, denoised])
    cv2.putText(side, "before", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    mode_lbl = {"skip": "(skipped, clean)", "light": "(light)",
                "medium": "(medium)", "heavy": "(heavy)"}.get(
                    lbm_info["mode"], "")
    cv2.putText(side, f"LBM {mode_lbl}", (Wd + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_1b_去噪对比.png"), side)
    stats["lbm_denoise"] = {
        "time_ms": round(t_lbm * 1000),
        "noise": round(lbm_info["noise"], 2),
        "mode": lbm_info["mode"],
        "applied": lbm_info["applied"],
        "n_steps": lbm_info["n_steps"],
        "tau_max": round(lbm_info["tau_max"], 3),
    }
    if verbose:
        print(f"  [1] LBM 去噪: 噪声={lbm_info['noise']:.1f} → mode={lbm_info['mode']} "
              f"applied={lbm_info['applied']} 用时 {t_lbm*1000:.0f} ms")

    # === 2. Canny 边缘检测 (在去噪图上) ===
    t0 = time.time()
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    edge = auto_canny(gray, sigma=0.33)
    t_edge = time.time() - t0
    imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_2_Canny边缘.png"), edge)
    overlay = denoised.copy()
    overlay[edge > 0] = (0, 0, 255)
    imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_2b_边缘overlay.png"), overlay)
    n_edge = int((edge > 0).sum())
    stats["canny"] = {
        "pixels": n_edge,
        "ratio": round(n_edge / (H * Wd), 4),
        "time_ms": round(t_edge * 1000),
    }
    if verbose:
        print(f"  [2] Canny 边缘: {n_edge} 像素 ({stats['canny']['ratio']*100:.2f}%) "
              f"用时 {t_edge*1000:.0f} ms")

    # === 3. 边缘划分 ===
    t0 = time.time()
    seg = segment_edges(edge)
    t_seg = time.time() - t0
    n_skel = int(seg.skeleton.sum() // 255)
    n_branch = int(seg.branch_pts.sum())
    n_adj = sum(len(s) for s in seg.adjacency) // 2
    n_near_closed = int(seg.near_closed[1:].sum()) if seg.near_closed is not None else 0
    if seg.near_closed is not None and seg.near_closed[1:].sum() > 0:
        mean_gap = float(seg.gap_distances[seg.near_closed].mean())
    else:
        mean_gap = 0.0
    stats["segment"] = {
        "n_segments": int(seg.n_segments),
        "skeleton_pixels": n_skel,
        "branch_points": n_branch,
        "n_adj_pairs": int(n_adj),
        "n_near_closed": n_near_closed,
        "near_closed_ratio": round(n_near_closed / max(1, seg.n_segments), 4),
        "mean_gap_px": round(mean_gap, 2),
        "time_ms": round(t_seg * 1000),
    }
    if verbose:
        print(f"  [3] 划分: 段数={seg.n_segments} 骨架={n_skel} 结点={n_branch} "
              f"邻接对={n_adj} 近闭合={n_near_closed}({stats['segment']['near_closed_ratio']*100:.1f}%) "
              f"用时 {t_seg*1000:.0f} ms")

    skel_img = cv2.cvtColor(seg.skeleton, cv2.COLOR_GRAY2BGR)
    skel_img[seg.branch_pts] = (0, 0, 255)
    imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_3_结点标注.png"), skel_img)

    # 3b. 近闭合段 (橙色高亮, 跨段端点对 ≤ 5px)
    if seg.near_closed is not None:
        near_closed_mask = np.isin(seg.labels,
                                    np.where(seg.near_closed)[0][1:])  # skip id=0
        nc_img = np.full((*edge.shape, 3), 255, dtype=np.uint8)
        nc_img[edge > 0] = (180, 180, 180)
        nc_img[near_closed_mask] = (40, 140, 230)
        imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_3b_近闭合段.png"), nc_img)

    # === 4. 黄金角逐段染色 ===
    palette = golden_palette(max(seg.n_segments, 1))
    colored = render_colored(seg.labels, palette)
    imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_4_逐段染色.png"), colored)
    if verbose:
        print(f"  [4] 黄金角染色: {seg.n_segments} 段")

    # === 5. 段级 4 色 ===
    seg_color = dsatur_color(seg.adjacency, seg.n_segments, max_colors=4)
    four_palette = np.array([
        [0, 0, 0],
        [40, 40, 220],
        [40, 200, 40],
        [220, 140, 40],
        [200, 60, 200],
    ], dtype=np.uint8)
    seg4 = render_colored(seg.labels, four_palette, bg=(255, 255, 255))
    for i in range(1, seg.n_segments + 1):
        if seg_color[i] > 0:
            seg4[seg.labels == i] = four_palette[seg_color[i]]
    imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_5_段级四色.png"), seg4)
    stats["segment_4color"] = {
        "used_colors": int(seg_color[1:].max() if seg_color[1:].size else 0),
        "conflicts": int(np.sum(seg_color[1:] == 0)),
    }
    if verbose:
        print(f"  [5] 段级 4 色: 用 {stats['segment_4color']['used_colors']} 色, "
              f"冲突 {stats['segment_4color']['conflicts']}")

    # === 6. 区域提取 (传统: 闭运算 + 8-CC) ===
    t0 = time.time()
    obj_labels, n_obj = extract_objects(edge, kernel_iter=2)
    t_obj = time.time() - t0
    stats["extract_objects"] = {
        "n_objects": int(n_obj),
        "object_pixels": int((obj_labels > 0).sum()),
        "time_ms": round(t_obj * 1000),
    }
    if verbose:
        print(f"  [6] 区域提取: {n_obj} 物体, "
              f"{stats['extract_objects']['object_pixels']} 像素, "
              f"用时 {t_obj*1000:.0f} ms")

    flood_vis = np.full((*edge.shape, 3), 240, dtype=np.uint8)
    flood_vis[obj_labels > 0] = (40, 40, 220)
    flood_vis[edge > 0] = (0, 0, 0)
    imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_6_物体提取.png"), flood_vis)

    # === 7. 区域级 4 色 ===
    region_adj: list[set[int]] = [set() for _ in range(n_obj + 1)]
    if n_obj > 0:
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1),           (0, 1),
                       (1, -1),  (1, 0),  (1, 1)]:
            shifted = np.roll(np.roll(obj_labels, dy, axis=0), dx, axis=1)
            mask = (obj_labels > 0) & (shifted > 0) & (shifted != obj_labels)
            for a, b in zip(obj_labels[mask].tolist(), shifted[mask].tolist()):
                region_adj[a].add(b)
                region_adj[b].add(a)

    region_color = dsatur_color(region_adj, n_obj, max_colors=4)
    region_vis = np.full((*edge.shape, 3), 240, dtype=np.uint8)
    region_vis[edge > 0] = (0, 0, 0)
    for i in range(1, n_obj + 1):
        if region_color[i] > 0:
            region_vis[obj_labels == i] = four_palette[region_color[i]]
    imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_7_区域级四色.png"), region_vis)
    stats["region_4color"] = {
        "n_regions": int(n_obj),
        "used_colors": int(region_color[1:].max() if region_color[1:].size else 0),
        "conflicts": int(np.sum(region_color[1:] == 0)),
    }
    if verbose:
        print(f"  [7] 区域 4 色: {n_obj} 区域, 用 {stats['region_4color']['used_colors']} 色, "
              f"冲突 {stats['region_4color']['conflicts']}")

    # === 统计 JSON ===
    stats_path = os.path.join(OUTPUT_DIR, f"{name}_统计.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"  统计 -> {stats_path}")
    return stats


# ===================== 主流程 =====================
def main():
    print("=" * 70)
    print("LBM 去噪 + 传统边缘/划分/染色管线  ——  docx 任务 1/2/3/4")
    print("LBM 仅用于 Perona-Malik 各向异性扩散去噪, 不参与边缘检测与划分")
    print("=" * 70)

    summary = []
    for path in TEST_IMAGES:
        if not os.path.exists(path):
            print(f"[warn] 图像不存在: {path}, 跳过")
            continue
        try:
            stats = process_one(path, verbose=True)
            summary.append(stats)
        except Exception as e:
            print(f"  [ERR] {path}: {e}")
            import traceback
            traceback.print_exc()

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 70)
    print(f"汇总 -> {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

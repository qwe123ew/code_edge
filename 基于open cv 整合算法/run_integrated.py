# -*- coding: utf-8 -*-
"""
整合 partition.py + colorize.py  实现 docx 任务 2/3/4
=====================================================

输入: 5 张测试图
流程:
  1. Canny 10/100 提取二值边缘  (复用 open cv 实现/canny_edge_detection.py 模块默认阈值)
  2. partition.partition_edges  -> 任务 2 边缘划分
       (Zhang-Suen 细化 + 8-邻域分类 + cKDTree 端点距离邻接)
  3. colorize.render_segments    -> 任务 3 黄金角逐段染色
  4. colorize.four_color         -> 任务 4 四色定理染色 (DSATUR + 回溯)
  5. 亮点: colorize.region_four_color  -> 区域级四色 (四色定理经典场景)
       (闭运算 -> 填充 -> 区域邻接 -> DSATUR)
  6. 亮点: 闭合轮廓 (ep_count == 0) = 物体边界, 体现"边缘-物体关系"

输出: outputs/<图名>_*  共 10 个 PNG + 1 个 JSON
"""
import os
import sys
import json
import time
import tempfile
import traceback

import cv2
import numpy as np

# 让本目录能 import partition / colorize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import partition  # noqa: E402
import colorize   # noqa: E402
import edge_closure as ec  # noqa: E402  新增: 边缘闭合完整性分析

# 引入现有的 OpenCV Canny 算法
CV_DIR = r"C:\Users\18607\Desktop\边缘识别模型方法\open cv 实现"
sys.path.insert(0, CV_DIR)
from canny_edge_detection import canny_edge_detection  # noqa: E402


# ===================== 输入图片 =====================
INPUT_IMAGES = [
    r"C:\Users\18607\Desktop\边缘识别模型方法\图片物体边缘的提取分割.png",
    r"C:\Users\18607\Desktop\bsds_368037_原图.jpg",
    r"C:\Users\18607\Desktop\bsds_97010_原图.jpg",
    r"C:\Users\18607\Desktop\nyud_5017_原图.png",
    r"C:\Users\18607\Desktop\nyud_6233_原图.png",
]


# ===================== 输出目录 =====================
OUTPUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== 工具函数 =====================

def imread_color(path: str):
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def run_canny(img_path: str, tmp_dir: str, threshold_low: int = 10,
              threshold_high: int = 100) -> np.ndarray:
    """复用 open cv 实现/canny_edge_detection.py, 拿到二值边缘图。

    阈值取模块默认的 10/100 (与 canny_edge_detection.py 中 CANNY_THRESHOLD_LOW=10
    和 CANNY_THRESHOLD_HIGH=100 一致)。
    """
    canny_edge_detection(img_path, tmp_dir,
                         threshold_low=threshold_low,
                         threshold_high=threshold_high)
    base = os.path.splitext(os.path.basename(img_path))[0]
    edge_path = os.path.join(tmp_dir, f"{base}_canny_edges.png")
    with open(edge_path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return g > 0


def save_image(arr: np.ndarray, path: str) -> None:
    """保存图像 (支持中文路径)。"""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imencode(".png", bgr)[1].tofile(path)
    else:
        cv2.imencode(".png", arr)[1].tofile(path)


def render_region_colored(regions: np.ndarray, n: int, colors_arr, palette: np.ndarray
                          ) -> np.ndarray:
    """渲染区域级染色 (背景用白色, 边缘线用浅灰)。

    colors_arr: 0-based 颜色数组 (长度 n) 由 region_four_color 返回;
                若 None 则用 sid-1 兜底 (只用于显示, 可能不满足 4-色)。
    """
    H, W = regions.shape
    out = np.full((H, W, 3), 255, dtype=np.uint8)
    if n == 0:
        return out
    if colors_arr is not None and len(colors_arr) >= n:
        # 用 region_four_color 给的 4-色解
        for sid in range(1, n + 1):
            out[regions == sid] = palette[colors_arr[sid - 1]]
    else:
        # 兜底: 黄金角循环
        for sid in range(1, n + 1):
            out[regions == sid] = palette[(sid - 1) % len(palette)]
    return out


# ===================== 主流程 =====================

def main():
    print("=" * 70)
    print("整合 partition.py + colorize.py —— 任务 2/3/4 + 区域四色 + 闭合轮廓")
    print("  Canny:  open cv 实现/canny_edge_detection.py (10/100)")
    print("  划分:   partition.partition_edges  (Zhang-Suen + 端点 KD-Tree)")
    print("  染色:   colorize.distinct_palette / four_color (DSATUR+回溯)")
    print("=" * 70)
    all_stats = []

    with tempfile.TemporaryDirectory() as canny_tmp:
        for idx, img_path in enumerate(INPUT_IMAGES, start=1):
            if not os.path.exists(img_path):
                print(f"\n[{idx}] 跳过 (不存在): {img_path}")
                continue
            print(f"\n[{idx}/{len(INPUT_IMAGES)}] 处理: {os.path.basename(img_path)}")
            try:
                t0 = time.perf_counter()
                base = os.path.splitext(os.path.basename(img_path))[0]

                # 1) Canny
                edge = run_canny(img_path, canny_tmp, 10, 100)
                h, w = edge.shape
                orig_bgr = imread_color(img_path)
                orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB) if orig_bgr is not None else None

                # 2) 任务 2: 边缘划分
                lab, jmask, adj, meta, skel = partition.partition_edges(
                    edge, min_len=4, endpoint_adj_r=2.5
                )
                K = meta["K"]

                # 3) 任务 3: 逐段染色 (黄金角)
                palette = colorize.distinct_palette(K) if K > 0 else np.zeros((0, 3), np.uint8)
                seg_rgb = colorize.render_segments(
                    lab, palette=palette, jmask=jmask, junction_color=(255, 255, 255)
                )

                # 4) 任务 4: 四色定理染色 (DSATUR + 回溯)
                if K > 0:
                    # adj 是 {1..K: set}, dsatur 用 0-based, 转一下
                    adj_0 = {i: {v - 1 for v in adj[i + 1]} for i in range(K)}
                    colors_arr, used_4, method_4 = colorize.four_color(adj_0, K, 4)
                    ok4, bad = colorize.check_coloring(adj_0, colors_arr) if colors_arr else (False, [])
                    # 渲染: 段 id (1-based) → 颜色 (0..3) → PALETTE4
                    colored4 = np.full((h, w, 3), 255, dtype=np.uint8)
                    if colors_arr is not None:
                        for sid in range(1, K + 1):
                            colored4[lab == sid] = colorize.PALETTE4[colors_arr[sid - 1]]
                    colored4[jmask] = (0, 0, 0)  # 结点黑色
                else:
                    colors_arr, used_4, method_4 = None, 0, "无段"
                    ok4, bad = True, []
                    colored4 = np.full((h, w, 3), 255, dtype=np.uint8)

                # 5) 区域级四色 (colorize.region_four_color 亮点)
                regions, n_region, region_colors, fill_mask = colorize.region_four_color(edge)
                region_rgb = render_region_colored(
                    regions, n_region, region_colors, colorize.PALETTE4
                )

                # 6) 闭合轮廓 = 物体边界 (meta['is_closed']: bool 数组, index 是段 id)
                closed_ids = [int(s) for s in np.where(meta["is_closed"])[0] if s >= 1]
                closed_count = len(closed_ids)
                closed_mask = np.isin(lab, closed_ids) if closed_ids else np.zeros_like(lab, dtype=bool)

                # 6b) 近闭合段 (两端距离 ≤ 5px, 可用 MORPH_CLOSE 缝合)
                near_closed_ids = [int(s) for s in np.where(meta["near_closed"])[0] if s >= 1]
                near_closed_count = len(near_closed_ids)
                near_closed_mask = (np.isin(lab, near_closed_ids)
                                    if near_closed_ids else np.zeros_like(lab, dtype=bool))

                # ---- 保存 PNG ----
                # _1_边缘图: 白底黑边
                edge_img = np.full((h, w), 255, dtype=np.uint8)
                edge_img[edge] = 0
                save_image(edge_img, os.path.join(OUTPUT_DIR, f"{base}_1_边缘图.png"))

                # _2_结点拆分: 黑骨架 + 红交叉点
                skel_img = np.full((h, w, 3), 255, dtype=np.uint8)
                skel_img[skel > 0] = (0, 0, 0)
                skel_img[jmask] = (235, 60, 60)
                save_image(skel_img, os.path.join(OUTPUT_DIR, f"{base}_2_结点拆分.png"))

                # _3_逐段染色: 任务 3
                save_image(seg_rgb, os.path.join(OUTPUT_DIR, f"{base}_3_逐段染色.png"))

                # _4_四色染色: 任务 4
                save_image(colored4, os.path.join(OUTPUT_DIR, f"{base}_4_四色染色.png"))

                # _5_原图叠加: 把任务 3 的彩色边叠到原图上
                if orig_rgb is not None:
                    overlay = colorize.overlay(orig_rgb, seg_rgb, alpha=0.6)
                    save_image(overlay, os.path.join(OUTPUT_DIR, f"{base}_5_原图叠加.png"))

                # _6_区域四色: 亮点 — 边缘包围的"区域"级四色 (地图染色)
                save_image(region_rgb, os.path.join(OUTPUT_DIR, f"{base}_6_区域四色.png"))

                # _7_闭合轮廓: 亮点 — 严格闭合边缘 (端点数==0, 红色高亮; Canny 几乎为 0)
                closed_img = np.full((h, w, 3), 255, dtype=np.uint8)
                closed_img[edge] = (180, 180, 180)
                closed_img[closed_mask] = (235, 60, 60)
                save_image(closed_img, os.path.join(OUTPUT_DIR, f"{base}_7_闭合轮廓.png"))

                # _7b_近闭合段: 端点对距离 ≤ 5px, 可用 MORPH_CLOSE 缝合的段 (橙色)
                near_closed_img = np.full((h, w, 3), 255, dtype=np.uint8)
                near_closed_img[edge] = (180, 180, 180)
                near_closed_img[near_closed_mask] = (40, 140, 230)
                save_image(near_closed_img, os.path.join(OUTPUT_DIR, f"{base}_7b_近闭合段.png"))

                # ---- 边缘闭合完整性分析 (新增) ----
                # (a) 端点缺口分析
                gaps, ep_mask = ec.analyze_endpoint_gaps(lab, edge, max_gap=5)
                gap_vis = ec.render_gap_visualization(edge, gaps, ep_mask, (h, w))
                save_image(gap_vis, os.path.join(OUTPUT_DIR, f"{base}_8_缺口分析.png"))

                # (b) 阶梯式 kernel 桥接 1-3px 缺口 (3→5→7)
                closed_e1 = ec.close_edge_gaps(edge, kernel_size=3, iterations=1)
                closed_e2 = ec.close_edge_gaps(closed_e1, kernel_size=5, iterations=1)
                closed_edge = ec.close_edge_gaps(closed_e2, kernel_size=7, iterations=1)
                closed_edge_img = ec.render_closed_edge(closed_edge, (h, w))
                save_image(closed_edge_img, os.path.join(OUTPUT_DIR, f"{base}_9_闭合边缘.png"))

                # (c) 闭合后再划分 -> 闭合段数会显著上升
                lab2, jmask2, adj2, meta2, skel2 = partition.partition_edges(
                    closed_edge, min_len=4, endpoint_adj_r=2.5
                )
                # 闭合后段染色
                K2 = meta2["K"]
                palette2 = colorize.distinct_palette(K2) if K2 > 0 else np.zeros((0, 3), np.uint8)
                seg_after_rgb = colorize.render_segments(
                    lab2, palette=palette2, jmask=jmask2, junction_color=(255, 255, 255)
                )
                save_image(seg_after_rgb, os.path.join(OUTPUT_DIR, f"{base}_10_闭合后划分.png"))

                # (d) 闭合性指标 (前后对比)
                m_before = ec.compute_closure_metrics(lab, edge)
                m_after = ec.compute_closure_metrics(lab2, closed_edge)

                # ---- 统计 ----
                stats = {
                    "image": os.path.basename(img_path),
                    "image_size": [w, h],
                    "edge_pixels": int(edge.sum()),
                    "skeleton_pixels": int(skel.sum()),
                    "junction_pixels": int(jmask.sum()),
                    "n_segments": K,
                    "n_adjacent_pairs": sum(len(s) for s in adj.values()) // 2,
                    "n_closed_contours": closed_count,         # 严格闭合 (端点数==0)
                    "closed_segment_ids": closed_ids[:50],     # 最多列 50 个
                    "n_near_closed_contours": near_closed_count,  # 近闭合 (端点对≤5px)
                    "near_closed_segment_ids": near_closed_ids[:50],
                    "n_regions": n_region,                     # 亮点
                    "region_4color_used": (
                        max(region_colors) + 1 if region_colors is not None else 0
                    ),
                    "segment_4color_used": used_4,             # 任务 4
                    "segment_4color_method": method_4,         # DSATUR / 回溯
                    "segment_4color_ok": bool(ok4),
                    "segment_4color_violations": len(bad),
                    # ---- 边缘闭合完整性 (新增) ----
                    "closure_n_gaps": len(gaps),               # 可缝合的端点对数
                    "closure_mean_gap_px": (
                        round(float(np.mean([g[4] for g in gaps])), 2) if gaps else 0.0
                    ),
                    "closure_before": m_before,                # 闭合前指标
                    "closure_after": m_after,                  # 闭合后指标
                    # 改善 = near_closed_ratio 变化 (Canny 几乎不可能产生
                    # 端点数==0 的严格闭合, 所以用近闭合作为"可缝合性"指标)
                    "closure_improvement": round(
                        m_after["near_closed_ratio"] - m_before["near_closed_ratio"], 3
                    ),
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                }
                stats_path = os.path.join(OUTPUT_DIR, f"{base}_统计.json")
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)
                all_stats.append(stats)

                print(f"  边缘像素: {stats['edge_pixels']}, 骨架: {stats['skeleton_pixels']}, "
                      f"结点: {stats['junction_pixels']}")
                print(f"  段数: {K}, 邻接对: {stats['n_adjacent_pairs']}")
                print(f"  严格闭合(端点数==0): {closed_count}    "
                      f"近闭合(端点对≤5px): {near_closed_count}")
                print(f"  区域数: {n_region}, 区域4色用色: {stats['region_4color_used']}")
                print(f"  段4色: 用 {used_4} 色 (方法={method_4}, "
                      f"通过={ok4}, 冲突={len(bad)})")
                print(f"  闭合性: 缺口 {stats['closure_n_gaps']} 对 "
                      f"(平均 {stats['closure_mean_gap_px']}px), "
                      f"段数 {m_before['n_segments']} -> {m_after['n_segments']}, "
                      f"端点 {m_before['n_endpoint_total']} -> {m_after['n_endpoint_total']}, "
                      f"严格闭合 {m_before['n_closed']} -> {m_after['n_closed']}, "
                      f"近闭合 {m_before['n_near_closed']}({m_before['near_closed_ratio']:.3f}) "
                      f"-> {m_after['n_near_closed']}({m_after['near_closed_ratio']:.3f}), "
                      f"平均 gap {m_after['mean_gap_px']}px")
                print(f"  耗时: {stats['elapsed_ms']} ms")
            except Exception as e:
                print(f"  失败: {e}")
                traceback.print_exc()

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 70)
    print(f"汇总 -> {summary_path}")
    print(f"共处理 {len(all_stats)} 张图片, 输出目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

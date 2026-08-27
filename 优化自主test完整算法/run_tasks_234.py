# -*- coding: utf-8 -*-
"""
edge_pipeline 算法  ——  docx 任务 2/3/4 实现
================================================
用 edge_pipeline.py 替代自研 MCT-Edge 算法, 完成 docx 任务:
  - 任务 2: 边缘划分 (segment_edges: branch_points + 8-CC)
  - 任务 3: 逐段染色 (render_colored, 黄金角 HSV)
  - 任务 4: 四色定理染色 (dsatur_coloring 4 色)
  - 亮点: 区域级四色 (extract_objects + region_adjacency + DSATUR)
  - 亮点: 折线有序化 (order_segments + render_polylines)
"""
import os
import sys
import json
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import edge_pipeline as ep  # noqa: E402

INPUT_IMAGES = [
    r"C:\Users\18607\Desktop\边缘识别模型方法\图片物体边缘的提取分割.png",
    r"C:\Users\18607\Desktop\bsds_368037_原图.jpg",
    r"C:\Users\18607\Desktop\bsds_97010_原图.jpg",
    r"C:\Users\18607\Desktop\nyud_5017_原图.png",
    r"C:\Users\18607\Desktop\nyud_6233_原图.png",
]

OUTPUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 72)
    print("edge_pipeline 算法  --  替代自研 MCT-Edge, 实现 docx 任务 2/3/4")
    print("  Canny:   edge_pipeline.canny (DoG + subpixel NMS + adaptive + hysteresis + gap link)")
    print("  划分:    edge_pipeline.segment_edges (Zhang-Suen + branch_points + 8-CC)")
    print("  染色:    edge_pipeline.render_colored (golden angle HSV)")
    print("  4-color: edge_pipeline.dsatur_coloring (DSATUR)")
    print("=" * 72)
    all_stats = []

    for idx, img_path in enumerate(INPUT_IMAGES, start=1):
        if not os.path.exists(img_path):
            print(f"[{idx}] skip (missing): {img_path}")
            continue
        print(f"[{idx}/{len(INPUT_IMAGES)}] {os.path.basename(img_path)}")
        try:
            t0 = time.perf_counter()
            stats = ep.run_pipeline(img_path, OUTPUT_DIR)
            elapsed = (time.perf_counter() - t0) * 1000
            stats["run_pipeline_elapsed_ms"] = round(elapsed, 1)
            base = os.path.splitext(os.path.basename(img_path))[0]
            stats_path = os.path.join(OUTPUT_DIR, f"{base}_统计.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            all_stats.append(stats)
            print(f"  edge: {stats.get('edge_pixels')}, skel: {stats.get('skeleton_pixels')}, "
                  f"junc: {stats.get('junction_pixels')}")
            print(f"  segments: {stats.get('n_segments')}, "
                  f"adjacent pairs: {stats.get('n_adjacent_pairs')}, "
                  f"closed: {stats.get('n_closed_segments')}")
            print(f"  mean_len: {stats.get('mean_segment_length')}, "
                  f"max: {stats.get('max_segment_length')}, "
                  f"total: {stats.get('total_segment_length')}")
            if stats.get("four_color_used") is not None:
                print(f"  4-color: used {stats['four_color_used']} (enough={stats['four_color_enough']}, "
                      f"viol={stats.get('four_color_violations', 0)})")
            print(f"  objects: {stats.get('n_objects')}, "
                  f"region adj: {stats.get('n_region_adjacent_pairs')}, "
                  f"region 4color: {stats.get('region_four_color_used')}")
            print(f"  elapsed: {elapsed:.0f} ms")
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print("=" * 72)
    print(f"summary -> {summary_path}")
    print(f"processed {len(all_stats)} images")
    print("=" * 72)


if __name__ == "__main__":
    main()

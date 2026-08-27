# -*- coding: utf-8 -*-
"""
对比模型 / Compare Models
=========================

在 5 张固定测试图上跑 5 种主流边缘检测算法并对比。
所有结果保存到本脚本所在目录的 outputs/ 子文件夹中。

用法:
    python compare.py                 # 跑全部 5 张图
    python compare.py --quick         # 只跑 1 张 (bsds_97010) 快速验证
    python compare.py --only nyud_5017_原图

输出:
    outputs/<图名>_0_original.png        原图
    outputs/<图名>_1_opencv_canny.png   OpenCV Canny (10,100)
    outputs/<图名>_2_skimage_canny.png  skimage Canny (sigma=1, 0.05/0.15)
    outputs/<图名>_pidinet_edges.png    PiDiNet 概率图
    outputs/<图名>_pidinet_binary.png   PiDiNet 二值图 (阈值 128)
    outputs/<图名>_dexined_*.png        DexiNed 4 张 (fused/avg × 单/双)
    outputs/<图名>_comparison.png       6 panel 对比图 (2 行 × 3 列网格)
    outputs/<图名>_统计.json            单图统计
    outputs/summary.json                5 张图汇总

依赖: numpy, opencv-python, scikit-image。
PiDiNet 和 DexiNed 通过子进程调用其原脚本(间接依赖 PyTorch)。
"""

import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np
import cv2
from skimage import feature


# ===================== 5 张固定测试图(与其他脚本保持一致) =====================
INPUT_IMAGES = [
    r"C:\Users\18607\Desktop\边缘识别模型方法\图片物体边缘的提取分割.png",
    r"C:\Users\18607\Desktop\bsds_368037_原图.jpg",
    r"C:\Users\18607\Desktop\bsds_97010_原图.jpg",
    r"C:\Users\18607\Desktop\nyud_5017_原图.png",
    r"C:\Users\18607\Desktop\nyud_6233_原图.png",
]

# ===================== 各算法实际用到的阈值(写进统计 JSON 便于复现) =====================
# 与 open cv 实现/canny_edge_detection.py 保持一致 (10, 100)
OPENCV_CANNY_LOW = 10
OPENCV_CANNY_HIGH = 100

SKIMAGE_SIGMA = 1.0
SKIMAGE_LOW = 0.05
SKIMAGE_HIGH = 0.15

PIDINET_BIN_THRESHOLD = 128
DEXINED_BIN_THRESHOLD = 128


# ===================== 路径常量 =====================
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 父目录(包含 PiDiNet、DexiNed 等算法)
PARENT = os.path.dirname(HERE)


# ===================== 工具函数 =====================
def imread_unicode(path: str):
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    cv2.imencode(os.path.splitext(path)[1], img)[1].tofile(path)


# ===================== 1. OpenCV Canny =====================
def opencv_canny(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, OPENCV_CANNY_LOW, OPENCV_CANNY_HIGH)


# ===================== 2. scikit-image Canny =====================
def skimage_canny(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    edges = feature.canny(gray, sigma=SKIMAGE_SIGMA,
                          low_threshold=SKIMAGE_LOW,
                          high_threshold=SKIMAGE_HIGH)
    return edges.astype(np.uint8) * 255


# ===================== 3/4/5. PyTorch 算法(子进程调用) =====================
def run_subprocess(script_path: str, *args, timeout: int = 600) -> dict:
    """调用子脚本, 返回 {ok, returncode, stdout, stderr, elapsed_ms}."""
    if not os.path.exists(script_path):
        return {"ok": False, "returncode": -1, "stdout": "",
                "stderr": f"script not found: {script_path}", "elapsed_ms": 0}
    cmd = [sys.executable, script_path, *args]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"ok": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed_ms": int((time.time() - t0) * 1000)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -1, "stdout": "",
                "stderr": f"timeout ({timeout}s)",
                "elapsed_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "returncode": -1, "stdout": "",
                "stderr": str(e), "elapsed_ms": int((time.time() - t0) * 1000)}


# ===================== 6 panel 网格化对比图 (2 行 × 3 列) =====================
def build_comparison(output_dir: str, base: str) -> str | None:
    panels = [
        ("Original",         f"{base}_0_original.png"),
        ("OpenCV Canny",     f"{base}_1_opencv_canny.png"),
        ("skimage Canny",    f"{base}_2_skimage_canny.png"),
        ("PiDiNet (bin)",    f"{base}_pidinet_binary.png"),
        ("DexiNed (bin)",    f"{base}_dexined_fused_binary.png"),
        ("DexiNed x2 (bin)", f"{base}_dexined_double_fused_binary.png"),
    ]

    H_target = 220
    rendered = []
    for name, fname in panels:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] {name}: file missing ({fname})")
            continue
        img = imread_unicode(path)
        if img is None:
            print(f"  [skip] {name}: decode failed")
            continue
        h, w = img.shape[:2]
        new_w = max(1, int(w * H_target / h))
        img = cv2.resize(img, (new_w, H_target))
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        canvas = np.full((H_target + 28, new_w, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, name, (4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        canvas[28:, :, :] = img
        rendered.append(canvas)

    if not rendered:
        print("  no images to compare")
        return None

    # 2 行 × 3 列网格(避免单行过宽, 屏幕友好)
    n_cols = 3
    rows = [rendered[i:i + n_cols] for i in range(0, len(rendered), n_cols)]
    row_imgs = []
    for row in rows:
        if not row:
            continue
        max_w = max(p.shape[1] for p in row)
        padded = []
        for p in row:
            if p.shape[1] < max_w:
                pad = np.full((p.shape[0], max_w - p.shape[1], 3), 255, dtype=np.uint8)
                p = np.hstack([p, pad])
            padded.append(p)
        row_imgs.append(np.hstack(padded))
    grid = np.vstack(row_imgs)
    out_path = os.path.join(output_dir, f"{base}_comparison.png")
    cv2.imencode(".png", grid)[1].tofile(out_path)
    print(f"  comparison -> {out_path}  ({grid.shape[1]}x{grid.shape[0]})")
    return out_path


# ===================== 单图主流程 =====================
def process_one(img_path: str) -> dict:
    """对一张图跑 5 种算法并返回统计 dict。"""
    name = os.path.splitext(os.path.basename(img_path))[0]
    stats = {
        "image": name,
        "path": img_path,
        "thresholds": {
            "opencv_canny": [OPENCV_CANNY_LOW, OPENCV_CANNY_HIGH],
            "skimage_canny": {"low": SKIMAGE_LOW, "high": SKIMAGE_HIGH, "sigma": SKIMAGE_SIGMA},
            "pidinet_bin": PIDINET_BIN_THRESHOLD,
            "dexined_bin": DEXINED_BIN_THRESHOLD,
        },
    }

    bgr = imread_unicode(img_path)
    if bgr is None:
        stats["error"] = f"无法解码: {img_path}"
        return stats
    H, W = bgr.shape[:2]
    stats["size"] = [W, H]

    print(f"\n>>> {name} ({W}x{H})")
    print("-" * 70)

    # 0. 原图
    imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_0_original.png"), bgr)
    print("  [0/5] 原图已保存")

    # 1. OpenCV Canny
    print("  [1/5] OpenCV Canny")
    t0 = time.time()
    try:
        edges = opencv_canny(bgr)
        imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_1_opencv_canny.png"), edges)
        n_pix = int((edges > 0).sum())
        ratio = n_pix / (H * W)
        stats["opencv_canny"] = {"pixels": n_pix, "ratio": round(ratio, 4),
                                 "time_ms": int((time.time() - t0) * 1000)}
        print(f"         边缘像素: {n_pix} ({ratio*100:.2f}%)  "
              f"耗时 {stats['opencv_canny']['time_ms']} ms")
    except Exception as e:
        stats["opencv_canny"] = {"error": str(e)}
        print(f"         失败: {e}")

    # 2. skimage Canny
    print("  [2/5] skimage Canny")
    t0 = time.time()
    try:
        edges = skimage_canny(bgr)
        imwrite_unicode(os.path.join(OUTPUT_DIR, f"{name}_2_skimage_canny.png"), edges)
        n_pix = int((edges > 0).sum())
        ratio = n_pix / (H * W)
        stats["skimage_canny"] = {"pixels": n_pix, "ratio": round(ratio, 4),
                                  "time_ms": int((time.time() - t0) * 1000)}
        print(f"         边缘像素: {n_pix} ({ratio*100:.2f}%)  "
              f"耗时 {stats['skimage_canny']['time_ms']} ms")
    except Exception as e:
        stats["skimage_canny"] = {"error": str(e)}
        print(f"         失败: {e}")

    # 3. PiDiNet (子进程)
    print("  [3/5] PiDiNet (PyTorch, 子进程)")
    pidinet_script = os.path.join(PARENT, "PyTorch PiDiNet 实现", "pidinet_inference.py")
    r = run_subprocess(pidinet_script, "--input", img_path, "--output", OUTPUT_DIR)
    stats["pidinet"] = {"ok": r["ok"], "elapsed_ms": r["elapsed_ms"],
                        "returncode": r["returncode"]}
    if r["ok"]:
        stdout = r["stdout"]
        stats["pidinet"]["stdout_tail"] = stdout[-400:] if len(stdout) > 400 else stdout
        print(f"         成功, 耗时 {r['elapsed_ms']} ms")
        for line in stdout.splitlines():
            if any(k in line for k in ("边缘强度", "已保存", "checkpoint")):
                print(f"           {line.strip()}")
    else:
        stats["pidinet"]["stderr_tail"] = r["stderr"][-400:]
        print(f"         失败 (rc={r['returncode']}): {r['stderr'][:200]}")

    # 4. DexiNed (子进程)
    print("  [4/5] DexiNed (PyTorch, 子进程)")
    dexined_script = os.path.join(PARENT, "PyTorch DexiNed 实现", "dexined_inference.py")
    r = run_subprocess(dexined_script, "--input", img_path, "--output", OUTPUT_DIR)
    stats["dexined"] = {"ok": r["ok"], "elapsed_ms": r["elapsed_ms"],
                        "returncode": r["returncode"]}
    if r["ok"]:
        stdout = r["stdout"]
        stats["dexined"]["stdout_tail"] = stdout[-400:] if len(stdout) > 400 else stdout
        print(f"         成功, 耗时 {r['elapsed_ms']} ms")
        for line in stdout.splitlines():
            if any(k in line for k in ("fused", "avg", "已保存", "原图尺寸")):
                print(f"           {line.strip()}")
    else:
        stats["dexined"]["stderr_tail"] = r["stderr"][-400:]
        print(f"         失败 (rc={r['returncode']}): {r['stderr'][:200]}")

    # 5. DexiNed double (子进程)
    print("  [5/5] DexiNed double (PyTorch, 子进程)")
    r = run_subprocess(dexined_script, "--input", img_path, "--output", OUTPUT_DIR, "--double")
    stats["dexined_double"] = {"ok": r["ok"], "elapsed_ms": r["elapsed_ms"],
                               "returncode": r["returncode"]}
    if r["ok"]:
        stats["dexined_double"]["stdout_tail"] = r["stdout"][-400:]
        print(f"         成功, 耗时 {r['elapsed_ms']} ms")
    else:
        stats["dexined_double"]["stderr_tail"] = r["stderr"][-400:]
        print(f"         失败 (rc={r['returncode']}): {r['stderr'][:200]}")

    # 6. 拼对比图
    print("  [对比] 生成对比图")
    cmp_path = build_comparison(OUTPUT_DIR, name)
    if cmp_path:
        stats["comparison_image"] = cmp_path

    return stats


# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser(description="对比 5 种边缘检测算法")
    parser.add_argument("--quick", action="store_true",
                        help="只跑 1 张 (bsds_97010) 快速验证")
    parser.add_argument("--only", type=str, default=None,
                        help="只跑指定名字的图 (如 bsds_97010_原图)")
    args = parser.parse_args()

    images = list(INPUT_IMAGES)
    if args.quick:
        images = [r"C:\Users\18607\Desktop\bsds_97010_原图.jpg"]
    if args.only:
        images = [p for p in INPUT_IMAGES
                  if os.path.splitext(os.path.basename(p))[0] == args.only]

    print("=" * 70)
    print(f"对比模型  ——  {len(images)} 张图 × 5 种算法")
    print(f"输出目录  :  {OUTPUT_DIR}")
    print(f"Canny 阈值:  OpenCV={OPENCV_CANNY_LOW}/{OPENCV_CANNY_HIGH}  "
          f"skimage=({SKIMAGE_LOW}/{SKIMAGE_HIGH}, σ={SKIMAGE_SIGMA})")
    print("=" * 70)

    summary = []
    t_start = time.time()
    for img_path in images:
        if not os.path.exists(img_path):
            print(f"\n[warn] 图像不存在, 跳过: {img_path}")
            continue
        try:
            stats = process_one(img_path)
        except Exception as e:
            import traceback
            print(f"\n[ERR] {img_path}: {e}")
            traceback.print_exc()
            stats = {"image": os.path.basename(img_path), "error": str(e)}

        summary.append(stats)
        # 单图统计 JSON(实时落盘, 中途崩了也不丢)
        name = os.path.splitext(os.path.basename(img_path))[0]
        per_path = os.path.join(OUTPUT_DIR, f"{name}_统计.json")
        with open(per_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    total_ms = int((time.time() - t_start) * 1000)
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"total_ms": total_ms, "n_images": len(summary),
                   "results": summary}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"完成!  共 {len(summary)} 张图,  总耗时 {total_ms} ms")
    print(f"汇总    -> {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
基于 scikit-image 库的图像边缘检测（优化版）
---------------------------------
相比初版的改进：
    1. 每个梯度算子除了保存原始幅值图外，还做：
        - Otsu 自适应二值化（自动选最佳阈值）
        - 骨架化（skeletonize）得到 1 像素宽的细线
    2. Laplace 算子（对噪声极敏感）先做重度高斯平滑再做二阶导；
    3. 拼一张 2x4 的对比大图，便于直观对比所有方法；
    4. 给出每种方法的边缘像素数和耗时统计。

使用：
    直接运行本脚本即可。
"""

import os
import time
import numpy as np
import cv2  # 仅用于读取/写出含中文路径的图像
from skimage import filters, feature
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte

# ---------- 路径配置 ----------
INPUT_IMAGE_PATH = r"C:\Users\18607\Desktop\边缘识别模型方法\图片物体边缘的提取分割.png"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 参数配置 ----------
# skimage.feature.canny 的双阈值（值越低，检测到的边缘越多）
CANNY_SIGMA = 1.0
CANNY_LOW = 0.05
CANNY_HIGH = 0.15

# Laplace 前置平滑的 sigma（必须够大才能压住二阶导数的噪声放大效应）
LAPLACE_SMOOTH_SIGMA = 2.0

# 各种算子归一化用的常数（绝大多数情况下用 255/归一化到 0~1 即可）
EPS = 1e-12


# ---------- 工具函数 ----------
def imread_unicode(path: str) -> np.ndarray:
    """读取含中文路径的图像。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    """写出图像到含中文路径。"""
    cv2.imencode(".png", img)[1].tofile(path)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """把浮点结果线性归一化到 0~255 再转 uint8，便于保存。"""
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < EPS:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - mn) / (mx - mn) * 255.0
    return out.astype(np.uint8)


def otsu_binary(magnitude: np.ndarray) -> np.ndarray:
    """对梯度幅值做 Otsu 阈值二值化，返回 bool 数组。"""
    # Otsu 需要 uint8 输入
    mag_u8 = to_uint8(magnitude)
    thresh = filters.threshold_otsu(mag_u8)
    return mag_u8 > thresh


def post_process_binary(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把二值图做骨架化（细线化），返回 (binary_uint8, skeleton_uint8)。"""
    binary_uint8 = (binary.astype(np.uint8)) * 255
    skel = skeletonize(binary)
    skel_uint8 = (skel.astype(np.uint8)) * 255
    return binary_uint8, skel_uint8


# ---------- 单个算子：原始 → 二值 → 骨架化 三件套 ----------
def gradient_trio(name: str,
                  gray: np.ndarray,
                  magnitude: np.ndarray,
                  results: dict,
                  timings: dict) -> None:
    """对单个梯度算子的结果做三件套处理，并把数据存到 results 字典。"""
    t0 = time.perf_counter()

    # 1) 原始幅值
    raw_u8 = to_uint8(magnitude)

    # 2) Otsu 二值
    binary = otsu_binary(magnitude)
    binary_u8, skel_u8 = post_process_binary(binary)

    elapsed = time.perf_counter() - t0
    timings[name] = elapsed * 1000  # ms

    results[name] = {
        "raw": raw_u8,
        "binary": binary_u8,
        "skel": skel_u8,
        "edge_count": int(binary.sum()),
    }


# ---------- 拼成 2x4 对比大图 ----------
def make_comparison(original_bgr: np.ndarray,
                    gray_u8: np.ndarray,
                    canny_u8: np.ndarray,
                    results: dict) -> np.ndarray:
    """把 8 张小图（原图、灰度、6 个算子的二值结果）拼成 2x4 对比大图。"""
    # 每张图都转成 BGR 以便和原图保持通道一致
    def to_bgr(u8: np.ndarray) -> np.ndarray:
        if u8.ndim == 2:
            return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        return u8

    # 二值图：黑色背景白色边缘 → 反过来成"白底黑边"更接近"线稿"观感
    def invert(u8: np.ndarray) -> np.ndarray:
        if u8.ndim == 2:
            return 255 - u8
        return 255 - u8

    # 缩放到统一高度，便于拼接
    H = 160
    panels = [
        ("Original",  cv2.resize(original_bgr, (int(original_bgr.shape[1] * H / original_bgr.shape[0]), H))),
        ("Gray",      cv2.cvtColor(cv2.resize(gray_u8, (int(gray_u8.shape[1] * H / gray_u8.shape[0]), H)), cv2.COLOR_GRAY2BGR)),
        ("Canny",     to_bgr(cv2.resize(canny_u8, (int(canny_u8.shape[1] * H / canny_u8.shape[0]), H)))),
        ("Sobel",     to_bgr(cv2.resize(invert(results["sobel"]["binary"]),   (int(canny_u8.shape[1] * H / canny_u8.shape[0]), H)))),
        ("Prewitt",   to_bgr(cv2.resize(invert(results["prewitt"]["binary"]), (int(canny_u8.shape[1] * H / canny_u8.shape[0]), H)))),
        ("Scharr",    to_bgr(cv2.resize(invert(results["scharr"]["binary"]),  (int(canny_u8.shape[1] * H / canny_u8.shape[0]), H)))),
        ("Roberts",   to_bgr(cv2.resize(invert(results["roberts"]["binary"]), (int(canny_u8.shape[1] * H / canny_u8.shape[0]), H)))),
        ("Laplace*",  to_bgr(cv2.resize(invert(results["laplace"]["binary"]), (int(canny_u8.shape[1] * H / canny_u8.shape[0]), H)))),
    ]

    # 在每张小图顶部写名称
    labelled = []
    for name, img in panels:
        h, w = img.shape[:2]
        canvas = np.full((h + 24, w, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, name, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        canvas[24:, :, :] = img
        labelled.append(canvas)

    # 2x4 拼接
    row1 = np.hstack(labelled[:4])
    row2 = np.hstack(labelled[4:])
    sep = np.full((4, row1.shape[1], 3), 200, dtype=np.uint8)
    return np.vstack([row1, sep, row2])


# ---------- 主体 ----------
def run_edge_detection(image_path: str, output_dir: str) -> None:
    bgr = imread_unicode(image_path)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_u8 = (gray * 255).astype(np.uint8)

    base = os.path.splitext(os.path.basename(image_path))[0]

    # 保存灰度原图
    gray_path = os.path.join(output_dir, f"{base}_gray.png")
    imwrite_unicode(gray_path, gray_u8)

    # ---------------- 1. Canny ----------------
    t0 = time.perf_counter()
    canny_edges = feature.canny(gray, sigma=CANNY_SIGMA,
                                low_threshold=CANNY_LOW,
                                high_threshold=CANNY_HIGH)
    canny_u8 = (canny_edges.astype(np.uint8)) * 255
    canny_ms = (time.perf_counter() - t0) * 1000

    canny_path = os.path.join(output_dir, f"{base}_skimage_canny.png")
    imwrite_unicode(canny_path, canny_u8)

    # ---------------- 2. 五个梯度算子 + Laplace ----------------
    results = {}
    timings = {}

    gradient_trio("sobel",   gray, filters.sobel(gray),  results, timings)
    gradient_trio("prewitt", gray, filters.prewitt(gray), results, timings)
    gradient_trio("scharr",  gray, filters.scharr(gray),  results, timings)
    gradient_trio("roberts", gray, filters.roberts(gray), results, timings)

    # Laplace 必须先高斯平滑（sigma 越大越能压住二阶导的噪声）
    smoothed_gray = filters.gaussian(gray, sigma=LAPLACE_SMOOTH_SIGMA)
    gradient_trio("laplace", gray, np.abs(filters.laplace(smoothed_gray)), results, timings)

    # ---------------- 3. 保存三件套结果 ----------------
    for name, d in results.items():
        # 原始幅值
        imwrite_unicode(os.path.join(output_dir, f"{base}_skimage_{name}.png"),          d["raw"])
        # Otsu 二值
        imwrite_unicode(os.path.join(output_dir, f"{base}_skimage_{name}_binary.png"),  d["binary"])
        # 骨架化（1 像素细线）
        imwrite_unicode(os.path.join(output_dir, f"{base}_skimage_{name}_skel.png"),    d["skel"])

    # ---------------- 4. 拼对比大图 ----------------
    comp = make_comparison(bgr, gray_u8, canny_u8, results)
    comp_path = os.path.join(output_dir, f"{base}_skimage_comparison.png")
    imwrite_unicode(comp_path, comp)

    # ---------------- 5. 输出汇总 ----------------
    total_pixels = gray_u8.size
    print(f"原图尺寸                : {bgr.shape}")
    print(f"灰度图已保存            : {gray_path}")
    print(f"Canny 边缘图已保存       : {canny_path}  ({canny_ms:.1f} ms, "
          f"边缘像素 {int(canny_edges.sum())}/{total_pixels} = "
          f"{canny_edges.mean()*100:.2f}%)")
    print("五个算子结果：")
    print(f"  {'方法':<10}{'耗时(ms)':<12}{'边缘像素':<14}{'占比':<10}")
    for name in ("sobel", "prewitt", "scharr", "roberts", "laplace"):
        d = results[name]
        ratio = d["edge_count"] / total_pixels * 100
        print(f"  {name:<10}{timings[name]:<12.2f}{d['edge_count']:<14}{ratio:<10.2f}")
    print(f"对比大图已保存          : {comp_path}  (2x4 拼图)")


if __name__ == "__main__":
    run_edge_detection(INPUT_IMAGE_PATH, OUTPUT_DIR)

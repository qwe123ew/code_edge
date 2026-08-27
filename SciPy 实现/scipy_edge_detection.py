# -*- coding: utf-8 -*-
"""
基于 SciPy 库的 Canny 边缘检测
---------------------------------
SciPy 自身没有封装好的 Canny 接口，但它提供了
    - scipy.ndimage.gaussian_filter        高斯平滑
    - scipy.ndimage.sobel / prewitt        一阶梯度
    - scipy.ndimage.gaussian_laplace       二阶梯度
因此本脚本用 SciPy 重新实现 Canny 的完整四步流程：
    1) 高斯平滑
    2) Sobel 一阶梯度幅值与方向
    3) 非极大值抑制
    4) 双阈值滞后跟踪
最终输出与 OpenCV / skimage Canny 等价的二值边缘图。

说明：
    OpenCV 不可用，全部 I/O 用 PIL + numpy 走 bytes 完成，兼容中文路径。
"""

import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel


# ---------- 路径配置 ----------
INPUT_IMAGE_PATH = r"C:\Users\18607\Desktop\边缘识别模型方法\图片物体边缘的提取分割.png"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Canny 参数 ----------
GAUSSIAN_SIGMA = 1.0
LOW_THRESHOLD = 0.05    # 归一化到 0~1
HIGH_THRESHOLD = 0.15


# ---------- 工具函数 ----------
def imread_unicode(path: str) -> np.ndarray:
    """PIL 可以读中文路径，返回 uint8 灰度数组。"""
    with Image.open(path) as img:
        gray = img.convert("L")
        return np.asarray(gray, dtype=np.uint8)


def imwrite_unicode(path: str, arr: np.ndarray) -> None:
    """写出含中文路径的灰度图（二值图也用此法）。"""
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    Image.fromarray(arr).save(path)


def _non_max_suppression(mag: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """非极大值抑制：只在梯度方向上保留局部最大值。"""
    H, W = mag.shape
    out = np.zeros_like(mag)

    # 把角度量化到 4 个方向：0°, 45°, 90°, 135°
    angle = angle * 180.0 / np.pi
    angle[angle < 0] += 180.0
    q = np.zeros_like(angle, dtype=np.uint8)
    q[(angle >= 0) & (angle < 22.5) | (angle >= 157.5) & (angle <= 180)] = 0       # 水平
    q[(angle >= 22.5) & (angle < 67.5)] = 1                                        # 45°
    q[(angle >= 67.5) & (angle < 112.5)] = 2                                       # 垂直
    q[(angle >= 112.5) & (angle < 157.5)] = 3                                      # 135°

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            m = mag[y, x]
            if q[y, x] == 0:
                if not (m >= mag[y, x - 1] and m >= mag[y, x + 1]):
                    continue
            elif q[y, x] == 1:
                if not (m >= mag[y - 1, x - 1] and m >= mag[y + 1, x + 1]):
                    continue
            elif q[y, x] == 2:
                if not (m >= mag[y - 1, x] and m >= mag[y + 1, x]):
                    continue
            else:
                if not (m >= mag[y - 1, x + 1] and m >= mag[y + 1, x - 1]):
                    continue
            out[y, x] = m
    return out


def _hysteresis(edges: np.ndarray, low: float, high: float) -> np.ndarray:
    """双阈值滞后：把连接到强边缘的弱边缘一并保留。"""
    strong = edges >= high
    weak = (edges >= low) & (edges < high)

    out = strong.copy()
    # 通过 8 邻域扩张，把与已确认边缘相邻的弱边缘激活
    for _ in range(100):
        # 8 邻域求或
        expanded = np.zeros_like(out)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                # 把 out 平移 dy,dx 后的结果加到 expanded
                shifted = np.zeros_like(out)
                ys_src = slice(max(0, -dy), out.shape[0] - max(0, dy))
                ys_dst = slice(max(0, dy), out.shape[0] - max(0, -dy))
                xs_src = slice(max(0, -dx), out.shape[1] - max(0, dx))
                xs_dst = slice(max(0, dx), out.shape[1] - max(0, -dx))
                shifted[ys_dst, xs_dst] = out[ys_src, xs_src]
                expanded |= shifted

        grown = out | (expanded & weak)
        if np.array_equal(grown, out):
            break
        out = grown
    return out


def canny_scipy(gray: np.ndarray,
                sigma: float = 1.0,
                low: float = 0.05,
                high: float = 0.15) -> np.ndarray:
    """用 SciPy 实现的 Canny 边缘检测。"""
    # 1. 高斯平滑
    smoothed = gaussian_filter(gray.astype(np.float32) / 255.0, sigma=sigma)

    # 2. Sobel 一阶梯度
    gx = sobel(smoothed, axis=1)
    gy = sobel(smoothed, axis=0)
    mag = np.hypot(gx, gy)
    mag /= (mag.max() + 1e-12)  # 归一化到 0~1
    angle = np.arctan2(gy, gx)

    # 3. 非极大值抑制
    nms = _non_max_suppression(mag, angle)

    # 4. 双阈值 + 滞后
    edges = _hysteresis(nms, low, high)
    return edges


def run_edge_detection(image_path: str, output_dir: str) -> None:
    gray = imread_unicode(image_path)
    if gray is None or gray.size == 0:
        raise FileNotFoundError(f"无法读取图像：{image_path}")

    base = os.path.splitext(os.path.basename(image_path))[0]

    gray_path = os.path.join(output_dir, f"{base}_gray.png")
    imwrite_unicode(gray_path, gray)

    edges = canny_scipy(gray, GAUSSIAN_SIGMA, LOW_THRESHOLD, HIGH_THRESHOLD)
    edges_path = os.path.join(output_dir, f"{base}_scipy_canny.png")
    imwrite_unicode(edges_path, edges)

    print(f"原图尺寸                : {gray.shape}")
    print(f"灰度图已保存            : {gray_path}")
    print(f"Canny 边缘图已保存       : {edges_path}")
    print(f"参数                    : sigma={GAUSSIAN_SIGMA}, low={LOW_THRESHOLD}, high={HIGH_THRESHOLD}")


if __name__ == "__main__":
    run_edge_detection(INPUT_IMAGE_PATH, OUTPUT_DIR)

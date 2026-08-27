# -*- coding: utf-8 -*-
"""
图片物体边缘的提取与分割 —— 完整管线
======================================

针对一张图片完成:
  1. 边缘检测   : 从零实现 Canny 算法(高斯模糊 -> Scharr/Sobel/DoG 梯度 -> 亚像素插值
                  非极大值抑制 -> 局部自适应双阈值 + 滞后连接), 得到二值边缘图;
                  默认对彩色图逐通道合成梯度, 能同时检出亮度边缘与纯色差边缘;
  2. 边缘划分   : 把边缘像素划分为“同一条边缘”的集合。方法: 计算每个边缘像素
                  的 8-邻接度, 度 >= 3 的像素是交叉点(结点); 去掉结点后对剩余
                  像素做连通域标记, 每个连通域就是一条边缘段; 结点像素再按
                  就近原则归属, 并用于构建边缘段之间的邻接关系;
  3. 染色展示   : 每条边缘段染一种颜色(hue 均匀采样), 输出逐段染色图;
  4. 四色定理染色(可选): 把“边缘段”看作图的顶点, 共享结点或 8-邻接的边缘段
                  之间连边, 用 DSATUR 贪心算法给整张图染 <=4 种颜色,
                  保证相邻边缘段颜色不同, 视觉区分最清晰;
  5. 汇总输出   : 保存原图、边缘图、逐段染色图、四色染色图与统计信息。

用法:
  python edge_pipeline.py 图片本地路径 [--output 输出目录] [--no-four-color]
  python edge_pipeline.py --input 图片本地路径   # 与位置参数等价
  python edge_pipeline.py "D:\\图片\\photo.png"   # 直接传本地路径, 默认输出到 ./output

仅依赖 numpy 与 Pillow, 不依赖 OpenCV。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw


# ----------------------------------------------------------------------------
# 第 1 步: 边缘检测 (从零实现 Canny)
# ----------------------------------------------------------------------------

def load_grayscale(path: str) -> np.ndarray:
    """读取图片并转为 float32 灰度数组, 值域 [0, 255]。"""
    with Image.open(path) as im:
        gray = im.convert("L")
    return np.asarray(gray, dtype=np.float32)


RGB_TO_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def load_rgb(path: str) -> np.ndarray:
    """读取图片并转为 float32 RGB 数组, 值域 [0, 255]。

    RGBA 图片先把半透明像素合成到白色背景, 避免透明通道干扰边缘检测。
    """
    with Image.open(path) as im:
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            rgb = bg
        else:
            rgb = im.convert("RGB")
    return np.asarray(rgb, dtype=np.float32)


def _gaussian_blur_impl(img64: np.ndarray, sigma: float, r: int) -> np.ndarray:
    """高斯模糊核心: 可分离一维核沿两个方向各卷积一次(反射填充, 严格居中)。"""
    x = np.arange(-r, r + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    # 用 float64 累加, 末尾转回 float32, 避免 float32 累加误差在平坦区域留下伪梯度
    h, w = img64.shape
    padded = np.pad(img64, ((0, 0), (r, r)), mode="reflect")
    tmp = np.zeros_like(img64)
    for m in range(2 * r + 1):
        tmp += kernel[m] * padded[:, m : m + w]
    tmp_padded = np.pad(tmp, ((r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(img64)
    for m in range(2 * r + 1):
        out += kernel[m] * tmp_padded[m : m + h, :]
    return out.astype(np.float32)


def gaussian_blur(img: np.ndarray, sigma: float = 1.4, radius: int | None = None) -> np.ndarray:
    """高斯模糊: 用反射填充 + 可分离一维核沿两个方向各卷积一次, 避免边界变暗。

    大尺度模糊(如自适应阈值的局部参照, sigma 可达图片短边/12)直接逐点卷积的
    开销随半径线性增长; 此时改用 FFT 卷积: 先反射填充到含完整核支撑的尺寸,
    再做周期卷积, 中央区域与逐点卷积在浮点误差内一致, 但快一到两个数量级。
    小 sigma 路径(radius < 60)保持逐点精确卷积。
    """
    if radius is None:
        radius = int(round(3 * sigma))
    r = max(1, radius)
    img64 = img.astype(np.float64)
    h, w = img.shape

    fft_size = (h + 2 * r) * (w + 2 * r)
    if r >= 60 and h >= 256 and w >= 256 and fft_size <= 2.0e7:
        ph, pw = h + 2 * r, w + 2 * r
        padded = np.pad(img64, ((r, r), (r, r)), mode="reflect")
        x = np.arange(-r, r + 1, dtype=np.float64)
        k1 = np.exp(-(x * x) / (2.0 * sigma * sigma))
        k1 /= k1.sum()
        # 把核按“g[t mod N] = k[t]”的周期布局放入, 使 FFT 结果中央区域与
        # 逐点反射卷积一致(核中心权重放在下标 0, 正偏移在前、负偏移绕到尾部)
        idx = np.arange(-r, r + 1)
        g = np.zeros((ph, pw))
        g[np.ix_(idx % ph, idx % pw)] = np.outer(k1, k1)
        f_img = np.fft.rfft2(padded)
        f_k = np.fft.rfft2(g)
        out = np.fft.irfft2(f_img * f_k, s=(ph, pw))
        return out[r : r + h, r : r + w].astype(np.float32)

    return _gaussian_blur_impl(img64, sigma, r)


SOBEL_KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
# Scharr 算子在 3x3 里旋转对称性最优, 梯度幅值与方向误差明显小于 Sobel
SCHARR_KX = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
SCHARR_KY = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)


def _conv3x3(img: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """向量化的 3x3 卷积(反射填充), 返回 (gx, gy), 用于任意 3x3 梯度核。"""
    h, w = img.shape
    p = np.pad(img, 1, mode="reflect")

    def _apply(k: np.ndarray) -> np.ndarray:
        return (
            k[0, 0] * p[0:-2, 0:-2] + k[0, 1] * p[0:-2, 1:-1] + k[0, 2] * p[0:-2, 2:]
            + k[1, 0] * p[1:-1, 0:-2] + k[1, 1] * p[1:-1, 1:-1] + k[1, 2] * p[1:-1, 2:]
            + k[2, 0] * p[2:, 0:-2] + k[2, 1] * p[2:, 1:-1] + k[2, 2] * p[2:, 2:]
        )

    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[...] = _apply(kx)
    gy[...] = _apply(ky)
    return gx, gy


def _conv1d_axis(img64: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """沿指定轴做严格居中的一维卷积(反射填充), 输入输出均为 float64。"""
    r = len(kernel) // 2
    h, w = img64.shape
    if axis == 0:  # 沿行(竖直方向)
        p = np.pad(img64, ((r, r), (0, 0)), mode="reflect")
        out = np.zeros_like(img64)
        for m in range(len(kernel)):
            out += kernel[m] * p[m : m + h, :]
    elif axis == 1:  # 沿列(水平方向)
        p = np.pad(img64, ((0, 0), (r, r)), mode="reflect")
        out = np.zeros_like(img64)
        for m in range(len(kernel)):
            out += kernel[m] * p[:, m : m + w]
    else:
        raise ValueError("axis 只能是 0(竖直) 或 1(水平)")
    return out


def gaussian_gradients(img: np.ndarray, sigma: float = 1.4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """高斯导数(DoG)梯度: 用高斯核在另一方向平滑、在其导数方向求导。

    相比 3x3 Scharr/Sobel, DoG 是连续可微的高斯核导数, 旋转对称性更好、
    抗噪能力更强、定位精度更高(标准的亚像素梯度估计), 因此边缘更贴近真实轮廓。
    """
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    g /= g.sum()
    dg = -x / (sigma * sigma) * g  # 高斯一阶导数的解析式

    img64 = img.astype(np.float64)
    # Gx = 先竖直平滑 g, 再水平求导 dg
    gx = _conv1d_axis(_conv1d_axis(img64, g, 0), dg, 1)
    # Gy = 先竖直求导 dg, 再水平平滑 g
    gy = _conv1d_axis(_conv1d_axis(img64, dg, 0), g, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
    return gx.astype(np.float32), gy.astype(np.float32), mag


def gradients(img: np.ndarray, operator: str = "scharr") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算梯度: 返回 (gx, gy, mag)。operator 可选 'scharr'(默认, 更准) 或 'sobel'。"""
    if operator == "scharr":
        kx, ky = SCHARR_KX, SCHARR_KY
    elif operator == "sobel":
        kx, ky = SOBEL_KX, SOBEL_KY
    else:
        raise ValueError(f"未知算子: {operator!r}, 可选 'scharr' 或 'sobel'")
    gx, gy = _conv3x3(img, kx, ky)
    mag = np.sqrt(gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2).astype(np.float32)
    return gx, gy, mag


def _tensor_gradients(
    img: np.ndarray, sigma: float, operator: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DiZenzo 结构张量彩色梯度: 逐通道求 ∇I_c 后合成结构张量
        J = Σ_c ∇I_c ∇I_cᵀ = [[Jxx, Jxy], [Jxy, Jyy]]
    梯度幅值 = sqrt(λ_max), 方向 = λ_max 对应特征向量方向。

    相比“逐通道取最大幅值”的旧做法, 结构张量把三个通道的能量相干叠加:
      - 纯色差边缘(如红/青相邻, 亮度相同)三个通道梯度同时指向分界线,
        λ_max 显著大于任一单通道, 检测更稳;
      - 各通道梯度方向不一致时(反彩色边缘) λ_max≈λ_min, 幅值自动衰减,
        不会像逐通道取最大那样给出方向矛盾的强响应;
      - 梯度方向是所有通道的“共识方向”, NMS 沿真实边缘法向抑制, 定位更准。
    """
    jxx = np.zeros(img.shape[:2], dtype=np.float64)
    jyy = np.zeros_like(jxx)
    jxy = np.zeros_like(jxx)
    for c in range(img.shape[2]):
        ch = img[..., c]
        if operator == "dog":
            gx, gy, _ = gaussian_gradients(ch, sigma=sigma)
        else:
            gx, gy, _ = gradients(gaussian_blur(ch, sigma=sigma), operator=operator)
        gxd = gx.astype(np.float64)
        gyd = gy.astype(np.float64)
        jxx += gxd * gxd
        jyy += gyd * gyd
        jxy += gxd * gyd

    # 2x2 对称矩阵的特征分解解析式
    tr = jxx + jyy
    gap = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    lam_max = 0.5 * (tr + gap)                       # 最大特征值
    mag = np.sqrt(np.maximum(lam_max, 0.0)).astype(np.float32)
    # λ_max 特征向量方向: θ = 0.5·atan2(2·Jxy, Jxx−Jyy)
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    gx = (mag * np.cos(theta)).astype(np.float32)
    gy = (mag * np.sin(theta)).astype(np.float32)
    return gx, gy, mag


def _channel_gradients(
    img: np.ndarray, sigma: float, operator: str, color_mode: str = "tensor"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """梯度计算(支持彩色图)。

    color_mode:
      'tensor' (默认) -- DiZenzo 结构张量合成, 通道能量相干叠加, 方向一致(见 _tensor_gradients);
      'max'          -- 旧做法: 逐通道求梯度, 逐像素取幅值最大的通道, 保留作对比。

    灰度化会丢失“颜色不同但亮度相同”的色差边缘(例如纯红与纯绿相邻), 彩色模式
    对每个 RGB 通道单独求梯度后合成, 能同时保留亮度边缘与色差边缘。
    """
    if img.ndim == 2:
        if operator == "dog":
            return gaussian_gradients(img, sigma=sigma)
        return gradients(gaussian_blur(img, sigma=sigma), operator=operator)

    if color_mode == "tensor":
        return _tensor_gradients(img, sigma=sigma, operator=operator)

    best_gx = best_gy = best_mag = None
    for c in range(img.shape[2]):
        ch = img[..., c]
        if operator == "dog":
            gx, gy, mag = gaussian_gradients(ch, sigma=sigma)
        else:
            gx, gy, mag = gradients(gaussian_blur(ch, sigma=sigma), operator=operator)
        if best_mag is None:
            best_gx, best_gy, best_mag = gx.copy(), gy.copy(), mag
        else:
            take = mag > best_mag
            best_gx[take] = gx[take]
            best_gy[take] = gy[take]
            np.maximum(best_mag, mag, out=best_mag)
    return best_gx, best_gy, best_mag


def _bilinear_interp(a: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """在连续坐标 (r, c) 上对数组 a 做双线性插值(向量化)。"""
    h, w = a.shape
    r = np.clip(r, 0, h - 1.0)
    c = np.clip(c, 0, w - 1.0)
    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    dr = (r - r0).astype(np.float32)
    dc = (c - c0).astype(np.float32)
    return (
        a[r0, c0] * (1 - dr) * (1 - dc)
        + a[r0, c1] * (1 - dr) * dc
        + a[r1, c0] * dr * (1 - dc)
        + a[r1, c1] * dr * dc
    )


def non_max_suppression_interp(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """亚像素非极大值抑制: 沿真实梯度方向做双线性插值后再比较, 定位误差远小于量化方向。"""
    h, w = mag.shape
    eps = 1e-12
    m = np.maximum(mag, eps)
    nx = gx / m  # 梯度单位向量的列分量
    ny = gy / m  # 梯度单位向量的行分量
    rr = np.arange(h, dtype=np.float32)[:, None]
    cc = np.arange(w, dtype=np.float32)[None, :]
    m1 = _bilinear_interp(mag, rr + ny, cc + nx)
    m2 = _bilinear_interp(mag, rr - ny, cc - nx)
    out = np.where((mag >= m1) & (mag >= m2), mag, 0.0).astype(np.float32)
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = 0.0
    return out


def non_max_suppression_quantized(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """(旧版) 把梯度方向量化到 4 个角度的非极大值抑制, 仅供对比基准使用。"""
    h, w = mag.shape
    angle_deg = np.degrees(np.arctan2(gy, gx)) % 180
    quant = np.zeros((h, w), dtype=np.int8)
    quant[(angle_deg >= 0) & (angle_deg < 22.5)] = 0
    quant[(angle_deg >= 22.5) & (angle_deg < 67.5)] = 1
    quant[(angle_deg >= 67.5) & (angle_deg < 112.5)] = 2
    quant[(angle_deg >= 112.5) & (angle_deg < 157.5)] = 3
    quant[(angle_deg >= 157.5)] = 0

    pad = np.pad(mag, 1, mode="constant", constant_values=-np.inf)
    n1 = {0: pad[1:-1, :-2], 1: pad[:-2, :-2], 2: pad[:-2, 1:-1], 3: pad[:-2, 2:]}
    n2 = {0: pad[1:-1, 2:], 1: pad[2:, 2:], 2: pad[2:, 1:-1], 3: pad[2:, :-2]}
    out = np.zeros_like(mag)
    for k in range(4):
        mask = quant == k
        out[mask] = np.where(
            (mag[mask] >= n1[k][mask]) & (mag[mask] >= n2[k][mask]), mag[mask], 0.0
        )
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = 0.0
    return out


def hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """滞后连接(向量化): strong 为强边缘种子, weak 为候选弱边缘, 返回二值边缘图。

    语义与经典 BFS 完全等价: 在候选图 cand = strong|weak 上做 8-连通域标记,
    保留“包含至少一个强种子”的连通分量。这样既保留清晰(强)边缘, 又能把被噪声
    削弱的同一条边缘续接完整, 而孤立噪声点(不与任何强边缘相连)会被丢弃。
    连通域标记用纯 numpy 指针跳跃实现, 避免逐像素 BFS 的 Python 循环。
    strong/weak 既可来自全局双阈值, 也可来自自适应局部阈值。
    """
    cand = strong | weak
    labels, _ = _connected_components(cand)
    strong_labels = np.unique(labels[strong])
    if strong_labels.size == 0:
        return np.zeros_like(strong)
    return np.isin(labels, strong_labels)


def _link_gaps(edge: np.ndarray, gap: int = 1) -> np.ndarray:
    """填补边缘上的小断口: 背景像素若在 8 邻域的某对“相对方向”上都有边缘像素,
    则认为它是 1 像素宽的断口, 补为边缘。让物体轮廓更连续, 更贴近实际形状。"""
    out = edge.copy()
    h, w = edge.shape
    p = np.pad(edge, 1, mode="constant", constant_values=False)
    opposite = [((0, -1), (0, 1)), ((-1, 0), (1, 0)), ((-1, -1), (1, 1)), ((-1, 1), (1, -1))]
    for _ in range(gap):
        fill = np.zeros_like(edge)
        for (dy1, dx1), (dy2, dx2) in opposite:
            n1 = p[1 + dy1 : 1 + dy1 + h, 1 + dx1 : 1 + dx1 + w]
            n2 = p[1 + dy2 : 1 + dy2 + h, 1 + dx2 : 1 + dx2 + w]
            fill |= (~out) & n1 & n2
        out |= fill
        p = np.pad(out, 1, mode="constant", constant_values=False)
    return out


def _adaptive_thresholds(
    nms: np.ndarray,
    mag: np.ndarray,
    local_sigma: float | None,
    strong_rel: float,
    weak_rel: float,
) -> np.ndarray:
    """自适应双阈值 + 滞后连接, 用于亮度/对比度在空间上不均匀的图片。

    核心思路(对比度归一化):
      单张图片里, 亮区域的边缘梯度幅值天然比暗区域大得多。若用同一个全局阈值
      (例如“梯度幅值的前 20% 分位”), 暗区域(如图片较暗的上半部分)的整体梯度低于
      该阈值, 就会因为没有强边缘种子而被整片丢掉 —— 这正是“原图上部分提取后
      未显示”的直接原因。

      这里把每个像素的梯度幅值 nms 除以其所在区域的“局部边缘强度” ref(对 mag
      做大尺度高斯模糊), 得到无量纲显著性 saliency = nms / ref。亮区域 ref 大、
      暗区域 ref 小, 相除后两者的真实边缘都落在相近的 saliency 区间, 于是可以用
      一组与绝对亮度无关的阈值同时检出; 平坦无内容区域 ref≈nms≈0, saliency≈0,
      不会产生伪边缘。
    """
    h, w = nms.shape
    nmax = float(nms.max())
    if nmax <= 0:
        return np.zeros((h, w), dtype=bool)

    noise_floor = nmax * 1e-4  # 绝对噪声底: 低于它的梯度不可能是边缘
    if local_sigma is None:
        local_sigma = max(4.0, min(h, w) / 12.0)

    ref = gaussian_blur(mag, sigma=local_sigma)
    ref = np.maximum(ref, noise_floor)  # 平坦区域 ref≈0, 用噪声底兜底避免除零/放大噪声
    saliency = nms / ref
    strong = (saliency >= strong_rel) & (nms > noise_floor)
    weak = (saliency >= weak_rel) & (nms > noise_floor) & ~strong
    return hysteresis(strong, weak)


def canny(
    img: np.ndarray,
    sigma: float = 1.4,
    low_frac: float = 0.4,
    high_frac: float = 0.8,
    operator: str = "dog",
    interp: bool = True,
    adaptive: bool = True,
    local_sigma: float | None = None,
    strong_rel: float = 2.0,
    weak_rel: float = 1.2,
    link_gaps: bool = True,
    gap_size: int = 1,
    color_mode: str = "tensor",
) -> np.ndarray:
    """完整 Canny 边缘检测, 返回二值边缘图 (True=边缘像素)。

    img       : float32 灰度图 (H,W) 或彩色图 (H,W,3); 彩色图默认用 DiZenzo
                结构张量合成梯度, 能检出纯色差边缘且方向一致(见 _tensor_gradients);
    operator  : 'dog'(默认, 高斯导数, 定位最准)、'scharr' 或 'sobel';
    interp    : True 用亚像素插值 NMS(定位更准), False 用旧版量化方向 NMS;
    adaptive  : True(默认) 用局部自适应双阈值(对比度归一化, 能同时检出明/暗区域的
                边缘), False 用全局分位数双阈值(向后兼容);
    local_sigma: 自适应阈值的局部参照模糊尺度, None 时按 min(h,w)/12 自动选取;
    strong_rel/weak_rel: 自适应模式下的强/弱显著性阈值(无量纲, 与图片亮度无关),
                仅 adaptive=True 时生效;
    link_gaps : True(默认) 填补 1px 断口, 让物体轮廓更连续;
    color_mode: 彩色图的梯度合成方式, 'tensor'(默认, 结构张量) 或 'max'(逐通道取最大)。
    """
    gx, gy, mag = _channel_gradients(img, sigma=sigma, operator=operator, color_mode=color_mode)
    if interp:
        nms = non_max_suppression_interp(mag, gx, gy)
    else:
        nms = non_max_suppression_quantized(mag, gx, gy)

    nmax = float(nms.max())
    if nmax <= 0:
        return np.zeros(img.shape[:2], dtype=bool)

    if adaptive:
        edge = _adaptive_thresholds(nms, mag, local_sigma, strong_rel, weak_rel)
    else:
        # 全局双阈值(向后兼容): 先剔除浮点噪声底, 再取分位数, 自适应不同图片
        noise_floor = nmax * 1e-4
        meaningful = nms[nms > noise_floor]
        if meaningful.size == 0:
            return np.zeros(img.shape[:2], dtype=bool)
        high = float(np.percentile(meaningful, high_frac * 100))
        low = high * low_frac
        strong = nms >= high
        weak = (nms >= low) & ~strong
        edge = hysteresis(strong, weak)

    if link_gaps and gap_size >= 1:
        edge = _link_gaps(edge, gap=gap_size)
    return edge


# ----------------------------------------------------------------------------
# 第 2 步: 边缘划分 (结点拆分 + 连通域标记)
# ----------------------------------------------------------------------------

@dataclass
class Segmentation:
    """边缘划分结果。

    labels       : 与图片同尺寸的 int 数组, 边缘像素取 [1..N] 段号, 非边缘为 0,
                   结点像素(交叉点)为 -1;
    segments     : 每段像素坐标列表, segments[k] 对应段号 k+1;
    junction     : bool 数组, True 表示交叉点像素;
    adjacency    : 邻接表, adjacency[i] 是与段 i+1 相邻的段号集合(1-based);
    skeleton     : 1px 骨架(细化后的边缘掩码), 划分在骨架上进行。
    """

    labels: np.ndarray
    segments: list[np.ndarray]
    junction: np.ndarray
    adjacency: list[set[int]] = field(default_factory=list)
    skeleton: np.ndarray = None

    @property
    def n_segments(self) -> int:
        # segments[0] 是占位元素, 真正段数 = len - 1
        return len(self.segments) - 1


_THIN_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def _thin_step(b: np.ndarray, step: int) -> tuple[np.ndarray, bool]:
    """Zhang-Suen 细化的单步(向量化): 返回 (新掩码, 是否有像素被删除)。

    删除条件(标准 Zhang-Suen): 8 邻域前景数 B 在 [2,6]; 环形序列 0->1 跳变数 A=1;
    step1 额外要求非 (p2&p4&p6 或 p4&p6&p8), step2 非 (p2&p4&p8 或 p2&p6&p8)。
    """
    P = np.pad(b, 1, mode="constant", constant_values=False)
    ring = [P[1 + dy : 1 + dy + b.shape[0], 1 + dx : 1 + dx + b.shape[1]]
            for dy, dx in _THIN_RING]
    B = np.zeros(b.shape, dtype=np.uint8)
    for p in ring:
        B += p
    A = np.zeros(b.shape, dtype=np.uint8)
    for i in range(8):
        A += ((~ring[i]) & ring[(i + 1) % 8]).astype(np.uint8)
    p2, p4, p6, p8 = ring[0], ring[2], ring[4], ring[6]
    mark = b & (B >= 2) & (B <= 6) & (A == 1)
    if step == 1:
        mark = mark & ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
    else:
        mark = mark & ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
    return b & ~mark, bool(mark.any())


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 骨架细化: 把 2~3px 宽的边缘带细化成 1px 骨架。

    Canny 输出在拐角/断口填补处常有 2px 宽的“带状”区域, 直接在其上检测交叉点
    会产生虚假拓扑(邻域度数高达 8), 折线有序化也无法得到干净的链。细化成
    1px 骨架后, 每条边缘段是真正的 8-连通链, 结点检测与有序化都更准确。
    """
    b = mask.astype(bool).copy()
    while True:
        b, c1 = _thin_step(b, 1)
        b, c2 = _thin_step(b, 2)
        if not (c1 or c2):
            return b


def edge_degree(edge: np.ndarray) -> np.ndarray:
    """计算每个边缘像素在 8-邻域中的边缘邻居数量(度)。"""
    h, w = edge.shape
    deg = np.zeros((h, w), dtype=np.int8)
    padded = np.pad(edge, 1, mode="constant", constant_values=False)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            deg += padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w].astype(np.int8)
    deg[~edge] = 0
    return deg


_BRANCH_RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _make_branch_lut() -> np.ndarray:
    """预计算 3x3 邻域 256 种位型的“臂数”(环形位置在 8-连通意义下的分量数)。"""
    lut = np.zeros(256, dtype=np.int8)
    for code in range(256):
        nbrs = {_BRANCH_RING[i] for i in range(8) if code & (1 << i)}
        arms = 0
        unvisited = set(nbrs)
        while unvisited:
            arms += 1
            stack = [unvisited.pop()]
            while stack:
                oy, ox = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        pos = (oy + dy, ox + dx)
                        if pos in unvisited:
                            unvisited.discard(pos)
                            stack.append(pos)
        lut[code] = arms
    return lut


_BRANCH_LUT = _make_branch_lut()


def branch_points(edge: np.ndarray) -> np.ndarray:
    """检测真正的分叉点(结点), 向量化查表版。

    把每个边缘像素 3x3 邻域的 8 个环形位置编码成 0-255 位型, 再用预计算查表
    得到该位型的“臂数”(8-连通分量数)。臂数 >= 3 说明有两条以上边缘交汇
    (如 T 形、X 形), 判定为分叉点; 普通拐角/弯曲处只有 1 条臂, 不会被误判。
    与原逐像素 BFS 版本结果一致, 但所有像素一次并行完成; 边界像素按
    “缺失邻居视为背景”处理, 顺带修掉旧版负下标回绕的越界 bug。
    """
    h, w = edge.shape
    p = np.pad(edge, 1, mode="constant", constant_values=False)
    code = np.zeros((h, w), dtype=np.int16)
    for i, (dy, dx) in enumerate(_BRANCH_RING):
        code |= p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w].astype(np.int16) << i
    return (_BRANCH_LUT[code] >= 3) & edge


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """对二值 mask 做 8-连通域标记(边表 + 并查集), 返回 (label_map, 分量数)。

    只在前景像素上建图: 用 4 个方向偏移找出所有 8-邻接的前景像素对(边表),
    再对边表做并查集(路径压缩)。边数通常远小于像素数(稀疏边缘图), 且并查集
    的复杂度近似 O(边数·α), 没有标签传播式算法在复杂连通结构上线性退化的
    风险, 替代逐像素 BFS 的 Python 循环。
    """
    h, w = mask.shape
    flat_mask = mask.ravel()
    if not flat_mask.any():
        return np.zeros((h, w), dtype=np.int32), 0

    fg = np.flatnonzero(flat_mask)
    n = fg.size
    y = fg // w
    x = fg - y * w

    # 建 8-连通无向边表: 每个像素只向右/下/右下/左下四个方向找邻居, 每条边出现一次
    src_parts, dst_parts = [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if dy == 0:
            cond = x < w - 1
        elif dx == 0:
            cond = y < h - 1
        elif dx == 1:
            cond = (y < h - 1) & (x < w - 1)
        else:
            cond = (y < h - 1) & (x >= 1)
        src = np.flatnonzero(cond)
        nb = fg[cond] + (dx + dy * w)  # 候选邻居的线性下标
        hit = flat_mask[nb]            # 邻居也是前景才算边
        src = src[hit]
        if src.size:
            src_parts.append(src)
            dst_parts.append(np.searchsorted(fg, nb[hit]))

    if not src_parts:  # 全是孤立像素
        out = np.zeros(h * w, dtype=np.int32)
        out[fg] = np.arange(1, n + 1, dtype=np.int32)
        return out.reshape(h, w), n

    a = np.concatenate(src_parts)
    b = np.concatenate(dst_parts)
    par = list(range(n))
    # 并查集(路径压缩); 边数通常远小于像素数, 因此比逐像素 BFS 快
    for u, v in zip(a.tolist(), b.tolist()):
        while par[u] != u:
            par[u] = par[par[u]]
            u = par[u]
        while par[v] != v:
            par[v] = par[par[v]]
            v = par[v]
        if u != v:
            par[u] = v
    # 固定森林上做指针跳跃, 让所有节点直接指向根(向量化, 对数轮收敛)
    par_arr = np.asarray(par, dtype=np.int64)
    while True:
        nxt = par_arr[par_arr]
        if np.array_equal(nxt, par_arr):
            break
        par_arr = nxt

    reps, inverse = np.unique(par_arr, return_inverse=True)
    out = np.zeros(h * w, dtype=np.int32)
    out[fg] = inverse + 1
    return out.reshape(h, w), int(reps.size)


def segment_edges(edge: np.ndarray, do_skeleton: bool = True) -> Segmentation:
    """把二值边缘图划分为若干条边缘段。

    思路:
      0. (可选, 默认开) Zhang-Suen 骨架细化: 2~3px 宽的边缘带 -> 1px 骨架,
         消除“带状”区域造成的虚假拓扑, 让每段成为干净的 8-连通链;
      1. 用 branch_points 检测真正的分叉点(两条以上边缘交汇处), 先把它们剔除;
      2. 对剩余像素做 8-连通域标记, 每个连通域即一条“边缘段”
         (个别段因骨架局部 2px 宽仍含小分支, 由 order_segments 的
         “主路径+子链”策略处理, 不在此粗暴拆碎);
      3. 交叉点像素按就近原则归属到相邻段(用于染色显示),
         同时用于建立段与段之间的邻接关系。
    """
    h, w = edge.shape
    skel = skeletonize(edge) if do_skeleton else edge
    junction = branch_points(skel)
    body = skel & ~junction

    raw_labels, _ = _connected_components(body)
    # 向量化剔除极短噪声段(<2 像素)并重新连续编号
    flat = raw_labels.ravel()
    ids = flat[flat > 0]
    labels_flat = np.zeros(h * w, dtype=np.int32)
    n_seg = 0
    if ids.size:
        counts = np.bincount(ids)
        keep = np.flatnonzero(counts >= 2)
        n_seg = int(keep.size)
        new_id = np.zeros(counts.size, dtype=np.int32)
        new_id[keep] = np.arange(1, n_seg + 1, dtype=np.int32)
        labels_flat = new_id[flat]
    labels = labels_flat.reshape(h, w)
    labels[junction] = -1

    # 按段号一次性分组收集像素坐标(argsort 分组, 避免逐段全图扫描)
    segments: list[np.ndarray] = [None]  # 1-based, 占位
    if n_seg > 0:
        pos = np.flatnonzero(labels_flat > 0)
        vals = labels_flat[pos]
        order = np.argsort(vals, kind="stable")
        pos, vals = pos[order], vals[order]
        bounds = np.flatnonzero(vals[1:] != vals[:-1]) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [pos.size]])
        for s, e in zip(starts, ends):
            ys, xs = np.divmod(pos[s:e], w)
            segments.append(np.stack([ys, xs], axis=1))

    # 建立邻接关系: (a) 共享交叉点; (b) 像素 8-邻接
    adjacency: list[set[int]] = [set() for _ in range(n_seg + 1)]

    def add_pair(a: int, b: int) -> None:
        if a >= 1 and b >= 1 and a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)

    # (a) 交叉点像素: 其 8 邻域出现哪些段, 这些段两两相邻
    for y, x in np.argwhere(junction):
        nbrs = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    lab = labels[ny, nx]
                    if lab >= 1:
                        nbrs.add(int(lab))
        nbrs = sorted(nbrs)
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                add_pair(nbrs[i], nbrs[j])

    # (b) 逐像素检查右下/左下 4 个方向, 捕获无交叉点但彼此贴邻的段
    padl = np.pad(labels, 1, constant_values=-2)
    pair_lists = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        nbr = padl[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        valid = (labels >= 1) & (nbr >= 1) & (labels != nbr)
        if valid.any():
            a = labels[valid]
            b = nbr[valid]
            pair_lists.append(np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1))
    if pair_lists:
        all_pairs = np.unique(np.concatenate(pair_lists, axis=0), axis=0)
        for a, b in all_pairs:
            add_pair(int(a), int(b))

    # 交叉点像素按就近原则归属: 交给与其 8 邻域中出现次数最多的段
    for y, x in np.argwhere(junction):
        counts: dict[int, int] = {}
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    lab = int(labels[ny, nx])
                    if lab >= 1:
                        counts[lab] = counts.get(lab, 0) + 1
        if counts:
            best = max(counts, key=lambda k: counts[k])
            labels[y, x] = best

    return Segmentation(labels=labels, segments=segments, junction=junction,
                        adjacency=adjacency, skeleton=skel)


# ----------------------------------------------------------------------------
# 第 2.5 步: 边缘段折线有序化 (无序像素集 -> 有序折线)
# ----------------------------------------------------------------------------

def _order_path(points: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """把一条边缘段的像素集合排成有序折线: 返回 (主路径, 子链列表)。

    思路: 段内像素是 8-连通图, 绝大多数是 1px 链, 但骨架局部 2px 宽处会有
    小分支。先建邻接表并统计度数:
      - 存在度 1 端点 -> 从端点起笔, 逐步走向未访问邻居(优先延续来向, 拐角少);
      - 无端点(闭合环) -> 从任意像素起笔绕一圈;
      - 游走结束后剩余的分支像素不再“就近乱接”(会产生跨图跳跃), 而是作为
        独立子链返回, 长度统计不丢、渲染不打结。
    """
    n = len(points)
    if n <= 2:
        return points, []

    index_of = {(int(y), int(x)): i for i, (y, x) in enumerate(points)}
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for i, (y, x) in enumerate(points):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = index_of.get((int(y) + dy, int(x) + dx))
                if j is not None:
                    nbrs[i].add(j)

    deg = np.array([len(s) for s in nbrs])
    endpoints = [i for i in range(n) if deg[i] == 1]
    start = endpoints[0] if endpoints else int(np.argmax(deg))

    visited = np.zeros(n, dtype=bool)
    order: list[int] = [start]
    visited[start] = True
    prev, cur = -1, start
    while True:
        cand = [j for j in nbrs[cur] if not visited[j]]
        if not cand:
            break
        if prev >= 0 and len(cand) > 1:
            # 优先延续来向(直行), 避免在 2px 宽处来回锯齿
            py, px = points[prev]
            cy, cx = points[cur]
            dy, dx = cy - py, cx - px
            cand.sort(key=lambda j: -abs((points[j][0] - cy) * dy
                                         + (points[j][1] - cx) * dx))
        nxt = cand[0]
        order.append(nxt)
        visited[nxt] = True
        prev, cur = cur, nxt

    main = points[np.asarray(order, dtype=np.int64)]

    # 剩余分支像素: 独立成链(不与主路径跨图拼接)
    subs: list[np.ndarray] = []
    remaining = {int(i) for i in np.flatnonzero(~visited)}
    while remaining:
        starts = [i for i in remaining if len(nbrs[i] & remaining) <= 1]
        s = starts[0] if starts else min(remaining)
        chain = [s]
        remaining.discard(s)
        cur = s
        while True:
            cand = [j for j in nbrs[cur] if j in remaining]
            if not cand:
                break
            chain.append(cand[0])
            remaining.discard(cand[0])
            cur = cand[0]
        subs.append(points[np.asarray(chain, dtype=np.int64)])
    return main, subs


def order_segments(seg: Segmentation) -> list[dict]:
    """把所有边缘段排成有序折线, 返回每段的几何信息。

    每段: {seg_id, points(有序主路径), length(邻接图精确总长, 每条边只计一次),
           path_length(主路径长), n_branch_pixels(不在主路径上的像素),
           closed(主路径首尾 8-邻接 => 闭合轮廓), n_pixels}
    """
    out: list[dict] = []
    for k, pts in enumerate(seg.segments[1:], start=1):
        main, subs = _order_path(pts)

        # 精确长度: 邻接图上每条边只计一次(主路径+子链覆盖全部边)
        def chain_len(arr: np.ndarray) -> float:
            if len(arr) < 2:
                return 0.0
            st = np.diff(arr.astype(np.float64), axis=0)
            return float(np.sqrt((st ** 2).sum(axis=1)).sum())

        total = chain_len(main) + sum(chain_len(c) for c in subs)
        closed = bool(len(main) >= 6 and (abs(main[0] - main[-1]).max() <= 1))
        out.append(dict(seg_id=k, points=main, length=round(total, 2),
                        path_length=round(chain_len(main), 2),
                        n_branch_pixels=int(len(pts) - len(main)),
                        closed=closed, n_pixels=int(len(pts))))
    return out


def render_polylines(shape: tuple[int, int], ordered: list[dict],
                     palette: list[tuple[int, int, int]],
                     junction: np.ndarray) -> np.ndarray:
    """折线渲染: 每段按序连成彩色折线, 起点绿点/终点红点/闭合段不打点。"""
    h, w = shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)
    for rec in ordered:
        c = palette[(rec["seg_id"] - 1) % len(palette)]
        pts = [(int(x), int(y)) for y, x in rec["points"]]
        if len(pts) >= 2:
            d.line(pts, fill=c, width=1)
        if not rec["closed"] and len(pts) >= 2:
            d.ellipse((pts[0][0] - 2, pts[0][1] - 2, pts[0][0] + 2, pts[0][1] + 2),
                      fill=(40, 160, 60))
            d.ellipse((pts[-1][0] - 2, pts[-1][1] - 2, pts[-1][0] + 2, pts[-1][1] + 2),
                      fill=(220, 50, 50))
    arr = np.asarray(img).copy()
    arr[junction] = (90, 90, 90)   # 交叉点统一灰色
    return arr


# ----------------------------------------------------------------------------
# 第 2.6 步: 边缘 -> 物体 (闭合轮廓围成的区域) + 区域级四色染色
# ----------------------------------------------------------------------------

def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """3x3 八邻域膨胀(纯 numpy 移位实现)。"""
    out = mask.copy()
    h, w = mask.shape
    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        nxt = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nxt |= p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        out = nxt
    return out


def _binary_closing(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """闭运算(先膨胀后腐蚀): 缝合边缘上的小断口, 让轮廓真正闭合。"""
    h, w = mask.shape
    out = _binary_dilate(mask, iterations)
    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant", constant_values=True)
        nxt = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nxt &= p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        out = nxt
    return out


def extract_objects(edge: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """由闭合边缘恢复物体区域 -- “边缘与物体的关系”落地。

    步骤:
      1. 闭运算缝合边缘上的 1~2px 小断口;
      2. 自由空间(非边缘)做连通域标记, 与图像边界连通的分量是外部背景;
      3. 其余分量即被闭合轮廓包围的“面” -- 物体(或物体内部的孔洞,
         如圆环会得到内外两个面, 恰对应平面图的两个 face)。
    返回 (物体区域标签图, 物体区域数, 外部背景掩码, 缝合后的边缘掩码)。
    """
    sealed = _binary_closing(edge, iterations=2)
    free = ~sealed
    lab, _ = _connected_components(free)
    border = set()
    border.update(int(v) for v in np.unique(np.concatenate(
        [lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])) if v > 0)
    bg_mask = np.isin(lab, list(border)) if border else np.zeros_like(free)
    obj = free & ~bg_mask
    obj_lab, n_obj = _connected_components(obj)
    return obj_lab, n_obj, bg_mask, sealed


def region_adjacency(regions: np.ndarray, n_regions: int,
                     dilate_iters: int = 2) -> list[set[int]]:
    """区域邻接表: 各区域膨胀后与其他区域相交 => 中间只隔着边缘 => 相邻。

    regions 取 1..n_regions(含作为“外面”的背景区域), 得到完整的平面图划分,
    也就是四色定理意义上的“地图”。
    """
    ids = list(range(1, n_regions + 1))
    masks = {i: (regions == i) for i in ids}
    dil = {i: _binary_dilate(masks[i], dilate_iters) for i in ids}
    adj: list[set[int]] = [set() for _ in range(n_regions + 1)]
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            ia, ib = ids[a], ids[b]
            if (dil[ia] & masks[ib]).any() or (dil[ib] & masks[ia]).any():
                adj[ia].add(ib)
                adj[ib].add(ia)
    return adj


def render_regions(shape: tuple[int, int], obj_lab: np.ndarray, n_obj: int,
                   colors: dict[int, int], bg_mask: np.ndarray, edge: np.ndarray,
                   ) -> np.ndarray:
    """区域四色渲染: 物体区域按染色结果填色, 外部背景用淡化色, 边缘深色勾线。"""
    h, w = shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = (250, 250, 250)
    pal = four_color_palette()
    for rid in range(1, n_obj + 1):
        out[obj_lab == rid] = pal[colors.get(rid, 0)]
    bg_c = np.array(pal[colors.get(n_obj + 1, 0)], dtype=np.float64)
    out[bg_mask] = (bg_c * 0.35 + 255 * 0.65).astype(np.uint8)  # 背景淡化
    out[edge] = (40, 40, 40)
    return out


# ----------------------------------------------------------------------------
# 第 3 步: 染色展示
# ----------------------------------------------------------------------------

def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """HSV -> RGB (h in [0,1)), 返回 0-255 整数三元组。"""
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i %= 6
    r, g, b = {
        0: (v, t, p),
        1: (q, v, p),
        2: (p, v, t),
        3: (p, q, v),
        4: (t, p, v),
        5: (v, p, q),
    }[i]
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def distinct_palette(n: int, s: float = 0.85, v: float = 0.95) -> list[tuple[int, int, int]]:
    """用黄金角在色相环上均匀采样, 生成 n 种互不相同的醒目颜色。"""
    golden = 0.618033988749895
    return [hsv_to_rgb((i * golden) % 1.0, s, v) for i in range(n)]


def render_colored(labels: np.ndarray, palette: list[tuple[int, int, int]]) -> np.ndarray:
    """按段号染色, 非边缘像素保持黑色。"""
    lut = np.zeros((len(palette) + 1, 3), dtype=np.uint8)
    for i, c in enumerate(palette, start=1):
        lut[i] = c
    clipped = np.clip(labels, 0, len(palette))
    return lut[clipped]


# ----------------------------------------------------------------------------
# 第 4 步: 四色定理染色 (DSATUR 贪心)
# ----------------------------------------------------------------------------

def dsatur_coloring(adjacency: list[set[int]], max_colors: int = 4) -> dict[int, int]:
    """对邻接表表示的图做 DSATUR 贪心染色, 返回 {顶点: 颜色号(0..max_colors-1)}。

    优先选择“已用颜色数最多、未染色邻接最多”的顶点, 用最小可用颜色染色;
    若 max_colors 种不够, 会继续使用更多颜色(返回的色号可 > max_colors-1),
    调用方据此判断四色是否足够。
    """
    n = len(adjacency) - 1  # 顶点数 (邻接表 1-based)
    color: dict[int, int] = {}
    uncolored = set(range(1, n + 1))

    while uncolored:
        best = None
        best_key = None
        for v in uncolored:
            used = {color[u] for u in adjacency[v] if u in color}
            sat = len(used)                     # 饱和度: 邻居已用颜色数
            deg = len(adjacency[v])             # 度数
            key = (sat, deg)
            if best_key is None or key > best_key:
                best_key = key
                best = v
        used = {color[u] for u in adjacency[best] if u in color}
        c = 0
        while c in used:
            c += 1
        color[best] = c
        uncolored.remove(best)
    return color


def four_color_palette() -> list[tuple[int, int, int]]:
    """四色定理常用的 4 种高对比颜色(红/绿/蓝/黄)。"""
    return [
        (235, 60, 60),    # 红
        (60, 180, 75),    # 绿
        (70, 120, 235),   # 蓝
        (245, 200, 40),   # 黄
    ]


# ----------------------------------------------------------------------------
# 第 5 步: 汇总输出
# ----------------------------------------------------------------------------

def save_image(arr: np.ndarray, path: str) -> None:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def run_pipeline(
    input_path: str,
    output_dir: str,
    sigma: float = 1.4,
    low_frac: float = 0.4,
    high_frac: float = 0.8,
    operator: str = "dog",
    interp: bool = True,
    adaptive: bool = True,
    local_sigma: float | None = None,
    strong_rel: float = 2.0,
    weak_rel: float = 1.2,
    link_gaps: bool = True,
    gap_size: int = 1,
    do_four_color: bool = True,
    use_color: bool = True,
    color_mode: str = "tensor",
) -> dict:
    """执行完整管线, 把结果保存到 output_dir, 返回统计信息。"""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    if use_color:
        rgb = load_rgb(input_path)
        gray = np.clip(rgb @ RGB_TO_GRAY, 0.0, 255.0).astype(np.float32)
        img = rgb
    else:
        gray = load_grayscale(input_path)
        img = gray
    edge = canny(
        img,
        sigma=sigma,
        low_frac=low_frac,
        high_frac=high_frac,
        operator=operator,
        interp=interp,
        adaptive=adaptive,
        local_sigma=local_sigma,
        strong_rel=strong_rel,
        weak_rel=weak_rel,
        link_gaps=link_gaps,
        gap_size=gap_size,
        color_mode=color_mode,
    )
    seg = segment_edges(edge)
    ordered = order_segments(seg)

    # 逐段染色: 不同段用不同颜色
    palette = distinct_palette(seg.n_segments)
    colored = render_colored(seg.labels, palette)

    lengths = [r["length"] for r in ordered] or [0.0]
    stats: dict = {
        "input": input_path,
        "image_size": [gray.shape[1], gray.shape[0]],
        "color_mode": bool(use_color),
        "gradient_mode": (color_mode if use_color else "gray"),
        "edge_pixels": int(edge.sum()),
        "skeleton_pixels": int(seg.skeleton.sum()),
        "junction_pixels": int(seg.junction.sum()),
        "n_segments": seg.n_segments,
        "n_adjacent_pairs": sum(len(s) for s in seg.adjacency) // 2,
        "n_closed_segments": sum(1 for r in ordered if r["closed"]),
        "mean_segment_length": round(float(np.mean(lengths)), 2),
        "max_segment_length": round(float(np.max(lengths)), 2),
        "total_segment_length": round(float(np.sum(lengths)), 2),
    }

    # 边缘图(白边黑底)
    edge_img = np.zeros_like(gray)
    edge_img[edge] = 255
    save_image(edge_img, os.path.join(output_dir, f"{base}_1_边缘图.png"))

    # 结点示意图(可选调试): 结点用红色标出
    junction_img = np.repeat(edge_img[:, :, None], 3, axis=2)
    junction_img[seg.junction] = (255, 60, 60)
    save_image(junction_img, os.path.join(output_dir, f"{base}_2_结点标注.png"))

    # 逐段染色图
    save_image(colored, os.path.join(output_dir, f"{base}_3_逐段染色.png"))

    # 四色定理染色图
    if do_four_color and seg.n_segments > 0:
        coloring = dsatur_coloring(seg.adjacency, max_colors=4)
        used = len(set(coloring.values()))
        stats["four_color_used"] = used
        stats["four_color_enough"] = used <= 4
        palette4 = four_color_palette()
        # 根据染色结果重建 label 图
        remap = np.zeros(seg.n_segments + 1, dtype=np.int32)
        for v, c in coloring.items():
            remap[v] = c + 1  # 段号 -> 颜色号
        labels4 = remap[np.clip(seg.labels, 0, seg.n_segments)]
        lut4 = np.zeros((5, 3), dtype=np.uint8)
        for i, c in enumerate(palette4, start=1):
            lut4[i] = c
        colored4 = lut4[np.clip(labels4, 0, 4)]
        save_image(colored4, os.path.join(output_dir, f"{base}_4_四色染色.png"))

        # 校验: 相邻段颜色必须不同
        violations = 0
        for v in range(1, seg.n_segments + 1):
            for u in seg.adjacency[v]:
                if u > v and coloring[v] == coloring[u]:
                    violations += 1
        stats["four_color_violations"] = violations
    else:
        stats["four_color_enough"] = None

    # 折线有序化图: 彩色折线 + 起点(绿)/终点(红), 闭合段两端不打点
    poly_img = render_polylines(gray.shape, ordered, palette, seg.junction)
    save_image(poly_img, os.path.join(output_dir, f"{base}_5_折线有序化.png"))
    with open(os.path.join(output_dir, f"{base}_折线.json"), "w", encoding="utf-8") as f:
        json.dump([{k: (v.tolist() if k == "points" else v) for k, v in r.items()}
                   for r in ordered], f, ensure_ascii=False)

    # 边缘 -> 物体区域: 闭合轮廓围成的面 + 含“外面”的完整地图四色染色
    obj_lab, n_obj, bg_mask, sealed = extract_objects(seg.skeleton)
    stats["n_objects"] = n_obj
    if n_obj > 0:
        sizes = np.bincount(obj_lab.ravel(), minlength=n_obj + 1)[1:]
        stats["object_sizes_top5"] = sorted(int(s) for s in sizes)[-5:]
        # 完整地图: 物体面 1..n_obj + 外部背景面 n_obj+1
        regions = np.where(bg_mask, n_obj + 1, obj_lab)
        reg_adj = region_adjacency(regions, n_obj + 1)
        stats["n_region_adjacent_pairs"] = sum(len(s) for s in reg_adj) // 2
        reg_colors = dsatur_coloring(reg_adj, max_colors=4)
        used = len(set(reg_colors.values()))
        stats["region_four_color_used"] = used
        stats["region_four_color_enough"] = used <= 4
        reg_img = render_regions(gray.shape, obj_lab, n_obj, reg_colors, bg_mask, sealed)
        save_image(reg_img, os.path.join(output_dir, f"{base}_6_物体区域四色.png"))
    else:
        stats["region_four_color_used"] = None

    with open(os.path.join(output_dir, f"{base}_统计.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="图片物体边缘的提取、划分与染色 (Canny + 结点拆分 + 四色定理)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="?", default=None, help="输入图片的本地路径")
    parser.add_argument("--input", dest="input_path", default=None,
                        help="输入图片的本地路径(与位置参数二选一)")
    parser.add_argument("--output", default="output", help="输出目录")
    parser.add_argument("--sigma", type=float, default=1.4, help="Canny 高斯模糊 sigma")
    parser.add_argument("--low-frac", type=float, default=0.4, help="低阈值 = high*low_frac")
    parser.add_argument("--high-frac", type=float, default=0.8, help="高阈值取梯度值的分位数")
    parser.add_argument("--operator", choices=["dog", "scharr", "sobel"], default="dog",
                        help="梯度算子(默认 dog=高斯导数, 定位最准)")
    parser.add_argument("--no-interp", action="store_true", help="改用旧版量化方向 NMS(默认用亚像素插值)")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="改用全局分位数双阈值(默认用局部自适应阈值)")
    parser.add_argument("--local-sigma", type=float, default=None,
                        help="自适应阈值的局部参照模糊尺度(默认按图片短边/12 自动选取)")
    parser.add_argument("--strong-rel", type=float, default=2.0,
                        help="自适应阈值下强边缘的显著性阈值(仅自适应模式生效)")
    parser.add_argument("--weak-rel", type=float, default=1.2,
                        help="自适应阈值下弱边缘的显著性阈值(仅自适应模式生效)")
    parser.add_argument("--no-link-gaps", action="store_true", help="不填补 1px 断口(默认填补)")
    parser.add_argument("--gap-size", type=int, default=1, help="断口填补半径(默认 1)")
    parser.add_argument("--no-four-color", action="store_true", help="跳过四色定理染色")
    parser.add_argument("--gray", action="store_true",
                        help="改用灰度图做边缘检测(默认彩色: 结构张量合成, 能检出纯色差边缘)")
    parser.add_argument("--color-max", action="store_true",
                        help="彩色梯度退回旧版逐通道取最大(默认 DiZenzo 结构张量)")
    args = parser.parse_args(argv)

    default_input = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "图片物体边缘的提取分割.png")
    input_path = args.input_path or args.input
    if not input_path and os.path.exists(default_input):
        input_path = default_input
    if not input_path:
        parser.error("请提供输入图片的本地路径(位置参数或 --input)")

    if not os.path.exists(input_path):
       print(f"错误: 输入文件不存在: {input_path}", file=sys.stderr)
       return 1

    stats = run_pipeline(
        input_path,
        args.output,
        sigma=args.sigma,
        low_frac=args.low_frac,
        high_frac=args.high_frac,
        operator=args.operator,
        interp=not args.no_interp,
        adaptive=not args.no_adaptive,
        local_sigma=args.local_sigma,
        strong_rel=args.strong_rel,
        weak_rel=args.weak_rel,
        link_gaps=not args.no_link_gaps,
        gap_size=args.gap_size,
        do_four_color=not args.no_four_color,
        use_color=not args.gray,
        color_mode=("max" if args.color_max else "tensor"),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n结果已保存到: {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

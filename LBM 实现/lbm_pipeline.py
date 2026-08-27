# -*- coding: utf-8 -*-
"""
LBM 去噪 (各向同性 + Perona-Malik 各向异性)
============================================

把图像当作密度场 ρ(x, t), 用 D2Q9 BGK 求解扩散方程:

    ∂ρ/∂t = div(c(|∇ρ|) · ∇ρ)

  - **各向同性**:  c = 1,    τ 固定 (等价于热方程 / 高斯平滑)
  - **Perona-Malik I**:  c(g) = exp(-(g/k)²)    (偏好高对比度边缘)
  - **Perona-Malik II**: c(g) = 1 / (1 + (g/k)²) (更平滑, 抗噪)
  - **Tukey biweight**:  c(g) = g²/(k² + g²)    (硬阈值)

LBM 中 τ 与扩散系数 D 的关系:  D = c_s² (τ - 0.5) Δt,  c_s² = 1/3.
设 c ∈ [0, 1] 控制扩散强度, 则:
    τ(x) = 0.5 + (τ_max - 0.5) · c(|∇ρ(x)|)

c → 1 (平滑区) ⇒ τ → τ_max, 大扩散
c → 0 (边缘)   ⇒ τ → 0.5, 小扩散, 实现**边缘保留**

参考:
- P. Perona, J. Malik, "Scale-space and edge detection using anisotropic
  diffusion" (1990), IEEE PAMI 12(7)
- S. Chen, G.D. Doolen, "Lattice Boltzmann Method for Fluid Flows" (1998)
- A. Ginzburg, "Equilibrium-type and link-type lattice Boltzmann models
  for generic advection and anisotropic-dispersion equation" (2005)
"""
from __future__ import annotations

import numpy as np
import cv2


# ===================== 噪声估计 =====================
def estimate_noise(gray: np.ndarray) -> float:
    """用 Laplacian 方差估计加性噪声强度 (Immerkaer 1996).

    干净图 (合成/卡通): σ² < 50  → 跳过 LBM
    轻微噪点 (JPEG):     50-200   → 轻度 LBM
    明显噪声:            200-1000 → 中度 LBM
    重噪声:              > 1000   → 强度 LBM
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    # 截尾均值, 去掉极端值
    p = np.percentile(np.abs(lap), 95.0)
    return float(p)


def _is_sparse_edge_map(gray: np.ndarray) -> bool:
    """判断输入是不是已经处理过的边缘图 (稀疏二值/低密度).

    启发: 0 值像素占比 > 70% 且 255 值像素占比 < 30%,
    且梯度集中在少数位置 → 不需要去噪, 跳过 LBM.
    """
    H, Wd = gray.shape
    n_dark = np.sum(gray < 10)
    n_bright = np.sum(gray > 200)
    dark_ratio = n_dark / (H * Wd)
    bright_ratio = n_bright / (H * Wd)
    return dark_ratio > 0.6 and bright_ratio < 0.4


# ===================== D2Q9 格子 =====================
E = np.array([
    [ 0,  0],   # 0  静止
    [ 1,  0],   # 1  东
    [-1,  0],   # 2  西
    [ 0,  1],   # 3  南
    [ 0, -1],   # 4  北
    [ 1,  1],   # 5  东南
    [-1,  1],   # 6  西南
    [ 1, -1],   # 7  东北
    [-1, -1],   # 8  西北
], dtype=np.int32)

W = np.array([
    4.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
], dtype=np.float32)

C_S2 = 1.0 / 3.0


# ===================== 内部工具 =====================
def _eq_iso(rho: np.ndarray) -> np.ndarray:
    """u=0 时的平衡分布: f_i^eq = w_i * ρ (各向同性扩散专用)."""
    return W[:, None, None] * rho[None, :, :]


def _streaming(f: np.ndarray) -> np.ndarray:
    """D2Q9 流动: f_i(x, t+1) = f_i(x - e_i, t). 用 np.roll 实现周期边界."""
    f_new = np.empty_like(f)
    for i in range(9):
        f_new[i] = np.roll(f[i], shift=(-int(E[i, 1]), -int(E[i, 0])), axis=(-2, -1))
    return f_new


def _diffusion_coeff(g_mag: np.ndarray, k_edge: float, scheme: str) -> np.ndarray:
    """Perona-Malik 扩散系数 c(|∇ρ|) ∈ [0, 1]."""
    r = g_mag / max(k_edge, 1e-12)
    if scheme == "isotropic":
        return np.ones_like(g_mag)
    if scheme == "perona_malik_1":
        return np.exp(-(r * r))                   # 偏好高对比度边缘
    if scheme == "perona_malik_2":
        return 1.0 / (1.0 + r * r)                # 更平滑、抗噪
    if scheme == "tukey":
        return (r * r) / (1.0 + r * r)            # 硬阈值
    raise ValueError(f"unknown scheme: {scheme}")


# ===================== 主入口 =====================
def lbm_denoise(image: np.ndarray,
                n_steps: int = 10,
                tau_max: float = 0.7,
                k_edge: float = 0.12,
                scheme: str = "perona_malik_2",
                auto: bool = True) -> tuple[np.ndarray, dict]:
    """LBM 去噪 (Perona-Malik 各向异性扩散, 边缘保留).

    Args:
        image:   H×W uint8 灰度, 或 H×W×3 uint8 RGB
        n_steps: LBM 迭代步数 (默认 10, 比旧版 30 更保守)
        tau_max: 平滑区最大松弛时间 (τ → 0.5 + (tau_max-0.5)·c),
                 越大扩散越强, 推荐 0.6-0.9
        k_edge:  边缘阈值 (相对 0-1 灰度梯度), 推荐 0.08-0.20
        scheme:  'isotropic' | 'perona_malik_1' | 'perona_malik_2' | 'tukey'
                 (默认 PM-2, 比 PM-1 更抗噪)
        auto:    若 True, 根据噪声强度自适应:
                   - 干净图 (σ<50)            → 跳过, 返回原图
                   - 稀疏边缘图 (黑底白线)    → 跳过, 返回原图
                   - 轻微噪点 (50-200)        → 轻度 (n_steps×0.5)
                   - 中度噪点 (200-1000)      → 中度 (n_steps)
                   - 重度噪点 (>1000)         → 强度 (n_steps×1.5, tau_max+0.2)

    Returns:
        (去噪后的图像, info dict):
            info = {
                "applied": bool,      # True=真去噪了, False=跳过了
                "noise":  float,      # 估计的噪声强度
                "mode":   "skip"|"light"|"medium"|"heavy",
                "n_steps": int,       # 实际用的步数
                "tau_max": float,     # 实际用的 τ
            }
    """
    is_rgb = image.ndim == 3
    if not is_rgb:
        channels = [image]
    else:
        channels = [image[..., c] for c in range(image.shape[2])]

    # === auto 模式: 决定用不用、用多狠 ===
    info = {"applied": True, "noise": 0.0, "mode": "medium",
            "n_steps": n_steps, "tau_max": tau_max}
    if auto:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_rgb else image
        noise = estimate_noise(gray)
        sparse = _is_sparse_edge_map(gray)
        info["noise"] = noise

        if noise < 50 or sparse:
            # 干净图 / 边缘图 — LBM 反而会破坏, 直接返回原图
            info.update({"applied": False, "mode": "skip",
                         "n_steps": 0, "tau_max": 0.0})
            return image.copy(), info
        elif noise < 200:
            info.update({"mode": "light",
                         "n_steps": max(2, int(n_steps * 0.5)),
                         "tau_max": max(0.55, tau_max - 0.1)})
        elif noise < 1000:
            info.update({"mode": "medium",
                         "n_steps": n_steps,
                         "tau_max": tau_max})
        else:
            info.update({"mode": "heavy",
                         "n_steps": int(n_steps * 1.5) + 2,
                         "tau_max": tau_max + 0.2})
        n_steps = info["n_steps"]
        tau_max = info["tau_max"]

    out_channels = []
    for ch in channels:
        if n_steps == 0:
            out_channels.append(ch.copy())
        else:
            denoised = _denoise_channel(ch, n_steps=n_steps,
                                        tau_max=tau_max, k_edge=k_edge,
                                        scheme=scheme)
            out_channels.append(denoised)

    if not is_rgb:
        return out_channels[0], info
    return np.stack(out_channels, axis=-1), info


def _denoise_channel(gray: np.ndarray,
                     n_steps: int,
                     tau_max: float,
                     k_edge: float,
                     scheme: str) -> np.ndarray:
    """单通道 LBM 去噪. gray: H×W uint8 → H×W uint8."""
    H, Wd = gray.shape
    rho = gray.astype(np.float32) / 255.0

    # 初始化 f_i = w_i * ρ (u=0 平衡)
    f = _eq_iso(rho)

    # 稳定性下限 (避免 τ=0.5 的边界不稳定)
    TAU_MIN = 0.51
    c_at_tau_max = 1.0
    c_at_tau_min = 0.0

    for _ in range(int(n_steps)):
        # 1. 局部梯度 |∇ρ|
        gy, gx = np.gradient(rho)
        g_mag = np.sqrt(gx * gx + gy * gy)

        # 2. 扩散系数 c(x) ∈ [0, 1]
        c = _diffusion_coeff(g_mag, k_edge=k_edge, scheme=scheme)

        # 3. τ(x) = TAU_MIN + (tau_max - TAU_MIN) · c
        tau = TAU_MIN + (tau_max - TAU_MIN) * c

        # 4. 碰撞 (BGK, u=0): f ← f - (f - f^eq) / τ
        feq = _eq_iso(rho)
        # 空间变化 τ: 广播 (9, H, W) / (H, W)
        f = f - (f - feq) / tau[None, :, :]

        # 5. 流动
        f = _streaming(f)

        # 6. 更新 ρ = Σ f_i
        rho = f.sum(axis=0)

    # 限幅回 uint8
    return np.clip(rho * 255.0, 0, 255).astype(np.uint8)

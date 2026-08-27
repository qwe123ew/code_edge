# -*- coding: utf-8 -*-
"""
基于 PyTorch + DexiNed 的图像边缘检测
------------------------------------
功能：
    1. 使用 xavysp/DexiNed 仓库的 DexiNed 模型（Dense Extreme Inception Network）；
    2. 加载用户给定的预训练权重 .pth；
    3. 在用户图像上做推理，输出：
        - DexiNed-fused  : 融合层（最后 1 个输出）的 sigmoid 概率图
        - DexiNed-avg    : 6 个侧边输出 + 1 个融合输出的平均
        - 0~255 的灰度边缘图（黑底白边）
        - 阈值 128 的二值图

注意：
    - DexiNed 下采样到 1/16，因此输入图像宽高必须是 16 的倍数；
    - 仓库通过 GitHub 镜像下载到 ./DexiNed-master/；
    - 训练时使用了 BGR 通道顺序 + mean 减均值 [103.939, 116.779, 123.68]，
      这一点必须与训练保持一致，否则输出会出错。
"""

import os
import sys
import argparse
from PIL import Image
import numpy as np
import cv2
import torch


# ---------- 路径配置 ----------
HERE = os.path.dirname(os.path.abspath(__file__))
DEXINED_ROOT = os.path.join(HERE, "DexiNed-master")
# 官方推荐目录结构：checkpoints/BIPED/10/10.pt
CHECKPOINT_PATH = os.path.join(DEXINED_ROOT, "checkpoints", "BIPED", "10", "10.pt")
INPUT_IMAGE_PATH = r"C:\Users\18607\Desktop\边缘识别模型方法\图片物体边缘的提取分割.png"
OUTPUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 把 DexiNed-master 加入 import 路径
sys.path.insert(0, DEXINED_ROOT)
# 同时让 datasets.py / utils 的相对导入能工作
sys.path.insert(0, os.path.join(DEXINED_ROOT, "utils"))


# DexiNed 训练时的 BGR 均值（必须严格一致）
MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


# ---------- 工具函数 ----------
def imread_unicode(path: str) -> np.ndarray:
    """读取含中文路径的图像（cv2.imread 在中文路径下会失败）。"""
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    """写出图像到含中文路径。"""
    cv2.imencode(os.path.splitext(path)[1], img)[1].tofile(path)


def pad_to_multiple(img: np.ndarray, multiple: int = 16) -> tuple[np.ndarray, tuple[int, int]]:
    """把 BGR 图右下角 pad 到 multiple 的整数倍，返回 padded 图与原尺寸。"""
    H, W = img.shape[:2]
    new_W = ((W + multiple - 1) // multiple) * multiple
    new_H = ((H + multiple - 1) // multiple) * multiple
    padded = np.zeros((new_H, new_W, 3), dtype=img.dtype)
    padded[:H, :W, :] = img
    return padded, (H, W)


def preprocess(image_path: str) -> tuple[torch.Tensor, tuple[int, int]]:
    """读取图像、pad 到 16 倍数、BGR 减均值、转成 (1, 3, H, W) 张量。"""
    bgr = imread_unicode(image_path)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    padded, orig_shape = pad_to_multiple(bgr, multiple=16)

    x = padded.astype(np.float32) - MEAN_BGR          # BGR 减均值
    x = x.transpose(2, 0, 1)                          # HWC -> CHW
    x = torch.from_numpy(x.copy()).float().unsqueeze(0)  # (1, 3, H, W)
    return x, orig_shape


def postprocess(sigmoid_map: np.ndarray, orig_shape: tuple[int, int]) -> np.ndarray:
    """把 sigmoid 概率图转成 0~255 灰度图并裁回原图尺寸。

    返回黑底白边的 uint8 边缘图（与 DexiNed 官方可视化一致：
    原概率高的位置在结果中反而是暗的，需要 cv2.bitwise_not）。
    """
    img = (sigmoid_map * 255.0).clip(0, 255).astype(np.uint8)
    img = cv2.bitwise_not(img)        # 边缘亮在白底 → 翻转成"黑底白边"
    H, W = orig_shape
    return img[:H, :W]


def build_and_load(checkpoint_path: str) -> torch.nn.Module:
    """构造 DexiNed 模型并加载预训练权重。"""
    from model import DexiNed  # 来自 DexiNed-master/model.py

    model = DexiNed()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # 去掉可能的 'module.' 前缀（多卡训练时才有）
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[警告] 缺失参数: {len(missing)} 个，例如 {missing[:3]}")
    if unexpected:
        print(f"[警告] 多余参数: {len(unexpected)} 个，例如 {unexpected[:3]}")
    model.eval()
    return model


def _run_single(model: torch.nn.Module, x: torch.Tensor) -> list[np.ndarray]:
    """单次前向：把 7 个输出转成 (H, W) numpy 概率图。"""
    outputs = model(x)
    return [torch.sigmoid(o).squeeze(0).squeeze(0).cpu().numpy() for o in outputs]


def _fuse_double(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """仿照 utils/image.py 的 AND-mask 融合：
        p1, p2 都是 sigmoid 后的 0~1 概率图。
        当 p1 判定为背景（高概率）但 p2 检出边缘（低概率）时，采纳 p2。
        其余位置用 p1。
    实现细节：先做与官方一致的 bitwise_not，再用 >128/<128 做掩码。
    """
    u1 = (p1 * 255.0).clip(0, 255).astype(np.uint8)
    u2 = (p2 * 255.0).clip(0, 255).astype(np.uint8)
    u1 = cv2.bitwise_not(u1)        # 边缘暗、背景亮
    u2 = cv2.bitwise_not(u2)
    mask = np.logical_and(u1 > 128, u2 < 128)
    merged_inverted = np.where(mask, u2, u1)
    merged = cv2.bitwise_not(merged_inverted)
    return merged.astype(np.float32) / 255.0


def run(image_path: str, checkpoint_path: str, output_dir: str,
        double: bool = False) -> None:
    model = build_and_load(checkpoint_path)
    x, orig_shape = preprocess(image_path)
    print(f"输入张量形状: {tuple(x.shape)},  原图尺寸: {orig_shape}")
    print(f"增强模式: {'双预测融合 (BGR + RGB)' if double else '单预测'}")

    with torch.no_grad():
        probs1 = _run_single(model, x)
        if double:
            # 把 BGR 顺序翻成 RGB（相当于 [2,1,0]），按官方 testPich 做法
            x_rgb = x[:, [2, 1, 0], :, :]
            probs2 = _run_single(model, x_rgb)
            # 融合：每路输出按 AND-mask 合并
            probs = [_fuse_double(p1, p2) for p1, p2 in zip(probs1, probs2)]
        else:
            probs = probs1

        fused = probs[-1]
        avg = np.mean(np.stack(probs, axis=0), axis=0)

    base = os.path.splitext(os.path.basename(image_path))[0]
    suffix = "_double" if double else ""

    fused_img = postprocess(fused, orig_shape)
    avg_img = postprocess(avg, orig_shape)

    fused_path = os.path.join(output_dir, f"{base}_dexined{suffix}_fused.png")
    avg_path = os.path.join(output_dir, f"{base}_dexined{suffix}_avg.png")
    imwrite_unicode(fused_path, fused_img)
    imwrite_unicode(avg_path, avg_img)

    fused_bin = (fused_img > 128).astype(np.uint8) * 255
    avg_bin = (avg_img > 128).astype(np.uint8) * 255
    fused_bin_path = os.path.join(output_dir, f"{base}_dexined{suffix}_fused_binary.png")
    avg_bin_path = os.path.join(output_dir, f"{base}_dexined{suffix}_avg_binary.png")
    imwrite_unicode(fused_bin_path, fused_bin)
    imwrite_unicode(avg_bin_path, avg_bin)

    print(f"权重            : {checkpoint_path}")
    print(f"输入图像        : {image_path}")
    print(f"原图尺寸        : {orig_shape}")
    print(f"fused 边缘图    : {fused_path}   ({fused.min():.3f}~{fused.max():.3f})")
    print(f"avg   边缘图    : {avg_path}   ({avg.min():.3f}~{avg.max():.3f})")
    print(f"fused 二值图    : {fused_bin_path}")
    print(f"avg   二值图    : {avg_bin_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DexiNed 边缘检测推理")
    parser.add_argument("--input", default=INPUT_IMAGE_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--double", action="store_true",
                        help="使用双预测增强策略（BGR + RGB 通道互换 + AND-mask 融合）")
    a = parser.parse_args()
    run(a.input, a.checkpoint, a.output, double=a.double)

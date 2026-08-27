# -*- coding: utf-8 -*-
"""
基于 PyTorch + PiDiNet 的图像边缘检测
------------------------------------
功能：
    1. 使用 zhuoinoulu/pidinet 仓库的 PiDiNet 模型；
    2. 加载预训练权重 trained_models/table5_pidinet.pth；
    3. 在用户给定的图像上做推理，输出 0~255 的二值化边缘图。

注意：
    - 仓库已通过 GitHub 镜像 ghfast.top 下载到 ./pidinet-master/；
    - 通过 sys.path 把仓库根目录加入，让 models/ 包可用。
"""

import os
import sys
import argparse
from types import SimpleNamespace
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T


# ---------- 路径配置 ----------
HERE = os.path.dirname(os.path.abspath(__file__))
PIDINET_ROOT = os.path.join(HERE, "pidinet-master")
CHECKPOINT_PATH = os.path.join(PIDINET_ROOT, "trained_models", "table5_pidinet.pth")
INPUT_IMAGE_PATH = r"C:\Users\18607\Desktop\边缘识别模型方法\图片物体边缘的提取分割.png"
OUTPUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 把 pidinet-master 加入 import 路径，以便能 `import models`
sys.path.insert(0, PIDINET_ROOT)


def build_model(checkpoint_path: str) -> torch.nn.Module:
    """构造 PiDiNet 模型并加载预训练权重（CPU 推理版本）。"""
    import models  # 来自 pidinet-master/models

    # 模型配置：carv4 + CSAM + CDCM（与 table5_pidinet 训练配置一致）
    args = SimpleNamespace(config="carv4", dil=True, sa=True)

    model = models.pidinet(args)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    # 去掉 DataParallel 带来的 "module." 前缀
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[警告] 缺失的参数: {len(missing)} 个，例如 {missing[:3]}")
    if unexpected:
        print(f"[警告] 多余的参数: {len(unexpected)} 个，例如 {unexpected[:3]}")

    model.eval()
    return model


def preprocess(image_path: str) -> tuple[torch.Tensor, tuple[int, int]]:
    """读取含中文路径的图像并做 ImageNet 归一化预处理。"""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)
    return tensor, (H, W)


def postprocess(sigmoid_map: np.ndarray) -> np.ndarray:
    """把 sigmoid 概率图转成 0~255 的 uint8 边缘图。"""
    edge = (sigmoid_map * 255.0).clip(0, 255).astype(np.uint8)
    return edge


def run(image_path: str, checkpoint_path: str, output_dir: str) -> None:
    model = build_model(checkpoint_path)
    tensor, (H, W) = preprocess(image_path)

    with torch.no_grad():
        outputs = model(tensor)              # list of 5 sigmoid maps
        fused = outputs[-1]                  # 最后一个是融合后的输出
        edge_map = fused.squeeze().cpu().numpy()

    edge = postprocess(edge_map)

    # 保存边缘图（用 PIL 保存，可支持中文路径）
    base = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{base}_pidinet_edges.png")
    Image.fromarray(edge).save(save_path)

    # 同时把单边二值图（threshold=128）也存一份，便于对比
    binary = (edge > 128).astype(np.uint8) * 255
    binary_path = os.path.join(output_dir, f"{base}_pidinet_binary.png")
    Image.fromarray(binary).save(binary_path)

    print(f"输入图像      : {image_path}")
    print(f"原图尺寸      : ({H}, {W})")
    print(f"checkpoint    : {checkpoint_path}")
    print(f"边缘图已保存   : {save_path}")
    print(f"二值图已保存   : {binary_path}")
    print(f"边缘强度范围   : min={edge_map.min():.4f}, max={edge_map.max():.4f}, mean={edge_map.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PiDiNet 边缘检测推理")
    parser.add_argument("--input", default=INPUT_IMAGE_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--output", default=OUTPUT_DIR)
    args_cli = parser.parse_args()

    run(args_cli.input, args_cli.checkpoint, args_cli.output)

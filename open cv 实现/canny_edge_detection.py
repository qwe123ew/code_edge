# -*- coding: utf-8 -*-
"""
基于 OpenCV Canny 算子的图像边缘检测
---------------------------------
功能：
    1. 读取指定路径的彩色图像；
    2. 转为灰度图并进行高斯模糊降噪；
    3. 调用 cv2.Canny() 提取边缘；
    4. 将原始图像、灰度图与边缘检测结果保存到输出目录。

使用：
    直接运行本脚本即可；如需修改输入/输出路径，编辑下方常量。
"""

import cv2
import numpy as np
import os

# ---------- 路径配置 ----------
# 输入图像绝对路径（注意 OpenCV 在 Windows 下不能直接读取含中文路径的图像，
# 因此使用 np.fromfile + cv2.imdecode 绕开此问题）
INPUT_IMAGE_PATH = r"C:\Users\18607\Desktop\边缘识别模型方法\图片物体边缘的提取分割.png"

# 输出目录（脚本所在目录下的 "open cv 实现" 文件夹）
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Canny 算子的双阈值（经验值，可按图像内容调整）
CANNY_THRESHOLD_LOW =10
CANNY_THRESHOLD_HIGH = 100

# 高斯模糊内核大小（必须是奇数）
GAUSSIAN_KERNEL_SIZE = 5


def canny_edge_detection(image_path: str, output_dir: str,
                        threshold_low: int = None,
                        threshold_high: int = None) -> None:
    """读取图像，进行 Canny 边缘检测并保存结果。

    Parameters
    ----------
    image_path : str
        待检测的彩色图像文件路径。
    output_dir : str
        边缘检测结果图片的输出目录。
    threshold_low : int, optional
        Canny 低阈值; 传 None 时取模块默认 CANNY_THRESHOLD_LOW (10)。
    threshold_high : int, optional
        Canny 高阈值; 传 None 时取模块默认 CANNY_THRESHOLD_HIGH (100)。
    """
    if threshold_low is None:
        threshold_low = CANNY_THRESHOLD_LOW
    if threshold_high is None:
        threshold_high = CANNY_THRESHOLD_HIGH

    # 1. 读取图像（使用 np.fromfile 方式以支持中文路径）
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像，请检查路径：{image_path}")

    # 2. 灰度化（Canny 只能处理单通道图像）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 高斯模糊降噪，避免边缘检测时产生过多噪点
    blurred = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)

    # 4. Canny 边缘检测
    edges = cv2.Canny(blurred, threshold_low, threshold_high)

    # 5. 文件名
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 6. 保存结果
    gray_path = os.path.join(output_dir, f"{base_name}_gray.png")
    edges_path = os.path.join(output_dir, f"{base_name}_canny_edges.png")

    # 中文路径下 cv2.imwrite 会失败，因此用 imencode + tofile 写入
    cv2.imencode(".png", gray)[1].tofile(gray_path)
    cv2.imencode(".png", edges)[1].tofile(edges_path)

    print(f"原图尺寸        : {img.shape}")
    print(f"灰度图已保存至  : {gray_path}")
    print(f"边缘图已保存至  : {edges_path}")
    print(f"Canny 阈值      : low={threshold_low}, high={threshold_high}")


if __name__ == "__main__":
    canny_edge_detection(INPUT_IMAGE_PATH, OUTPUT_DIR)

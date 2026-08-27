# 边缘识别模型方法集合 (Edge Detection Methods Collection)

> 📚 一个面向"图片物体边缘提取与分割"任务的**多算法对比与集成**项目，覆盖从经典算法到深度学习的多种实现方案。
>
> **作者 / Author**: Neos

---

## 📑 项目背景

本仓库系统性地实现并对比了 **9 种边缘检测/图像分割方案**，围绕两份方案文档：

- `图片物体边缘的提取分割.docx` — 任务定义文档
- `边缘识别算法流程.docx` — 算法流程总览

任务包括：**去噪**、**边缘检测**、**边缘骨架化与划分**、**黄金角染色**、**四色定理染色**、**边缘-物体关系分析** 等。

---

## 🗂️ 目录结构

| 目录 | 类型 | 主要算法 | 依赖 |
|---|---|---|---|
| [LBM 实现](./LBM%20实现/) | 物理模型 | D2Q9 BGK Lattice Boltzmann + Perona-Malik 各向异性扩散去噪 + Canny 边缘 + 骨架划分 + 黄金角/四色染色 | `numpy`, `opencv-python`, `scipy` |
| [PyTorch DexiNed 实现](./PyTorch%20DexiNed%20实现/) | 深度学习 | DexiNed (Dense Extreme Inception Network) 边缘检测 | `torch`, `opencv-python` |
| [PyTorch PiDiNet 实现](./PyTorch%20PiDiNet%20实现/) | 深度学习 | PiDiNet (Pixel Difference Networks) 边缘检测 | `torch`, `opencv-python` |
| [SciPy 实现](./SciPy%20实现/) | 经典数值 | 基于 SciPy 的边缘检测 | `scipy`, `opencv-python` |
| [open cv 实现](./open%20cv%20实现/) | 经典算法 | OpenCV Canny 边缘检测 | `opencv-python` |
| [scikit-image 实现](./scikit-image%20实现/) | 经典算法 | scikit-image Canny / Sobel / 等滤波器 | `scikit-image`, `opencv-python` |
| [基于open cv 整合算法](./基于open%20cv%20整合算法/) | 集成管线 | 边缘划分 (Zhang-Suen) + 区域四色染色 + 边缘闭合性分析 | `opencv-python`, `numpy`, `scipy` |
| [优化自主test完整算法](./优化自主test完整算法/) | 端到端管线 | 自研完整算法：去噪→边缘→骨架→划分→染色 | `opencv-python`, `numpy`, `scipy` |
| [对比模型](./对比模型/) | 评估对比 | 多模型横向对比评估脚本 | `opencv-python`, `numpy` |

---

## 🔬 算法分类

### 1. 经典算法 (Classical)
- **Canny** (OpenCV / scikit-image)
- **Sobel / Scharr / Prewitt / Roberts** (scikit-image)
- **Laplacian of Gaussian (LoG)**
- **Zhang-Suen 骨架化**
- **8-连通域标记**

### 2. 数值方法 (Numerical)
- **SciPy ndimage** 滤波器族
- **形态学操作** (开/闭运算、膨胀、腐蚀)

### 3. 物理模型 (Physics-based)
- **LBM (格子玻尔兹曼方法)** D2Q9 BGK 求解扩散方程
- **Perona-Malik 各向异性扩散**（含 4 个扩散系数: isotropic / PM-1 / PM-2 / Tukey biweight）

### 4. 深度学习 (Deep Learning)
- **DexiNed** — 密集极值 inception 网络，sub-pixel 精度
- **PiDiNet** — 像素差分网络，轻量高效

### 5. 集成管线 (Pipeline)
- **边缘-物体关系**：闭运算 + 8-CC + 抠掉外部背景
- **黄金角 137.5° HSV 染色**
- **DSATUR 贪心 4 色染色**（四色定理）
- **边缘闭合完整性分析**：缺口检测 + 阶梯式形态学闭合

---

## 🚀 快速开始

### 环境要求

```bash
pip install numpy opencv-python scipy scikit-image matplotlib
# 深度学习方法（可选）
pip install torch torchvision
```

### 运行示例

```bash
# LBM 去噪 + 经典边缘管线
cd "LBM 实现" && python run_lbm.py

# 单一 Canny 实现
cd "open cv 实现" && python canny_edge_detection.py

# 深度学习 DexiNed
cd "PyTorch DexiNed 实现" && python dexined_inference.py

# 完整端到端集成
cd "优化自主test完整算法" && python edge_pipeline.py

# 多模型对比
cd "对比模型" && python compare.py
```

各目录的 `outputs/` 子目录存放了对应算法的可视化结果 PNG。

---

## 📊 输出说明

每个实现目录都包含 `outputs/` 子目录，按"阶段名"组织可视化结果，例如：

- `0_原图` — 原始输入
- `1_Canny轮廓` — 边缘检测结果
- `2_结点标注` — 骨架端点/分支点标注
- `3_连通骨架` / `4_逐级骨架` — 骨架细化过程
- `5_LBM去噪结果` — 去噪对比
- `region_colored` — 区域四色染色结果

深度学习方法的 `outputs/` 还包含 fused / binary 等多种后处理产物。

---

## 🛠️ 模型权重

⚠️ **模型权重文件未上传到本仓库**（GitHub 单文件 100MB 限制）。

- DexiNed 的 BIPED 训练权重 `10.pt` 需从原仓库下载：[DexiNed/BIPED](https://github.com/xavysp/DexiNed)
- PiDiNet 的预训练权重需从原仓库下载：[PiDiNet](https://github.com/zhuoinoulu/pidinet)

下载后将权重文件放到对应 `checkpoints/` 目录即可运行。

---

## 📝 引用与致谢

- **DexiNed**: Soria et al., "Dense Extreme Inception Network for Edge Detection"
- **PiDiNet**: Su et al., "Pixel Difference Networks for Efficient Edge Detection"
- **LBM 格子玻尔兹曼方法**: 经典流体数值模拟方法
- **四色定理染色**: DSATUR 贪心算法

---

## ✍️ 作者信息

| | |
|---|---|
| **作者** | **Neos** |
| **项目主题** | 图片物体边缘的提取与分割 |
| **创建时间** | 2026 |

如有问题或建议，欢迎通过 GitHub Issues 联系。

---

> 💡 **提示**: 每个子目录的 `README.md` 提供了该模块的**详细函数说明、参数解释和算法原理**，请按需查阅。

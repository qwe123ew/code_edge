# LBM 去噪 + 传统边缘/划分/染色管线

按反馈,LBM **只用于 Perona-Malik 各向异性扩散去噪**。后续所有环节用经典算法,不再用 LBM 解决边缘/划分问题。

**第二轮迭代**: 旧默认参数 (n_steps=30, τ_max=1.5, PM-1) 在干净图上过度平滑, 把细节全抹平了。
改成 **`auto=True` 自适应模式**: 按噪声强度自动选 skip / light / medium / heavy, 默认参数也下调一档。

| 阶段 | docx 任务 | 算法 | 是否 LBM |
|---|---|---|---|
| 去噪 | — (前置) | LBM-Perona-Malik 各向异性扩散 (**auto 模式**) | **✅ LBM** |
| 边缘检测 | 任务 1 | OpenCV Canny (auto-threshold 基于中位数) | ❌ |
| 边缘划分 | 任务 2 | Zhang-Suen 骨架 + branch_points + 8-CC | ❌ |
| 染色 | 任务 3 | 黄金角 137.5° HSV | ❌ |
| 4 色 | 任务 4 (段级) | DSATUR 贪心 4 色 | ❌ |
| 物体提取 | "边缘-物体关系" | 闭运算 + 8-CC + 抠掉外部背景 | ❌ |
| 4 色 | 任务 4 (区域级) | DSATUR 贪心 4 色 | ❌ |

依赖: `numpy`, `opencv-python`, `scipy`。**零 PyTorch、零神经网络**——LBM 是纯物理模型。

---

## 文件

- [lbm_pipeline.py](lbm_pipeline.py) — LBM 核心 (D2Q9 BGK, Perona-Malik 空间变化 τ)
  - `lbm_denoise(image, n_steps=10, tau_max=0.7, k_edge=0.12, scheme='perona_malik_2', auto=True) → (uint8, info)`
  - `estimate_noise(gray) → float` — Laplacian 95% 分位估计
  - `_is_sparse_edge_map(gray) → bool` — 暗像素 >60% 判定
  - 4 个扩散系数: isotropic / perona_malik_1 / perona_malik_2 / tukey
- [run_lbm.py](run_lbm.py) — 5 张图端到端驱动, 输出 9 PNG + 1 统计 JSON + 1 summary
- [outputs/](outputs/) — 产物
  - [outputs/archive_v1/](outputs/archive_v1/) — 旧 v1 "全包" / v2 "固定参数" 方案的产物 (已废弃, 仅留档)

---

## LBM-Perona-Malik 原理

把图像当作密度场 ρ(x, t), 用 D2Q9 BGK 求解扩散方程:

```
∂ρ/∂t = div(c(|∇ρ|) · ∇ρ)
```

LBM 中 τ 与扩散系数 D 的关系: `D = c_s² (τ - 0.5) Δt`, `c_s² = 1/3`.

**Perona-Malik 关键**: τ 随空间变化, 边缘处扩散被抑制。

```
c(x) = exp(-(|∇ρ(x)| / k_edge)²)        # Perona-Malik I, 偏好高对比度边缘
c(x) = 1 / (1 + (|∇ρ|/k_edge)²)         # Perona-Malik II, 更平滑抗噪
τ(x) = 0.51 + (τ_max - 0.51) · c(x)      # 0.51 是稳定性下限 (τ > 0.5)
```

每步:
1. `g = |∇ρ|`
2. `c = diffusion_coeff(g, k_edge, scheme)` ∈ [0, 1]
3. `τ = 0.51 + (τ_max - 0.51) · c`
4. `f ← f - (f - f^eq) / τ` (BGK, u=0)
5. `f ← streaming(f)`
6. `ρ ← Σ f_i`

直观: 平滑区 c→1, τ→τ_max, 大扩散; 边缘区 c→0, τ→0.5, 小扩散, 边缘保留。

参考:
- P. Perona, J. Malik, "Scale-space and edge detection using anisotropic diffusion" (1990)
- S. Chen, G.D. Doolen, "Lattice Boltzmann Method for Fluid Flows" (1998)
- A. Ginzburg, "Equilibrium-type and link-type lattice Boltzmann models for generic advection and anisotropic-dispersion equation" (2005)

---

## Auto 模式: 按噪声强度自适应

PM 各向异性扩散**对干净图是伤害** (细节被磨掉), **对真噪声才有用**。
所以默认开 `auto=True`, 先估噪声再决定动不动:

| 噪声估计 (Laplacian 95% 分位) | 模式 | 步数 | τ_max | 行为 |
|---:|---|---:|---:|---|
| < 50 (或检测为稀疏边缘图) | **skip** | 0 | — | 完全不跑, 返回原图 |
| 50 – 200 | **light** | 5 | 0.6 | 轻度磨皮, 保留细节 |
| 200 – 1000 | **medium** | 10 | 0.7 | 中度去噪 |
| > 1000 | **heavy** | 17 | 0.9 | 强去噪 |

为什么这样切:
- 噪声 < 50 几乎就是干净图, LBM 跑下去只破坏, 不如直接跳过
- 噪声 50-200 是普通 JPEG 压缩/传感器噪声, light 模式足够
- 噪声 > 1000 才是真的脏图, 才值得上多步

`sparse edge map` 检测 (暗像素 > 60% 且亮像素 < 40%) 会强制 skip, 因为输入已经是边缘图了,
再 PM 一遍会把稀疏的线洗掉 (教训: 旧版 `图片物体边缘的提取分割.png` 就被这样毁过)。

参考: J. Immerkaer, "Fast Noise Variance Estimation" (1996), CVIU 64(2).

---

## 输出文件

| 文件 | 内容 |
|---|---|
| `<图名>_0_原图.png` | 原图 |
| `<图名>_1_LBM去噪.png` | LBM Perona-Malik 去噪后 (skip 时与原图相同) |
| `<图名>_1b_去噪对比.png` | before / after 并排 (绿字标注, 含 skip / light / medium / heavy) |
| `<图名>_2_Canny边缘.png` | 在去噪图上跑 auto-Canny |
| `<图名>_2b_边缘overlay.png` | 边缘叠在去噪图上 (红色) |
| `<图名>_3_结点标注.png` | 骨架 + 红色 branch points |
| `<图名>_4_逐段染色.png` | 黄金角逐段染色 |
| `<图名>_5_段级四色.png` | DSATUR 段级 4 色 |
| `<图名>_6_物体提取.png` | 闭运算 + 8-CC 抠掉外部背景 |
| `<图名>_7_区域级四色.png` | DSATUR 区域级 4 色 |
| `<图名>_统计.json` | 全部指标 (含 lbm_denoise 的 noise/mode/applied) |
| `summary.json` | 5 张图汇总 |

---

## 跑法

```bash
cd "C:\Users\18607\Desktop\边缘识别模型方法\LBM 实现"
python run_lbm.py
```

或单独调用:

```python
import cv2
from lbm_pipeline import lbm_denoise

bgr = cv2.imread("photo.png")

# auto=True (推荐): 自适应 skip / light / medium / heavy
denoised, info = lbm_denoise(bgr, auto=True)
print(info)  # {"applied": True, "noise": 81.0, "mode": "light",
             #  "n_steps": 5, "tau_max": 0.6}

# auto=False: 强制按给定参数跑
denoised, info = lbm_denoise(bgr, n_steps=10, tau_max=0.7,
                              k_edge=0.12, scheme="perona_malik_2",
                              auto=False)
```

---

## LBM 关键参数选择

| 参数 | 含义 | 默认 | 推荐范围 | 调参提示 |
|---|---|---|---|---|
| `n_steps` | 迭代步数 | 10 | 5-20 | 越大越平滑, light 模式下只用 5 |
| `tau_max` | 平滑区最大松弛时间 | 0.7 | 0.6-1.0 | 越大扩散越强, light 用 0.6 |
| `k_edge` | 边缘阈值 (相对 0-1 灰度) | 0.12 | 0.08-0.20 | 越大保留的边缘越多 |
| `scheme` | 扩散系数 | PM-2 | PM-1 / PM-2 / Tukey | PM-2 更抗噪, Tukey 最硬阈值 |
| `auto` | 自适应 | True | True / False | **强烈建议开** |

---

## 5 张图实测数据 (Python 3.14 / numpy / opencv / scipy, auto=True)

| 图片 | 尺寸 | 噪声 | LBM 模式 | 步数 | 耗时 ms | Canny 边缘像素 | 段数 | 段 4 色 (用/冲突) | 物体数 | 区域 4 色 (用/冲突) |
|---|---|---:|---|---:|---:|---:|---:|:-:|---:|:-:|
| 图片物体边缘的提取分割 | 438×158 | 37.0 | **skip** | 0 | 7 | 10506 (15.18%) | 590 | 3 / 0 | 32 | 1 / 0 |
| bsds_368037_原图 | 481×321 | 81.0 | light | 5 | 109 | 11900 (7.71%) | 952 | 3 / 0 | 30 | 1 / 0 |
| bsds_97010_原图 | 481×321 | 196.0 | light | 5 | 132 | 22763 (14.74%) | 1455 | 3 / 0 | 37 | 1 / 0 |
| nyud_5017_原图 | 224×170 | 82.0 | light | 5 | 26 | 5083 (13.35%) | 320 | 3 / 0 | 28 | 1 / 0 |
| nyud_6233_原图 | 224×170 | 61.0 | light | 5 | 27 | 6380 (16.75%) | 621 | 3 / 0 | 33 | 1 / 0 |

**段级 4 色冲突 = 0 (5/5)**, **区域级 4 色冲突 = 0 (5/5)**。
**auto 模式让卡通图自动 skip, 真实图自动 light**, 不再需要人工选参数。

---

## 迭代对比: 三版方案

| 维度 | v1 (旧) | v2 (固定参数) | **v3 (auto 模式, 当前)** |
|---|---|---|---|
| LBM 参数 | 30 步 τ=1.5 PM-1 | 30 步 τ=1.5 PM-1 | **auto: 5 步 τ=0.6 PM-2 (按图选)** |
| 干净图行为 | 全部细节抹平 | 全部细节抹平 | **跳过, 保留原图** |
| 边缘图行为 | 白线被洗没 | 白线被洗没 | **跳过, 保留白线** |
| 噪声图行为 | 强去噪, 但误伤 | 强去噪, 但误伤 | **轻度去噪, 保留细节** |
| 谷仓图 LBM 耗时 | 565 ms | 565 ms | **132 ms (light, 5 步)** |
| 卡通图 LBM 耗时 | 212 ms | 212 ms | **7 ms (skip)** |
| 调试复杂度 | 4 个 LBM 参数 | 4 个 LBM 参数 | **0 个 (auto 接管)** |

### 旧 vs 新 视觉对比 (bsds_97010 谷仓)

- **旧版 (30 步, τ=1.5, PM-1)**: 谷草纹理、屋顶木板、远处树枝全部磨平, 画面像油画化
- **新版 (auto → light, 5 步, τ=0.6, PM-2)**: 谷草纹理清晰, 屋顶木板条纹可见, 轻微磨皮即可

旧版 `1_LBM去噪.png` 已无意义, 现已用新版覆盖。

---

## 与早期 LBM-canny + LBM-flood-fill 方案对比

| 维度 | 旧方案 (LBM 全包) | 当前方案 (LBM 只去噪) |
|---|---|---|
| 边缘定位 | LBM 应力张量 → 稀疏、断续 | LBM 去噪 + OpenCV Canny → **连续、完整** |
| 段数 (bsds_97010) | 728 | 1455 (更细, 因 Canny 边缘更密) |
| 段 4 色用色 | 3 | 3 (持平) |
| 物体提取 | LBM 漫灌 6.5s/张 (慢) | 闭运算+8-CC 2-3 ms/张 (**快 2000×**) |
| 区域 4 色用色 | 1 | 1 (持平, 都是孤立区域) |
| 调试复杂度 | 5 个 LBM 参数 | **0 个手动 (auto 接管) + 标准 Canny** |

# 图像处理作业：下采样与抗混叠实验

本项目用 Python 实现以下三部分实验：

| 部分 | 内容 |
|------|------|
| 第一部分 | 生成棋盘格/Chirp 测试图，演示混叠，用高斯滤波消除混叠，FFT 频域验证 |
| 第二部分 | 固定 M=4，对比 σ=0.5/1.0/2.0/4.0，验证理论公式 σ ≈ 0.45M = 1.8 |
| 第三部分 | 梯度自适应估计局部 σ，实现自适应下采样并与统一下采样对比误差图 |

---

## 环境要求

- [Anaconda](https://www.anaconda.com/) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.10+

---

## 第一步：创建 Conda 环境

```bash
conda create -n imgproc python=3.11 -y
conda activate imgproc
```

---

## 第二步：安装依赖

```bash
pip install -r requirements.txt
```

依赖列表（`requirements.txt`）：

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

---

## 第三步：运行代码

```bash
python main.py
```

程序会依次运行三个部分，每部分结束后弹出图像窗口（关闭窗口后继续），并将结果保存为 PNG 文件。

---

## 输出文件

| 文件名 | 内容 |
|--------|------|
| `part1_aliasing_demo.png` | 原图 / 直接下采样 / 高斯下采样 / 差异图，每列对应图像和 FFT 频谱 |
| `part2_sigma_verification.png` | M=4 条件下不同 σ 的视觉效果与 FFT 对比 |
| `part3_adaptive_downsampling.png` | 梯度图、自适应 σ 图、三种下采样结果、误差图与误差分布直方图 |

---

## 代码结构

```
main.py
├── generate_checkerboard()   # 生成棋盘格
├── generate_chirp()          # 生成二维 Chirp 图
├── downsample()              # 直接下采样
├── gaussian_downsample()     # 高斯滤波 + 下采样
├── fft_spectrum()            # 计算对数幅度谱
├── mse()                     # 均方误差
│
├── part1_aliasing_demo()     # 第一部分
├── part2_sigma_verification() # 第二部分
└── part3_adaptive_downsampling() # 第三部分
```

---

## 原理说明

### 混叠（Aliasing）

对图像做因子 M 的下采样时，若图像含有高于奈奎斯特频率 `f_s/(2M)` 的分量，这些分量会"折叠"回低频区域，产生混叠失真。

### 抗混叠高斯滤波

在下采样前施加高斯低通滤波，截断超出奈奎斯特频率的成分：

- **理论最优 σ**：`σ ≈ 0.45 × M`
- M=4 时，σ_theory = **1.8**
- σ 太小 → 高频未完全截断，混叠残留
- σ 太大 → 截断过多低频，图像过度模糊

### 自适应下采样

利用梯度幅度估计图像各区域的局部频率复杂度：
- 高复杂度区域（边缘/高频纹理）：使用较大 σ 充分平滑
- 低复杂度区域（平坦区域）：使用较小 σ 保留细节

自适应策略兼顾抗混叠与细节保留，优于全图统一 σ。

---

## 常见问题

**Q: 图像窗口不弹出？**  
A: 在服务器或无显示器环境中，可在 `main.py` 顶部将 `matplotlib.use('Agg')` 取消注释，程序将直接保存图片而不弹窗。

**Q: 中文显示为方框？**  
A: macOS 默认支持中文字体。若在 Linux 上运行，可安装字体后修改 `matplotlib.rcParams['font.sans-serif']`。

**Q: 第三部分运行较慢？**  
A: 自适应滤波对每个像素做插值，256×256 图像约需 30-60 秒，属正常现象。

"""
图像处理作业：下采样与抗混叠实验
=================================
第一部分：生成棋盘格/Chirp图，演示混叠与高斯滤波消除混叠，FFT频谱验证
第二部分：固定M=4，验证σ公式，对比σ=0.5/1.0/2.0/4.0与理论值σ≈0.45M=1.8
第三部分：梯度自适应下采样，与全图统一下采样对比误差
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift

# 中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def generate_checkerboard(size=256, block_size=8):
    """生成棋盘格图像（高频周期信号，易产生混叠）"""
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]
    img = ((rows // block_size + cols // block_size) % 2) * 255.0
    return img

def generate_chirp(size=256):
    """
    生成二维 Chirp 测试图：水平/垂直方向频率从低到高线性增加。
    低频区在左上角，高频区在右下角，可直观看到混叠从何处开始。
    """
    t = np.linspace(0, 1, size)
    X, Y = np.meshgrid(t, t)
    # 最高频率 = size/4 周期/图（接近奈奎斯特频率）
    f_max = size / 4
    chirp_x = np.cos(2 * np.pi * f_max * X ** 2)
    chirp_y = np.cos(2 * np.pi * f_max * Y ** 2)
    img = chirp_x * chirp_y
    # 归一化到 [0, 255]
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img

def downsample(img, M):
    """直接下采样（每 M 个像素取 1 个，无预滤波）"""
    return img[::M, ::M]

def gaussian_downsample(img, sigma, M):
    """高斯低通滤波后再下采样（抗混叠标准流程）"""
    return gaussian_filter(img, sigma=sigma)[::M, ::M]

def fft_spectrum(img):
    """返回以对数尺度显示的中心化 FFT 幅度谱"""
    return np.log1p(np.abs(fftshift(fft2(img))))

def mse(a, b):
    """均方误差"""
    n = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return np.mean((a[:n[0], :n[1]].astype(float) - b[:n[0], :n[1]].astype(float)) ** 2)


# ─────────────────────────────────────────────
# 第一部分：混叠演示
# ─────────────────────────────────────────────

def part1_aliasing_demo(M=4):
    print("=" * 60)
    print("第一部分：混叠演示")
    print("=" * 60)

    cb   = generate_checkerboard(256, block_size=8)
    chirp = generate_chirp(256)
    sigma_opt = 0.45 * M  # 理论最优 σ

    test_images = [("棋盘格", cb), ("Chirp图", chirp)]

    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    fig.suptitle(f"第一部分：混叠演示  (M={M}, 高斯σ={sigma_opt:.1f})", fontsize=15, fontweight='bold')

    row_labels = ["原图", f"直接下采样 M={M}", f"高斯下采样 σ={sigma_opt:.1f}", "差异图（直接 vs 高斯）"]
    for ax, label in zip(axes[:, 0], row_labels):
        ax.set_ylabel(label, fontsize=11, rotation=90, labelpad=10)

    for col_idx, (name, img) in enumerate(test_images):
        c = col_idx * 2  # 每张图占两列：图像 + FFT

        ds_direct   = downsample(img, M)
        ds_gaussian = gaussian_downsample(img, sigma_opt, M)
        h = min(ds_direct.shape[0], ds_gaussian.shape[0])
        w = min(ds_direct.shape[1], ds_gaussian.shape[1])
        diff = np.abs(ds_direct[:h, :w].astype(float) - ds_gaussian[:h, :w].astype(float))

        # 行 0：原图
        axes[0, c].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, c].set_title(f"{name} 原图 (256×256)")
        axes[0, c].axis('off')
        axes[0, c+1].imshow(fft_spectrum(img), cmap='hot')
        axes[0, c+1].set_title(f"{name} 原图 FFT")
        axes[0, c+1].axis('off')

        # 行 1：直接下采样
        axes[1, c].imshow(ds_direct, cmap='gray', vmin=0, vmax=255)
        axes[1, c].set_title(f"直接下采样 ({ds_direct.shape[0]}×{ds_direct.shape[1]})")
        axes[1, c].axis('off')
        axes[1, c+1].imshow(fft_spectrum(ds_direct), cmap='hot')
        axes[1, c+1].set_title("FFT — 有混叠（高频分量折叠）")
        axes[1, c+1].axis('off')

        # 行 2：高斯下采样
        axes[2, c].imshow(ds_gaussian, cmap='gray', vmin=0, vmax=255)
        axes[2, c].set_title(f"高斯下采样 ({ds_gaussian.shape[0]}×{ds_gaussian.shape[1]})")
        axes[2, c].axis('off')
        axes[2, c+1].imshow(fft_spectrum(ds_gaussian), cmap='hot')
        axes[2, c+1].set_title("FFT — 混叠消失（高频已截断）")
        axes[2, c+1].axis('off')

        # 行 3：差异图
        axes[3, c].imshow(diff, cmap='hot')
        axes[3, c].set_title(f"差异图  max={diff.max():.1f}  MSE={mse(ds_direct,ds_gaussian):.2f}")
        axes[3, c].axis('off')
        axes[3, c+1].axis('off')

    plt.tight_layout()
    plt.savefig("part1_aliasing_demo.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(">>> 已保存：part1_aliasing_demo.png\n")


# ─────────────────────────────────────────────
# 第二部分：σ 公式验证
# ─────────────────────────────────────────────

def part2_sigma_verification(M=4):
    print("=" * 60)
    print("第二部分：σ 公式验证")
    print("=" * 60)

    sigma_theory = 0.45 * M
    sigmas = [0.5, 1.0, 2.0, 4.0]
    print(f"理论最优 σ = 0.45 × M = 0.45 × {M} = {sigma_theory}")

    cb    = generate_checkerboard(256, block_size=8)
    chirp = generate_chirp(256)
    test_images = [("棋盘格", cb), ("Chirp图", chirp)]

    fig, axes = plt.subplots(len(sigmas) + 1, 4, figsize=(18, (len(sigmas)+1) * 4))
    fig.suptitle(f"第二部分：σ 参数验证  (M={M}, 理论σ={sigma_theory})", fontsize=15, fontweight='bold')

    # 第 0 行：原图
    for col_idx, (name, img) in enumerate(test_images):
        c = col_idx * 2
        axes[0, c].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, c].set_title(f"{name} 原图")
        axes[0, c].axis('off')
        axes[0, c+1].imshow(fft_spectrum(img), cmap='hot')
        axes[0, c+1].set_title(f"{name} 原图 FFT")
        axes[0, c+1].axis('off')

    # 第 1-4 行：不同 σ
    for row_idx, sigma in enumerate(sigmas):
        if sigma < sigma_theory * 0.7:
            note = "  ← σ太小：混叠残留"
        elif sigma > sigma_theory * 1.6:
            note = "  ← σ太大：过度模糊"
        elif abs(sigma - sigma_theory) < 0.4:
            note = "  ← ≈ 最合适"
        else:
            note = ""

        for col_idx, (name, img) in enumerate(test_images):
            c = col_idx * 2
            ds = gaussian_downsample(img, sigma, M)
            axes[row_idx+1, c].imshow(ds, cmap='gray', vmin=0, vmax=255)
            axes[row_idx+1, c].set_title(f"{name}  σ={sigma}{note}")
            axes[row_idx+1, c].axis('off')
            axes[row_idx+1, c+1].imshow(fft_spectrum(ds), cmap='hot')
            axes[row_idx+1, c+1].set_title(f"FFT  σ={sigma}")
            axes[row_idx+1, c+1].axis('off')

    plt.tight_layout()
    plt.savefig("part2_sigma_verification.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(">>> 已保存：part2_sigma_verification.png\n")

    # ── 定量对比：以理论 σ 的结果为参考 ──────────────────────
    print(f"定量对比（以 σ={sigma_theory} 理论值结果为参考基准）：")
    print(f"  {'σ':>5}  {'棋盘格 MSE':>12}  {'Chirp MSE':>12}  评估")
    print("  " + "-" * 50)
    ref_cb    = gaussian_downsample(cb,    sigma_theory, M)
    ref_chirp = gaussian_downsample(chirp, sigma_theory, M)
    mse_table = {}
    for sigma in sigmas:
        ds_cb    = gaussian_downsample(cb,    sigma, M)
        ds_chirp = gaussian_downsample(chirp, sigma, M)
        e_cb    = mse(ds_cb,    ref_cb)
        e_chirp = mse(ds_chirp, ref_chirp)
        mse_table[sigma] = (e_cb, e_chirp)
        if sigma < sigma_theory * 0.7:
            label = "混叠残留（σ太小）"
        elif sigma > sigma_theory * 1.6:
            label = "过度模糊（σ太大）"
        else:
            label = "接近最优"
        print(f"  {sigma:>5.1f}  {e_cb:>12.3f}  {e_chirp:>12.3f}  {label}")
    print(f"  {sigma_theory:>5.1f}  {'0.000':>12}  {'0.000':>12}  ← 理论最优（基准）\n")

    # ── 细粒度搜索"最合适" σ，与理论值对比 ──────────────────
    print("细粒度搜索最合适 σ（棋盘格 + Chirp 平均 MSE）...")
    sigmas_fine = np.linspace(0.3, 5.0, 100)
    errors_fine = []
    for s in sigmas_fine:
        e = (mse(gaussian_downsample(cb,    s, M), ref_cb) +
             mse(gaussian_downsample(chirp, s, M), ref_chirp)) / 2
        errors_fine.append(e)
    errors_fine = np.array(errors_fine)
    best_idx   = int(np.argmin(errors_fine))
    best_sigma = sigmas_fine[best_idx]
    print(f"  细粒度搜索最合适 σ = {best_sigma:.2f}  (理论值 σ = {sigma_theory:.2f})\n")

    # ── 图2：MSE vs σ 曲线 + 视觉对比 ──────────────────────
    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))
    fig2.suptitle(
        f"第二部分：找最合适 σ 与理论值对比  (M={M})",
        fontsize=15, fontweight='bold'
    )

    # 上半行左侧：MSE vs σ 曲线（棋盘格）
    errors_cb_fine = [mse(gaussian_downsample(cb, s, M), ref_cb) for s in sigmas_fine]
    errors_chirp_fine = [mse(gaussian_downsample(chirp, s, M), ref_chirp) for s in sigmas_fine]

    ax_curve = axes2[0, 0]
    ax_curve.plot(sigmas_fine, errors_cb_fine,    color='steelblue',  label='棋盘格 MSE')
    ax_curve.plot(sigmas_fine, errors_chirp_fine, color='darkorange', label='Chirp MSE')
    ax_curve.axvline(best_sigma,   color='green',  linestyle='--', linewidth=1.5,
                     label=f'搜索最优 σ={best_sigma:.2f}')
    ax_curve.axvline(sigma_theory, color='red',    linestyle='-',  linewidth=1.5,
                     label=f'理论值 σ={sigma_theory:.2f}')
    # 标记 4 个测试点
    for s in sigmas:
        e_avg = (mse_table[s][0] + mse_table[s][1]) / 2
        ax_curve.scatter(s, e_avg, color='black', zorder=5, s=40)
        ax_curve.annotate(f'σ={s}', (s, e_avg), textcoords='offset points',
                          xytext=(4, 4), fontsize=8)
    ax_curve.set_xlabel('σ')
    ax_curve.set_ylabel('MSE（相对理论值）')
    ax_curve.set_title('MSE vs σ（细粒度搜索）')
    ax_curve.legend(fontsize=8)
    ax_curve.set_ylim(bottom=0)

    # 上半行右侧：4 个 σ 的棋盘格 MSE 柱状图
    ax_bar = axes2[0, 1]
    bar_colors = ['tomato' if e > 100 else 'gold' if e > 10 else 'mediumseagreen'
                  for e, _ in mse_table.values()]
    ax_bar.bar([str(s) for s in sigmas], [e for e, _ in mse_table.values()],
               color=bar_colors, edgecolor='black')
    ax_bar.axhline(0, color='red', linestyle='--', linewidth=1, label=f'理论值 σ={sigma_theory}')
    ax_bar.set_xlabel('σ')
    ax_bar.set_ylabel('棋盘格 MSE')
    ax_bar.set_title('4 个测试 σ 的误差柱状图')
    ax_bar.legend(fontsize=8)

    # 最合适 σ vs 理论值：棋盘格和 Chirp 各两张（图像 + FFT）
    comparisons = [
        (f'最合适 σ={best_sigma:.2f}（搜索）', best_sigma),
        (f'理论值 σ={sigma_theory:.2f}',       sigma_theory),
    ]
    for row_i, (img_name, img_data) in enumerate([('棋盘格', cb), ('Chirp图', chirp)]):
        for col_offset, (label, s) in enumerate(comparisons):
            col = col_offset + (2 if row_i == 0 else 2)
            ds = gaussian_downsample(img_data, s, M)
            r_img  = axes2[row_i, col_offset + 2]
            r_img.imshow(ds, cmap='gray', vmin=0, vmax=255)
            r_img.set_title(f'{img_name}  {label}')
            r_img.axis('off')

    # 下半行左两列：FFT 对比
    for col_offset, (label, s) in enumerate(comparisons):
        ds_cb_cmp    = gaussian_downsample(cb,    s, M)
        ds_chirp_cmp = gaussian_downsample(chirp, s, M)
        axes2[0, col_offset].axis('off')   # 已被曲线/柱状图占用，这里跳过

    # 重新布局下半行：棋盘格 FFT 对比
    for col_offset, (label, s) in enumerate(comparisons):
        ds = gaussian_downsample(cb, s, M)
        axes2[1, col_offset].imshow(fft_spectrum(ds), cmap='hot')
        axes2[1, col_offset].set_title(f'棋盘格 FFT  {label}')
        axes2[1, col_offset].axis('off')

    plt.tight_layout()
    plt.savefig("part2_best_vs_theory.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f">>> 已保存：part2_best_vs_theory.png")
    print(f"    细粒度最优 σ={best_sigma:.2f}  vs  理论值 σ={sigma_theory:.2f}  "
          f"（偏差 {abs(best_sigma - sigma_theory):.2f}）\n")


# ─────────────────────────────────────────────
# 第三部分：自适应下采样
# ─────────────────────────────────────────────

def part3_adaptive_downsampling(M=4):
    print("=" * 60)
    print("第三部分：自适应下采样")
    print("=" * 60)

    # 使用 Chirp 图：不同位置频率不同，非常适合验证自适应滤波
    img = generate_chirp(256)
    sigma_theory = 0.45 * M

    # ── 步骤 1：梯度分析估计局部复杂度 ──────────────────
    print("步骤1：梯度分析，估计局部频率复杂度...")
    grad_x = np.gradient(img, axis=1)
    grad_y = np.gradient(img, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 用较大 σ 的高斯核做局部平均，得到平滑的复杂度图
    local_complexity = gaussian_filter(grad_mag, sigma=12)
    complexity_norm = (local_complexity - local_complexity.min()) / \
                      (local_complexity.max() - local_complexity.min() + 1e-8)

    # ── 步骤 2：建立自适应 σ 图 ───────────────────────
    # 低复杂度区域（平坦/低频）：σ 可小一些，保留细节
    # 高复杂度区域（高频/边缘）：σ 需大一些，充分抗混叠
    sigma_min = sigma_theory * 0.6
    sigma_max = sigma_theory * 1.8
    sigma_map = sigma_min + complexity_norm * (sigma_max - sigma_min)
    print(f"  σ 范围：[{sigma_min:.2f}, {sigma_max:.2f}]，理论均值={sigma_theory:.2f}")

    # ── 步骤 3：自适应高斯滤波（分块实现）─────────────
    # 策略：预计算几个 σ 级别的滤波图，然后按 σ_map 插值混合
    sigma_levels = np.linspace(sigma_min, sigma_max, 6)
    filtered_stack = np.stack([gaussian_filter(img, s) for s in sigma_levels], axis=0)

    # 对每个像素，根据 sigma_map 在最近两档 σ 之间线性插值
    adaptive_filtered = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            s = sigma_map[i, j]
            # 找到最近的两档
            idx = np.searchsorted(sigma_levels, s)
            idx = np.clip(idx, 1, len(sigma_levels) - 1)
            s0, s1 = sigma_levels[idx-1], sigma_levels[idx]
            alpha = (s - s0) / (s1 - s0 + 1e-8)
            adaptive_filtered[i, j] = (1 - alpha) * filtered_stack[idx-1, i, j] + \
                                              alpha  * filtered_stack[idx,   i, j]

    adaptive_ds = adaptive_filtered[::M, ::M]

    # ── 步骤 4：对比基准 ─────────────────────────────
    direct_ds  = downsample(img, M)
    uniform_ds = gaussian_downsample(img, sigma_theory, M)

    h = min(adaptive_ds.shape[0], uniform_ds.shape[0])
    w = min(adaptive_ds.shape[1], uniform_ds.shape[1])

    err_direct   = np.abs(direct_ds[:h,  :w].astype(float) - uniform_ds[:h, :w].astype(float))
    err_adaptive = np.abs(adaptive_ds[:h, :w].astype(float) - uniform_ds[:h, :w].astype(float))

    mse_direct   = np.mean(err_direct**2)
    mse_adaptive = np.mean(err_adaptive**2)

    # ── 可视化 ───────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    fig.suptitle(f"第三部分：自适应下采样  (M={M})", fontsize=15, fontweight='bold')

    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("原始 Chirp 图 (256×256)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(grad_mag, cmap='jet')
    axes[0, 1].set_title("梯度幅度图（局部频率指示器）")
    axes[0, 1].axis('off')

    im = axes[0, 2].imshow(sigma_map, cmap='viridis', vmin=sigma_min, vmax=sigma_max)
    axes[0, 2].set_title(f"自适应 σ 图  [{sigma_min:.1f}, {sigma_max:.1f}]")
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(direct_ds, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f"直接下采样 ({direct_ds.shape[0]}×{direct_ds.shape[1]})")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(uniform_ds, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title(f"统一高斯下采样  σ={sigma_theory:.1f}")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(adaptive_ds, cmap='gray', vmin=0, vmax=255)
    axes[1, 2].set_title("自适应高斯下采样")
    axes[1, 2].axis('off')

    axes[2, 0].imshow(err_direct, cmap='hot')
    axes[2, 0].set_title(f"直接 vs 统一  MSE={mse_direct:.3f}")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(err_adaptive, cmap='hot')
    axes[2, 1].set_title(f"自适应 vs 统一  MSE={mse_adaptive:.3f}")
    axes[2, 1].axis('off')

    axes[2, 2].hist(err_direct.flatten(),   bins=60, alpha=0.6, label=f"直接下采样  MSE={mse_direct:.2f}",   color='red',  density=True)
    axes[2, 2].hist(err_adaptive.flatten(), bins=60, alpha=0.6, label=f"自适应下采样 MSE={mse_adaptive:.2f}", color='blue', density=True)
    axes[2, 2].set_title("误差分布对比（以统一高斯为参考）")
    axes[2, 2].set_xlabel("误差值")
    axes[2, 2].set_ylabel("概率密度")
    axes[2, 2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("part3_adaptive_downsampling.png", dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n误差统计（均以统一高斯下采样 σ={sigma_theory} 为参考基准）：")
    print(f"  直接下采样   MSE = {mse_direct:.4f}")
    print(f"  自适应下采样 MSE = {mse_adaptive:.4f}")
    improvement = (mse_direct - mse_adaptive) / mse_direct * 100
    print(f"  自适应下采样相比直接下采样改善了 {improvement:.1f}%")
    print(">>> 已保存：part3_adaptive_downsampling.png\n")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("图像处理作业：下采样与抗混叠实验")
    print("=" * 60)

    M = 4  # 全局下采样因子

    part1_aliasing_demo(M=M)
    part2_sigma_verification(M=M)
    part3_adaptive_downsampling(M=M)

    print("全部实验完成！输出图像已保存在当前目录。")

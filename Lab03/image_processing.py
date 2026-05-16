#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像缩小、恢复与频域分析
=====================================
使用 OpenCV 和 NumPy 完成图像处理实验
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import warnings

warnings.filterwarnings("ignore")

# 设置 matplotlib 中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# 1. 图像生成与预处理
# ============================================================================


def create_test_image(size=512):
    """
    创建测试图像 - 包含各种频率成分的灰度图像
    用于观察下采样和插值的效果
    """
    img = np.zeros((size, size), dtype=np.uint8)

    # 添加低频成分 - 渐变背景
    x = np.linspace(0, 4 * np.pi, size)
    y = np.linspace(0, 4 * np.pi, size)
    X, Y = np.meshgrid(x, y)

    # 低频正弦波
    img += (127 * (1 + np.sin(X / 2) * np.cos(Y / 2))).astype(np.uint8) // 4

    # 添加高频成分 - 精细纹理
    high_freq = (63 * (1 + np.sin(X * 4) * np.cos(Y * 4))).astype(np.uint8)
    img = cv2.add(img, high_freq // 4)

    # 添加几何图形（边缘信息）
    cv2.circle(img, (size // 3, size // 3), size // 6, 200, -1)
    cv2.rectangle(img, (size // 2, size // 2), (size * 3 // 4, size * 3 // 4), 50, -1)
    cv2.line(img, (0, 0), (size, size), 255, 3)

    return img


def load_image(path):
    """
    加载灰度图像
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法加载图像: {path}")
    return img


# ============================================================================
# 2. 下采样
# ============================================================================


def downsample(image, scale=0.5, use_gaussian=False):
    """
    下采样图像

    Args:
        image: 输入图像
        scale: 缩放比例 (0.5=1/2, 0.25=1/4)
        use_gaussian: 是否先进行高斯平滑

    Returns:
        缩小后的图像
    """
    if use_gaussian:
        # 高斯平滑，减少混叠效应
        # 核大小根据缩放比例选择
        kernel_size = int(1 / scale) * 2 + 1
        smoothed = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        result = cv2.resize(smoothed, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        # 直接缩小（可能产生混叠）
        result = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return result


# ============================================================================
# 3. 图像恢复（插值）
# ============================================================================


def restore_image(downsampled, original_shape, method="bilinear"):
    """
    使用指定内插方法恢复图像到原始尺寸

    Args:
        downsampled: 缩小后的图像
        original_shape: 原始图像尺寸 (height, width)
        method: 插值方法 - 'nearest', 'bilinear', 'bicubic'

    Returns:
        恢复后的图像
    """
    interpolation_methods = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}  # 最近邻  # 双线性  # 双三次

    if method not in interpolation_methods:
        raise ValueError(f"未知的插值方法: {method}")

    restored = cv2.resize(downsampled, (original_shape[1], original_shape[0]), interpolation=interpolation_methods[method])

    return restored


# ============================================================================
# 4. 空间域比较指标
# ============================================================================


def calculate_mse(original, restored):
    """
    计算均方误差 (Mean Square Error)
    """
    mse = np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2)
    return mse


def calculate_psnr(original, restored):
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio)
    """
    mse = calculate_mse(original, restored)
    if mse == 0:
        return float("inf")  # 图像完全相同

    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel**2) / mse)
    return psnr


def calculate_ssim(original, restored):
    """
    计算结构相似性指数 (Structural Similarity Index)
    简单实现版本
    """
    # 转换为浮点数
    img1 = original.astype(np.float64)
    img2 = restored.astype(np.float64)

    # 常数
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # 计算均值
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


# ============================================================================
# 5. 傅里叶变换分析
# ============================================================================


def fft_analysis(image):
    """
    二维傅里叶变换分析

    步骤：
    1. 进行二维 FFT
    2. 将频谱中心移动到图像中心 (fftshift)
    3. 计算幅度谱并取对数显示

    Returns:
        magnitude_spectrum: 对数幅度谱（用于显示）
        fft_shifted: 中心化后的频谱（用于分析）
    """
    # 转换为浮点数
    f = np.fft.fft2(image.astype(np.float64))

    # 将低频移到中心
    fshift = np.fft.fftshift(f)

    # 计算幅度谱并取对数
    magnitude_spectrum = 20 * np.log10(np.abs(fshift) + 1)

    return magnitude_spectrum, fshift


def analyze_high_frequency(fft_shifted, radius_ratio=0.1):
    """
    分析高频成分比例

    Args:
        fft_shifted: 中心化后的频谱
        radius_ratio: 低频区域半径比例（相对于图像尺寸）

    Returns:
        high_freq_energy: 高频能量占比
        low_freq_energy: 低频能量占比
    """
    h, w = fft_shifted.shape
    center_y, center_x = h // 2, w // 2
    radius = int(min(h, w) * radius_ratio)

    # 创建圆形掩码，中心为低频区域
    Y, X = np.ogrid[:h, :w]
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius**2

    # 计算能量
    energy = np.abs(fft_shifted) ** 2
    low_freq_energy = np.sum(energy[mask])
    total_energy = np.sum(energy)
    high_freq_energy = total_energy - low_freq_energy

    if total_energy > 0:
        return high_freq_energy / total_energy, low_freq_energy / total_energy
    return 0, 0


# ============================================================================
# 6. DCT 分析
# ============================================================================


def dct2(image):
    """
    二维 DCT 变换
    """
    # scipy 的 dct 默认是沿着最后一个轴进行
    # 先对行进行 DCT
    tmp = dct(image.astype(np.float64), type=2, norm="ortho", axis=0)
    # 再对列进行 DCT
    result = dct(tmp, type=2, norm="ortho", axis=1)
    return result


def idct2(coeff):
    """
    二维逆 DCT 变换
    """
    tmp = idct(coeff, type=2, norm="ortho", axis=0)
    result = idct(tmp, type=2, norm="ortho", axis=1)
    return result


def dct_analysis(image, low_freq_size=8):
    """
    DCT 分析

    Args:
        image: 输入图像
        low_freq_size: 低频区域大小（左上角 low_freq_size x low_freq_size）

    Returns:
        dct_coeff: DCT 系数
        low_freq_ratio: 低频能量占比
        dct_log: 对数幅度（用于显示）
    """
    dct_coeff = dct2(image)

    # 计算低频能量占比
    total_energy = np.sum(dct_coeff**2)
    low_freq_energy = np.sum(dct_coeff[:low_freq_size, :low_freq_size] ** 2)

    if total_energy > 0:
        low_freq_ratio = low_freq_energy / total_energy
    else:
        low_freq_ratio = 0

    # 对数幅度谱（便于显示）
    dct_log = 20 * np.log10(np.abs(dct_coeff) + 1)

    return dct_coeff, low_freq_ratio, dct_log


# ============================================================================
# 7. 可视化工具
# ============================================================================


def save_comparison_figure(images, titles, filename, suptitle=""):
    """
    保存对比图像
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  已保存: {filename}")
    plt.close()


def save_spectrum_figure(original, downsampled, restored, orig_fft, down_fft, rest_fft, filename, title_suffix=""):
    """
    保存频谱对比图
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    images_row = [original, downsampled, restored]
    fft_row = [orig_fft, down_fft, rest_fft]
    labels = ["Original", "Downsampled", "Restored"]

    # 第一行：原始图像
    for ax, img, label in zip(axes[0], images_row, labels):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{label}", fontsize=12)
        ax.axis("off")

    # 第二行：频谱图
    for ax, fft_img, label in zip(axes[1], fft_row, labels):
        im = ax.imshow(fft_img, cmap="hot")
        ax.set_title(f"{label} FFT Spectrum", fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"FFT Analysis {title_suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  已保存: {filename}")
    plt.close()


def save_dct_figure(dct_results, filename, title_suffix=""):
    """
    保存 DCT 对比图
    """
    n = len(dct_results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))

    if n == 1:
        axes = axes.reshape(-1, 1)

    labels = list(dct_results.keys())

    for i, (label, (dct_log, low_freq_ratio)) in enumerate(dct_results.items()):
        # 第一行：DCT 系数图
        im1 = axes[0, i].imshow(dct_log, cmap="viridis", aspect="auto")
        axes[0, i].set_title(f"{label}\nLow-freq Energy: {low_freq_ratio:.2%}", fontsize=10)
        axes[0, i].axis("off")
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

        # 第二行：DCT 能量分布（按频率）
        # 将 DCT 系数按频率分组
        dct_coeff = np.exp(dct_log / 20) - 1  # 还原近似系数
        h, w = dct_coeff.shape

        # 计算径向频率分布
        freqs = []
        energies = []

        for r in range(min(h, w) // 2):
            mask = np.zeros((h, w), dtype=bool)
            for y in range(h):
                for x in range(w):
                    if int(np.sqrt((y) ** 2 + (x) ** 2)) == r:
                        mask[y, x] = True
            if np.sum(mask) > 0:
                freqs.append(r)
                energies.append(np.sum(dct_coeff[mask] ** 2))

        axes[1, i].plot(freqs, energies, "b-", linewidth=2)
        axes[1, i].set_xlabel("Frequency", fontsize=10)
        axes[1, i].set_ylabel("Energy", fontsize=10)
        axes[1, i].set_title("DCT Energy Distribution", fontsize=10)
        axes[1, i].grid(True, alpha=0.3)

    fig.suptitle(f"DCT Analysis {title_suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  已保存: {filename}")
    plt.close()


# ============================================================================
# 8. 主程序
# ============================================================================


def main():
    """
    主函数：执行所有实验步骤
    """
    print("=" * 60)
    print("图像缩小、恢复与频域分析")
    print("=" * 60)

    # 创建输出目录
    output_dir = "output"
    os.makedirs(f"{output_dir}/spatial_comparison", exist_ok=True)
    os.makedirs(f"{output_dir}/fft_analysis", exist_ok=True)
    os.makedirs(f"{output_dir}/dct_analysis", exist_ok=True)

    # ========================================================================
    # 1. 创建/加载测试图像
    # ========================================================================
    print("\n[1] 创建测试图像...")
    # original = create_test_image(size=512)
    original = load_image("7.png")
    cv2.imwrite(f"{output_dir}/original.png", original)
    print(f"  原始图像尺寸: {original.shape}")

    # ========================================================================
    # 2. 下采样 (1/2 和 1/4)
    # ========================================================================
    print("\n[2] 图像下采样...")

    results = {}

    for scale in [0.5, 0.25]:
        scale_name = f"1_{int(1/scale)}"
        print(f"\n  --- 缩放比例: {scale_name} ---")

        results[scale_name] = {}

        # 2a. 不做预滤波直接缩小
        down_direct = downsample(original, scale=scale, use_gaussian=False)
        print(f"    直接缩小: {down_direct.shape}")

        # 2b. 高斯平滑后缩小
        down_gaussian = downsample(original, scale=scale, use_gaussian=True)
        print(f"    高斯平滑后缩小: {down_gaussian.shape}")

        results[scale_name]["direct"] = down_direct
        results[scale_name]["gaussian"] = down_gaussian

        # 保存缩小后的图像
        cv2.imwrite(f"{output_dir}/downsampled_{scale_name}_direct.png", down_direct)
        cv2.imwrite(f"{output_dir}/downsampled_{scale_name}_gaussian.png", down_gaussian)

    # ========================================================================
    # 3. 图像恢复 (三种插值方法)
    # ========================================================================
    print("\n[3] 图像恢复（插值）...")

    interpolation_methods = ["nearest", "bilinear", "bicubic"]

    for scale_name in results.keys():
        print(f"\n  --- 缩放比例: {scale_name} ---")

        for prefilter in ["direct", "gaussian"]:
            print(f"    预滤波: {prefilter}")
            downsampled = results[scale_name][prefilter]

            for method in interpolation_methods:
                restored = restore_image(downsampled, original.shape, method)
                results[scale_name][f"{prefilter}_{method}"] = restored

                mse = calculate_mse(original, restored)
                psnr = calculate_psnr(original, restored)

                print(f"      {method:10s}: MSE={mse:8.2f}, PSNR={psnr:6.2f} dB")

    # ========================================================================
    # 4. 空间域比较可视化
    # ========================================================================
    print("\n[4] 生成空间域比较图...")

    # 对每个缩放比例和预滤波组合生成对比图
    for scale_name in results.keys():
        for prefilter in ["direct", "gaussian"]:
            # 获取缩小后的图像
            downsampled = results[scale_name][prefilter]

            # 收集三种插值结果
            restored_images = [original, downsampled]
            titles = ["Original", f"Downsampled\n({prefilter})"]

            for method in interpolation_methods:
                restored = results[scale_name][f"{prefilter}_{method}"]
                restored_images.append(restored)
                mse = calculate_mse(original, restored)
                psnr = calculate_psnr(original, restored)
                titles.append(f"{method.capitalize()}\nMSE={mse:.1f}\nPSNR={psnr:.1f}dB")

            # 保存对比图
            filename = f"{output_dir}/spatial_comparison/comparison_{scale_name}_{prefilter}.png"
            save_comparison_figure(restored_images, titles, filename, suptitle=f"Spatial Domain Comparison ({scale_name}, {prefilter})")

    # ========================================================================
    # 5. 傅里叶变换分析
    # ========================================================================
    print("\n[5] 傅里叶变换分析...")

    # 对原图进行 FFT
    orig_fft, orig_fft_shifted = fft_analysis(original)
    orig_high_freq, orig_low_freq = analyze_high_frequency(orig_fft_shifted)
    print(f"  原图 - 高频能量占比: {orig_high_freq:.2%}, 低频能量占比: {orig_low_freq:.2%}")

    # 对每个配置进行 FFT 分析（以双线性插值为例）
    for scale_name in results.keys():
        for prefilter in ["direct", "gaussian"]:
            downsampled = results[scale_name][prefilter]
            restored = results[scale_name][f"{prefilter}_bilinear"]

            # 将缩小图插值到原图尺寸以便比较
            downsampled_resized = cv2.resize(downsampled, original.shape[::-1], interpolation=cv2.INTER_LINEAR)

            # FFT 分析
            down_fft, down_fft_shifted = fft_analysis(downsampled_resized)
            rest_fft, rest_fft_shifted = fft_analysis(restored)

            # 高频成分分析
            down_high_freq, down_low_freq = analyze_high_frequency(down_fft_shifted)
            rest_high_freq, rest_low_freq = analyze_high_frequency(rest_fft_shifted)

            print(f"\n  --- {scale_name}, {prefilter} ---")
            print(f"    缩小图 - 高频占比: {down_high_freq:.2%}")
            print(f"    恢复图 - 高频占比: {rest_high_freq:.2%}")

            # 保存频谱对比图
            filename = f"{output_dir}/fft_analysis/fft_{scale_name}_{prefilter}.png"
            save_spectrum_figure(
                original, downsampled_resized, restored, orig_fft, down_fft, rest_fft, filename, title_suffix=f"({scale_name}, {prefilter})"
            )

    print("\n  高频成分差异分析:")
    print("  - 缩小后的图像高频成分会减少（混叠效应）")
    print("  - 使用高斯预滤波可以减少混叠")
    print("  - 插值恢复可以部分重建高频成分，但无法完全恢复丢失的信息")

    # ========================================================================
    # 6. DCT 分析
    # ========================================================================
    print("\n[6] DCT 分析...")

    # 对原图进行 DCT
    orig_dct, orig_low_ratio, orig_dct_log = dct_analysis(original, low_freq_size=16)
    print(f"  原图低频能量占比 (16x16): {orig_low_ratio:.2%}")

    # 对每个配置进行 DCT 分析
    for scale_name in results.keys():
        for prefilter in ["direct", "gaussian"]:
            dct_results = {"Original": (orig_dct_log, orig_low_ratio)}

            for method in interpolation_methods:
                restored = results[scale_name][f"{prefilter}_{method}"]
                _, low_ratio, dct_log = dct_analysis(restored, low_freq_size=16)
                dct_results[method.capitalize()] = (dct_log, low_ratio)

                print(f"  {scale_name}, {prefilter}, {method}: 低频占比={low_ratio:.2%}")

            # 保存 DCT 对比图
            filename = f"{output_dir}/dct_analysis/dct_{scale_name}_{prefilter}.png"
            save_dct_figure(dct_results, filename, title_suffix=f"({scale_name}, {prefilter})")

    print("\n  DCT 能量分布分析:")
    print("  - 自然图像的大部分能量集中在低频区域")
    print("  - 下采样会丢失高频 DCT 系数")
    print("  - 双三次插值通常比最近邻和双线性保留更多的频率信息")

    # ========================================================================
    # 7. 生成汇总表格
    # ========================================================================
    print("\n[7] 结果汇总...")
    print("\n" + "=" * 80)
    print("MSE 和 PSNR 汇总表")
    print("=" * 80)

    print(f"\n{'Scale':<10} {'Prefilter':<10} {'Method':<10} {'MSE':<12} {'PSNR (dB)':<12}")
    print("-" * 60)

    for scale_name in results.keys():
        for prefilter in ["direct", "gaussian"]:
            for method in interpolation_methods:
                restored = results[scale_name][f"{prefilter}_{method}"]
                mse = calculate_mse(original, restored)
                psnr = calculate_psnr(original, restored)
                print(f"{scale_name:<10} {prefilter:<10} {method:<10} {mse:<12.2f} {psnr:<12.2f}")

    print("\n" + "=" * 80)
    print("分析结论:")
    print("=" * 80)
    print("1. 下采样会丢失高频信息，导致图像模糊")
    print("2. 使用高斯预滤波可以减少混叠效应")
    print("3. 双三次插值通常能获得最好的恢复质量（最低的 MSE，最高的 PSNR）")
    print("4. 傅里叶变换显示缩小后的图像高频成分显著减少")
    print("5. DCT 分析表明插值恢复无法完全重建原始图像的高频成分")
    print("=" * 80)

    # 显示最终结果图
    print("\n[8] 显示部分结果（按任意键关闭窗口）...")

    # 显示原始图像和最佳恢复结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Original", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(results["1_2"]["direct"], cmap="gray")
    axes[0, 1].set_title("Downsampled 1/2 (direct)", fontsize=12)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(results["1_4"]["direct"], cmap="gray")
    axes[0, 2].set_title("Downsampled 1/4 (direct)", fontsize=12)
    axes[0, 2].axis("off")

    # 显示三种插值方法（以 1/2 缩小为例）
    for i, method in enumerate(interpolation_methods):
        restored = results["1_2"][f"direct_{method}"]
        mse = calculate_mse(original, restored)
        psnr = calculate_psnr(original, restored)
        axes[1, i].imshow(restored, cmap="gray")
        axes[1, i].set_title(f"{method.capitalize()}\nMSE={mse:.1f}, PSNR={psnr:.1f}dB", fontsize=12)
        axes[1, i].axis("off")

    plt.suptitle("Image Processing Results Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n✓ 所有结果已保存到 output/ 目录")


if __name__ == "__main__":
    main()

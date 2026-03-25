# 实验一：Python视觉开发环境搭建与图像基本读写
# ===============================================
"""
任务列表：
1. 使用OpenCV读取一张测试图片
2. 输出图像基本信息（尺寸、通道数、数据类型）
3. 显示原图（使用Matplotlib或OpenCV）
4. 转换为灰度图并显示
5. 保存处理结果（灰度图为新文件）
6. 用NumPy做简单操作（输出像素值、裁剪区域）
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse  # 添加argparse模块用于命令行参数解析


def parse_arguments():
    """
    解析命令行参数

    支持以下参数:
        --image, -i: 指定输入图片路径
        --output, -o: 指定输出目录
        --no-display: 不显示图像（在无GUI环境使用）
    """
    parser = argparse.ArgumentParser(
        description='实验一：Python视觉开发环境搭建与图像基本读写',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 使用默认测试图片
  python exp1_image_processing.py

  # 指定自定义图片
  python exp1_image_processing.py -i path/to/your/image.jpg

  # 指定输出目录
  python exp1_image_processing.py -i image.jpg -o my_results/

  # 无GUI模式运行（不显示图像）
  python exp1_image_processing.py -i image.jpg --no-display
        '''
    )

    parser.add_argument(
        '--image', '-i',
        type=str,
        default='images/test_image.jpg',
        help='输入图片路径 (默认: images/test_image.jpg)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='输出目录路径 (默认: results)'
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        help='不显示图像，只保存结果（适用于无GUI环境）'
    )

    return parser.parse_args()


def task1_read_image(image_path):
    """
    任务1：使用OpenCV读取一张测试图片

    参数:
        image_path: 图片文件路径

    返回:
        img: OpenCV读取的图像（BGR格式）
    """
    # 使用cv2.imread()读取图像，默认以BGR格式加载
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}，请检查路径是否正确")

    print(f"[任务1] 成功读取图片: {image_path}")
    return img


def task2_print_info(img):
    """
    任务2：输出图像基本信息

    参数:
        img: OpenCV图像对象
    """
    print("\n" + "=" * 50)
    print("[任务2] 图像基本信息")
    print("=" * 50)

    # 获取图像形状 (高度, 宽度, 通道数)
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1

    # 获取数据类型
    dtype = img.dtype

    print(f"图像尺寸 (宽 x 高): {width} x {height} 像素")
    print(f"图像通道数: {channels}")
    print(f"数据类型: {dtype}")
    print(f"数组形状 (shape): {img.shape}")
    print(f"像素总数: {img.size}")
    print("=" * 50 + "\n")


def task3_show_original(img, output_dir='results'):
    """
    任务3：显示原图

    使用Matplotlib显示原图（需要将BGR转换为RGB）

    参数:
        img: OpenCV图像对象(BGR格式)
        output_dir: 输出目录路径
    """
    print("[任务3] 显示原图")

    # 创建图形窗口
    plt.figure(figsize=(10, 8))

    # OpenCV读取的是BGR格式，需要转换为RGB格式供Matplotlib显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 显示图像
    plt.imshow(img_rgb)
    plt.title("Original Image (RGB)")
    plt.axis('off')  # 不显示坐标轴
    plt.tight_layout()

    # 保存显示结果
    output_path = os.path.join(output_dir, '01_original.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  原图已保存到: {output_path}")

    plt.show()


def task3_save_original(img, output_dir='results'):
    """
    任务3（无GUI模式）：保存原图而不显示

    参数:
        img: OpenCV图像对象(BGR格式)
        output_dir: 输出目录路径
    """
    print("[任务3] 保存原图（无显示模式）")

    output_path = os.path.join(output_dir, '01_original.png')

    # 将BGR转换为RGB保存
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imsave(output_path, img_rgb)

    print(f"  原图已保存到: {output_path}")


def task4_convert_to_grayscale(img, output_dir='results', no_display=False):
    """
    任务4：转换为灰度图

    参数:
        img: 彩色图像(BGR格式)
        output_dir: 输出目录路径
        no_display: 是否不显示图像

    返回:
        gray: 灰度图像
    """
    print("[任务4] 转换为灰度图")

    # 使用cv2.cvtColor将BGR图像转换为灰度图
    # cv2.COLOR_BGR2GRAY: 从BGR颜色空间转换到灰度空间
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 显示灰度图
    plt.figure(figsize=(10, 8))
    plt.imshow(gray, cmap='gray')  # 使用灰度颜色映射
    plt.title("Grayscale Image")
    plt.axis('off')
    plt.tight_layout()

    # 保存显示结果
    output_path = os.path.join(output_dir, '02_grayscale.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  灰度图已保存到: {output_path}")

    if not no_display:
        plt.show()
    else:
        plt.close()

    return gray


def task5_save_result(gray_img, output_path='results/gray_output.jpg'):
    """
    任务5：保存处理结果

    参数:
        gray_img: 灰度图像
        output_path: 输出文件路径
    """
    print("[任务5] 保存处理结果")

    # 使用cv2.imwrite保存图像
    # 参数: 文件路径, 图像数据
    success = cv2.imwrite(output_path, gray_img)

    if success:
        print(f"  成功保存灰度图到: {output_path}")
    else:
        print(f"  保存失败: {output_path}")

    return success


def task6_numpy_operations(img, gray_img, output_dir='results', no_display=False):
    """
    任务6：用NumPy做简单操作

    包括：
    1. 输出指定位置的像素值
    2. 裁剪图像左上角区域
    3. 图像基本统计信息

    参数:
        img: 原始彩色图像
        gray_img: 灰度图像
        output_dir: 输出目录路径
        no_display: 是否不显示图像
    """
    print("\n" + "=" * 50)
    print("[任务6] NumPy简单操作")
    print("=" * 50)

    # 6.1 输出特定位置的像素值
    print("\n[6.1] 像素值查看:")

    # 获取图像中心点坐标
    h, w = gray_img.shape
    center_y, center_x = h // 2, w // 2

    # 输出中心点的灰度值
    center_pixel = gray_img[center_y, center_x]
    print(f"  中心点 ({center_x}, {center_y}) 的灰度值: {center_pixel}")

    # 输出左上角像素值
    top_left_pixel = gray_img[0, 0]
    print(f"  左上角 (0, 0) 的灰度值: {top_left_pixel}")

    # 如果是彩色图像，输出某个像素的BGR值
    b, g, r = img[center_y, center_x]
    print(f"  中心点 ({center_x}, {center_y}) 的BGR值: B={b}, G={g}, R={r}")

    # 6.2 裁剪图像左上角区域
    print("\n[6.2] 图像裁剪:")

    # 裁剪左上角 200x200 区域
    # NumPy切片语法: [y_start:y_end, x_start:x_end]
    crop_size = min(200, h // 2, w // 2)  # 确保不超出图像边界
    cropped = img[0:crop_size, 0:crop_size]

    print(f"  裁剪区域: 左上角 {crop_size}x{crop_size} 像素")
    print(f"  裁剪后数组形状: {cropped.shape}")

    # 保存裁剪结果
    crop_output = os.path.join(output_dir, '03_cropped.png')
    cv2.imwrite(crop_output, cropped)
    print(f"  裁剪图像已保存到: {crop_output}")

    # 显示裁剪结果
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title(f"Cropped Region ({crop_size}x{crop_size})")
    plt.axis('off')
    plt.tight_layout()
    crop_display_path = os.path.join(output_dir, '04_cropped_display.png')
    plt.savefig(crop_display_path, dpi=150, bbox_inches='tight')

    if not no_display:
        plt.show()
    else:
        plt.close()

    # 6.3 图像基本统计信息
    print("\n[6.3] 图像统计信息:")
    print(f"  灰度图最小值: {gray_img.min()}")
    print(f"  灰度图最大值: {gray_img.max()}")
    print(f"  灰度图平均值: {gray_img.mean():.2f}")
    print(f"  灰度图标准差: {gray_img.std():.2f}")

    # 6.4 简单的图像操作：反转
    print("\n[6.4] 图像反转操作:")
    inverted = 255 - gray_img  # 像素值反转
    inverted_path = os.path.join(output_dir, '05_inverted.png')
    cv2.imwrite(inverted_path, inverted)
    print(f"  反转图像已保存到: {inverted_path}")

    # 显示反转结果
    plt.figure(figsize=(10, 8))
    plt.imshow(inverted, cmap='gray')
    plt.title("Inverted Image")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(inverted_path, dpi=150, bbox_inches='tight')

    if not no_display:
        plt.show()
    else:
        plt.close()

    print("=" * 50 + "\n")


def create_sample_image():
    """
    创建示例测试图片（如果用户没有提供图片）

    返回:
        sample_path: 示例图片路径
    """
    # 创建一个彩色渐变图像作为测试图片
    print("[准备] 创建示例测试图片...")

    height, width = 600, 800

    # 创建渐变图像
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 水平渐变
    for i in range(width):
        img[:, i, 0] = int(255 * i / width)  # B通道 - 蓝色渐变
        img[:, i, 1] = int(255 * (1 - i / width))  # G通道 - 绿色渐变

    # 垂直渐变到红色
    for j in range(height):
        img[j, :, 2] = int(255 * j / height)  # R通道 - 红色渐变

    # 添加一些形状
    # 画一个矩形
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)

    # 画一个圆
    cv2.circle(img, (600, 400), 100, (0, 255, 255), -1)

    # 添加文字
    cv2.putText(img, 'OpenCV Test', (250, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    sample_path = 'images/sample_test.jpg'
    cv2.imwrite(sample_path, img)
    print(f"  示例图片已保存到: {sample_path}")

    return sample_path


def main():
    """
    主函数：执行所有任务

    使用argparse解析命令行参数，支持自定义图片路径和输出目录
    """
    # 解析命令行参数
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("  实验一：Python视觉开发环境搭建与图像基本读写")
    print("=" * 60 + "\n")

    # 输出参数信息
    print("参数配置:")
    print(f"  输入图片: {args.image}")
    print(f"  输出目录: {args.output}")
    print(f"  显示图像: {'否' if args.no_display else '是'}")
    print()

    # 创建必要的目录
    os.makedirs('images', exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    # 定义测试图片路径
    test_image_path = args.image

    # 检查是否存在测试图片，如果不存在则创建示例图片
    if not os.path.exists(test_image_path):
        print(f"未找到测试图片: {test_image_path}")
        print("将创建示例图片进行测试...\n")
        test_image_path = create_sample_image()

    try:
        # 任务1: 读取图片
        img = task1_read_image(test_image_path)

        # 任务2: 输出图像基本信息
        task2_print_info(img)

        # 任务3: 显示原图
        if not args.no_display:
            task3_show_original(img, args.output)
        else:
            # 无GUI模式，只保存不显示
            task3_save_original(img, args.output)

        # 任务4: 转换为灰度图
        gray_img = task4_convert_to_grayscale(img, args.output, args.no_display)

        # 任务5: 保存灰度图
        gray_output_path = os.path.join(args.output, 'gray_output.jpg')
        task5_save_result(gray_img, gray_output_path)

        # 任务6: NumPy操作
        task6_numpy_operations(img, gray_img, args.output, args.no_display)

        print("\n" + "=" * 60)
        print("  所有任务已完成！")
        print(f"  结果文件保存在 '{args.output}/' 目录下")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n错误: {e}")
        raise


if __name__ == '__main__':
    main()

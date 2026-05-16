# 实验一：C++ OpenCV 图像处理实验

使用C++和OpenCV完成图像基本读写操作。

## 实验要求（对应6个任务）

| 任务 | 内容 | 实现函数 |
|------|------|----------|
| 任务1 | 读取测试图片 | `cv::imread()` |
| 任务2 | 输出图像基本信息 | `Mat.cols`, `Mat.rows`, `Mat.channels()`, `Mat.depth()` |
| 任务3 | 显示原图 | `cv::imshow()`, `cv::waitKey()` |
| 任务4 | 转换为灰度图 | `cv::cvtColor()` |
| 任务5 | 保存处理结果 | `cv::imwrite()` |
| 任务6 | 像素操作 | `Mat.at<>()`, `Mat(Rect)` |

## 环境依赖

### 1. 安装OpenCV C++库

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libopencv-dev

# 验证安装
pkg-config --modversion opencv4
```

### 2. 编译工具

```bash
sudo apt install cmake g++
```

## 项目结构

```
exp1_image_processing/
├── CMakeLists.txt              # CMake配置文件
├── build.sh                    # 自动构建脚本
├── run.sh                      # 运行脚本
├── exp1_image_processing.cpp   # 源代码
├── images/
│   └── test_image.jpg          # 测试图片
└── output/                     # 输出目录
    ├── 01_original.jpg         # 原图副本
    ├── 02_grayscale.jpg        # 灰度图
    ├── gray_image.jpg          # 任务5输出
    ├── cropped_image.jpg       # 任务6裁剪输出
    └── modified_image.jpg      # 中心红色方块图
```

## 编译运行

### 方式一：使用脚本（推荐）

```bash
# 构建
./build.sh

# 运行
./run.sh
```

### 方式二：手动编译

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
./exp1_image_processing
```

## 输出结果示例

```
===== 任务1: 成功读取图片 =====

===== 任务2: 图像基本信息 =====
图像宽度 (Width): 227 像素
图像高度 (Height): 148 像素
图像通道数 (Channels): 3
图像数据类型 (Data Type): CV_8U (8位无符号整数)
图像总像素数: 33596
图像元素大小: 3 字节/像素

===== 任务3: 显示原图 =====
[跳过] 无GUI环境，原图已保存到 output/01_original.jpg

===== 任务4: 转换为灰度图 =====
成功转换为灰度图
灰度图尺寸: 227x148
灰度图通道数: 1

===== 任务5: 保存处理结果 =====
灰度图已保存到: ../output/gray_image.jpg

===== 任务6: 简单像素操作 =====
中心点像素值 (113, 74): BGR(95, 123, 153)
灰度图中心像素值: 129

裁剪左上角区域:
裁剪区域: (0, 0) 到 (75, 49)
裁剪后尺寸: 75x49
裁剪图片已保存到: ../output/cropped_image.jpg

===== 额外: 像素值修改示例 =====
在图像中心画了红色方块
修改后的图片已保存到: ../output/modified_image.jpg

===== 所有任务完成! =====
```

## 关键代码说明

### 1. 读取图像
```cpp
Mat image = imread("../images/test_image.jpg");
if (image.empty()) {
    cerr << "无法加载图片" << endl;
    return -1;
}
```

### 2. 图像信息获取
```cpp
cout << "宽度: " << image.cols << endl;
cout << "高度: " << image.rows << endl;
cout << "通道数: " << image.channels() << endl;
cout << "数据类型: " << image.depth() << endl;
```

### 3. 灰度转换
```cpp
Mat grayImage;
cvtColor(image, grayImage, COLOR_BGR2GRAY);
```

### 4. 像素访问
```cpp
// 访问彩色图像像素
Vec3b pixel = image.at<Vec3b>(y, x);

// 访问灰度图像像素
uchar grayPixel = grayImage.at<uchar>(y, x);
```

### 5. 图像裁剪
```cpp
Rect roi(0, 0, cropWidth, cropHeight);
Mat croppedImage = image(roi).clone();
```

### 6. 保存图像
```cpp
imwrite("output/gray_image.jpg", grayImage);
```

## 提交GitHub的文件清单

根据实验要求，需要提交：

1. **源代码**: `exp1_image_processing.cpp`, `CMakeLists.txt`
2. **原始图片**: `images/test_image.jpg`
3. **处理结果图片**:
   - `output/gray_image.jpg` (灰度图)
   - `output/cropped_image.jpg` (裁剪图)
   - `output/modified_image.jpg` (修改后的图)
4. **README**: 本文件
5. **实验报告**: 课后提交

## 注意事项

1. **GUI环境**: 如果在WSL或无显示器环境运行，程序会自动检测并跳过窗口显示，仅保存图片文件
2. **路径问题**: 程序支持从`build`目录或项目根目录运行，会自动切换路径
3. **依赖版本**: 代码基于OpenCV 4.x编写，与OpenCV 3.x API兼容

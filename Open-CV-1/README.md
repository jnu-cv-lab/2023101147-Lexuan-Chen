# 实验一：Python视觉开发环境搭建与图像基本读写

## 项目简介

本项目是计算机视觉入门实验，涵盖Python视觉开发环境搭建与图像基本读写操作。

## 实验任务

1. **使用OpenCV读取一张测试图片**
2. **输出图像基本信息**（尺寸、通道数、数据类型）
3. **显示原图**（使用Matplotlib）
4. **转换为灰度图**并显示
5. **保存处理结果**（灰度图）
6. **用NumPy做简单操作**（输出像素值、裁剪区域）

## 项目结构

```
exp1_image_processing/
├── exp1_image_processing.py    # 主程序源代码
├── environment.yml              # Conda环境配置文件
├── images/                      # 测试图片目录
│   ├── test_image.jpg          # 用户提供的测试图片
│   └── sample_test.jpg         # 自动生成的示例图片
├── results/                     # 处理结果输出目录
│   ├── 01_original.png         # 原图显示结果
│   ├── 02_grayscale.png        # 灰度图显示结果
│   ├── 03_cropped.png          # 裁剪结果
│   ├── 04_cropped_display.png  # 裁剪结果显示
│   ├── 05_inverted.png         # 反转图像
│   └── gray_output.jpg         # 保存的灰度图
└── README.md                    # 项目说明文档
```

## 环境搭建

### 方法一：使用Conda创建环境

```bash
# 1. 创建conda环境
conda env create -f environment.yml

# 2. 激活环境
conda activate cv_exp1

# 3. 验证安装
python -c "import cv2; import numpy; import matplotlib; print('所有依赖安装成功！')"
```

### 方法二：手动安装依赖

```bash
# 创建新环境
conda create -n cv_exp1 python=3.10 -y

# 激活环境
conda activate cv_exp1

# 安装依赖包
conda install numpy matplotlib opencv jupyter -y
```

## 使用方法

### 命令行参数

```bash
python exp1_image_processing.py [-h] [--image IMAGE] [--output OUTPUT] [--no-display]
```

参数说明:
- `-i, --image`: 指定输入图片路径 (默认: images/test_image.jpg)
- `-o, --output`: 指定输出目录路径 (默认: results)
- `--no-display`: 不显示图像，只保存结果（适用于无GUI环境）
- `-h, --help`: 显示帮助信息

### 使用示例

```bash
# 使用默认测试图片
python exp1_image_processing.py

# 指定自定义图片
python exp1_image_processing.py -i path/to/your/image.jpg

# 指定输出目录
python exp1_image_processing.py -i image.jpg -o my_results/

# 无GUI模式运行（不显示图像）
python exp1_image_processing.py -i image.jpg --no-display

# 显示帮助
python exp1_image_processing.py --help
```

### 运行结果

程序会依次执行：
1. 读取并显示图片信息
2. 显示原图并保存
3. 转换为灰度图并保存
4. 保存灰度图文件
5. 执行NumPy操作（查看像素值、裁剪、反转）

所有结果图片保存在 `results/` 目录下。

## 核心代码说明

### 1. 读取图像
```python
img = cv2.imread(image_path)  # BGR格式
```

### 2. 获取图像信息
```python
height, width = img.shape[:2]
channels = img.shape[2] if len(img.shape) == 3 else 1
dtype = img.dtype
```

### 3. 显示图像（Matplotlib）
```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
```

### 4. 转换为灰度图
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### 5. 保存图像
```python
cv2.imwrite(output_path, gray_img)
```

### 6. NumPy操作
```python
# 访问像素
pixel_value = gray_img[y, x]

# 裁剪区域
cropped = img[y1:y2, x1:x2]

# 图像统计
min_val = gray_img.min()
max_val = gray_img.max()
mean_val = gray_img.mean()
```

## 依赖版本信息

- Python: 3.10
- OpenCV: 4.x
- NumPy: 1.x
- Matplotlib: 3.x

## 注意事项

1. **图像格式**: OpenCV默认使用BGR格式，Matplotlib使用RGB格式，显示时需要转换
2. **路径问题**: Windows用户使用反斜杠时需要注意转义，建议使用原始字符串 `r"path"` 或正斜杠
3. **显示问题**: 如果在无GUI环境运行，Matplotlib可能无法正常显示图像，可以保存后查看

## 作者信息
-学号姓名：2023101147-陈乐轩
- 实验课程：Python计算机视觉
- 日期：2026-03-24

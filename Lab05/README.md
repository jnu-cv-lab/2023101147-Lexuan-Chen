# 图像几何变换实验

本项目按照作业要求，使用 Python 与 OpenCV 完成以下内容：

1. 构造一张包含矩形、圆、平行线、垂直线的测试图像。
2. 分别对测试图像施加：
   - 相似变换
   - 仿射变换
   - 透视变换
3. 观察并总结三类变换对几何性质的影响：
   - 直线是否仍为直线
   - 平行线是否仍保持平行
   - 垂直线是否仍保持垂直
   - 圆是否仍保持为圆
4. 构造一张模拟“桌面拍摄 A4 纸”的透视图，并进行透视校正。

---

## 项目文件说明

```text
image_processing/
├── main.py                # 主脚本，运行后自动生成全部结果
├── requirements.txt       # pip 依赖列表
├── README.md              # 实验说明与环境配置步骤
├── 作业要求.jpg            # 作业截图
└── outputs/               # 运行脚本后自动生成
    ├── original_test_image.png
    ├── similarity_transform.png
    ├── affine_transform.png
    ├── perspective_transform.png
    ├── document_photo.png
    ├── document_photo_marked.png
    ├── document_rectified.png
    ├── overview.png
    └── analysis_summary.md
```

---

## 一、Conda 环境配置

建议使用 Conda 单独创建实验环境，避免污染本机 Python。

### 1. 创建环境

```bash
conda create -n image-processing-hw python=3.11 -y
```

### 2. 激活环境

```bash
conda activate image-processing-hw
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

如果你希望把 OpenCV 也交给 conda 安装，可以使用：

```bash
conda install -c conda-forge opencv matplotlib numpy -y
```

但为了与 `requirements.txt` 保持一致，默认推荐使用 `pip install -r requirements.txt`。

---

## 二、依赖说明

`requirements.txt` 内容如下：

```txt
numpy>=1.24
opencv-python>=4.8
matplotlib>=3.7
```

各依赖用途：

- `numpy`：矩阵与图像数组处理
- `opencv-python`：执行 `cv2.warpAffine`、`cv2.warpPerspective` 等几何变换
- `matplotlib`：拼图展示并保存总览结果

---

## 三、运行方法

在项目根目录执行：

```bash
python main.py
```

运行完成后，终端会打印输出文件列表，所有结果保存在 `outputs/` 目录下。

---

## 四、脚本实现内容

### 1. 生成测试图像

`main.py` 会自动生成一张白底测试图，包含：

- 一个矩形
- 一个圆
- 一组平行线
- 一组垂直线
- 额外的交叉线与文字标注

不需要你手工画图或准备输入图片。

### 2. 三种变换

- **相似变换**：旋转 + 缩放 + 平移
- **仿射变换**：通过 3 对控制点生成变换
- **透视变换**：通过 4 对控制点生成透视畸变

### 3. 透视校正

脚本还会自动生成一张“桌面上的 A4 纸”模拟图像，并执行透视校正，得到拉正后的结果图。

---

## 五、结果文件说明

### `original_test_image.png`
原始测试图。

### `similarity_transform.png`
相似变换结果。可用于观察：
- 直线是否保持直线
- 平行关系是否保持
- 垂直关系是否保持
- 圆是否仍为圆

### `affine_transform.png`
仿射变换结果。可用于观察：
- 直线仍为直线
- 平行线仍保持平行
- 垂直关系一般不再保持
- 圆通常变为椭圆

### `perspective_transform.png`
透视变换结果。可用于观察：
- 直线仍为直线
- 平行线一般会汇聚
- 垂直关系通常不再保持
- 圆通常不再保持为圆

### `document_photo.png`
模拟拍摄的透视畸变纸张图像。

### `document_photo_marked.png`
在透视图像上标出四个角点后的结果，便于说明校正输入点。

### `document_rectified.png`
透视校正后的结果图。

### `overview.png`
将主要结果拼接到一张总览图中，适合直接插入实验报告。

### `analysis_summary.md`
自动生成的性质总结表，可直接作为报告观察结论的基础。

---

## 六、实验结论可直接写入报告

### 相似变换

- 直线保持为直线
- 平行线保持平行
- 垂直线保持垂直
- 圆保持为圆

### 仿射变换

- 直线保持为直线
- 平行线保持平行
- 垂直关系通常不再保持
- 圆通常不再保持为圆，而会变成椭圆

### 透视变换

- 直线保持为直线
- 平行线通常不再保持平行
- 垂直关系通常不再保持
- 圆通常不再保持为圆

---

## 七、如果你要换成自己拍摄的图片

如果老师要求最后一步必须使用你自己拍摄的 A4 纸照片，你可以在 `main.py` 里替换“模拟纸张图像”部分：

1. 用 `cv2.imread()` 读取你拍摄的图片。
2. 手动指定纸张四个角点坐标。
3. 调用 `cv2.getPerspectiveTransform()` 和 `cv2.warpPerspective()` 完成矫正。

当前版本已经把完整流程写好了，你只需要把输入图像和四个点替换掉即可。

---

## 八、常见问题

### 1. 运行时报错 `ModuleNotFoundError: No module named 'cv2'`
说明当前 Python 环境还没有安装 OpenCV，请先执行：

```bash
pip install -r requirements.txt
```

### 2. 中文标题显示异常
不同系统可用字体不同。当前脚本已经优先尝试 macOS 常见中文字体；若你的系统没有这些字体，图像标题可能显示为方框，但不影响核心结果生成。

### 3. 输出目录不存在
程序会自动创建 `outputs/`，不需要手工新建。


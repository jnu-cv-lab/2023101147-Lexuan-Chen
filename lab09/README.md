# CNN Optimizer And Visualization Experiment

本项目完成第 10 次实验要求：在 CNN 图像分类实验基础上，分析优化器、学习率、卷积核、feature maps、错误分类样本和混淆矩阵。

## 实验内容

脚本 `digits_homework.py` 会自动完成以下任务：

1. 使用 CNN 模型进行手写数字分类。
2. 使用相同模型和数据集比较三种优化器：
   - SGD
   - SGD + Momentum
   - Adam
3. 固定优化器为 Adam，比较三种学习率：
   - 0.1
   - 0.01
   - 0.001
4. 记录每轮训练的：
   - training loss
   - validation loss
   - training accuracy
   - validation accuracy
   - test accuracy
5. 可视化第一层卷积核。
6. 可视化第一层卷积输出的 feature maps。
7. 展示测试集中错误分类的样本。
8. 绘制测试集混淆矩阵。

## 数据集

本实验使用 `sklearn.datasets.load_digits` 自带的手写数字数据集。

该数据集不需要联网下载，适合课堂实验和本地快速复现。每张图片大小为 `8 x 8`，类别为数字 `0-9`。

## 环境依赖

需要安装：

```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

当前脚本会自动创建 matplotlib 可写缓存目录，避免 Windows 用户目录权限导致绘图报错。

## 运行方法

默认运行：

```bash
python digits_homework.py
```

指定训练轮数：

```bash
python digits_homework.py --epochs 10
```

指定 batch size：

```bash
python digits_homework.py --batch-size 64
```

指定输出目录：

```bash
python digits_homework.py --output-dir outputs_digits_cnn
```

## 输出文件

运行完成后，结果默认保存在 `outputs_digits_cnn/`：

| 文件 | 说明 |
|---|---|
| `optimizer_comparison.png` | SGD、SGD+Momentum、Adam 的 loss 和 accuracy 对比曲线 |
| `learning_rate_comparison.png` | Adam 在不同学习率下的 loss 和 accuracy 对比曲线 |
| `conv_kernels.png` | 第一层卷积核可视化 |
| `feature_maps.png` | 第一层卷积输出 feature maps 可视化 |
| `misclassified_samples.png` | 测试集错误分类样本 |
| `confusion_matrix.png` | 测试集混淆矩阵 |
| `best_digit_cnn.pth` | 测试集表现最好的 CNN 模型参数 |
| `experiment_report.md` | 自动生成的实验结果摘要与分析 |


## 主要代码入口

核心代码位于 `digits_homework.py`：

- `DigitCNN`：CNN 模型结构。
- `train_one_experiment`：单次训练实验。
- `plot_training_curves`：绘制 loss 和 accuracy 曲线。
- `plot_conv_kernels`：绘制卷积核。
- `plot_feature_maps`：绘制 feature maps。
- `plot_misclassified`：绘制错误分类样本。
- `plot_confusion`：绘制混淆矩阵。
- `write_report`：生成实验报告。

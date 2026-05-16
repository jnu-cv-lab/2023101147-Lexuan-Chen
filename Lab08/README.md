# 实验 8：PyTorch 入门与图像分类

## 1. 实验环境

- Python: 3.13.4
- PyTorch: 2.12.0+cpu
- torchvision: 0.27.0+cpu
- matplotlib: 3.10.9
- 运行设备：CPU

## 2. 模型设计

本实验使用一个简单 CNN 完成 MNIST 与 CIFAR-10 图像分类。模型结构如下：

- Conv2d + BatchNorm2d + ReLU + MaxPool2d
- Conv2d + BatchNorm2d + ReLU + MaxPool2d
- Conv2d + BatchNorm2d + ReLU + AdaptiveAvgPool2d
- Flatten + Dropout + Linear 输出 10 类

损失函数使用 `CrossEntropyLoss`。优化器分别比较 `SGD(momentum=0.9)` 与 `Adam`。

## 3. 实验设置

- MNIST：batch size = 256，epochs = 5
- CIFAR-10：batch size = 512，epochs = 5
- 验证集比例：10%
- SGD learning rate = 0.01
- Adam learning rate = 0.001
- 随机种子：42

## 4. 实验结果

| 数据集 | 优化器 | Learning Rate | Validation Accuracy | Test Accuracy |
| --- | --- | ---: | ---: | ---: |
| MNIST | SGD | 0.01 | 0.9672 | 0.9700 |
| MNIST | Adam | 0.001 | 0.9797 | 0.9780 |
| CIFAR-10 | SGD | 0.01 | 0.4810 | 0.4785 |
| CIFAR-10 | Adam | 0.001 | 0.5562 | 0.5569 |

## 5. 曲线图文件

- `outputs/MNIST_SGD_lr0.01_ep5_loss.png`
- `outputs/MNIST_SGD_lr0.01_ep5_accuracy.png`
- `outputs/MNIST_Adam_lr0.001_ep5_loss.png`
- `outputs/MNIST_Adam_lr0.001_ep5_accuracy.png`
- `outputs/CIFAR10_SGD_lr0.01_ep5_loss.png`
- `outputs/CIFAR10_SGD_lr0.01_ep5_accuracy.png`
- `outputs/CIFAR10_Adam_lr0.001_ep5_loss.png`
- `outputs/CIFAR10_Adam_lr0.001_ep5_accuracy.png`

## 6. 结果分析

MNIST 是灰度手写数字数据集，图像结构简单，类别差异明显，因此 CNN 在 5 个 epoch 后已经能达到较高准确率。Adam 在 MNIST 上的 test accuracy 为 0.9780，略高于 SGD 的 0.9700，说明 Adam 在该实验设置下收敛更快。

CIFAR-10 是彩色自然图像数据集，类别更复杂，背景和物体变化更大，所以整体准确率明显低于 MNIST。Adam 在 CIFAR-10 上的 test accuracy 为 0.5569，高于 SGD 的 0.4785，说明 Adam 对该任务的优化效果更好。

从训练结果可以看出，loss 整体随 epoch 增加而下降，accuracy 整体上升。MNIST 的训练与验证准确率较接近，模型拟合效果较好；CIFAR-10 的准确率仍有提升空间，可以通过更深的网络、更多 epoch、数据增强、学习率调度或使用 ResNet 等结构继续改进。

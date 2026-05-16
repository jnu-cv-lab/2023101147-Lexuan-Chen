"""
实验 8：PyTorch 入门与图像分类

用法示例：
    python digits_homework.py --dataset MNIST --optimizer SGD --epochs 5
    python digits_homework.py --dataset CIFAR10 --optimizer Adam --epochs 5
    python digits_homework.py --compare

脚本会自动完成：
1. 加载 MNIST 或 CIFAR-10 数据集
2. 定义并训练一个简单 CNN
3. 输出每个 epoch 的 training loss / training accuracy
4. 输出 validation loss / validation accuracy
5. 在 test set 上测试 accuracy
6. 保存 loss 和 accuracy 曲线图到 outputs/ 目录
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms
except ModuleNotFoundError as exc:
    missing = exc.name or "required package"
    raise SystemExit(
        f"缺少依赖：{missing}\n"
        "请先安装实验环境，例如：\n"
        "    pip install torch torchvision matplotlib\n"
        "如果使用 Conda，也可以按 PyTorch 官网命令安装对应 CUDA/CPU 版本。"
    ) from exc


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


@dataclass
class ExperimentConfig:
    dataset: str = "MNIST"
    optimizer: str = "SGD"
    learning_rate: float = 0.01
    epochs: int = 5
    batch_size: int = 64
    validation_ratio: float = 0.1
    seed: int = 42


@dataclass
class History:
    train_loss: list[float]
    train_accuracy: list[float]
    validation_loss: list[float]
    validation_accuracy: list[float]
    test_accuracy: float


class SimpleCNN(nn.Module):
    """适用于 MNIST 和 CIFAR-10 的小型卷积神经网络。"""

    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(dataset_name: str) -> tuple[Callable, Callable]:
    if dataset_name == "MNIST":
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        test_transform = train_transform
    elif dataset_name == "CIFAR10":
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )
    else:
        raise ValueError(f"不支持的数据集：{dataset_name}")

    return train_transform, test_transform


def build_dataloaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, int, int]:
    train_transform, test_transform = build_transforms(config.dataset)
    dataset_class = datasets.MNIST if config.dataset == "MNIST" else datasets.CIFAR10
    input_channels = 1 if config.dataset == "MNIST" else 3
    num_classes = 10

    full_train_set = dataset_class(
        root=DATA_DIR,
        train=True,
        transform=train_transform,
        download=True,
    )
    test_set = dataset_class(
        root=DATA_DIR,
        train=False,
        transform=test_transform,
        download=True,
    )

    validation_size = int(len(full_train_set) * config.validation_ratio)
    train_size = len(full_train_set) - validation_size
    generator = torch.Generator().manual_seed(config.seed)
    train_set, validation_set = random_split(
        full_train_set,
        [train_size, validation_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader, test_loader, input_channels, num_classes


def build_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
) -> optim.Optimizer:
    if optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    raise ValueError(f"不支持的优化器：{optimizer_name}")


def run_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total
    return average_loss, accuracy


def evaluate_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    criterion = nn.CrossEntropyLoss()
    _, accuracy = run_one_epoch(model, data_loader, criterion, device)
    return accuracy


def plot_history(history: History, config: ExperimentConfig) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    prefix = f"{config.dataset}_{config.optimizer}_lr{config.learning_rate}_ep{config.epochs}"

    epochs = range(1, config.epochs + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.train_loss, marker="o", label="training loss")
    plt.plot(epochs, history.validation_loss, marker="o", label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{config.dataset} loss curve ({config.optimizer})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.train_accuracy, marker="o", label="training accuracy")
    plt.plot(epochs, history.validation_accuracy, marker="o", label="validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"{config.dataset} accuracy curve ({config.optimizer})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_accuracy.png", dpi=160)
    plt.close()


def train_and_evaluate(config: ExperimentConfig) -> History:
    set_seed(config.seed)
    device = get_device()
    print(f"\n实验配置：{config}")
    print(f"当前设备：{device}")

    train_loader, validation_loader, test_loader, input_channels, num_classes = (
        build_dataloaders(config)
    )
    model = SimpleCNN(input_channels=input_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config.optimizer, model, config.learning_rate)

    history = History([], [], [], [], test_accuracy=0.0)

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = run_one_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        validation_loss, validation_accuracy = run_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_accuracy)
        history.validation_loss.append(validation_loss)
        history.validation_accuracy.append(validation_accuracy)

        print(
            f"epoch {epoch:02d}/{config.epochs} | "
            f"training loss: {train_loss:.4f} | "
            f"training accuracy: {train_accuracy:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.4f}"
        )

    history.test_accuracy = evaluate_accuracy(model, test_loader, device)
    print(f"test accuracy: {history.test_accuracy:.4f}")
    plot_history(history, config)
    return history


def run_comparison(base_config: ExperimentConfig) -> None:
    results: list[tuple[str, str, float, float]] = []
    experiments = [
        ("MNIST", "SGD", 0.01),
        ("MNIST", "Adam", 0.001),
        ("CIFAR10", "SGD", 0.01),
        ("CIFAR10", "Adam", 0.001),
    ]

    for dataset_name, optimizer_name, learning_rate in experiments:
        config = ExperimentConfig(
            dataset=dataset_name,
            optimizer=optimizer_name,
            learning_rate=learning_rate,
            epochs=base_config.epochs,
            batch_size=base_config.batch_size,
            validation_ratio=base_config.validation_ratio,
            seed=base_config.seed,
        )
        history = train_and_evaluate(config)
        results.append(
            (
                dataset_name,
                optimizer_name,
                history.validation_accuracy[-1],
                history.test_accuracy,
            )
        )

    print("\n对比实验结果")
    print("-" * 62)
    print(f"{'Dataset':<10}{'Optimizer':<12}{'Val Accuracy':<18}{'Test Accuracy':<18}")
    print("-" * 62)
    for dataset_name, optimizer_name, validation_accuracy, test_accuracy in results:
        print(
            f"{dataset_name:<10}{optimizer_name:<12}"
            f"{validation_accuracy:<18.4f}{test_accuracy:<18.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch CNN 图像分类实验")
    parser.add_argument("--dataset", choices=["MNIST", "CIFAR10"], default="MNIST")
    parser.add_argument("--optimizer", choices=["SGD", "Adam"], default="SGD")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="依次运行 MNIST/CIFAR-10 与 SGD/Adam 的对比实验",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 0.001 if args.optimizer == "Adam" else 0.01

    config = ExperimentConfig(
        dataset=args.dataset,
        optimizer=args.optimizer,
        learning_rate=learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    if args.compare:
        run_comparison(config)
    else:
        train_and_evaluate(config)


if __name__ == "__main__":
    main()

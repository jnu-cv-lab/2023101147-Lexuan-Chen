import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class DigitCNN(nn.Module):
    """Simple CNN for handwritten digit classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x, return_features=False):
        conv1_out = self.conv1(x)
        x = self.pool(F.relu(conv1_out))
        conv2_out = self.conv2(x)
        x = self.pool(F.relu(conv2_out))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        if return_features:
            return logits, {"conv1": conv1_out, "conv2": conv2_out}
        return logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_data(batch_size, seed):
    digits = load_digits()
    images = digits.images.astype(np.float32)
    labels = digits.target.astype(np.int64)

    scaler = StandardScaler()
    flat_images = images.reshape(len(images), -1)
    scaled_images = scaler.fit_transform(flat_images).reshape(-1, 1, 8, 8)

    x_train, x_temp, y_train, y_temp = train_test_split(
        scaled_images,
        labels,
        test_size=0.3,
        random_state=seed,
        stratify=labels,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp,
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val), torch.tensor(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test), torch.tensor(y_test)),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader, scaler


def build_optimizer(name, model, lr):
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if name == "SGD+Momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
    }


def train_one_experiment(optimizer_name, lr, loaders, epochs, device, seed):
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(optimizer_name, model, lr)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(
            f"{optimizer_name:12s} lr={lr:<6g} epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

    test_metrics = evaluate(model, test_loader, criterion, device)
    return model, history, test_metrics


def plot_training_curves(results, title, output_path):
    epochs = range(1, len(next(iter(results.values()))["history"]["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for label, result in results.items():
        history = result["history"]
        axes[0].plot(epochs, history["train_loss"], marker="o", label=f"{label} train")
        axes[0].plot(epochs, history["val_loss"], marker="s", linestyle="--", label=f"{label} val")
        axes[1].plot(epochs, history["train_accuracy"], marker="o", label=f"{label} train")
        axes[1].plot(epochs, history["val_accuracy"], marker="s", linestyle="--", label=f"{label} val")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_conv_kernels(model, output_path, max_kernels=8):
    kernels = model.conv1.weight.detach().cpu().numpy()
    count = min(max_kernels, kernels.shape[0])
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))

    for idx, ax in enumerate(axes.flat):
        if idx < count:
            kernel = kernels[idx, 0]
            ax.imshow(kernel, cmap="coolwarm")
            ax.set_title(f"kernel {idx}")
        ax.axis("off")

    fig.suptitle("First Conv Layer Kernels")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_maps(model, test_loader, output_path, device, max_maps=8):
    model.eval()
    images, labels = next(iter(test_loader))
    image = images[:1].to(device)

    with torch.no_grad():
        logits, features = model(image, return_features=True)
        pred = logits.argmax(dim=1).item()
        maps = features["conv1"][0, :max_maps].detach().cpu().numpy()

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    axes = axes.flat
    axes[0].imshow(images[0, 0].numpy(), cmap="gray")
    axes[0].set_title(f"input true={labels[0].item()} pred={pred}")
    axes[0].axis("off")

    for idx in range(max_maps):
        axes[idx + 1].imshow(maps[idx], cmap="viridis")
        axes[idx + 1].set_title(f"map {idx}")
        axes[idx + 1].axis("off")

    fig.suptitle("Feature Maps From First Conv Layer")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_misclassified(model, test_loader, output_path, device, max_images=8):
    model.eval()
    wrong_samples = []

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            wrong_mask = preds != labels
            for image, true_label, pred_label in zip(images[wrong_mask], labels[wrong_mask], preds[wrong_mask]):
                wrong_samples.append((image[0].numpy(), true_label.item(), pred_label.item()))
                if len(wrong_samples) >= max_images:
                    break
            if len(wrong_samples) >= max_images:
                break

    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for idx, ax in enumerate(axes.flat):
        if idx < len(wrong_samples):
            image, true_label, pred_label = wrong_samples[idx]
            ax.imshow(image, cmap="gray")
            ax.set_title(f"true={true_label}, pred={pred_label}")
        else:
            ax.set_title("no error")
        ax.axis("off")

    fig.suptitle("Misclassified Test Samples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return wrong_samples


def plot_confusion(test_metrics, output_path):
    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(
        cmap="Blues",
        values_format="d",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return cm


def write_report(output_dir, optimizer_results, lr_results, best_label, wrong_samples, cm):
    lines = []
    lines.append("# CNN Optimizer And Learning Rate Experiment")
    lines.append("")
    lines.append("## Task 1: CNN Model")
    lines.append("Used a two-layer CNN: Conv2d(1->16), MaxPool, Conv2d(16->32), MaxPool, FC(64), FC(10).")
    lines.append("")
    lines.append("## Task 2: Optimizer Comparison")
    lines.append("| Optimizer | LR | Final Train Loss | Final Val Loss | Final Train Acc | Final Val Acc | Test Acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, result in optimizer_results.items():
        h = result["history"]
        lines.append(
            f"| {label} | {result['lr']} | {h['train_loss'][-1]:.4f} | {h['val_loss'][-1]:.4f} | "
            f"{h['train_accuracy'][-1]:.4f} | {h['val_accuracy'][-1]:.4f} | {result['test']['accuracy']:.4f} |"
        )
    lines.append("")
    lines.append("## Task 3: Learning Rate Comparison")
    lines.append("| Optimizer | LR | Final Train Loss | Final Val Loss | Final Train Acc | Final Val Acc | Test Acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, result in lr_results.items():
        h = result["history"]
        lines.append(
            f"| Adam | {result['lr']} | {h['train_loss'][-1]:.4f} | {h['val_loss'][-1]:.4f} | "
            f"{h['train_accuracy'][-1]:.4f} | {h['val_accuracy'][-1]:.4f} | {result['test']['accuracy']:.4f} |"
        )
    lines.append("")
    lines.append("## Task 4: Convolution Kernel Visualization")
    lines.append("Saved first-layer convolution kernels to `conv_kernels.png`.")
    lines.append("Kernels with positive/negative color changes often respond to local edge, stroke direction, and brightness contrast.")
    lines.append("")
    lines.append("## Task 5: Feature Map Visualization")
    lines.append("Saved first-layer feature maps to `feature_maps.png`.")
    lines.append("Different maps emphasize different digit strokes, corners, edge positions, and local texture responses.")
    lines.append("")
    lines.append("## Task 6: Misclassified Samples")
    lines.append("Saved misclassified examples to `misclassified_samples.png`.")
    if wrong_samples:
        pairs = ", ".join([f"true {true}/pred {pred}" for _, true, pred in wrong_samples])
        lines.append(f"Observed wrong labels: {pairs}.")
    else:
        lines.append("No misclassified samples were found in the displayed test batch.")
    lines.append("Errors usually happen when different digits share similar strokes, when writing is ambiguous, or when the sample is very sparse.")
    lines.append("")
    lines.append("## Task 7: Confusion Matrix")
    lines.append("Saved confusion matrix to `confusion_matrix.png`.")
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    confused = np.argwhere(off_diag == off_diag.max())
    if off_diag.max() > 0:
        examples = ", ".join([f"{true}->{pred}" for true, pred in confused[:5]])
        lines.append(f"Most visible confusion count: {off_diag.max()} for {examples}.")
    else:
        lines.append("The test set had no off-diagonal confusion for the selected model.")
    lines.append("")
    lines.append(f"Best model selected for visual analysis: {best_label}.")

    (output_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="CNN experiment for handwritten digit classification.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs for each experiment.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="outputs_digits_cnn", help="Directory for plots and report.")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, _ = prepare_data(args.batch_size, args.seed)
    loaders = (train_loader, val_loader, test_loader)

    optimizer_settings = [
        ("SGD", 0.01),
        ("SGD+Momentum", 0.01),
        ("Adam", 0.01),
    ]
    optimizer_results = {}
    trained_models = {}

    print("\nTask 2: optimizer comparison")
    for optimizer_name, lr in optimizer_settings:
        model, history, test_metrics = train_one_experiment(
            optimizer_name, lr, loaders, args.epochs, device, args.seed
        )
        optimizer_results[optimizer_name] = {
            "lr": lr,
            "history": history,
            "test": test_metrics,
        }
        trained_models[optimizer_name] = model
        print(f"Test accuracy ({optimizer_name}, lr={lr}): {test_metrics['accuracy']:.4f}\n")

    plot_training_curves(
        optimizer_results,
        "Optimizer Comparison",
        output_dir / "optimizer_comparison.png",
    )

    lr_settings = [0.1, 0.01, 0.001]
    lr_results = {}
    print("\nTask 3: Adam learning rate comparison")
    for lr in lr_settings:
        label = f"Adam lr={lr:g}"
        model, history, test_metrics = train_one_experiment(
            "Adam", lr, loaders, args.epochs, device, args.seed
        )
        lr_results[label] = {
            "lr": lr,
            "history": history,
            "test": test_metrics,
        }
        trained_models[label] = model
        print(f"Test accuracy ({label}): {test_metrics['accuracy']:.4f}\n")

    plot_training_curves(
        lr_results,
        "Adam Learning Rate Comparison",
        output_dir / "learning_rate_comparison.png",
    )

    all_results = {**optimizer_results, **lr_results}
    best_label = max(all_results, key=lambda key: all_results[key]["test"]["accuracy"])
    best_model = trained_models[best_label]
    best_test = all_results[best_label]["test"]

    print(f"\nBest model for visualization: {best_label}")
    torch.save(best_model.state_dict(), output_dir / "best_digit_cnn.pth")

    plot_conv_kernels(best_model, output_dir / "conv_kernels.png")
    plot_feature_maps(best_model, test_loader, output_dir / "feature_maps.png", device)
    wrong_samples = plot_misclassified(best_model, test_loader, output_dir / "misclassified_samples.png", device)
    cm = plot_confusion(best_test, output_dir / "confusion_matrix.png")
    write_report(output_dir, optimizer_results, lr_results, best_label, wrong_samples, cm)

    print("\nFinished. Generated files:")
    for path in sorted(output_dir.iterdir()):
        print(f"- {path}")


if __name__ == "__main__":
    main()

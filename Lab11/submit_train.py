import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "checkpoints"
RESULT_DIR = ROOT_DIR / "results"

INPUT_DIM = 132
NUM_CLASSES = 6
RANDOM_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_terms = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_terms)
        pe[:, 1::2] = torch.cos(positions * div_terms[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PoseTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 132,
        num_classes: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.position(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.classifier(x)


def load_clean_dataset(data_dir: Path):
    X_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    y_train = np.load(data_dir / "y_train.npy").astype(np.int64)
    X_val = np.load(data_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(data_dir / "y_val.npy").astype(np.int64)

    train_mask = np.abs(X_train).sum(axis=(1, 2)) > 0
    val_mask = np.abs(X_val).sum(axis=(1, 2)) > 0
    dropped = {
        "train": int((~train_mask).sum()),
        "val": int((~val_mask).sum()),
    }

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_val, y_val = X_val[val_mask], y_val[val_mask]

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_set = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    return train_set, val_set, mean.astype(np.float32), std.astype(np.float32), dropped


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    loss_meter = AverageMeter()
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            loss_meter.update(loss.item(), batch_size)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return loss_meter.avg, correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train badminton action classifier.")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dirs(MODEL_DIR, RESULT_DIR)
    device = get_device()

    train_set, val_set, mean, std, dropped = load_clean_dataset(args.data_dir)
    print(
        f"Using {len(train_set)} train / {len(val_set)} val samples "
        f"(dropped empty train={dropped['train']}, val={dropped['val']})"
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = PoseTransformer(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    history_path = RESULT_DIR / "training_log.csv"
    history_path.write_text("epoch,train_loss,train_acc,val_loss,val_acc\n", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        line = (
            f"{epoch},{train_loss:.6f},{train_acc:.6f},"
            f"{val_loss:.6f},{val_acc:.6f}\n"
        )
        with history_path.open("a", encoding="utf-8") as f:
            f.write(line)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "d_model": args.d_model,
                        "nhead": args.nhead,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                    },
                    "feature_mean": mean,
                    "feature_std": std,
                    "dropped_empty": dropped,
                    "best_acc": best_acc,
                },
                MODEL_DIR / "best_model.pth",
            )

    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()

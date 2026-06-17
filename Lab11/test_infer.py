import argparse
import math
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "checkpoints"
RESULT_DIR = ROOT_DIR / "results"
MODEL_ASSET_PATH = ROOT_DIR / "models" / "pose_landmarker_lite.task"

CLASSES = [
    "forehand_drive",
    "forehand_lift",
    "forehand_net_shot",
    "forehand_clear",
    "backhand_drive",
    "backhand_net_shot",
]
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}

SEQUENCE_LENGTH = 32
INPUT_DIM = 132
NUM_CLASSES = len(CLASSES)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def torch_load_checkpoint(checkpoint_path: Path, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})
    model = PoseTransformer(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        d_model=config.get("d_model", 128),
        nhead=config.get("nhead", 4),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def sample_or_pad(sequence: np.ndarray, target_len: int) -> np.ndarray:
    if len(sequence) == 0:
        return np.zeros((target_len, INPUT_DIM), dtype=np.float32)
    if len(sequence) >= target_len:
        indices = np.linspace(0, len(sequence) - 1, target_len).astype(int)
        return sequence[indices].astype(np.float32)

    pad = np.repeat(sequence[-1][None, :], target_len - len(sequence), axis=0)
    return np.concatenate([sequence, pad], axis=0).astype(np.float32)


def result_to_vector(result) -> np.ndarray:
    if not result.pose_landmarks:
        return np.zeros(INPUT_DIM, dtype=np.float32)

    values = []
    for landmark in result.pose_landmarks[0]:
        values.extend(
            [
                landmark.x,
                landmark.y,
                landmark.z,
                getattr(landmark, "visibility", 0.0),
            ]
        )
    return np.asarray(values, dtype=np.float32)


def evaluate(args) -> None:
    ensure_dirs(RESULT_DIR)
    device = get_device()
    model, checkpoint = load_model(args.checkpoint, device)

    X_val = np.load(args.data_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(args.data_dir / "y_val.npy").astype(np.int64)
    val_mask = np.abs(X_val).sum(axis=(1, 2)) > 0
    X_val, y_val = X_val[val_mask], y_val[val_mask]

    mean = checkpoint.get("feature_mean")
    std = checkpoint.get("feature_std")
    if mean is not None and std is not None:
        X_val = (X_val - mean) / std

    dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device))
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print(report)
    (RESULT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()


def predict_sequence(model: PoseTransformer, sequence: list[np.ndarray], device: torch.device):
    features = sample_or_pad(np.asarray(sequence, dtype=np.float32), SEQUENCE_LENGTH)
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    pred = int(probs.argmax().item())
    return IDX_TO_CLASS[pred], float(probs[pred].item())


def run_inference(args) -> None:
    if not args.model_asset.exists():
        raise FileNotFoundError(f"Pose model not found: {args.model_asset}")

    device = get_device()
    model, _ = load_model(args.checkpoint, device)
    source = int(args.source) if str(args.source).isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source: {args.source}")

    window = deque(maxlen=SEQUENCE_LENGTH)
    label, score = "warming_up", 0.0

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model_asset)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(image)
            window.append(result_to_vector(result))

            if len(window) == SEQUENCE_LENGTH:
                label, score = predict_sequence(model, list(window), device)

            cv2.putText(
                frame,
                f"{label}: {score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Badminton Action Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def plot_training_curves(args) -> None:
    history = pd.read_csv(args.log_path)
    best_idx = history["val_acc"].idxmax()
    best_epoch = int(history.loc[best_idx, "epoch"])
    best_acc = float(history.loc[best_idx, "val_acc"])

    ensure_dirs(args.output_path.parent)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(alpha=0.25)

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_acc"], label="Train Acc")
    plt.plot(history["epoch"], history["val_acc"], label="Val Acc")
    plt.scatter([best_epoch], [best_acc], color="red", label=f"Best {best_acc:.2%}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=200)
    plt.close()

    print(f"Saved {args.output_path}")
    print(f"Best epoch: {best_epoch}, best val_acc: {best_acc:.4f}")


def collect_demo_videos(raw_dir: Path, limit: int) -> list[tuple[Path, str]]:
    videos = []
    for class_name in CLASSES:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append((path, class_name))
                break
        if len(videos) >= limit:
            break
    return videos


def draw_skeleton(frame, landmarks, min_visibility: float) -> None:
    height, width = frame.shape[:2]
    points = []
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        visibility = getattr(landmark, "visibility", 1.0)
        visible = visibility >= min_visibility and 0 <= x < width and 0 <= y < height
        points.append((x, y, visible))

    for start, end in POSE_CONNECTIONS:
        x1, y1, ok1 = points[start]
        x2, y2, ok2 = points[end]
        if ok1 and ok2:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 220, 255), 3, cv2.LINE_AA)

    for x, y, visible in points:
        if visible:
            cv2.circle(frame, (x, y), 4, (20, 255, 80), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 6, (0, 0, 0), 1, cv2.LINE_AA)


def visualize_video(
    video_path: Path,
    class_name: str,
    output_path: Path,
    model_asset_path: Path,
    max_frames: int,
    min_visibility: float,
) -> list[tuple[str, object]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_asset_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_count = 0
    snapshots = []
    snapshot_indices = {0, max_frames // 2, max_frames - 1}
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened() and frame_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(image)
            if result.pose_landmarks:
                draw_skeleton(frame, result.pose_landmarks[0], min_visibility)

            cv2.putText(
                frame,
                f"{class_name} | {video_path.name}",
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{class_name} | {video_path.name}",
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if frame_count in snapshot_indices:
                snapshots.append((f"{class_name} frame {frame_count + 1}", frame.copy()))
            writer.write(frame)
            frame_count += 1

    cap.release()
    writer.release()
    return snapshots


def save_contact_sheet(snapshots: list[tuple[str, object]], output_path: Path) -> None:
    if not snapshots:
        return

    cell_width = 360
    cell_height = 240
    label_height = 34
    cells = []
    for label, frame in snapshots:
        frame = cv2.resize(frame, (cell_width, cell_height))
        label_bar = np.zeros((label_height, cell_width, 3), dtype=np.uint8)
        cv2.putText(
            label_bar,
            label,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells.append(np.vstack([label_bar, frame]))

    rows = []
    columns = 3
    blank = np.zeros_like(cells[0])
    for start in range(0, len(cells), columns):
        row = cells[start : start + columns]
        while len(row) < columns:
            row.append(blank.copy())
        rows.append(np.hstack(row))

    sheet = np.vstack(rows)
    cv2.imwrite(str(output_path), sheet)


def visualize_skeleton(args) -> None:
    if not args.model_asset.exists():
        raise FileNotFoundError(f"Pose model not found: {args.model_asset}")

    ensure_dirs(args.visual_output_dir)
    videos = collect_demo_videos(args.raw_dir, args.limit)
    if not videos:
        raise FileNotFoundError(f"No videos found under {args.raw_dir}")

    all_snapshots = []
    for index, (video_path, class_name) in enumerate(videos, start=1):
        output_path = args.visual_output_dir / f"skeleton_demo_{index}_{class_name}_{video_path.stem}.mp4"
        snapshots = visualize_video(
            video_path=video_path,
            class_name=class_name,
            output_path=output_path,
            model_asset_path=args.model_asset,
            max_frames=args.max_frames,
            min_visibility=args.min_visibility,
        )
        all_snapshots.extend(snapshots)
        print(f"Saved {output_path}")

    contact_sheet = args.contact_sheet or args.visual_output_dir / "skeleton_contact_sheet.jpg"
    save_contact_sheet(all_snapshots, contact_sheet)
    print(f"Saved {contact_sheet}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate, infer, plot curves, or visualize skeletons."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="evaluate",
        choices=["evaluate", "infer", "curves", "visualize"],
    )
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "best_model.pth")
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--source", default="0", help="video path or webcam index")
    parser.add_argument("--model-asset", type=Path, default=MODEL_ASSET_PATH)

    parser.add_argument("--log-path", type=Path, default=RESULT_DIR / "training_log.csv")
    parser.add_argument("--output-path", type=Path, default=RESULT_DIR / "training_curves.png")

    parser.add_argument("--raw-dir", type=Path, default=RAW_VIDEO_DIR)
    parser.add_argument("--visual-output-dir", type=Path, default=RESULT_DIR / "visualizations")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--min-visibility", type=float, default=0.25)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate(args)
    elif args.command == "infer":
        run_inference(args)
    elif args.command == "curves":
        plot_training_curves(args)
    elif args.command == "visualize":
        visualize_skeleton(args)


if __name__ == "__main__":
    main()

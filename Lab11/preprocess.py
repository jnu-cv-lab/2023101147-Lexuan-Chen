import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_ASSET_PATH = ROOT_DIR / "models" / "pose_landmarker_lite.task"

CLASSES = [
    "forehand_drive",
    "forehand_lift",
    "forehand_net_shot",
    "forehand_clear",
    "backhand_drive",
    "backhand_net_shot",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

SEQUENCE_LENGTH = 32
NUM_LANDMARKS = 33
LANDMARK_DIMS = 4
INPUT_DIM = NUM_LANDMARKS * LANDMARK_DIMS
TRAIN_RATIO = 0.8
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


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


def extract_pose_sequence(
    video_path: Path, sequence_length: int, model_asset_path: Path
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_asset_path)),
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
            frames.append(result_to_vector(result))

    cap.release()
    return sample_or_pad(np.asarray(frames, dtype=np.float32), sequence_length)


def collect_videos(raw_dir: Path) -> list[tuple[Path, int]]:
    samples = []
    for class_name in CLASSES:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            continue
        for path in class_dir.rglob("*"):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                samples.append((path, CLASS_TO_IDX[class_name]))
    return samples


def split_and_save(
    features: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    train_ratio: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labels))
    train_size = int(len(indices) * train_ratio)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    np.save(output_dir / "X_train.npy", features[train_idx])
    np.save(output_dir / "y_train.npy", labels[train_idx])
    np.save(output_dir / "X_val.npy", features[val_idx])
    np.save(output_dir / "y_val.npy", labels[val_idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe Pose features.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--model-asset", type=Path, default=MODEL_ASSET_PATH)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dirs(args.output_dir)
    if not args.model_asset.exists():
        raise FileNotFoundError(f"Pose model not found: {args.model_asset}")

    samples = collect_videos(args.raw_dir)
    if not samples:
        raise FileNotFoundError(
            f"No videos found. Put videos under {args.raw_dir}/<class_name>/"
        )

    features, labels = [], []
    for video_path, label in tqdm(samples, desc="Extracting pose"):
        features.append(
            extract_pose_sequence(video_path, args.sequence_length, args.model_asset)
        )
        labels.append(label)

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    split_and_save(X, y, args.output_dir, args.train_ratio, args.seed)

    print(f"Saved {len(y)} samples to {args.output_dir}")
    print(f"Feature shape: {X.shape}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "NumPy is required to train the pose-sequence TCN model. "
        "Install it in the current environment before running this script."
    ) from exc

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "PyTorch is required to train the pose-sequence TCN model. "
        "Install it in the current environment before running this script."
    ) from exc


PROJECT_SENTINELS = ("Data/LLSP", "artifacts")

KPT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

KEYPOINT_PROFILE_SPECS: dict[str, dict[str, float]] = {
    "battle_rope_v1": {
        "nose": 0.45,
        "left_eye": 0.45,
        "right_eye": 0.45,
        "left_ear": 0.45,
        "right_ear": 0.45,
        "left_shoulder": 1.45,
        "right_shoulder": 1.45,
        "left_elbow": 1.35,
        "right_elbow": 1.35,
        "left_wrist": 1.50,
        "right_wrist": 1.50,
        "left_hip": 0.95,
        "right_hip": 0.95,
        "left_knee": 0.75,
        "right_knee": 0.75,
        "left_ankle": 0.70,
        "right_ankle": 0.70,
    },
    "bench_pressing_v1": {
        "nose": 0.40,
        "left_eye": 0.40,
        "right_eye": 0.40,
        "left_ear": 0.40,
        "right_ear": 0.40,
        "left_shoulder": 1.55,
        "right_shoulder": 1.55,
        "left_elbow": 1.45,
        "right_elbow": 1.45,
        "left_wrist": 1.35,
        "right_wrist": 1.35,
        "left_hip": 0.95,
        "right_hip": 0.95,
        "left_knee": 0.60,
        "right_knee": 0.60,
        "left_ankle": 0.55,
        "right_ankle": 0.55,
    },
    "front_raise_v1": {
        "nose": 0.45,
        "left_eye": 0.45,
        "right_eye": 0.45,
        "left_ear": 0.45,
        "right_ear": 0.45,
        "left_shoulder": 1.50,
        "right_shoulder": 1.50,
        "left_elbow": 1.30,
        "right_elbow": 1.30,
        "left_wrist": 1.45,
        "right_wrist": 1.45,
        "left_hip": 0.85,
        "right_hip": 0.85,
        "left_knee": 0.60,
        "right_knee": 0.60,
        "left_ankle": 0.55,
        "right_ankle": 0.55,
    },
    "jump_jacks_v1": {
        "nose": 0.45,
        "left_eye": 0.45,
        "right_eye": 0.45,
        "left_ear": 0.45,
        "right_ear": 0.45,
        "left_shoulder": 1.35,
        "right_shoulder": 1.35,
        "left_elbow": 1.10,
        "right_elbow": 1.10,
        "left_wrist": 1.40,
        "right_wrist": 1.40,
        "left_hip": 1.20,
        "right_hip": 1.20,
        "left_knee": 1.35,
        "right_knee": 1.35,
        "left_ankle": 1.45,
        "right_ankle": 1.45,
    },
    "pommelhorse_v1": {
        "nose": 0.45,
        "left_eye": 0.45,
        "right_eye": 0.45,
        "left_ear": 0.45,
        "right_ear": 0.45,
        "left_shoulder": 1.35,
        "right_shoulder": 1.35,
        "left_elbow": 1.25,
        "right_elbow": 1.25,
        "left_wrist": 1.25,
        "right_wrist": 1.25,
        "left_hip": 1.35,
        "right_hip": 1.35,
        "left_knee": 1.10,
        "right_knee": 1.10,
        "left_ankle": 1.05,
        "right_ankle": 1.05,
    },
    "pull_up_v1": {
        "nose": 0.45,
        "left_eye": 0.45,
        "right_eye": 0.45,
        "left_ear": 0.45,
        "right_ear": 0.45,
        "left_shoulder": 1.55,
        "right_shoulder": 1.55,
        "left_elbow": 1.45,
        "right_elbow": 1.45,
        "left_wrist": 1.35,
        "right_wrist": 1.35,
        "left_hip": 1.00,
        "right_hip": 1.00,
        "left_knee": 0.75,
        "right_knee": 0.75,
        "left_ankle": 0.70,
        "right_ankle": 0.70,
    },
    "push_up_v1": {
        "nose": 0.40,
        "left_eye": 0.40,
        "right_eye": 0.40,
        "left_ear": 0.40,
        "right_ear": 0.40,
        "left_shoulder": 1.45,
        "right_shoulder": 1.45,
        "left_elbow": 1.55,
        "right_elbow": 1.55,
        "left_wrist": 1.45,
        "right_wrist": 1.45,
        "left_hip": 1.20,
        "right_hip": 1.20,
        "left_knee": 0.85,
        "right_knee": 0.85,
        "left_ankle": 0.75,
        "right_ankle": 0.75,
    },
    "sit_up_v1": {
        "nose": 0.45,
        "left_eye": 0.45,
        "right_eye": 0.45,
        "left_ear": 0.45,
        "right_ear": 0.45,
        "left_shoulder": 1.35,
        "right_shoulder": 1.35,
        "left_elbow": 0.95,
        "right_elbow": 0.95,
        "left_wrist": 0.85,
        "right_wrist": 0.85,
        "left_hip": 1.45,
        "right_hip": 1.45,
        "left_knee": 0.80,
        "right_knee": 0.80,
        "left_ankle": 0.70,
        "right_ankle": 0.70,
    },
    "squat_v1": {
        "nose": 0.45,
        "left_eye": 0.45,
        "right_eye": 0.45,
        "left_ear": 0.45,
        "right_ear": 0.45,
        "left_shoulder": 0.90,
        "right_shoulder": 0.90,
        "left_elbow": 0.75,
        "right_elbow": 0.75,
        "left_wrist": 0.70,
        "right_wrist": 0.70,
        "left_hip": 1.25,
        "right_hip": 1.25,
        "left_knee": 1.55,
        "right_knee": 1.55,
        "left_ankle": 1.45,
        "right_ankle": 1.45,
    },
}

AUTO_EXERCISE_PROFILE_V1 = {
    "battle_rope": "battle_rope_v1",
    "bench_pressing": "bench_pressing_v1",
    "front_raise": "front_raise_v1",
    "jump_jacks": "jump_jacks_v1",
    "pommelhorse": "pommelhorse_v1",
    "pull_up": "pull_up_v1",
    "push_up": "push_up_v1",
    "sit_up": "sit_up_v1",
    "squat": "squat_v1",
}


def resolve_project_dir(project_dir_arg: str | None = None) -> Path:
    if project_dir_arg:
        project_dir = Path(project_dir_arg).expanduser().resolve()
        if all((project_dir / sentinel).exists() for sentinel in PROJECT_SENTINELS):
            return project_dir
        raise FileNotFoundError(
            f"Provided --project-dir does not look like the project root: {project_dir}"
        )

    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        cand = (base / "CV_Image_pose_detection").resolve()
        if all((cand / sentinel).exists() for sentinel in PROJECT_SENTINELS):
            return cand

    for base in [cwd, *cwd.parents]:
        if all((base / sentinel).exists() for sentinel in PROJECT_SENTINELS):
            return base.resolve()

    raise FileNotFoundError("Could not resolve project directory containing Data/LLSP and artifacts.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass
class Sample:
    name: str
    split: str
    count: float
    exercise: str
    sequence_path: Path


def parse_count(raw: str | None) -> float | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def resolve_sequence_path(project_dir: Path, raw_path: str) -> Path:
    annotation_dir = project_dir / "Data" / "LLSP" / "annotation_cleaned"
    raw = Path(raw_path)

    direct_local = annotation_dir / "pose_sequences" / raw.name
    if direct_local.exists():
        return direct_local.resolve()

    if raw.is_absolute() and raw.exists():
        return raw.resolve()

    if not raw.is_absolute():
        joined = (annotation_dir / raw).resolve()
        if joined.exists():
            return joined

    raise FileNotFoundError(f"Could not resolve pose sequence path from {raw_path!r}")


def load_samples(project_dir: Path, index_path: Path, exercise_filter: str | None) -> list[Sample]:
    rows = load_csv_rows(index_path)
    samples: list[Sample] = []
    skipped_bad_count: list[str] = []
    for row in rows:
        split = row["split"].strip().lower()
        if split not in {"train", "valid"}:
            continue
        exercise = row["type"].strip()
        if exercise_filter and exercise != exercise_filter:
            continue
        count = parse_count(row.get("count"))
        if count is None:
            skipped_bad_count.append(row["name"].strip())
            continue
        samples.append(
            Sample(
                name=row["name"].strip(),
                split=split,
                count=count,
                exercise=exercise,
                sequence_path=resolve_sequence_path(project_dir, row["sequence_path"]),
            )
        )
    if skipped_bad_count:
        preview = ", ".join(skipped_bad_count[:5])
        print(
            f"Skipped {len(skipped_bad_count)} row(s) with missing/non-numeric count"
            f" in {index_path.name}: {preview}"
        )
    if not samples:
        scope = exercise_filter or "all exercises"
        raise RuntimeError(f"No train/valid samples found for {scope} in {index_path}")
    return samples


def resample_sequence(array: np.ndarray, target_len: int) -> np.ndarray:
    if array.shape[0] == target_len:
        return array.astype(np.float32, copy=False)
    if array.shape[0] == 1:
        return np.repeat(array, target_len, axis=0).astype(np.float32, copy=False)

    src_x = np.linspace(0.0, 1.0, num=array.shape[0], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.empty((target_len, array.shape[1]), dtype=np.float32)
    for feat_idx in range(array.shape[1]):
        out[:, feat_idx] = np.interp(dst_x, src_x, array[:, feat_idx]).astype(np.float32)
    return out


def load_sequence_array(path: Path, target_len: int) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [T, F] sequence array, got {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return resample_sequence(arr, target_len=target_len)


def resolve_keypoint_profile_name(profile_name: str, exercise_name: str | None) -> str | None:
    if profile_name == "none":
        return None
    if profile_name == "auto_exercise_v1":
        if not exercise_name:
            raise ValueError("--keypoint-profile auto_exercise_v1 requires --exercise.")
        return AUTO_EXERCISE_PROFILE_V1.get(exercise_name)
    if profile_name in KEYPOINT_PROFILE_SPECS:
        return profile_name
    raise ValueError(
        "Unsupported --keypoint-profile. Use none, auto_exercise_v1, or one of: "
        + ", ".join(sorted(KEYPOINT_PROFILE_SPECS))
    )


def build_feature_weight_vector(
    input_dim: int,
    profile_name: str,
    exercise_name: str | None,
    strength: float,
) -> tuple[np.ndarray | None, str | None]:
    resolved_profile = resolve_keypoint_profile_name(profile_name, exercise_name)
    if resolved_profile is None:
        return None, None
    if input_dim % 3 != 0:
        raise ValueError(f"Expected feature dimension divisible by 3, got {input_dim}")

    keypoint_weights = np.ones(input_dim // 3, dtype=np.float32)
    profile_spec = KEYPOINT_PROFILE_SPECS[resolved_profile]
    for keypoint_name, weight in profile_spec.items():
        keypoint_weights[KPT[keypoint_name]] = float(weight)

    strength = float(max(strength, 0.0))
    keypoint_weights = 1.0 + strength * (keypoint_weights - 1.0)
    feature_weights = np.repeat(keypoint_weights, 3).astype(np.float32)
    return feature_weights, resolved_profile


def time_warp_sequence(array: np.ndarray, max_scale_delta: float) -> np.ndarray:
    if max_scale_delta <= 0.0 or array.shape[0] <= 1:
        return array

    scale = random.uniform(max(1.0 - max_scale_delta, 0.5), 1.0 + max_scale_delta)
    src_x = np.linspace(0.0, 1.0, num=array.shape[0], dtype=np.float32)
    warped_x = np.clip(((src_x - 0.5) / scale) + 0.5, 0.0, 1.0)
    warped = np.empty_like(array, dtype=np.float32)
    for feat_idx in range(array.shape[1]):
        warped[:, feat_idx] = np.interp(warped_x, src_x, array[:, feat_idx]).astype(np.float32)
    return warped


def apply_camera_motion_augmentation(
    array: np.ndarray,
    *,
    camera_motion_std: float,
    camera_zoom_std: float,
) -> np.ndarray:
    if (camera_motion_std <= 0.0 and camera_zoom_std <= 0.0) or array.ndim != 2 or array.shape[1] % 3 != 0:
        return array

    augmented = array.astype(np.float32, copy=True).reshape(array.shape[0], -1, 3)
    coords = augmented[:, :, :2]
    alphas = np.linspace(0.0, 1.0, num=augmented.shape[0], dtype=np.float32)[:, None, None]

    if camera_motion_std > 0.0:
        start_shift = np.random.normal(0.0, camera_motion_std, size=(1, 1, 2)).astype(np.float32)
        end_shift = np.random.normal(0.0, camera_motion_std, size=(1, 1, 2)).astype(np.float32)
        coords = coords + ((1.0 - alphas) * start_shift) + (alphas * end_shift)

    if camera_zoom_std > 0.0:
        start_scale = float(np.clip(np.random.normal(1.0, camera_zoom_std), 0.75, 1.25))
        end_scale = float(np.clip(np.random.normal(1.0, camera_zoom_std), 0.75, 1.25))
        scale = ((1.0 - alphas) * start_scale) + (alphas * end_scale)
        coords = coords * scale

    augmented[:, :, :2] = coords
    return augmented.reshape(array.shape[0], array.shape[1]).astype(np.float32, copy=False)


def apply_joint_occlusion_augmentation(
    array: np.ndarray,
    *,
    joint_occlusion_prob: float,
    joint_occlusion_min_ratio: float,
    joint_occlusion_max_ratio: float,
    joint_occlusion_max_joints: int,
) -> np.ndarray:
    if (
        joint_occlusion_prob <= 0.0
        or joint_occlusion_max_joints <= 0
        or array.ndim != 2
        or array.shape[1] % 3 != 0
        or np.random.rand() >= joint_occlusion_prob
    ):
        return array

    augmented = array.astype(np.float32, copy=True).reshape(array.shape[0], -1, 3)
    seq_len = augmented.shape[0]
    num_joints = augmented.shape[1]
    min_len = max(1, int(round(seq_len * max(joint_occlusion_min_ratio, 0.0))))
    max_len = max(min_len, int(round(seq_len * max(joint_occlusion_max_ratio, joint_occlusion_min_ratio))))
    span_len = min(seq_len, random.randint(min_len, max_len))
    start_idx = random.randint(0, max(seq_len - span_len, 0))
    end_idx = start_idx + span_len
    joint_count = random.randint(1, min(joint_occlusion_max_joints, num_joints))
    joint_indices = random.sample(range(num_joints), k=joint_count)

    augmented[start_idx:end_idx, joint_indices, :] = 0.0
    return augmented.reshape(array.shape[0], array.shape[1]).astype(np.float32, copy=False)


def apply_pose_space_augmentation(
    array: np.ndarray,
    *,
    time_warp_range: float,
    camera_motion_std: float,
    camera_zoom_std: float,
    joint_occlusion_prob: float,
    joint_occlusion_min_ratio: float,
    joint_occlusion_max_ratio: float,
    joint_occlusion_max_joints: int,
) -> np.ndarray:
    augmented = time_warp_sequence(array, max_scale_delta=time_warp_range)
    augmented = apply_camera_motion_augmentation(
        augmented,
        camera_motion_std=camera_motion_std,
        camera_zoom_std=camera_zoom_std,
    )
    augmented = apply_joint_occlusion_augmentation(
        augmented,
        joint_occlusion_prob=joint_occlusion_prob,
        joint_occlusion_min_ratio=joint_occlusion_min_ratio,
        joint_occlusion_max_ratio=joint_occlusion_max_ratio,
        joint_occlusion_max_joints=joint_occlusion_max_joints,
    )
    return augmented.astype(np.float32, copy=False)


def apply_feature_space_augmentation(
    array: np.ndarray,
    *,
    feature_noise_std: float,
    frame_dropout_prob: float,
) -> np.ndarray:
    augmented = array.astype(np.float32, copy=False)
    if frame_dropout_prob > 0.0:
        keep_mask = (np.random.rand(augmented.shape[0], 1) >= frame_dropout_prob).astype(np.float32)
        augmented = augmented * keep_mask
    if feature_noise_std > 0.0:
        noise = np.random.normal(0.0, feature_noise_std, size=augmented.shape).astype(np.float32)
        augmented = augmented + noise
    return augmented.astype(np.float32, copy=False)


def compute_feature_stats(arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(list(arrays), axis=0)
    mean = stacked.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = stacked.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


class PoseSequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        target_len: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        feature_weight_vector: np.ndarray | None = None,
        augment: bool = False,
        time_warp_range: float = 0.0,
        camera_motion_std: float = 0.0,
        camera_zoom_std: float = 0.0,
        feature_noise_std: float = 0.0,
        frame_dropout_prob: float = 0.0,
        joint_occlusion_prob: float = 0.0,
        joint_occlusion_min_ratio: float = 0.10,
        joint_occlusion_max_ratio: float = 0.35,
        joint_occlusion_max_joints: int = 2,
    ) -> None:
        self.samples = samples
        self.target_len = target_len
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.feature_weight_vector = feature_weight_vector
        self.augment = augment
        self.time_warp_range = time_warp_range
        self.camera_motion_std = camera_motion_std
        self.camera_zoom_std = camera_zoom_std
        self.feature_noise_std = feature_noise_std
        self.frame_dropout_prob = frame_dropout_prob
        self.joint_occlusion_prob = joint_occlusion_prob
        self.joint_occlusion_min_ratio = joint_occlusion_min_ratio
        self.joint_occlusion_max_ratio = joint_occlusion_max_ratio
        self.joint_occlusion_max_joints = joint_occlusion_max_joints

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        sample = self.samples[idx]
        arr = load_sequence_array(sample.sequence_path, self.target_len)
        if self.augment:
            arr = apply_pose_space_augmentation(
                arr,
                time_warp_range=self.time_warp_range,
                camera_motion_std=self.camera_motion_std,
                camera_zoom_std=self.camera_zoom_std,
                joint_occlusion_prob=self.joint_occlusion_prob,
                joint_occlusion_min_ratio=self.joint_occlusion_min_ratio,
                joint_occlusion_max_ratio=self.joint_occlusion_max_ratio,
                joint_occlusion_max_joints=self.joint_occlusion_max_joints,
            )
        arr = (arr - self.feature_mean) / self.feature_std
        if self.feature_weight_vector is not None:
            arr = arr * self.feature_weight_vector
        if self.augment:
            arr = apply_feature_space_augmentation(
                arr,
                feature_noise_std=self.feature_noise_std,
                frame_dropout_prob=self.frame_dropout_prob,
            )
        x = torch.from_numpy(arr)
        y = torch.tensor(sample.count, dtype=torch.float32)
        return x, y, sample.name, sample.exercise


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PoseCountTCNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int,
        kernel_size: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=1)
        blocks = []
        for block_idx in range(num_blocks):
            blocks.append(
                TemporalBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=2**block_idx,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.tcn(x)
        y = self.head(x)
        return y.squeeze(-1)


def transform_predictions(pred: Iterable[float], mode: str) -> list[float]:
    values = [float(x) for x in pred]
    if mode == "raw":
        return values
    if mode == "round":
        return [float(round(x)) for x in values]
    if mode == "round_clip_nonneg":
        return [float(max(0, round(x))) for x in values]
    raise ValueError(f"Unsupported eval transform: {mode}")


def regression_metrics(pred: list[float], true: list[float], eval_transform: str = "raw") -> dict[str, float]:
    if not pred:
        return {"rows": 0, "mae": math.nan, "rmse": math.nan, "within_1": math.nan}
    eval_pred = transform_predictions(pred, eval_transform)
    abs_err = [abs(a - b) for a, b in zip(eval_pred, true)]
    mse = sum((a - b) ** 2 for a, b in zip(eval_pred, true)) / len(eval_pred)
    within_1 = sum(1 for err in abs_err if err <= 1.0) / len(eval_pred)
    return {
        "rows": float(len(eval_pred)),
        "mae": float(sum(abs_err) / len(abs_err)),
        "rmse": float(math.sqrt(mse)),
        "within_1": float(within_1),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, list[float], list[float], list[str], list[str]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    preds: list[float] = []
    targets: list[float] = []
    names: list[str] = []
    exercises: list[str] = []

    for x, y, batch_names, batch_exercises in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        pred = model(x)
        loss = criterion(pred, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * x.shape[0]
        preds.extend(pred.detach().cpu().tolist())
        targets.extend(y.detach().cpu().tolist())
        names.extend(list(batch_names))
        exercises.extend(list(batch_exercises))

    mean_loss = total_loss / max(len(loader.dataset), 1)
    return mean_loss, preds, targets, names, exercises


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_balanced_count_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    count_freq = Counter(sample.count for sample in samples)
    weights = torch.tensor([1.0 / count_freq[sample.count] for sample in samples], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(samples), replacement=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a counting-only TCN on normalized pose sequences.")
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Optional explicit path to the CV_Image_pose_detection project root.",
    )
    parser.add_argument(
        "--index-csv",
        default=None,
        help="Optional explicit pose_sequence_index.csv path. Defaults to Data/LLSP/annotation_cleaned/pose_sequence_index.csv.",
    )
    parser.add_argument(
        "--exercise",
        default=None,
        help="Optional exercise filter, e.g. squat or push_up. If omitted, trains on all rows in the index.",
    )
    parser.add_argument("--run-name", default="pose_count_tcn_v1", help="Output subdirectory name.")
    parser.add_argument("--seq-len", type=int, default=192, help="Resampled sequence length.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--channels", type=int, default=96, help="TCN channel width.")
    parser.add_argument("--kernel-size", type=int, default=3, help="Conv1d kernel size.")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of residual TCN blocks.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience on the selection metric.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--loss",
        default="l1",
        choices=["smooth_l1", "l1", "mse"],
        help="Regression loss function.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device. Use auto, cpu, cuda, or an explicit torch device like cuda:0.",
    )
    parser.add_argument(
        "--eval-transform",
        default="raw",
        choices=["raw", "round", "round_clip_nonneg"],
        help="How to transform predicted counts before computing evaluation metrics.",
    )
    parser.add_argument(
        "--selection-metric",
        default="mae",
        choices=["mae", "rmse", "within_1"],
        help="Validation metric used for early stopping and best-checkpoint selection.",
    )
    parser.add_argument(
        "--sampler",
        default="shuffle",
        choices=["shuffle", "balanced_count"],
        help="Train sampling strategy. balanced_count upsamples rare rep-count targets.",
    )
    parser.add_argument(
        "--time-warp-range",
        type=float,
        default=0.0,
        help="Max relative temporal warp applied during training augmentation.",
    )
    parser.add_argument(
        "--feature-noise-std",
        type=float,
        default=0.0,
        help="Stddev of Gaussian noise added to normalized training features.",
    )
    parser.add_argument(
        "--frame-dropout-prob",
        type=float,
        default=0.0,
        help="Probability of dropping a whole timestep during training augmentation.",
    )
    parser.add_argument(
        "--camera-motion-std",
        type=float,
        default=0.0,
        help="Stddev of global x/y pose drift used to approximate camera motion in normalized pose coordinates.",
    )
    parser.add_argument(
        "--camera-zoom-std",
        type=float,
        default=0.0,
        help="Stddev of global pose scale drift used to approximate camera zoom in normalized pose coordinates.",
    )
    parser.add_argument(
        "--joint-occlusion-prob",
        type=float,
        default=0.0,
        help="Probability of masking a random set of joints for a contiguous span to mimic occlusion.",
    )
    parser.add_argument(
        "--joint-occlusion-min-ratio",
        type=float,
        default=0.10,
        help="Minimum occlusion span as a fraction of the sequence length.",
    )
    parser.add_argument(
        "--joint-occlusion-max-ratio",
        type=float,
        default=0.35,
        help="Maximum occlusion span as a fraction of the sequence length.",
    )
    parser.add_argument(
        "--joint-occlusion-max-joints",
        type=int,
        default=2,
        help="Maximum number of joints to mask in one occlusion event.",
    )
    parser.add_argument(
        "--keypoint-profile",
        default="none",
        help=(
            "Keypoint weighting profile. Use none, auto_exercise_v1, or a named "
            "profile such as squat_v1, push_up_v1, or pull_up_v1."
        ),
    )
    parser.add_argument(
        "--profile-strength",
        type=float,
        default=1.0,
        help="Interpolation strength toward the selected keypoint profile. 0.0 disables weighting.",
    )
    return parser


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(device_arg)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Unsupported device argument: {device_arg}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def build_loss(loss_name: str) -> nn.Module:
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss(beta=1.0)
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def is_better_metric(current: float, best: float, metric_name: str) -> bool:
    if metric_name == "within_1":
        return current > best
    return current < best


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    project_dir = resolve_project_dir(args.project_dir)
    annotation_dir = project_dir / "Data" / "LLSP" / "annotation_cleaned"
    index_path = (
        Path(args.index_csv).expanduser().resolve()
        if args.index_csv
        else (annotation_dir / "pose_sequence_index.csv").resolve()
    )
    if not index_path.exists():
        raise FileNotFoundError(f"Missing required file: {index_path}")

    output_dir = project_dir / "artifacts" / "3_Modeling" / "training_outputs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(project_dir, index_path, exercise_filter=args.exercise)
    train_samples = [sample for sample in samples if sample.split == "train"]
    valid_samples = [sample for sample in samples if sample.split == "valid"]
    if not train_samples or not valid_samples:
        raise RuntimeError("Both train and valid samples are required.")

    train_arrays = [load_sequence_array(sample.sequence_path, args.seq_len) for sample in train_samples]
    feature_mean, feature_std = compute_feature_stats(train_arrays)
    input_dim = int(train_arrays[0].shape[1])
    feature_weight_vector, resolved_profile_name = build_feature_weight_vector(
        input_dim=input_dim,
        profile_name=args.keypoint_profile,
        exercise_name=args.exercise,
        strength=args.profile_strength,
    )
    if resolved_profile_name:
        print(
            f"Applying keypoint profile {resolved_profile_name} "
            f"(requested={args.keypoint_profile}, strength={args.profile_strength:.2f})"
        )

    train_ds = PoseSequenceDataset(
        samples=train_samples,
        target_len=args.seq_len,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_weight_vector=feature_weight_vector,
        augment=(
            args.time_warp_range > 0.0
            or args.feature_noise_std > 0.0
            or args.frame_dropout_prob > 0.0
            or args.camera_motion_std > 0.0
            or args.camera_zoom_std > 0.0
            or args.joint_occlusion_prob > 0.0
        ),
        time_warp_range=args.time_warp_range,
        camera_motion_std=args.camera_motion_std,
        camera_zoom_std=args.camera_zoom_std,
        feature_noise_std=args.feature_noise_std,
        frame_dropout_prob=args.frame_dropout_prob,
        joint_occlusion_prob=args.joint_occlusion_prob,
        joint_occlusion_min_ratio=args.joint_occlusion_min_ratio,
        joint_occlusion_max_ratio=args.joint_occlusion_max_ratio,
        joint_occlusion_max_joints=args.joint_occlusion_max_joints,
    )
    valid_ds = PoseSequenceDataset(
        samples=valid_samples,
        target_len=args.seq_len,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_weight_vector=feature_weight_vector,
    )
    # Save train artifacts from one deterministic full pass, not the sampled/augmented loader.
    train_eval_ds = PoseSequenceDataset(
        samples=train_samples,
        target_len=args.seq_len,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_weight_vector=feature_weight_vector,
    )

    train_sampler = build_balanced_count_sampler(train_samples) if args.sampler == "balanced_count" else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    device = choose_device(args.device)
    model = PoseCountTCNRegressor(
        input_dim=input_dim,
        channels=args.channels,
        kernel_size=args.kernel_size,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = build_loss(args.loss)

    history_rows: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_valid_metric = -math.inf if args.selection_metric == "within_1" else math.inf
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_pred, train_true, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_pred, valid_true, _, _ = run_epoch(model, valid_loader, criterion, None, device)
        train_metrics = regression_metrics(train_pred, train_true, eval_transform=args.eval_transform)
        valid_metrics = regression_metrics(valid_pred, valid_true, eval_transform=args.eval_transform)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_within_1": train_metrics["within_1"],
                "valid_mae": valid_metrics["mae"],
                "valid_rmse": valid_metrics["rmse"],
                "valid_within_1": valid_metrics["within_1"],
                "selection_metric": valid_metrics[args.selection_metric],
            }
        )

        print(
            f"[{epoch:03d}] "
            f"train_mae={train_metrics['mae']:.4f} "
            f"valid_mae={valid_metrics['mae']:.4f} "
            f"valid_within_1={valid_metrics['within_1']:.4f} "
            f"selection({args.selection_metric})={valid_metrics[args.selection_metric]:.4f}"
        )

        current_valid_metric = valid_metrics[args.selection_metric]
        if is_better_metric(current_valid_metric, best_valid_metric, args.selection_metric):
            best_valid_metric = current_valid_metric
            best_epoch = epoch
            epochs_without_improve = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    _, train_pred, train_true, train_names, train_exercises = run_epoch(
        model, train_eval_loader, criterion, None, device
    )
    _, valid_pred, valid_true, valid_names, valid_exercises = run_epoch(model, valid_loader, criterion, None, device)
    train_eval_pred = transform_predictions(train_pred, args.eval_transform)
    valid_eval_pred = transform_predictions(valid_pred, args.eval_transform)
    train_metrics = regression_metrics(train_pred, train_true, eval_transform=args.eval_transform)
    valid_metrics = regression_metrics(valid_pred, valid_true, eval_transform=args.eval_transform)

    combined_rows: list[dict[str, object]] = []
    for split, names, exercises, raw_pred, eval_pred, true in [
        ("train", train_names, train_exercises, train_pred, train_eval_pred, train_true),
        ("valid", valid_names, valid_exercises, valid_pred, valid_eval_pred, valid_true),
    ]:
        for name, exercise, raw_pred_value, eval_pred_value, true_value in zip(
            names, exercises, raw_pred, eval_pred, true
        ):
            combined_rows.append(
                {
                    "name": name,
                    "type": exercise,
                    "split": split,
                    "true_count": true_value,
                    "raw_pred_count": raw_pred_value,
                    "eval_pred_count": eval_pred_value,
                    "abs_error": abs(eval_pred_value - true_value),
                }
            )

    split_counts = {
        "train_rows": len(train_samples),
        "valid_rows": len(valid_samples),
    }
    exercise_counter = Counter(sample.exercise for sample in samples)

    config = {
        "run_name": args.run_name,
        "index_csv": str(index_path),
        "exercise_filter": args.exercise,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "channels": args.channels,
        "kernel_size": args.kernel_size,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
        "patience": args.patience,
        "seed": args.seed,
        "loss": args.loss,
        "eval_transform": args.eval_transform,
        "selection_metric": args.selection_metric,
        "sampler": args.sampler,
        "time_warp_range": args.time_warp_range,
        "feature_noise_std": args.feature_noise_std,
        "frame_dropout_prob": args.frame_dropout_prob,
        "camera_motion_std": args.camera_motion_std,
        "camera_zoom_std": args.camera_zoom_std,
        "joint_occlusion_prob": args.joint_occlusion_prob,
        "joint_occlusion_min_ratio": args.joint_occlusion_min_ratio,
        "joint_occlusion_max_ratio": args.joint_occlusion_max_ratio,
        "joint_occlusion_max_joints": args.joint_occlusion_max_joints,
        "keypoint_profile": args.keypoint_profile,
        "resolved_keypoint_profile": resolved_profile_name,
        "profile_strength": args.profile_strength,
        "device": str(device),
        "input_dim": input_dim,
        "train_rows": split_counts["train_rows"],
        "valid_rows": split_counts["valid_rows"],
        "exercise_counts": dict(exercise_counter),
    }

    summary = {
        "best_epoch": best_epoch,
        "best_selection_metric": best_valid_metric,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
    }

    write_csv(output_dir / "history.csv", list(history_rows[0].keys()), history_rows)
    write_csv(
        output_dir / "predictions.csv",
        ["name", "type", "split", "true_count", "raw_pred_count", "eval_pred_count", "abs_error"],
        combined_rows,
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(output_dir / "feature_mean.npy", feature_mean)
    np.save(output_dir / "feature_std.npy", feature_std)
    if feature_weight_vector is not None:
        np.save(output_dir / "feature_weight_vector.npy", feature_weight_vector)
    torch.save(best_state, output_dir / "best_model.pt")

    print("\nSaved outputs:")
    print(" -", output_dir / "config.json")
    print(" -", output_dir / "history.csv")
    print(" -", output_dir / "predictions.csv")
    print(" -", output_dir / "metrics_summary.json")
    if feature_weight_vector is not None:
        print(" -", output_dir / "feature_weight_vector.npy")
    print(" -", output_dir / "best_model.pt")
    print("\nBest valid metrics:")
    print(json.dumps(valid_metrics, indent=2))


if __name__ == "__main__":
    main()

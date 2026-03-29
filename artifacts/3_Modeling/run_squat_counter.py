#!/usr/bin/env python3
"""
Squat-only counting runtime for the project prototype.

This script packages the strongest deployable part of the repo into a single
runtime surface:
1. Optional YOLO pose extraction from a video.
2. Squat-specific engineered feature construction.
3. Dedicated squat TCN inference, with FSM retained as a fallback/reference.

It is intentionally narrow. It does not attempt multi-exercise recognition or
the routed architecture; it exposes the current best practical squat-only
prototype.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime dependency guard
    np = None
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None

try:
    import cv2
except Exception as exc:  # pragma: no cover - runtime dependency guard
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - runtime dependency guard
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - runtime dependency guard
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def load_pose_extractor_module():
    module_path = Path(__file__).resolve().with_name("pose_feature_extraction.py")
    module_name = "pose_feature_extraction_runtime"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pose_feature_extraction.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


POSE_EXTRACTOR = load_pose_extractor_module()
MODEL_DEFAULT = POSE_EXTRACTOR.MODEL_DEFAULT
extract_from_video = POSE_EXTRACTOR.extract_from_video


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
KPT = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}

FEATURE_COLUMNS = [
    "frame_idx",
    "left_knee_angle",
    "right_knee_angle",
    "avg_knee_angle",
    "knee_flex",
    "left_hip_angle",
    "right_hip_angle",
    "avg_hip_angle",
    "hip_center_y",
    "knee_center_y",
    "ankle_center_y",
    "hip_drop",
    "leg_extension",
    "hip_velocity",
    "frame_valid",
    "mean_conf",
]
FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_COLUMNS)}

CONF_THRESHOLD = 0.25
EMA_ALPHA = 0.2
MIN_SCALE = 1e-6
LOWER_BODY_WEIGHT_MAP = {
    "left_hip": 2.0,
    "right_hip": 2.0,
    "left_knee": 2.0,
    "right_knee": 2.0,
    "left_ankle": 1.0,
    "right_ankle": 1.0,
}
HIP_KNEE_WEIGHT_MAP = {
    "left_hip": 1.0,
    "right_hip": 1.0,
    "left_knee": 1.0,
    "right_knee": 1.0,
}
LOWER_BODY_NAMES = list(LOWER_BODY_WEIGHT_MAP.keys())
HIP_KNEE_NAMES = list(HIP_KNEE_WEIGHT_MAP.keys())
LOWER_BODY_WEIGHTS = np.array([LOWER_BODY_WEIGHT_MAP[name] for name in LOWER_BODY_NAMES], dtype=np.float32) if np is not None else None
HIP_KNEE_WEIGHTS = np.array([HIP_KNEE_WEIGHT_MAP[name] for name in HIP_KNEE_NAMES], dtype=np.float32) if np is not None else None
LOWER_BODY_IDX = np.array([KPT[name] for name in LOWER_BODY_NAMES], dtype=np.int32) if np is not None else None
HIP_KNEE_IDX = np.array([KPT[name] for name in HIP_KNEE_NAMES], dtype=np.int32) if np is not None else None

BASE_FSM_CFG = {
    "min_conf": 0.8,
    "min_valid_ratio": 0.5,
    "enter_down": 20.0,
    "enter_bottom": 55.0,
    "exit_bottom": 40.0,
    "back_to_up": 15.0,
    "min_bottom_frames": 2,
}

BEST_FSM_CFG = {
    "min_conf": 0.7,
    "min_valid_ratio": 0.35,
    "enter_down": 25.0,
    "enter_bottom": 45.0,
    "exit_bottom": 30.0,
    "back_to_up": 20.0,
    "min_bottom_frames": 2,
}

DEFAULT_TCN_RUN_NAME = "squat_tcn_l1_channels96"


@dataclass
class CountResult:
    pred_count: int
    event_frames: list[int]
    state_trace: list[str]
    valid_frames: int
    mean_conf: float


@dataclass
class TCNCountResult:
    run_name: str
    raw_pred_count: float
    eval_pred_count: float
    practical_pred_count: int
    seq_len: int
    input_dim: int
    eval_transform: str


def resolve_project_dir(project_dir_arg: str | None = None) -> Path:
    if project_dir_arg:
        candidate = Path(project_dir_arg).expanduser().resolve()
        if (candidate / "Data" / "LLSP").exists() and (candidate / "artifacts").exists():
            return candidate
        raise FileNotFoundError(f"Provided --project-dir does not look like the project root: {candidate}")

    script_path = Path(__file__).resolve()
    for base in [script_path.parent, *script_path.parents]:
        if (base / "Data" / "LLSP").exists() and (base / "artifacts").exists():
            return base.resolve()
    raise FileNotFoundError("Could not resolve project directory containing Data/LLSP and artifacts.")


def ensure_numpy() -> None:
    if np is None:
        raise SystemExit(
            "NumPy is required for squat counting runtime. "
            "Install it in the project environment before running this script."
        ) from NUMPY_IMPORT_ERROR


def ensure_video_runtime() -> None:
    missing: list[str] = []
    if cv2 is None:
        missing.append(f"opencv-python: {CV2_IMPORT_ERROR}")
    if YOLO is None:
        missing.append(f"ultralytics: {ULTRALYTICS_IMPORT_ERROR}")
    if missing:
        joined = "; ".join(missing)
        raise SystemExit(
            "Video-based squat counting requires pose extraction dependencies. "
            f"Missing or broken imports: {joined}"
        )


def ensure_torch_runtime() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for TCN squat inference. "
            "Install torch in the project environment before running the TCN backend."
        ) from TORCH_IMPORT_ERROR


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def forward_fill_nan(xy: np.ndarray) -> np.ndarray:
    out = xy.copy()
    for t in range(1, out.shape[0]):
        missing = np.isnan(out[t])
        out[t][missing] = out[t - 1][missing]
    return out


def backward_fill_nan(xy: np.ndarray) -> np.ndarray:
    out = xy.copy()
    for t in range(out.shape[0] - 2, -1, -1):
        missing = np.isnan(out[t])
        out[t][missing] = out[t + 1][missing]
    return out


def ema_smooth(xy: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    out = xy.copy()
    for t in range(1, out.shape[0]):
        out[t] = alpha * out[t] + (1.0 - alpha) * out[t - 1]
    return out


def vector_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    dot = np.sum(ba * bc, axis=-1)
    norm_ba = np.linalg.norm(ba, axis=-1)
    norm_bc = np.linalg.norm(bc, axis=-1)
    denom = np.clip(norm_ba * norm_bc, 1e-6, None)
    cosine = np.clip(dot / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def weighted_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denom = np.clip(weights.sum(), 1e-6, None)
    return (values * weights[None, :]).sum(axis=1) / denom


def load_pose_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 51:
        raise ValueError(f"Expected [T, 51] pose array, got {arr.shape} for {path}")
    return arr.astype(np.float32)


def preprocess_pose(pose: np.ndarray, conf_threshold: float = CONF_THRESHOLD, alpha: float = EMA_ALPHA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = pose[:, :, :2].copy()
    conf = pose[:, :, 2].copy()

    xy[conf < conf_threshold] = np.nan
    valid_mask = conf >= conf_threshold

    xy = forward_fill_nan(xy)
    xy = backward_fill_nan(xy)
    xy = np.nan_to_num(xy, nan=0.0)
    xy = ema_smooth(xy, alpha=alpha)
    return xy, conf, valid_mask


def normalize_pose(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_hip = xy[:, KPT["left_hip"]]
    right_hip = xy[:, KPT["right_hip"]]
    left_shoulder = xy[:, KPT["left_shoulder"]]
    right_shoulder = xy[:, KPT["right_shoulder"]]

    hip_center = midpoint(left_hip, right_hip)
    shoulder_center = midpoint(left_shoulder, right_shoulder)
    scale = np.linalg.norm(shoulder_center - hip_center, axis=-1)
    scale = np.clip(scale, MIN_SCALE, None)
    normalized = (xy - hip_center[:, None, :]) / scale[:, None, None]
    return normalized, hip_center, scale


def build_squat_feature_array_from_pose_array(pose_arr: np.ndarray) -> np.ndarray:
    pose = pose_arr.reshape(pose_arr.shape[0], 17, 3)
    xy, conf, valid_mask = preprocess_pose(pose)
    normalized_xy, _, _ = normalize_pose(xy)

    left_knee_angle = vector_angle(
        normalized_xy[:, KPT["left_hip"]],
        normalized_xy[:, KPT["left_knee"]],
        normalized_xy[:, KPT["left_ankle"]],
    )
    right_knee_angle = vector_angle(
        normalized_xy[:, KPT["right_hip"]],
        normalized_xy[:, KPT["right_knee"]],
        normalized_xy[:, KPT["right_ankle"]],
    )
    left_hip_angle = vector_angle(
        normalized_xy[:, KPT["left_shoulder"]],
        normalized_xy[:, KPT["left_hip"]],
        normalized_xy[:, KPT["left_knee"]],
    )
    right_hip_angle = vector_angle(
        normalized_xy[:, KPT["right_shoulder"]],
        normalized_xy[:, KPT["right_hip"]],
        normalized_xy[:, KPT["right_knee"]],
    )

    hip_center_y = midpoint(normalized_xy[:, KPT["left_hip"]], normalized_xy[:, KPT["right_hip"]])[:, 1]
    knee_center_y = midpoint(normalized_xy[:, KPT["left_knee"]], normalized_xy[:, KPT["right_knee"]])[:, 1]
    ankle_center_y = midpoint(normalized_xy[:, KPT["left_ankle"]], normalized_xy[:, KPT["right_ankle"]])[:, 1]

    lower_body_conf = weighted_average(conf[:, LOWER_BODY_IDX], LOWER_BODY_WEIGHTS)
    lower_body_valid_score = weighted_average(valid_mask[:, LOWER_BODY_IDX].astype(np.float32), LOWER_BODY_WEIGHTS)
    hip_knee_valid_score = weighted_average(valid_mask[:, HIP_KNEE_IDX].astype(np.float32), HIP_KNEE_WEIGHTS)

    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0
    avg_hip_angle = (left_hip_angle + right_hip_angle) / 2.0
    knee_flex = 180.0 - avg_knee_angle
    hip_drop = knee_center_y - hip_center_y
    leg_extension = ankle_center_y - hip_center_y
    if len(hip_center_y) >= 2:
        hip_velocity = np.gradient(hip_center_y)
    else:
        hip_velocity = np.zeros_like(hip_center_y, dtype=np.float32)
    frame_valid = np.where(
        hip_knee_valid_score >= 0.5,
        np.maximum(lower_body_valid_score, hip_knee_valid_score),
        0.0,
    ).astype(np.float32)

    return np.column_stack(
        [
            np.arange(len(normalized_xy), dtype=np.float32),
            left_knee_angle,
            right_knee_angle,
            avg_knee_angle,
            knee_flex,
            left_hip_angle,
            right_hip_angle,
            avg_hip_angle,
            hip_center_y,
            knee_center_y,
            ankle_center_y,
            hip_drop,
            leg_extension,
            hip_velocity,
            frame_valid,
            lower_body_conf,
        ]
    ).astype(np.float32)


def load_squat_feature_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected [T, {len(FEATURE_COLUMNS)}] squat feature array, got {arr.shape} for {path}")
    return arr.astype(np.float32)


def select_feature_columns(feature_arr: np.ndarray, drop_frame_idx: bool) -> np.ndarray:
    if feature_arr.ndim != 2:
        raise ValueError(f"Expected [T, F] feature array, got {feature_arr.shape}")
    if drop_frame_idx and feature_arr.shape[1] >= 2:
        return feature_arr[:, 1:]
    return feature_arr


def resample_sequence(feature_arr: np.ndarray, target_len: int) -> np.ndarray:
    if feature_arr.shape[0] == target_len:
        return feature_arr.astype(np.float32, copy=False)
    if feature_arr.shape[0] == 1:
        return np.repeat(feature_arr, target_len, axis=0).astype(np.float32, copy=False)

    src_x = np.linspace(0.0, 1.0, num=feature_arr.shape[0], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.empty((target_len, feature_arr.shape[1]), dtype=np.float32)
    for feat_idx in range(feature_arr.shape[1]):
        out[:, feat_idx] = np.interp(dst_x, src_x, feature_arr[:, feat_idx]).astype(np.float32)
    return out


def practical_count_from_raw(raw_value: float) -> int:
    return max(0, int(round(float(raw_value))))


def eval_count_from_raw(raw_value: float, eval_transform: str) -> float:
    if eval_transform == "raw":
        return float(raw_value)
    if eval_transform == "round":
        return float(round(raw_value))
    if eval_transform == "round_clip_nonneg":
        return float(max(0, round(raw_value)))
    raise ValueError(f"Unsupported eval_transform={eval_transform!r}")


def resolve_tcn_run_dir(project_dir: Path, run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg).expanduser().resolve()
    else:
        run_dir = (
            project_dir
            / "artifacts"
            / "3_Modeling"
            / "training_outputs"
            / DEFAULT_TCN_RUN_NAME
        )

    required = [
        run_dir / "config.json",
        run_dir / "best_model.pt",
        run_dir / "feature_mean.npy",
        run_dir / "feature_std.npy",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"TCN run directory is missing required files in {run_dir}: {', '.join(missing)}"
        )
    return run_dir


if nn is not None:
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


    class SquatTCNRegressor(nn.Module):
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
else:  # pragma: no cover - import guard for environments without torch
    TemporalBlock = None
    SquatTCNRegressor = None


def prepare_tcn_input_array(
    feature_arr: np.ndarray,
    *,
    seq_len: int,
    drop_frame_idx: bool,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    input_dim: int,
) -> np.ndarray:
    arr = np.nan_to_num(feature_arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr = select_feature_columns(arr, drop_frame_idx=drop_frame_idx)
    if arr.shape[1] != input_dim:
        raise ValueError(
            f"TCN expected {input_dim} input features after selection, got {arr.shape[1]}"
        )
    arr = resample_sequence(arr, target_len=seq_len)
    if feature_mean.shape != (input_dim,) or feature_std.shape != (input_dim,):
        raise ValueError(
            "TCN feature normalization stats do not match the configured input dimension."
        )
    return ((arr - feature_mean) / feature_std).astype(np.float32, copy=False)


def run_tcn_inference(
    feature_arr: np.ndarray,
    *,
    run_dir: Path,
    device_name: str,
) -> TCNCountResult:
    ensure_torch_runtime()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    feature_mean = np.load(run_dir / "feature_mean.npy").astype(np.float32)
    feature_std = np.load(run_dir / "feature_std.npy").astype(np.float32)

    seq_len = int(config["seq_len"])
    input_dim = int(config["input_dim"])
    eval_transform = str(config.get("eval_transform", "raw"))
    prepared = prepare_tcn_input_array(
        feature_arr,
        seq_len=seq_len,
        drop_frame_idx=bool(config.get("drop_frame_idx", False)),
        feature_mean=feature_mean,
        feature_std=feature_std,
        input_dim=input_dim,
    )

    device = torch.device(device_name)
    model = SquatTCNRegressor(
        input_dim=input_dim,
        channels=int(config["channels"]),
        kernel_size=int(config["kernel_size"]),
        num_blocks=int(config["num_blocks"]),
        dropout=float(config["dropout"]),
    ).to(device)
    state_dict = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(prepared[None, :, :]).to(device)
        raw_pred = float(model(x).detach().cpu().item())

    return TCNCountResult(
        run_name=str(config.get("run_name", run_dir.name)),
        raw_pred_count=raw_pred,
        eval_pred_count=eval_count_from_raw(raw_pred, eval_transform),
        practical_pred_count=practical_count_from_raw(raw_pred),
        seq_len=seq_len,
        input_dim=input_dim,
        eval_transform=eval_transform,
    )


def count_squat_reps(feature_arr: np.ndarray, cfg: dict[str, Any]) -> CountResult:
    count = 0
    state = "UP"
    state_trace: list[str] = []
    event_frames: list[int] = []
    bottom_frames = 0

    frame_valid_col = FEATURE_INDEX["frame_valid"]
    mean_conf_col = FEATURE_INDEX["mean_conf"]
    knee_flex_col = FEATURE_INDEX["knee_flex"]
    frame_idx_col = FEATURE_INDEX["frame_idx"]

    for row in feature_arr:
        valid_frame = (
            float(row[frame_valid_col]) >= float(cfg["min_valid_ratio"])
            and float(row[mean_conf_col]) >= float(cfg["min_conf"])
        )
        knee_flex = float(row[knee_flex_col])

        if not valid_frame:
            state_trace.append(state)
            continue

        if state == "UP":
            bottom_frames = 0
            if knee_flex > float(cfg["enter_down"]):
                state = "DESCENDING"

        elif state == "DESCENDING":
            if knee_flex > float(cfg["enter_bottom"]):
                state = "BOTTOM"
                bottom_frames = 1
            elif knee_flex < float(cfg["back_to_up"]):
                state = "UP"

        elif state == "BOTTOM":
            if knee_flex > float(cfg["exit_bottom"]):
                bottom_frames += 1
            else:
                if bottom_frames >= int(cfg["min_bottom_frames"]):
                    state = "ASCENDING"
                else:
                    state = "DESCENDING"

        elif state == "ASCENDING":
            if knee_flex < float(cfg["back_to_up"]):
                count += 1
                event_frames.append(int(row[frame_idx_col]))
                state = "UP"
                bottom_frames = 0
            elif knee_flex > float(cfg["enter_bottom"]):
                state = "BOTTOM"
                bottom_frames = 1

        state_trace.append(state)

    valid_frames = int(np.sum(feature_arr[:, frame_valid_col] >= cfg["min_valid_ratio"]))
    mean_conf = float(np.mean(feature_arr[:, mean_conf_col])) if len(feature_arr) else 0.0
    return CountResult(
        pred_count=count,
        event_frames=event_frames,
        state_trace=state_trace,
        valid_frames=valid_frames,
        mean_conf=mean_conf,
    )


def capture_video_metadata(video_path: Path) -> dict[str, float | int | None]:
    ensure_video_runtime()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"fps": None, "frames_total": None, "duration_sec": None}
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = (frames_total / fps) if fps > 0 else None
        return {"fps": fps if fps > 0 else None, "frames_total": frames_total, "duration_sec": duration_sec}
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the squat-only repetition counter.")
    parser.add_argument("--project-dir", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--video-path", type=Path, default=None, help="Input squat video. If set, pose extraction will run.")
    parser.add_argument("--pose-path", type=Path, default=None, help="Existing [T,51] pose .npy file.")
    parser.add_argument("--feature-path", type=Path, default=None, help="Existing squat feature .npy file.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--save-pose-path", type=Path, default=None, help="Optional path to save extracted pose features.")
    parser.add_argument("--save-feature-path", type=Path, default=None, help="Optional path to save built squat features.")
    parser.add_argument(
        "--counter-backend",
        choices=("auto", "tcn", "fsm"),
        default="auto",
        help="Counting backend. 'auto' prefers the dedicated squat TCN and falls back to FSM if needed.",
    )
    parser.add_argument(
        "--tcn-run-dir",
        type=str,
        default=None,
        help="Optional path to a local squat TCN training-output directory containing best_model.pt.",
    )
    parser.add_argument(
        "--tcn-device",
        type=str,
        default="cpu",
        help="PyTorch device for TCN inference, for example cpu, cuda:0, or mps.",
    )
    parser.add_argument("--model", type=str, default=str(MODEL_DEFAULT), help="YOLO pose model path/name for --video-path mode.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", type=str, default=None, help="Inference device, for example cpu, cuda:0, or mps.")
    parser.add_argument("--disable-track-person", action="store_false", dest="track_person", help="Disable temporal person tracking.")
    parser.add_argument("--track-search-expand", type=float, default=1.6, help="Search-box expansion around the tracked person.")
    parser.add_argument("--track-max-misses", type=int, default=8, help="How many missed pose frames before resetting the tracker.")
    parser.add_argument("--use-base-fsm", action="store_true", help="Use the untuned FSM thresholds instead of the tuned config.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout.")
    return parser.parse_args()


def resolve_input_mode(args: argparse.Namespace) -> str:
    provided = [bool(args.video_path), bool(args.pose_path), bool(args.feature_path)]
    if sum(provided) != 1:
        raise SystemExit("Provide exactly one of --video-path, --pose-path, or --feature-path.")
    if args.video_path:
        return "video"
    if args.pose_path:
        return "pose"
    return "feature"


def main() -> int:
    ensure_numpy()
    args = parse_args()
    project_dir = resolve_project_dir(args.project_dir)
    mode = resolve_input_mode(args)
    cfg = dict(BASE_FSM_CFG if args.use_base_fsm else BEST_FSM_CFG)

    pose_arr: np.ndarray | None = None
    feature_arr: np.ndarray | None = None
    source_info: dict[str, Any] = {"mode": mode}
    video_meta: dict[str, float | int | None] = {"fps": None, "frames_total": None, "duration_sec": None}

    if mode == "feature":
        feature_path = args.feature_path.expanduser().resolve()
        feature_arr = load_squat_feature_array(feature_path)
        source_info["feature_path"] = str(feature_path)

    elif mode == "pose":
        pose_path = args.pose_path.expanduser().resolve()
        pose_arr = load_pose_array(pose_path)
        feature_arr = build_squat_feature_array_from_pose_array(pose_arr)
        source_info["pose_path"] = str(pose_path)

    else:
        ensure_video_runtime()
        video_path = args.video_path.expanduser().resolve()
        if not video_path.exists():
            raise SystemExit(f"Video file not found: {video_path}")
        video_meta = capture_video_metadata(video_path)
        model = YOLO(str(args.model))
        pose_arr, frames_total, frames_used = extract_from_video(
            model=model,
            video_path=video_path,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            track_person=args.track_person,
            track_search_expand=args.track_search_expand,
            track_max_misses=args.track_max_misses,
        )
        feature_arr = build_squat_feature_array_from_pose_array(pose_arr)
        source_info.update(
            {
                "video_path": str(video_path),
                "pose_frames_total": frames_total,
                "pose_frames_used": frames_used,
            }
        )

    assert feature_arr is not None

    if args.save_pose_path and pose_arr is not None:
        args.save_pose_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_pose_path, pose_arr.astype(np.float32))
        source_info["saved_pose_path"] = str(args.save_pose_path.resolve())

    if args.save_feature_path:
        args.save_feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_feature_path, feature_arr.astype(np.float32))
        source_info["saved_feature_path"] = str(args.save_feature_path.resolve())

    fsm_result = count_squat_reps(feature_arr, cfg)
    fsm_prediction = {
        "backend": "fsm",
        "pred_count": fsm_result.pred_count,
        "raw_pred_count": float(fsm_result.pred_count),
        "eval_pred_count": float(fsm_result.pred_count),
        "event_frames": fsm_result.event_frames,
        "valid_frames": fsm_result.valid_frames,
        "mean_conf": fsm_result.mean_conf,
        "feature_frames_total": int(feature_arr.shape[0]),
    }

    requested_backend = args.counter_backend
    resolved_backend = "fsm"
    backend_fallback_reason: str | None = None
    tcn_prediction: dict[str, Any] | None = None
    tcn_run_dir_str: str | None = None

    if requested_backend in {"auto", "tcn"}:
        try:
            tcn_run_dir = resolve_tcn_run_dir(project_dir, args.tcn_run_dir)
            tcn_run_dir_str = str(tcn_run_dir)
            tcn_result = run_tcn_inference(
                feature_arr,
                run_dir=tcn_run_dir,
                device_name=args.tcn_device,
            )
            tcn_prediction = {
                "backend": "tcn",
                "pred_count": tcn_result.practical_pred_count,
                "raw_pred_count": tcn_result.raw_pred_count,
                "eval_pred_count": tcn_result.eval_pred_count,
                "feature_frames_total": int(feature_arr.shape[0]),
                "seq_len": tcn_result.seq_len,
                "input_dim": tcn_result.input_dim,
                "eval_transform": tcn_result.eval_transform,
                "run_name": tcn_result.run_name,
            }
            resolved_backend = "tcn"
        except Exception as exc:
            if requested_backend == "tcn":
                raise SystemExit(f"TCN backend failed: {exc}") from exc
            backend_fallback_reason = str(exc)
            resolved_backend = "fsm"

    primary_prediction = tcn_prediction if resolved_backend == "tcn" and tcn_prediction is not None else fsm_prediction
    payload = {
        "task": "squat_only_rep_counting",
        "project_dir": str(project_dir),
        "source": source_info,
        "backend_selection": {
            "requested": requested_backend,
            "resolved": resolved_backend,
            "fallback_reason": backend_fallback_reason,
            "tcn_run_dir": tcn_run_dir_str,
        },
        "prediction": primary_prediction,
        "reference_predictions": {
            "fsm": fsm_prediction,
            "tcn": tcn_prediction,
        },
        "reference_configs": {
            "fsm": cfg,
        },
        "video_metadata": video_meta,
        "notes": [
            "This runtime is squat-only and prefers the dedicated squat TCN when the local training-output directory is available.",
            "The FSM branch remains in the payload as a transparent fallback/reference path.",
            "FSM event frame indices refer to the engineered squat-feature sequence, not guaranteed original raw-video frame numbers.",
            "The practical target for this branch is prototype-level counting, not production deployment.",
        ],
    }

    text = json.dumps(payload, indent=2 if args.pretty or args.output_json else None)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote squat count result to {args.output_json.resolve()}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

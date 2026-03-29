#!/usr/bin/env python3
"""
Build a generic normalized pose-sequence dataset from YOLO pose features.

This stage is the widening successor to the squat-only engineered-feature step.
It keeps the representation generic so downstream counting models can be
exercise-agnostic or later switch away from pose-specific assumptions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "NumPy is required to build the pose-sequence dataset."
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = PROJECT_DIR / "Data" / "LLSP" / "annotation_cleaned"

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

TORSO_NAMES = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generic normalized pose sequences.")
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=ANNOTATION_DIR / "pose_feature_index.csv",
        help="Input pose feature index CSV.",
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=ANNOTATION_DIR / "pose_sequences",
        help="Directory where normalized pose sequence .npy files will be written.",
    )
    parser.add_argument(
        "--output-index-csv",
        type=Path,
        default=ANNOTATION_DIR / "pose_sequence_index.csv",
        help="Output CSV mapping videos to normalized sequence paths.",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=ANNOTATION_DIR / "pose_sequence_summary.csv",
        help="Output summary CSV for the generated sequences.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Keypoint confidence threshold used for gap filling and quality metrics.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor for the xy tracks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing normalized sequence files.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"name", "feature_path", "type", "split", "count"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    filtered = [row for row in rows if (row.get("count") or "").strip()]
    skipped = len(rows) - len(filtered)
    if skipped:
        missing_names = [row["name"].strip() for row in rows if not (row.get("count") or "").strip()]
        preview = ", ".join(missing_names[:5])
        print(f"Skipped {skipped} row(s) with missing count in {path.name}: {preview}")
    return filtered


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


def ema_smooth(xy: np.ndarray, alpha: float) -> np.ndarray:
    out = xy.copy()
    for t in range(1, out.shape[0]):
        out[t] = alpha * out[t] + (1.0 - alpha) * out[t - 1]
    return out


def preprocess_pose(pose: np.ndarray, conf_threshold: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = pose[:, :, :2].copy()
    conf = pose[:, :, 2].copy()
    valid = conf >= conf_threshold

    xy[~valid] = np.nan
    xy = forward_fill_nan(xy)
    xy = backward_fill_nan(xy)
    xy = np.nan_to_num(xy, nan=0.0)
    xy = ema_smooth(xy, alpha=alpha)
    return xy, conf, valid


def compute_torso_center_and_scale(xy: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    torso_idx = np.array([KPT[name] for name in TORSO_NAMES], dtype=np.int32)
    torso_xy = xy[:, torso_idx, :]
    torso_valid = valid[:, torso_idx]

    counts = torso_valid.sum(axis=1, keepdims=True).astype(np.float32)
    weighted_sum = (torso_xy * torso_valid[..., None].astype(np.float32)).sum(axis=1)
    center = weighted_sum / np.maximum(counts, 1.0)

    diffs = torso_xy - center[:, None, :]
    dists = np.linalg.norm(diffs, axis=2)
    dists = np.where(torso_valid, dists, np.nan)
    scale = np.nanmedian(dists, axis=1)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    scale = np.where(scale < 1e-3, 1.0, scale)
    return center.astype(np.float32), scale.astype(np.float32)


def normalize_pose(xy: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center, scale = compute_torso_center_and_scale(xy, valid)
    normalized = (xy - center[:, None, :]) / scale[:, None, None]
    return normalized.astype(np.float32), center, scale


def build_sequence_array(normalized_xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    return np.concatenate([normalized_xy, conf[..., None]], axis=2).reshape(normalized_xy.shape[0], -1)


@dataclass
class SequenceSummary:
    name: str
    pose_path: str
    sequence_path: str
    type: str
    split: str
    count: str
    status: str
    frames_total: int
    frames_valid: int
    valid_ratio: float
    mean_conf: float
    message: str


def process_row(
    row: dict[str, str],
    sequence_dir: Path,
    conf_threshold: float,
    alpha: float,
    overwrite: bool,
) -> SequenceSummary:
    name = row["name"].strip()
    pose_path = Path(row["feature_path"]).expanduser().resolve()
    sequence_path = sequence_dir / f"{pose_path.stem}_sequence.npy"

    if sequence_path.exists() and not overwrite:
        arr = np.load(sequence_path)
        return SequenceSummary(
            name=name,
            pose_path=str(pose_path),
            sequence_path=str(sequence_path),
            type=row["type"].strip(),
            split=row["split"].strip(),
            count=row["count"].strip(),
            status="skipped_exists",
            frames_total=int(arr.shape[0]),
            frames_valid=0,
            valid_ratio=0.0,
            mean_conf=0.0,
            message="",
        )

    try:
        raw = np.load(pose_path).astype(np.float32)
        if raw.ndim != 2 or raw.shape[1] != 51:
            raise ValueError(f"Expected [T, 51], got {raw.shape}")
        pose = raw.reshape(raw.shape[0], 17, 3)
        xy, conf, valid = preprocess_pose(pose, conf_threshold=conf_threshold, alpha=alpha)
        normalized_xy, _, _ = normalize_pose(xy, valid)
        sequence = build_sequence_array(normalized_xy, conf)
        sequence_dir.mkdir(parents=True, exist_ok=True)
        np.save(sequence_path, sequence.astype(np.float32))

        frame_valid = valid.mean(axis=1) >= 0.5
        frames_total = int(sequence.shape[0])
        frames_valid = int(frame_valid.sum())
        valid_ratio = float(frames_valid / frames_total) if frames_total else 0.0
        mean_conf = float(conf.mean()) if frames_total else 0.0

        return SequenceSummary(
            name=name,
            pose_path=str(pose_path),
            sequence_path=str(sequence_path),
            type=row["type"].strip(),
            split=row["split"].strip(),
            count=row["count"].strip(),
            status="ok",
            frames_total=frames_total,
            frames_valid=frames_valid,
            valid_ratio=valid_ratio,
            mean_conf=mean_conf,
            message="",
        )
    except Exception as exc:
        return SequenceSummary(
            name=name,
            pose_path=str(pose_path),
            sequence_path=str(sequence_path),
            type=row["type"].strip(),
            split=row["split"].strip(),
            count=row["count"].strip(),
            status="failed",
            frames_total=0,
            frames_valid=0,
            valid_ratio=0.0,
            mean_conf=0.0,
            message=str(exc),
        )


def write_outputs(
    output_index_csv: Path,
    output_summary_csv: Path,
    summaries: list[SequenceSummary],
) -> None:
    output_index_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_index_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "sequence_path", "type", "split", "count"],
        )
        writer.writeheader()
        for summary in summaries:
            if summary.status in {"ok", "skipped_exists"}:
                writer.writerow(
                    {
                        "name": summary.name,
                        "sequence_path": summary.sequence_path,
                        "type": summary.type,
                        "split": summary.split,
                        "count": summary.count,
                    }
                )

    with output_summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "type",
                "split",
                "count",
                "pose_path",
                "sequence_path",
                "status",
                "frames_total",
                "frames_valid",
                "valid_ratio",
                "mean_conf",
                "message",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.index_csv.expanduser().resolve())
    summaries = [
        process_row(
            row,
            sequence_dir=args.sequence_dir.expanduser().resolve(),
            conf_threshold=args.conf_threshold,
            alpha=args.ema_alpha,
            overwrite=args.overwrite,
        )
        for row in rows
    ]
    write_outputs(
        args.output_index_csv.expanduser().resolve(),
        args.output_summary_csv.expanduser().resolve(),
        summaries,
    )

    counts: dict[str, int] = {}
    for summary in summaries:
        counts[summary.status] = counts.get(summary.status, 0) + 1
    print(f"Wrote {len(summaries)} sequence summaries")
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

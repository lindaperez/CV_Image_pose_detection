#!/usr/bin/env python3
"""
Extract frozen RGB frame features for a controlled exercise subset.

This script reads the existing pose-feature index for split/count metadata,
resolves the corresponding raw video files by name, samples frames uniformly,
and writes per-video RGB feature sequences using a frozen torchvision backbone.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("OpenCV is required to extract RGB frame features.") from exc

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("NumPy is required to extract RGB frame features.") from exc

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("PyTorch is required to extract RGB frame features.") from exc

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("Pillow is required to extract RGB frame features.") from exc

try:
    from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "torchvision is required to extract RGB frame features."
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = PROJECT_DIR / "Data" / "LLSP" / "annotation_cleaned"


@dataclass
class ExtractionSummary:
    name: str
    video_path: str
    feature_path: str
    type: str
    split: str
    count: str
    status: str
    frames_total: int
    frames_used: int
    feature_dim: int
    backbone: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen RGB frame features from raw videos.")
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=ANNOTATION_DIR / "pose_feature_index.csv",
        help="Input CSV with columns name, type, split, count.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=PROJECT_DIR / "Data" / "LLSP" / "video",
        help="Directory containing raw .mp4 videos.",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=None,
        help="Output directory for RGB feature sequences. Defaults to annotation_cleaned/rgb_<backbone>_features.",
    )
    parser.add_argument(
        "--output-index-csv",
        type=Path,
        default=None,
        help="Output CSV mapping videos to RGB feature paths. Defaults to annotation_cleaned/rgb_feature_index_<backbone>_selected.csv.",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=None,
        help="Output summary CSV for extraction status. Defaults to annotation_cleaned/rgb_feature_summary_<backbone>_selected.csv.",
    )
    parser.add_argument(
        "--exercise",
        action="append",
        default=[],
        help="Exercise to include. Repeat the flag to include multiple exercises.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=256,
        help="Maximum number of uniformly sampled frames per video.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for frozen backbone inference.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Extraction device. Use auto, cpu, cuda, or an explicit torch device like cuda:0.",
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet50"],
        default="resnet18",
        help="Frozen torchvision backbone used to encode sampled frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing RGB feature files.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print progress every N processed videos.",
    )
    parser.add_argument(
        "--save-progress-every",
        type=int,
        default=10,
        help="Rewrite the output index and summary CSVs every N processed videos. Use 0 to disable intermediate saves.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"name", "type", "split", "count"}
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


def normalize_exercises(raw_exercises: list[str]) -> set[str]:
    return {value.strip() for value in raw_exercises if value.strip()}


def filter_rows(rows: list[dict[str, str]], exercises: set[str]) -> list[dict[str, str]]:
    if not exercises:
        return rows
    return [row for row in rows if row["type"].strip() in exercises]


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    backbone = args.backbone
    feature_dir = (
        args.feature_dir.expanduser().resolve()
        if args.feature_dir
        else (ANNOTATION_DIR / f"rgb_{backbone}_features").resolve()
    )
    output_index_csv = (
        args.output_index_csv.expanduser().resolve()
        if args.output_index_csv
        else (ANNOTATION_DIR / f"rgb_feature_index_{backbone}_selected.csv").resolve()
    )
    output_summary_csv = (
        args.output_summary_csv.expanduser().resolve()
        if args.output_summary_csv
        else (ANNOTATION_DIR / f"rgb_feature_summary_{backbone}_selected.csv").resolve()
    )
    return feature_dir, output_index_csv, output_summary_csv


def build_video_lookup(video_dir: Path) -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = {}
    for path in sorted(video_dir.rglob("*.mp4"), key=lambda p: str(p)):
        lookup.setdefault(path.name.lower(), []).append(path.resolve())
    return lookup


def resolve_video_path(video_dir: Path, name: str, lookup: dict[str, list[Path]]) -> Path | None:
    candidates = lookup.get(name.lower())
    if not candidates:
        direct = (video_dir / name).resolve()
        return direct if direct.exists() else None
    return candidates[0]


def sample_frame_indices(num_frames: int, max_frames: int) -> list[int]:
    if num_frames <= 0:
        return []
    if num_frames <= max_frames:
        return list(range(num_frames))
    return np.linspace(0, num_frames - 1, num=max_frames, dtype=np.int32).tolist()


def load_sampled_frames(video_path: Path, max_frames: int) -> tuple[list[np.ndarray], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 0

    indices = sample_frame_indices(total_frames, max_frames)
    target_index_set = set(indices)
    frames: list[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx in target_index_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) >= len(indices):
                break
        frame_idx += 1
    cap.release()
    return frames, total_frames


def build_backbone(backbone_name: str, device: torch.device) -> tuple[nn.Module, object, int]:
    if backbone_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        feature_dim = 512
    elif backbone_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        feature_dim = 2048
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    transform = weights.transforms()
    return model, transform, feature_dim


def encode_frames(
    frames: list[np.ndarray],
    model: nn.Module,
    transform: object,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not frames:
        return np.zeros((0, 512), dtype=np.float32)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            inputs = torch.stack([transform(Image.fromarray(frame)) for frame in chunk], dim=0).to(device)
            features = model(inputs).detach().cpu().numpy().astype(np.float32)
            batches.append(features)
    return np.concatenate(batches, axis=0) if batches else np.zeros((0, 512), dtype=np.float32)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(summaries: list[ExtractionSummary]) -> list[dict[str, object]]:
    return [
        {
            "name": item.name,
            "video_path": item.video_path,
            "feature_path": item.feature_path,
            "type": item.type,
            "split": item.split,
            "count": item.count,
            "status": item.status,
            "frames_total": item.frames_total,
            "frames_used": item.frames_used,
            "feature_dim": item.feature_dim,
            "backbone": item.backbone,
            "message": item.message,
        }
        for item in summaries
    ]


def flush_progress_outputs(
    *,
    output_index_csv: Path,
    output_summary_csv: Path,
    ok_rows: list[dict[str, object]],
    summaries: list[ExtractionSummary],
) -> None:
    write_csv(
        output_index_csv,
        ["name", "feature_path", "type", "split", "count", "backbone"],
        ok_rows,
    )
    write_csv(
        output_summary_csv,
        [
            "name",
            "video_path",
            "feature_path",
            "type",
            "split",
            "count",
            "status",
            "frames_total",
            "frames_used",
            "feature_dim",
            "backbone",
            "message",
        ],
        summarize_rows(summaries),
    )


def main() -> None:
    args = parse_args()
    index_path = args.index_csv.expanduser().resolve()
    video_dir = args.video_dir.expanduser().resolve()
    feature_dir, output_index_csv, output_summary_csv = resolve_output_paths(args)

    if not index_path.exists():
        raise FileNotFoundError(f"Missing required file: {index_path}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Missing required directory: {video_dir}")

    rows = load_rows(index_path)
    exercises = normalize_exercises(args.exercise)
    rows = filter_rows(rows, exercises)
    if not rows:
        scope = ", ".join(sorted(exercises)) if exercises else "the full index"
        raise SystemExit(f"No rows selected from {index_path.name} for {scope}.")

    lookup = build_video_lookup(video_dir)
    device = choose_device(args.device)
    model, transform, feature_dim = build_backbone(args.backbone, device)

    summaries: list[ExtractionSummary] = []
    ok_rows: list[dict[str, object]] = []
    total_rows = len(rows)
    log_every = max(int(args.log_every), 1)
    save_every = int(args.save_progress_every)

    print(
        f"Starting RGB extraction for {total_rows} videos"
        f" across exercises={sorted(exercises) if exercises else 'all'}"
        f" | backbone={args.backbone} | max_frames={args.max_frames}"
        f" | batch_size={args.batch_size} | device={device}",
        flush=True,
    )

    for row_idx, row in enumerate(rows, start=1):
        name = row["name"].strip()
        exercise = row["type"].strip()
        split = row["split"].strip()
        count = row["count"].strip()
        video_path = resolve_video_path(video_dir, name, lookup)
        feature_path = (feature_dir / f"{Path(name).stem}_rgb_{args.backbone}.npy").resolve()
        progress_summary: ExtractionSummary | None = None

        if feature_path.exists() and not args.overwrite:
            arr = np.load(feature_path)
            progress_summary = ExtractionSummary(
                name=name,
                video_path=str(video_path) if video_path else "",
                feature_path=str(feature_path),
                type=exercise,
                split=split,
                count=count,
                status="skipped_exists",
                frames_total=int(arr.shape[0]),
                frames_used=int(arr.shape[0]),
                feature_dim=int(arr.shape[1]) if arr.ndim == 2 else 0,
                backbone=args.backbone,
                message="",
            )
            summaries.append(progress_summary)
            ok_rows.append(
                {
                    "name": name,
                    "feature_path": str(feature_path),
                    "type": exercise,
                    "split": split,
                    "count": count,
                    "backbone": args.backbone,
                }
            )
        elif video_path is None or not video_path.exists():
            progress_summary = ExtractionSummary(
                name=name,
                video_path=str(video_path) if video_path else "",
                feature_path=str(feature_path),
                type=exercise,
                split=split,
                count=count,
                status="failed",
                frames_total=0,
                frames_used=0,
                feature_dim=0,
                backbone=args.backbone,
                message="video_not_found",
            )
            summaries.append(progress_summary)
        else:
            try:
                frames, frames_total = load_sampled_frames(video_path, args.max_frames)
                feature_array = encode_frames(frames, model, transform, device, args.batch_size)
                if feature_array.ndim != 2 or feature_array.shape[0] == 0:
                    raise RuntimeError("no_frames_encoded")
                if feature_array.shape[1] != feature_dim:
                    raise RuntimeError(
                        f"unexpected_feature_dim:{feature_array.shape[1]} expected:{feature_dim}"
                    )
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(feature_path, feature_array.astype(np.float32))
                progress_summary = ExtractionSummary(
                    name=name,
                    video_path=str(video_path),
                    feature_path=str(feature_path),
                    type=exercise,
                    split=split,
                    count=count,
                    status="ok",
                    frames_total=frames_total,
                    frames_used=int(feature_array.shape[0]),
                    feature_dim=int(feature_array.shape[1]),
                    backbone=args.backbone,
                    message="",
                )
                summaries.append(progress_summary)
                ok_rows.append(
                    {
                        "name": name,
                        "feature_path": str(feature_path),
                        "type": exercise,
                        "split": split,
                        "count": count,
                        "backbone": args.backbone,
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime extraction errors
                progress_summary = ExtractionSummary(
                    name=name,
                    video_path=str(video_path),
                    feature_path=str(feature_path),
                    type=exercise,
                    split=split,
                    count=count,
                    status="failed",
                    frames_total=0,
                    frames_used=0,
                    feature_dim=0,
                    backbone=args.backbone,
                    message=str(exc),
                )
                summaries.append(progress_summary)

        if progress_summary and (row_idx % log_every == 0 or row_idx == total_rows):
            message = progress_summary.message if progress_summary.message else "-"
            print(
                f"[{row_idx}/{total_rows}] {progress_summary.status}: {name}"
                f" | exercise={exercise} split={split}"
                f" | frames_total={progress_summary.frames_total}"
                f" used={progress_summary.frames_used}"
                f" feat_dim={progress_summary.feature_dim}"
                f" | msg={message}",
                flush=True,
            )

        if save_every > 0 and row_idx % save_every == 0:
            flush_progress_outputs(
                output_index_csv=output_index_csv,
                output_summary_csv=output_summary_csv,
                ok_rows=ok_rows,
                summaries=summaries,
            )

    flush_progress_outputs(
        output_index_csv=output_index_csv,
        output_summary_csv=output_summary_csv,
        ok_rows=ok_rows,
        summaries=summaries,
    )

    status_counts: dict[str, int] = {}
    for item in summaries:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1

    print(f"Wrote {len(ok_rows)} rows to {output_index_csv}", flush=True)
    print(f"Wrote {len(summaries)} summary rows to {output_summary_csv}", flush=True)
    print(status_counts, flush=True)


if __name__ == "__main__":
    main()

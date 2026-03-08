#!/usr/bin/env python3
"""
Pose feature extraction for RepCount train/valid pipeline.

Reads pose_feature_index.csv, runs YOLO pose on each video, and saves
per-video temporal features as .npy arrays with shape [T, F].

Default feature vector per frame:
- 17 keypoints * (x, y, conf) => F=51

Usage example:
python CV_Image_pose_detection/artifacts/modeling/pose_feature_extraction.py \
  --index-csv CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_feature_index.csv \
  --video-dir CV_Image_pose_detection/Data/LLSP/video \
  --model yolo11n-pose.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

# Resolve repository-local defaults from this script path (not from shell cwd)
PROJECT_DIR = Path(__file__).resolve().parents[2]  # .../CV_Image_pose_detection
ANNOTATION_CLEANED_DIR = PROJECT_DIR / "Data" / "LLSP" / "annotation_cleaned"
VIDEO_DIR_DEFAULT = PROJECT_DIR / "Data" / "LLSP" / "video"


@dataclass
class ExtractResult:
    name: str
    video_path: str
    feature_path: str
    status: str
    frames_total: int
    frames_used: int
    feat_dim: int
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract YOLO pose features to .npy files.")
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=ANNOTATION_CLEANED_DIR / "pose_feature_index.csv",
        help="Path to pose_feature_index.csv with columns: name, feature_path",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=VIDEO_DIR_DEFAULT,
        help="Directory containing input videos named as in CSV `name` column.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-pose.pt",
        help="Ultralytics YOLO pose model path/name.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Optional cap for debugging. 0 = process all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy features.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ANNOTATION_CLEANED_DIR / "pose_extraction_report.csv",
        help="CSV report path.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=ANNOTATION_CLEANED_DIR / "pose_extraction_summary.json",
        help="JSON summary path.",
    )
    return parser.parse_args()


def load_index(index_csv: Path) -> List[Dict[str, str]]:
    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")
    rows: List[Dict[str, str]] = []
    with index_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"name", "feature_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Index CSV missing columns: {missing}")
        for row in reader:
            rows.append({"name": row["name"].strip(), "feature_path": row["feature_path"].strip()})
    return rows


def build_video_lookup(video_dir: Path) -> Dict[str, List[Path]]:
    """
    Build a basename->fullpath lookup recursively under video_dir.
    Key is lowercase filename (e.g., stu6_57.mp4).
    """
    lookup: Dict[str, List[Path]] = {}
    for p in video_dir.rglob("*.mp4"):
        key = p.name.strip().lower()
        lookup.setdefault(key, []).append(p.resolve())
    return lookup


def resolve_video_path(video_dir: Path, name: str, lookup: Dict[str, List[Path]]) -> Optional[Path]:
    """
    Resolve video path with:
    1) direct path: video_dir/name
    2) recursive lookup by basename
    If multiple matches exist, prefer deterministic lexicographic first path.
    """
    direct = (video_dir / name).resolve()
    if direct.exists():
        return direct

    key = name.strip().lower()
    candidates = lookup.get(key, [])
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda x: str(x))
    return candidates[0]


def pick_main_person(result) -> Optional[int]:
    """Pick one person index per frame (largest bbox area, fallback highest score)."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))

    widths = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])
    heights = np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])
    areas = widths * heights

    if len(areas) == 0:
        return None
    # rank by area first, then conf as tie-breaker
    idx = int(np.lexsort((conf, areas))[-1])
    return idx


def extract_from_video(
    model: YOLO,
    video_path: Path,
    conf: float,
    imgsz: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Returns:
    - feature array [T, F] (float32)
    - total frames read
    - frames used (with successful pose extraction)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    feats: List[np.ndarray] = []
    frames_total = 0
    frames_used = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_total += 1

            pred = model.predict(
                source=frame,
                conf=conf,
                imgsz=imgsz,
                verbose=False,
            )
            if not pred:
                continue
            res = pred[0]
            kpts = getattr(res, "keypoints", None)
            if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
                continue

            person_idx = pick_main_person(res)
            if person_idx is None:
                continue

            xy = kpts.xy[person_idx].detach().cpu().numpy()  # [K,2]
            if kpts.conf is not None:
                kc = kpts.conf[person_idx].detach().cpu().numpy()[:, None]  # [K,1]
            else:
                kc = np.ones((xy.shape[0], 1), dtype=np.float32)
            frame_feat = np.concatenate([xy, kc], axis=1).reshape(-1).astype(np.float32)  # [K*3]
            feats.append(frame_feat)
            frames_used += 1
    finally:
        cap.release()

    if not feats:
        # If nothing detected, keep a single zero frame so downstream can proceed.
        feat_dim = 51
        arr = np.zeros((1, feat_dim), dtype=np.float32)
    else:
        arr = np.stack(feats, axis=0).astype(np.float32)
    return arr, frames_total, frames_used


def write_report(report_path: Path, rows: List[ExtractResult]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "video_path",
                "feature_path",
                "status",
                "frames_total",
                "frames_used",
                "feat_dim",
                "message",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.name,
                    r.video_path,
                    r.feature_path,
                    r.status,
                    r.frames_total,
                    r.frames_used,
                    r.feat_dim,
                    r.message,
                ]
            )


def write_summary(summary_path: Path, rows: List[ExtractResult], args: argparse.Namespace) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    ok = sum(1 for r in rows if r.status == "ok")
    skipped = sum(1 for r in rows if r.status == "skipped_exists")
    failed = sum(1 for r in rows if r.status == "failed")
    empty_pose = sum(1 for r in rows if r.status == "ok" and r.frames_used == 0)
    payload = {
        "total_rows": total,
        "ok": ok,
        "skipped_exists": skipped,
        "failed": failed,
        "ok_with_zero_pose_frames": empty_pose,
        "args": {
            "index_csv": str(args.index_csv),
            "video_dir": str(args.video_dir),
            "model": args.model,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "overwrite": args.overwrite,
            "max_videos": args.max_videos,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def main() -> int:
    args = parse_args()

    if YOLO is None:
        print("ERROR: ultralytics is not available.")
        print("Import error:", IMPORT_ERROR)
        print("Install with: pip install ultralytics")
        return 2

    rows = load_index(args.index_csv)
    if args.max_videos > 0:
        rows = rows[: args.max_videos]

    args.video_dir = args.video_dir.resolve()
    video_lookup = build_video_lookup(args.video_dir)
    print(f"Indexed {sum(len(v) for v in video_lookup.values())} videos under {args.video_dir}")
    model = YOLO(args.model)

    results: List[ExtractResult] = []
    total = len(rows)
    print(f"Processing {total} videos...")

    for i, row in enumerate(rows, start=1):
        name = row["name"]
        feature_path = Path(row["feature_path"]).expanduser().resolve()
        resolved = resolve_video_path(args.video_dir, name, video_lookup)
        video_path = resolved if resolved is not None else (args.video_dir / name).resolve()

        feature_path.parent.mkdir(parents=True, exist_ok=True)

        if feature_path.exists() and not args.overwrite:
            results.append(
                ExtractResult(
                    name=name,
                    video_path=str(video_path),
                    feature_path=str(feature_path),
                    status="skipped_exists",
                    frames_total=0,
                    frames_used=0,
                    feat_dim=0,
                    message="feature file already exists",
                )
            )
            if i % 25 == 0 or i == total:
                print(f"[{i}/{total}] skipped_exists: {name}")
            continue

        if resolved is None or not video_path.exists():
            results.append(
                ExtractResult(
                    name=name,
                    video_path=str(video_path),
                    feature_path=str(feature_path),
                    status="failed",
                    frames_total=0,
                    frames_used=0,
                    feat_dim=0,
                    message="video file not found",
                )
            )
            print(f"[{i}/{total}] FAILED (missing video): {name}")
            continue

        try:
            arr, frames_total, frames_used = extract_from_video(
                model=model,
                video_path=video_path,
                conf=args.conf,
                imgsz=args.imgsz,
            )
            np.save(feature_path, arr)
            results.append(
                ExtractResult(
                    name=name,
                    video_path=str(video_path),
                    feature_path=str(feature_path),
                    status="ok",
                    frames_total=int(frames_total),
                    frames_used=int(frames_used),
                    feat_dim=int(arr.shape[1]),
                    message="",
                )
            )
            if i % 25 == 0 or i == total:
                print(
                    f"[{i}/{total}] ok: {name} | frames={frames_total} used={frames_used} "
                    f"shape={tuple(arr.shape)}"
                )
        except Exception as exc:  # pragma: no cover
            results.append(
                ExtractResult(
                    name=name,
                    video_path=str(video_path),
                    feature_path=str(feature_path),
                    status="failed",
                    frames_total=0,
                    frames_used=0,
                    feat_dim=0,
                    message=str(exc),
                )
            )
            print(f"[{i}/{total}] FAILED: {name} | {exc}")

    write_report(args.report_path, results)
    write_summary(args.summary_path, results, args)

    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skipped_exists")
    failed = sum(1 for r in results if r.status == "failed")
    print("\nDone.")
    print(f"ok={ok}, skipped_exists={skipped}, failed={failed}")
    print("report:", args.report_path.resolve())
    print("summary:", args.summary_path.resolve())
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

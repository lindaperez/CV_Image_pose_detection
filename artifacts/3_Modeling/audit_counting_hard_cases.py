#!/usr/bin/env python3
"""
Audit likely data-side failure modes for pose-vs-RGB counting runs.

This script joins:
- pose predictions
- RGB predictions
- pose sequence quality summaries
- optional RGB feature summaries
- local video metadata

and writes a ranked review table so hard cases can be tagged before we make
representation decisions from aggregate metrics alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = PROJECT_DIR / "Data" / "LLSP" / "annotation_cleaned"
VIDEO_DIR = PROJECT_DIR / "Data" / "LLSP" / "video"


@dataclass
class AuditCase:
    name: str
    exercise: str
    split: str
    true_count: float
    pose_pred_count: float
    rgb_pred_count: float
    pose_abs_error: float
    rgb_abs_error: float
    error_gap_rgb_minus_pose: float
    pose_valid_ratio: float | None
    pose_mean_conf: float | None
    rgb_frames_used: float | None
    rgb_backbone: str
    width: int
    height: int
    duration_sec: float
    fps: float
    orientation: str
    issue_tags: list[str]
    audit_bucket: str
    severity: str
    video_path: str
    contact_sheet: str

    def as_row(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": self.exercise,
            "split": self.split,
            "true_count": self.true_count,
            "pose_pred_count": self.pose_pred_count,
            "rgb_pred_count": self.rgb_pred_count,
            "pose_abs_error": self.pose_abs_error,
            "rgb_abs_error": self.rgb_abs_error,
            "error_gap_rgb_minus_pose": self.error_gap_rgb_minus_pose,
            "pose_valid_ratio": "" if self.pose_valid_ratio is None else f"{self.pose_valid_ratio:.6f}",
            "pose_mean_conf": "" if self.pose_mean_conf is None else f"{self.pose_mean_conf:.6f}",
            "rgb_frames_used": "" if self.rgb_frames_used is None else f"{self.rgb_frames_used:.1f}",
            "rgb_backbone": self.rgb_backbone,
            "width": self.width,
            "height": self.height,
            "duration_sec": f"{self.duration_sec:.3f}",
            "fps": f"{self.fps:.3f}",
            "orientation": self.orientation,
            "issue_tags": ",".join(self.issue_tags),
            "audit_bucket": self.audit_bucket,
            "severity": self.severity,
            "video_path": self.video_path,
            "contact_sheet": self.contact_sheet,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit pose-vs-RGB counting hard cases.")
    parser.add_argument(
        "--pose-predictions-csv",
        type=Path,
        required=True,
        help="Pose predictions.csv file for one exercise/run.",
    )
    parser.add_argument(
        "--rgb-predictions-csv",
        type=Path,
        required=True,
        help="RGB predictions.csv file for one exercise/run.",
    )
    parser.add_argument(
        "--pose-summary-csv",
        type=Path,
        default=ANNOTATION_DIR / "pose_sequence_summary.csv",
        help="Pose sequence quality summary CSV.",
    )
    parser.add_argument(
        "--rgb-summary-csv",
        type=Path,
        default=None,
        help="Optional RGB feature summary CSV.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=VIDEO_DIR,
        help="Root LLSP video directory.",
    )
    parser.add_argument(
        "--exercise",
        action="append",
        default=[],
        help="Optional exercise filter. Repeat to include multiple exercises.",
    )
    parser.add_argument(
        "--split",
        default="valid",
        help="Prediction split to audit. Defaults to valid.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of ranked rows to keep in the JSON summary.",
    )
    parser.add_argument(
        "--contact-sheet-limit",
        type=int,
        default=12,
        help="How many ranked rows to render as contact sheets. 0 disables contact sheets.",
    )
    parser.add_argument(
        "--contact-width",
        type=int,
        default=320,
        help="Per-tile width for ffmpeg contact sheets.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults beside the RGB predictions file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults beside the RGB predictions file.",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str | float | int | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def parse_optional_float(value: str | float | int | None) -> float | None:
    parsed = safe_float(value)
    if math.isnan(parsed):
        return None
    return parsed


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def rational_to_float(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    if "/" not in value:
        return safe_float(value)
    num, den = value.split("/", 1)
    den_f = safe_float(den)
    return safe_float(num) / den_f if den_f else 0.0


def build_video_lookup(video_dir: Path) -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = defaultdict(list)
    for video_path in video_dir.rglob("*.mp4"):
        lookup[video_path.name.lower()].append(video_path.resolve())
    return lookup


def ffprobe_video(video_path: Path) -> dict[str, object]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration = safe_float(fmt.get("duration"))
    fps = rational_to_float(stream.get("avg_frame_rate")) or rational_to_float(stream.get("r_frame_rate"))
    return {
        "width": width,
        "height": height,
        "duration_sec": 0.0 if math.isnan(duration) else duration,
        "fps": fps,
        "orientation": "portrait" if height > width else "landscape",
    }


def write_contact_sheet(video_path: Path, output_path: Path, duration_sec: float, contact_width: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = 4.0 / duration_sec if duration_sec > 0.0 else 1.0
    vf = f"fps={fps:.6f},scale={contact_width}:-1:force_original_aspect_ratio=decrease,tile=2x2"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        vf,
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def pose_quality_tags(valid_ratio: float | None, mean_conf: float | None) -> list[str]:
    tags: list[str] = []
    if valid_ratio is None:
        tags.append("missing_pose_summary")
        return tags
    if valid_ratio == 0.0:
        tags.append("no_valid_pose_frames")
    elif valid_ratio < 0.25:
        tags.append("severe_pose_visibility_loss")
    elif valid_ratio < 0.75:
        tags.append("partial_pose_visibility_loss")
    elif valid_ratio < 0.90:
        tags.append("moderate_pose_coverage_loss")

    if mean_conf is None:
        tags.append("missing_pose_confidence")
        return tags
    if mean_conf < 0.25:
        tags.append("extreme_pose_confidence_failure")
    elif mean_conf < 0.40:
        tags.append("very_low_pose_confidence")
    elif mean_conf < 0.50:
        tags.append("low_pose_confidence")
    elif mean_conf < 0.70:
        tags.append("moderate_pose_confidence")

    if valid_ratio < 0.50 and mean_conf < 0.50:
        tags.append("likely_framing_or_occlusion_issue")
    elif valid_ratio < 0.75 and mean_conf >= 0.70:
        tags.append("likely_pose_contract_mismatch")
    elif valid_ratio >= 0.90 and mean_conf < 0.70:
        tags.append("confidence_noise_but_geometry_available")
    return tags


def classify_case(
    pose_abs_error: float,
    rgb_abs_error: float,
    true_count: float,
    valid_ratio: float | None,
    mean_conf: float | None,
    orientation: str,
    width: int,
    height: int,
) -> tuple[list[str], str, str]:
    tags = pose_quality_tags(valid_ratio, mean_conf)
    if orientation == "portrait":
        tags.append("portrait_framing")
    if min(width, height) and min(width, height) < 360:
        tags.append("low_resolution")

    major_error = max(2.0, 0.20 * true_count)
    weak_pose = (
        valid_ratio is not None
        and mean_conf is not None
        and (valid_ratio < 0.85 or mean_conf < 0.65)
    )
    clean_pose = (
        valid_ratio is not None
        and mean_conf is not None
        and valid_ratio >= 0.90
        and mean_conf >= 0.70
    )
    both_high = pose_abs_error > major_error and rgb_abs_error > major_error
    rgb_clear_win = rgb_abs_error + 1.0 < pose_abs_error
    pose_clear_win = pose_abs_error + 1.0 < rgb_abs_error

    if both_high and ("likely_framing_or_occlusion_issue" in tags or "severe_pose_visibility_loss" in tags):
        tags.append("both_models_struggle")
        return tags, "likely_visibility_limited_case", "high"
    if both_high and clean_pose:
        tags.append("both_models_struggle")
        tags.append("likely_semantic_or_definition_ambiguity")
        return tags, "likely_semantic_or_definition_ambiguity", "high"
    if rgb_clear_win and weak_pose:
        tags.append("rgb_recovers_when_pose_is_weak")
        return tags, "rgb_advantage_on_weak_pose", "medium"
    if pose_clear_win and clean_pose:
        tags.append("pose_stronger_on_clean_geometry")
        return tags, "pose_advantage_on_clean_geometry", "medium"
    if both_high:
        tags.append("both_models_struggle")
        return tags, "shared_model_difficulty", "review"
    if rgb_clear_win:
        tags.append("rgb_advantage_without_clear_pose_failure")
        return tags, "rgb_advantage_context_or_semantics", "review"
    if pose_clear_win:
        tags.append("pose_advantage_without_clear_rgb_need")
        return tags, "pose_advantage_structure_friendly", "review"
    return tags, "no_clear_data_issue_signal", "ok"


def rank_key(case: AuditCase) -> tuple[int, float]:
    severity_rank = {"high": 0, "medium": 1, "review": 2, "ok": 3}
    return (
        severity_rank.get(case.severity, 9),
        -max(case.pose_abs_error, case.rgb_abs_error),
    )


def main() -> None:
    args = parse_args()
    exercises = {value.strip() for value in args.exercise if value.strip()}
    split = args.split.strip().lower()

    pose_predictions_path = args.pose_predictions_csv.expanduser().resolve()
    rgb_predictions_path = args.rgb_predictions_csv.expanduser().resolve()
    pose_summary_path = args.pose_summary_csv.expanduser().resolve()
    rgb_summary_path = args.rgb_summary_csv.expanduser().resolve() if args.rgb_summary_csv else None
    video_dir = args.video_dir.expanduser().resolve()

    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else rgb_predictions_path.with_name(f"{rgb_predictions_path.stem}_hard_case_audit.csv")
    )
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else rgb_predictions_path.with_name(f"{rgb_predictions_path.stem}_hard_case_audit_summary.json")
    )
    contact_sheet_dir = output_csv.with_suffix("").with_name(f"{output_csv.stem}_contact_sheets")

    pose_prediction_rows = load_csv_rows(pose_predictions_path)
    rgb_prediction_rows = load_csv_rows(rgb_predictions_path)
    pose_summary_rows = load_csv_rows(pose_summary_path)
    rgb_summary_rows = load_csv_rows(rgb_summary_path) if rgb_summary_path and rgb_summary_path.exists() else []

    pose_by_name = {
        row["name"].strip(): row
        for row in pose_prediction_rows
        if row.get("split", "").strip().lower() == split
    }
    rgb_by_name = {
        row["name"].strip(): row
        for row in rgb_prediction_rows
        if row.get("split", "").strip().lower() == split
    }
    pose_summary_by_name = {row["name"].strip(): row for row in pose_summary_rows}
    rgb_summary_by_name = {row["name"].strip(): row for row in rgb_summary_rows}

    video_lookup = build_video_lookup(video_dir)
    shared_names = sorted(set(pose_by_name) & set(rgb_by_name))
    cases: list[AuditCase] = []

    for name in shared_names:
        pose_row = pose_by_name[name]
        rgb_row = rgb_by_name[name]
        exercise = (rgb_row.get("type") or pose_row.get("type") or "").strip()
        if not exercise:
            if len(exercises) == 1:
                exercise = next(iter(exercises))
            else:
                exercise = "squat"
        if exercises and exercise not in exercises:
            continue

        true_count = parse_optional_float(rgb_row.get("true_count") or pose_row.get("true_count"))
        pose_pred = parse_optional_float(pose_row.get("eval_pred_count"))
        rgb_pred = parse_optional_float(rgb_row.get("eval_pred_count"))
        pose_abs_error = parse_optional_float(pose_row.get("abs_error"))
        rgb_abs_error = parse_optional_float(rgb_row.get("abs_error"))
        if (
            true_count is None
            or pose_pred is None
            or rgb_pred is None
            or pose_abs_error is None
            or rgb_abs_error is None
        ):
            continue

        pose_summary = pose_summary_by_name.get(name, {})
        rgb_summary = rgb_summary_by_name.get(name, {})
        pose_valid_ratio = parse_optional_float(pose_summary.get("valid_ratio"))
        pose_mean_conf = parse_optional_float(pose_summary.get("mean_conf"))
        rgb_frames_used = parse_optional_float(rgb_summary.get("frames_used"))
        rgb_backbone = rgb_summary.get("backbone", "").strip()

        candidate_paths = video_lookup.get(name.lower(), [])
        video_path = candidate_paths[0] if candidate_paths else None
        metadata = {"width": 0, "height": 0, "duration_sec": 0.0, "fps": 0.0, "orientation": "unknown"}
        if video_path is not None:
            try:
                metadata = ffprobe_video(video_path)
            except Exception:
                metadata = {"width": 0, "height": 0, "duration_sec": 0.0, "fps": 0.0, "orientation": "unknown"}
                tags = ["video_metadata_probe_failed"]
                bucket = "video_metadata_probe_failed"
                severity = "review"
            else:
                tags, bucket, severity = classify_case(
                    pose_abs_error=pose_abs_error,
                    rgb_abs_error=rgb_abs_error,
                    true_count=true_count,
                    valid_ratio=pose_valid_ratio,
                    mean_conf=pose_mean_conf,
                    orientation=str(metadata["orientation"]),
                    width=int(metadata["width"]),
                    height=int(metadata["height"]),
                )
        else:
            tags, bucket, severity = classify_case(
                pose_abs_error=pose_abs_error,
                rgb_abs_error=rgb_abs_error,
                true_count=true_count,
                valid_ratio=pose_valid_ratio,
                mean_conf=pose_mean_conf,
                orientation=str(metadata["orientation"]),
                width=int(metadata["width"]),
                height=int(metadata["height"]),
            )

        cases.append(
            AuditCase(
                name=name,
                exercise=exercise,
                split=split,
                true_count=true_count,
                pose_pred_count=pose_pred,
                rgb_pred_count=rgb_pred,
                pose_abs_error=pose_abs_error,
                rgb_abs_error=rgb_abs_error,
                error_gap_rgb_minus_pose=rgb_abs_error - pose_abs_error,
                pose_valid_ratio=pose_valid_ratio,
                pose_mean_conf=pose_mean_conf,
                rgb_frames_used=rgb_frames_used,
                rgb_backbone=rgb_backbone,
                width=int(metadata["width"]),
                height=int(metadata["height"]),
                duration_sec=float(metadata["duration_sec"]),
                fps=float(metadata["fps"]),
                orientation=str(metadata["orientation"]),
                issue_tags=tags,
                audit_bucket=bucket,
                severity=severity,
                video_path="" if video_path is None else str(video_path),
                contact_sheet="",
            )
        )

    if not cases:
        scope = ", ".join(sorted(exercises)) if exercises else split
        raise SystemExit(f"No shared pose/RGB prediction rows found for {scope}.")

    cases.sort(key=rank_key)

    contact_limit = min(args.contact_sheet_limit, len(cases)) if args.contact_sheet_limit >= 0 else len(cases)
    review_limit = min(args.top_k, len(cases)) if args.top_k >= 0 else len(cases)

    for case in cases[:contact_limit]:
        if not case.video_path:
            continue
        output_path = contact_sheet_dir / f"{Path(case.name).stem}_contact.jpg"
        try:
            write_contact_sheet(Path(case.video_path), output_path, case.duration_sec, args.contact_width)
        except Exception:
            case.issue_tags.append("contact_sheet_failed")
        else:
            case.contact_sheet = str(output_path)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0].as_row().keys()))
        writer.writeheader()
        writer.writerows(case.as_row() for case in cases)

    bucket_counts = Counter(case.audit_bucket for case in cases)
    severity_counts = Counter(case.severity for case in cases)
    per_exercise_buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in cases:
        per_exercise_buckets[case.exercise][case.audit_bucket] += 1

    summary = {
        "pose_predictions_csv": str(pose_predictions_path),
        "rgb_predictions_csv": str(rgb_predictions_path),
        "pose_summary_csv": str(pose_summary_path),
        "rgb_summary_csv": "" if rgb_summary_path is None else str(rgb_summary_path),
        "video_dir": str(video_dir),
        "split": split,
        "exercise_filter": sorted(exercises),
        "rows": len(cases),
        "bucket_counts": dict(bucket_counts),
        "severity_counts": dict(severity_counts),
        "mean_pose_abs_error": mean([case.pose_abs_error for case in cases]),
        "mean_rgb_abs_error": mean([case.rgb_abs_error for case in cases]),
        "per_exercise_bucket_counts": {
            exercise: dict(bucket_counts)
            for exercise, bucket_counts in sorted(per_exercise_buckets.items())
        },
        "top_review_rows": [case.as_row() for case in cases[:review_limit]],
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote hard-case audit rows to {output_csv}")
    print(f"Wrote hard-case audit summary to {output_json}")
    print(
        json.dumps(
            {
                "rows": summary["rows"],
                "bucket_counts": summary["bucket_counts"],
                "severity_counts": summary["severity_counts"],
                "mean_pose_abs_error": summary["mean_pose_abs_error"],
                "mean_rgb_abs_error": summary["mean_rgb_abs_error"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze whether RGB tends to outperform pose when pose quality is weak.

This script joins:
- pose sequence quality summaries
- RGB feature extraction summaries
- pose predictions
- RGB predictions

and produces a per-video comparison table plus a compact JSON summary so we can
check whether the chosen representation matches the data conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = PROJECT_DIR / "Data" / "LLSP" / "annotation_cleaned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze pose-vs-RGB representation fit.")
    parser.add_argument(
        "--pose-summary-csv",
        type=Path,
        default=ANNOTATION_DIR / "pose_sequence_summary.csv",
        help="Pose sequence summary CSV from Stage 5.",
    )
    parser.add_argument(
        "--rgb-summary-csv",
        type=Path,
        default=ANNOTATION_DIR / "rgb_feature_summary_selected.csv",
        help="RGB feature summary CSV from Stage 7 or 7B.",
    )
    parser.add_argument(
        "--pose-predictions-csv",
        type=Path,
        required=True,
        help="Pose predictions.csv artifact for one run.",
    )
    parser.add_argument(
        "--rgb-predictions-csv",
        type=Path,
        required=True,
        help="RGB predictions.csv artifact for one run.",
    )
    parser.add_argument(
        "--exercise",
        action="append",
        default=[],
        help="Optional exercise filter. Repeat to include multiple exercises.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults beside the RGB predictions file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults beside the RGB predictions file.",
    )
    parser.add_argument(
        "--rgb-label",
        default="rgb",
        help="Label used in the JSON summary for the RGB branch being analyzed.",
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


def visibility_bucket(valid_ratio: float | None) -> str:
    if valid_ratio is None:
        return "unknown"
    if valid_ratio < 0.85:
        return "low"
    if valid_ratio < 0.93:
        return "medium"
    return "high"


def confidence_bucket(mean_conf: float | None) -> str:
    if mean_conf is None:
        return "unknown"
    if mean_conf < 0.65:
        return "low"
    if mean_conf < 0.75:
        return "medium"
    return "high"


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def main() -> None:
    args = parse_args()
    exercises = {value.strip() for value in args.exercise if value.strip()}

    pose_summary_path = args.pose_summary_csv.expanduser().resolve()
    rgb_summary_path = args.rgb_summary_csv.expanduser().resolve()
    pose_predictions_path = args.pose_predictions_csv.expanduser().resolve()
    rgb_predictions_path = args.rgb_predictions_csv.expanduser().resolve()

    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else rgb_predictions_path.with_name(f"{rgb_predictions_path.stem}_representation_fit.csv")
    )
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else rgb_predictions_path.with_name(f"{rgb_predictions_path.stem}_representation_fit_summary.json")
    )

    pose_summary_rows = load_csv_rows(pose_summary_path)
    rgb_summary_rows = load_csv_rows(rgb_summary_path)
    pose_pred_rows = load_csv_rows(pose_predictions_path)
    rgb_pred_rows = load_csv_rows(rgb_predictions_path)

    pose_summary_by_name = {row["name"].strip(): row for row in pose_summary_rows}
    rgb_summary_by_name = {row["name"].strip(): row for row in rgb_summary_rows}

    pose_valid_rows = {
        row["name"].strip(): row
        for row in pose_pred_rows
        if row.get("split", "").strip().lower() == "valid"
    }
    rgb_valid_rows = {
        row["name"].strip(): row
        for row in rgb_pred_rows
        if row.get("split", "").strip().lower() == "valid"
    }

    shared_names = sorted(set(pose_valid_rows) & set(rgb_valid_rows))
    compare_rows: list[dict[str, object]] = []

    for name in shared_names:
        pose_row = pose_valid_rows[name]
        rgb_row = rgb_valid_rows[name]
        exercise = (rgb_row.get("type") or pose_row.get("type") or "").strip()
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
        rgb_feature_dim = parse_optional_float(rgb_summary.get("feature_dim"))

        compare_rows.append(
            {
                "name": name,
                "type": exercise,
                "true_count": true_count,
                "pose_pred_count": pose_pred,
                "rgb_pred_count": rgb_pred,
                "pose_abs_error": pose_abs_error,
                "rgb_abs_error": rgb_abs_error,
                "delta_rgb_minus_pose_abs_error": rgb_abs_error - pose_abs_error,
                "rgb_better": int(rgb_abs_error < pose_abs_error),
                "pose_better": int(pose_abs_error < rgb_abs_error),
                "tie": int(pose_abs_error == rgb_abs_error),
                "pose_valid_ratio": pose_valid_ratio if pose_valid_ratio is not None else "",
                "pose_mean_conf": pose_mean_conf if pose_mean_conf is not None else "",
                "pose_visibility_bucket": visibility_bucket(pose_valid_ratio),
                "pose_conf_bucket": confidence_bucket(pose_mean_conf),
                "rgb_frames_used": rgb_frames_used if rgb_frames_used is not None else "",
                "rgb_feature_dim": int(rgb_feature_dim) if rgb_feature_dim is not None else "",
                "rgb_backbone": rgb_summary.get("backbone", "").strip(),
            }
        )

    if not compare_rows:
        scope = ", ".join(sorted(exercises)) if exercises else "shared valid rows"
        raise SystemExit(f"No comparable pose/RGB valid rows found for {scope}.")

    compare_rows.sort(key=lambda row: (row["type"], row["delta_rgb_minus_pose_abs_error"], row["name"]))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compare_rows)

    exercise_summary: dict[str, dict[str, object]] = {}
    bucket_summary: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)

    by_exercise: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_bucket: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in compare_rows:
        by_exercise[str(row["type"])].append(row)
        bucket_key = f"{row['pose_visibility_bucket']}__{row['pose_conf_bucket']}"
        by_bucket[bucket_key].append(row)

    for exercise, rows in by_exercise.items():
        exercise_summary[exercise] = {
            "rows": len(rows),
            "rgb_better_rows": int(sum(int(row["rgb_better"]) for row in rows)),
            "pose_better_rows": int(sum(int(row["pose_better"]) for row in rows)),
            "ties": int(sum(int(row["tie"]) for row in rows)),
            "mean_pose_abs_error": mean([float(row["pose_abs_error"]) for row in rows]),
            "mean_rgb_abs_error": mean([float(row["rgb_abs_error"]) for row in rows]),
            "mean_delta_rgb_minus_pose_abs_error": mean(
                [float(row["delta_rgb_minus_pose_abs_error"]) for row in rows]
            ),
            "mean_pose_valid_ratio": mean(
                [float(row["pose_valid_ratio"]) for row in rows if row["pose_valid_ratio"] != ""]
            ),
            "mean_pose_mean_conf": mean(
                [float(row["pose_mean_conf"]) for row in rows if row["pose_mean_conf"] != ""]
            ),
        }

    for bucket_key, rows in by_bucket.items():
        visibility, confidence = bucket_key.split("__", 1)
        bucket_summary[visibility][confidence] = {
            "rows": len(rows),
            "rgb_better_rows": int(sum(int(row["rgb_better"]) for row in rows)),
            "pose_better_rows": int(sum(int(row["pose_better"]) for row in rows)),
            "mean_pose_abs_error": mean([float(row["pose_abs_error"]) for row in rows]),
            "mean_rgb_abs_error": mean([float(row["rgb_abs_error"]) for row in rows]),
            "mean_delta_rgb_minus_pose_abs_error": mean(
                [float(row["delta_rgb_minus_pose_abs_error"]) for row in rows]
            ),
        }

    summary = {
        "pose_predictions_csv": str(pose_predictions_path),
        "rgb_predictions_csv": str(rgb_predictions_path),
        "pose_summary_csv": str(pose_summary_path),
        "rgb_summary_csv": str(rgb_summary_path),
        "rgb_label": args.rgb_label,
        "exercise_filter": sorted(exercises),
        "rows": len(compare_rows),
        "rgb_better_rows": int(sum(int(row["rgb_better"]) for row in compare_rows)),
        "pose_better_rows": int(sum(int(row["pose_better"]) for row in compare_rows)),
        "ties": int(sum(int(row["tie"]) for row in compare_rows)),
        "mean_pose_abs_error": mean([float(row["pose_abs_error"]) for row in compare_rows]),
        "mean_rgb_abs_error": mean([float(row["rgb_abs_error"]) for row in compare_rows]),
        "mean_delta_rgb_minus_pose_abs_error": mean(
            [float(row["delta_rgb_minus_pose_abs_error"]) for row in compare_rows]
        ),
        "by_exercise": exercise_summary,
        "by_pose_quality_bucket": bucket_summary,
        "top_rgb_win_cases": compare_rows[:10],
        "top_pose_win_cases": sorted(
            compare_rows,
            key=lambda row: float(row["delta_rgb_minus_pose_abs_error"]),
            reverse=True,
        )[:10],
    }

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote representation-fit rows to {output_csv}")
    print(f"Wrote representation-fit summary to {output_json}")
    print(json.dumps({
        "rows": summary["rows"],
        "rgb_better_rows": summary["rgb_better_rows"],
        "pose_better_rows": summary["pose_better_rows"],
        "ties": summary["ties"],
        "mean_pose_abs_error": summary["mean_pose_abs_error"],
        "mean_rgb_abs_error": summary["mean_rgb_abs_error"],
    }, indent=2))


if __name__ == "__main__":
    main()

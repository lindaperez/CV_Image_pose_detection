#!/usr/bin/env python3
"""
Apply the manual validation-review policy to a predictions.csv artifact.

This script is intended for reruns of the squat TCN stage. It uses the
manually reviewed validation cases to:
- exclude unusable upstream failures from filtered metrics
- keep a record of flagged hard cases
- export a compact before/after summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {"rows": 0.0, "mae": math.nan, "rmse": math.nan, "within_1": math.nan}

    errors = [abs(float(row["eval_pred_count"]) - float(row["true_count"])) for row in rows]
    mse = sum(err * err for err in errors) / len(errors)
    within_1 = sum(1 for err in errors if err <= 1.0) / len(errors)
    return {
        "rows": float(len(rows)),
        "mae": float(sum(errors) / len(errors)),
        "rmse": float(math.sqrt(mse)),
        "within_1": float(within_1),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply manual validation-review policy to predictions.csv.")
    parser.add_argument("--predictions-csv", type=Path, required=True, help="Path to a predictions.csv artifact.")
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("CV_Image_pose_detection/artifacts/3_Modeling/validation_failure_review.csv"),
        help="Path to the machine-readable validation review policy CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary output path. Defaults beside predictions.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional filtered predictions CSV output path. Defaults beside predictions.csv.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    predictions_csv = args.predictions_csv.resolve()
    review_csv = args.review_csv.resolve()

    predictions = load_csv(predictions_csv)
    review_rows = load_csv(review_csv)

    review_by_name = {row["name"]: row for row in review_rows}
    excluded_names = {row["name"] for row in review_rows if row["action"].strip().lower() == "exclude"}

    valid_rows = [row for row in predictions if row.get("split", "").strip().lower() == "valid"]
    filtered_valid_rows = [row for row in valid_rows if row["name"] not in excluded_names]

    flagged_rows: list[dict[str, object]] = []
    for row in filtered_valid_rows:
        review = review_by_name.get(row["name"])
        flagged_rows.append(
            {
                "name": row["name"],
                "split": row["split"],
                "true_count": float(row["true_count"]),
                "eval_pred_count": float(row["eval_pred_count"]),
                "abs_error": float(row["abs_error"]),
                "policy_action": review["action"] if review else "keep",
                "policy_tag": review["policy_tag"] if review else "",
                "classification": review["classification"] if review else "",
                "notes": review["notes"] if review else "",
            }
        )

    summary = {
        "predictions_csv": str(predictions_csv),
        "review_csv": str(review_csv),
        "excluded_names": sorted(excluded_names),
        "valid_metrics_before_policy": compute_metrics(valid_rows),
        "valid_metrics_after_policy": compute_metrics(filtered_valid_rows),
        "reviewed_cases_present": sorted(name for name in review_by_name if any(row["name"] == name for row in valid_rows)),
    }

    output_json = args.output_json or predictions_csv.with_name("policy_filtered_metrics_summary.json")
    output_csv = args.output_csv or predictions_csv.with_name("policy_filtered_valid_predictions.csv")

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_csv, flagged_rows)

    print("Saved:")
    print(" -", output_json)
    print(" -", output_csv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

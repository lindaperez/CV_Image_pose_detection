#!/usr/bin/env python3
"""
Compare a TCN counting run against a trivial count baseline.

The default baseline is the per-exercise train-split mean count, which is a
better sanity check than judging MAE alone in isolation.
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
    parser = argparse.ArgumentParser(
        description="Compare a counting run against a trivial train-split baseline."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=ANNOTATION_DIR / "pose_sequence_index.csv",
        help="Input pose_sequence_index.csv file.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
        help="Input predictions.csv produced by a counting trainer.",
    )
    parser.add_argument(
        "--baseline",
        choices=["mean", "median"],
        default="mean",
        help="Train-split baseline statistic.",
    )
    parser.add_argument(
        "--exercise",
        default=None,
        help="Optional exercise filter. If omitted, uses all valid rows in predictions.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults beside predictions.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path with row-level comparison. Defaults beside predictions.csv.",
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


def parse_count(value: str | float | int | None) -> float | None:
    parsed = safe_float(value)
    if math.isnan(parsed):
        return None
    return parsed


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def median(values: list[float]) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def regression_metrics(pred: list[float], true: list[float]) -> dict[str, float]:
    if not pred:
        return {"rows": 0.0, "mae": math.nan, "rmse": math.nan, "within_1": math.nan}
    abs_err = [abs(a - b) for a, b in zip(pred, true)]
    mse = sum((a - b) ** 2 for a, b in zip(pred, true)) / len(pred)
    within_1 = sum(1 for err in abs_err if err <= 1.0) / len(pred)
    return {
        "rows": float(len(pred)),
        "mae": float(sum(abs_err) / len(abs_err)),
        "rmse": float(math.sqrt(mse)),
        "within_1": float(within_1),
    }


def main() -> None:
    args = parse_args()
    index_path = args.index_csv.expanduser().resolve()
    predictions_path = args.predictions_csv.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else predictions_path.with_name("baseline_comparison_summary.json")
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else predictions_path.with_name("baseline_comparison_rows.csv")
    )

    index_rows = load_csv_rows(index_path)
    prediction_rows = load_csv_rows(predictions_path)

    index_by_name = {row["name"].strip(): row for row in index_rows}

    train_counts_by_type: dict[str, list[float]] = defaultdict(list)
    for row in index_rows:
        split = row["split"].strip().lower()
        if split != "train":
            continue
        exercise = row["type"].strip()
        if args.exercise and exercise != args.exercise:
            continue
        count = parse_count(row.get("count"))
        if count is None:
            continue
        train_counts_by_type[exercise].append(count)

    if not train_counts_by_type:
        scope = args.exercise or "all exercises"
        raise SystemExit(f"No train rows found in {index_path} for {scope}.")

    baseline_by_type: dict[str, float] = {}
    for exercise, counts in train_counts_by_type.items():
        baseline_by_type[exercise] = mean(counts) if args.baseline == "mean" else median(counts)

    compare_rows: list[dict[str, object]] = []
    model_pred: list[float] = []
    baseline_pred: list[float] = []
    true_count: list[float] = []

    for pred_row in prediction_rows:
        split = pred_row.get("split", "").strip().lower()
        if split != "valid":
            continue

        name = pred_row["name"].strip()
        index_row = index_by_name.get(name)
        if index_row is None:
            continue

        exercise = pred_row.get("type", "").strip() or index_row["type"].strip()
        if args.exercise and exercise != args.exercise:
            continue
        if exercise not in baseline_by_type:
            continue

        true_value = safe_float(pred_row.get("true_count", index_row["count"]))
        model_value = safe_float(pred_row.get("eval_pred_count"))
        baseline_value = baseline_by_type[exercise]
        if math.isnan(true_value) or math.isnan(model_value):
            continue

        model_err = abs(model_value - true_value)
        baseline_err = abs(baseline_value - true_value)

        compare_rows.append(
            {
                "name": name,
                "type": exercise,
                "true_count": true_value,
                "model_pred_count": model_value,
                "baseline_pred_count": baseline_value,
                "model_abs_error": model_err,
                "baseline_abs_error": baseline_err,
                "model_beats_baseline": int(model_err < baseline_err),
                "model_ties_baseline": int(model_err == baseline_err),
            }
        )
        model_pred.append(model_value)
        baseline_pred.append(baseline_value)
        true_count.append(true_value)

    if not compare_rows:
        scope = args.exercise or "the selected prediction rows"
        raise SystemExit(f"No valid rows available to compare for {scope}.")

    model_metrics = regression_metrics(model_pred, true_count)
    baseline_metrics = regression_metrics(baseline_pred, true_count)

    summary = {
        "predictions_csv": str(predictions_path),
        "index_csv": str(index_path),
        "exercise_filter": args.exercise,
        "baseline_type": args.baseline,
        "baseline_by_type": baseline_by_type,
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "delta_vs_baseline": {
            "mae": model_metrics["mae"] - baseline_metrics["mae"],
            "rmse": model_metrics["rmse"] - baseline_metrics["rmse"],
            "within_1": model_metrics["within_1"] - baseline_metrics["within_1"],
        },
        "row_level": {
            "valid_rows": len(compare_rows),
            "model_beats_baseline": int(sum(row["model_beats_baseline"] for row in compare_rows)),
            "model_ties_baseline": int(sum(row["model_ties_baseline"] for row in compare_rows)),
            "mean_model_prediction": mean(model_pred),
            "mean_true_count": mean(true_count),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compare_rows)

    print(f"Wrote summary to {output_json}")
    print(f"Wrote row-level comparison to {output_csv}")
    print()
    print("Model metrics:", json.dumps(model_metrics, indent=2))
    print("Baseline metrics:", json.dumps(baseline_metrics, indent=2))
    print("Delta vs baseline:", json.dumps(summary["delta_vs_baseline"], indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Estimate bootstrap confidence intervals for counting metrics from predictions.csv.

This utility is intended for small validation surfaces where point estimates alone
can be misleading. It resamples rows with replacement and reports percentile
intervals for MAE, RMSE, and Within-1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate bootstrap confidence intervals from a counting predictions.csv artifact."
    )
    parser.add_argument("--predictions-csv", type=Path, required=True, help="Path to predictions.csv.")
    parser.add_argument(
        "--split",
        default="valid",
        help="Optional split filter. Use 'all' to disable split filtering. Defaults to valid.",
    )
    parser.add_argument(
        "--exercise",
        default=None,
        help="Optional exercise filter. If omitted, uses all matching rows.",
    )
    parser.add_argument(
        "--prediction-column",
        default="eval_pred_count",
        help="Prediction column to evaluate. Defaults to eval_pred_count.",
    )
    parser.add_argument(
        "--target-column",
        default="true_count",
        help="Ground-truth count column. Defaults to true_count.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of bootstrap resamples. Defaults to 5000.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level between 0 and 1. Defaults to 0.95.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible bootstrap draws.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults beside predictions.csv.",
    )
    return parser


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def regression_metrics(pred: list[float], true: list[float]) -> dict[str, float]:
    if not pred:
        return {"rows": 0.0, "mae": math.nan, "rmse": math.nan, "within_1": math.nan}
    errors = [abs(a - b) for a, b in zip(pred, true)]
    mse = sum((a - b) ** 2 for a, b in zip(pred, true)) / len(pred)
    within_1 = sum(1 for err in errors if err <= 1.0) / len(pred)
    return {
        "rows": float(len(pred)),
        "mae": float(sum(errors) / len(errors)),
        "rmse": float(math.sqrt(mse)),
        "within_1": float(within_1),
    }


def filter_rows(
    rows: list[dict[str, str]],
    split: str,
    exercise: str | None,
    prediction_column: str,
    target_column: str,
) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    split_key = split.strip().lower()
    for row in rows:
        row_split = row.get("split", "").strip().lower()
        if split_key != "all" and row_split and row_split != split_key:
            continue
        row_exercise = row.get("type", "").strip() or (exercise or "")
        if exercise and row_exercise != exercise:
            continue
        pred = safe_float(row.get(prediction_column))
        true = safe_float(row.get(target_column))
        if math.isnan(pred) or math.isnan(true):
            continue
        filtered.append(row)
    return filtered


def bootstrap_metrics(
    rows: list[dict[str, str]],
    prediction_column: str,
    target_column: str,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    pred = [safe_float(row[prediction_column]) for row in rows]
    true = [safe_float(row[target_column]) for row in rows]
    point_estimate = regression_metrics(pred, true)

    rng = random.Random(seed)
    metric_samples = {"mae": [], "rmse": [], "within_1": []}
    sample_size = len(rows)
    for _ in range(bootstrap_samples):
        sample_pred: list[float] = []
        sample_true: list[float] = []
        for _ in range(sample_size):
            row = rows[rng.randrange(sample_size)]
            sample_pred.append(safe_float(row[prediction_column]))
            sample_true.append(safe_float(row[target_column]))
        metrics = regression_metrics(sample_pred, sample_true)
        for key in metric_samples:
            metric_samples[key].append(metrics[key])

    alpha = 1.0 - confidence_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q

    summary_metrics: dict[str, object] = {}
    for metric_name, values in metric_samples.items():
        values = sorted(values)
        summary_metrics[metric_name] = {
            "point_estimate": point_estimate[metric_name],
            "ci_low": percentile(values, lower_q),
            "ci_high": percentile(values, upper_q),
        }

    return {
        "rows": sample_size,
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
        "metrics": summary_metrics,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive.")
    if not (0.0 < args.confidence_level < 1.0):
        raise SystemExit("--confidence-level must be between 0 and 1.")

    predictions_csv = args.predictions_csv.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else predictions_csv.with_name("bootstrap_confidence_intervals.json")
    )

    rows = load_csv(predictions_csv)
    filtered_rows = filter_rows(
        rows=rows,
        split=args.split,
        exercise=args.exercise,
        prediction_column=args.prediction_column,
        target_column=args.target_column,
    )
    if not filtered_rows:
        scope = args.exercise or "all exercises"
        raise SystemExit(
            f"No rows available for bootstrap evaluation in {predictions_csv} "
            f"for split={args.split!r}, exercise={scope!r}."
        )

    summary = {
        "predictions_csv": str(predictions_csv),
        "split_filter": args.split,
        "exercise_filter": args.exercise,
        "prediction_column": args.prediction_column,
        "target_column": args.target_column,
        "seed": args.seed,
        **bootstrap_metrics(
            rows=filtered_rows,
            prediction_column=args.prediction_column,
            target_column=args.target_column,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            seed=args.seed,
        ),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote bootstrap summary to {output_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

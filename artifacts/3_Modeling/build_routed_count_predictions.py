#!/usr/bin/env python3
"""
Build an exercise-dependent counting surface from existing run artifacts.

This stage does not train a new generic model. It routes each supported
exercise to the strongest current branch and stitches the row-level predictions
into one combined artifact that can be reviewed, compared against the trivial
baseline, and queried for specific videos.
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
TRAINING_OUTPUTS_DIR = PROJECT_DIR / "artifacts" / "3_Modeling" / "training_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an exercise-dependent routed counting artifact from existing run outputs."
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Optional explicit path to the CV_Image_pose_detection project root.",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=None,
        help="Optional dataset index CSV. Defaults to pose_sequence_index.csv.",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help="Routing rule in the form exercise=run_name. Repeat for multiple exercises.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults under training_outputs/routed_exercise_dependent_counting/.",
    )
    parser.add_argument(
        "--lookup-name",
        action="append",
        default=[],
        help="Optional video name lookup. Repeat to print routed predictions for selected videos.",
    )
    return parser.parse_args()


def resolve_project_dir(raw: str | None) -> Path:
    if raw:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Project directory does not exist: {path}")
        return path
    return PROJECT_DIR.resolve()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str | float | int | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def rmse(pred: list[float], true: list[float]) -> float:
    if not pred:
        return math.nan
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pred, true)) / len(pred))


def regression_metrics(pred: list[float], true: list[float]) -> dict[str, float]:
    if not pred:
        return {"rows": 0.0, "mae": math.nan, "rmse": math.nan, "within_1": math.nan}
    abs_err = [abs(a - b) for a, b in zip(pred, true)]
    return {
        "rows": float(len(pred)),
        "mae": float(mean(abs_err)),
        "rmse": float(rmse(pred, true)),
        "within_1": float(sum(1 for err in abs_err if err <= 1.0) / len(abs_err)),
    }


def parse_routes(route_args: list[str]) -> dict[str, str]:
    if not route_args:
        raise SystemExit("At least one --route exercise=run_name rule is required.")
    routes: dict[str, str] = {}
    for item in route_args:
        if "=" not in item:
            raise SystemExit(f"Invalid --route value {item!r}. Expected exercise=run_name.")
        exercise, run_name = item.split("=", 1)
        exercise = exercise.strip()
        run_name = run_name.strip()
        if not exercise or not run_name:
            raise SystemExit(f"Invalid --route value {item!r}. Expected exercise=run_name.")
        routes[exercise] = run_name
    return routes


def infer_exercise(row: dict[str, str], fallback: str) -> str:
    exercise = (row.get("type") or "").strip()
    return exercise or fallback


def main() -> None:
    args = parse_args()
    project_dir = resolve_project_dir(args.project_dir)
    annotation_dir = project_dir / "Data" / "LLSP" / "annotation_cleaned"
    training_outputs_dir = project_dir / "artifacts" / "3_Modeling" / "training_outputs"
    index_csv = (
        args.index_csv.expanduser().resolve()
        if args.index_csv
        else (annotation_dir / "pose_sequence_index.csv").resolve()
    )
    if not index_csv.exists():
        raise FileNotFoundError(f"Missing required dataset index CSV: {index_csv}")

    routes = parse_routes(args.route)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (training_outputs_dir / "routed_exercise_dependent_counting").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = load_csv_rows(index_csv)
    dataset_by_name = {row["name"].strip(): row for row in dataset_rows}

    combined_rows: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str]] = set()
    route_summary_rows: list[dict[str, object]] = []

    for exercise, run_name in sorted(routes.items()):
        predictions_csv = training_outputs_dir / run_name / "predictions.csv"
        if not predictions_csv.exists():
            raise FileNotFoundError(f"Missing predictions.csv for route {exercise} -> {run_name}: {predictions_csv}")

        rows = load_csv_rows(predictions_csv)
        added_rows = 0
        split_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            split = row.get("split", "").strip().lower()
            if split not in {"train", "valid"}:
                continue
            row_exercise = infer_exercise(row, exercise)
            if row_exercise != exercise:
                continue

            name = row["name"].strip()
            key = (name, split)
            if key in seen_keys:
                raise RuntimeError(
                    f"Duplicate routed row for {name} ({split}). "
                    f"Check route overlap around exercise {exercise}."
                )
            seen_keys.add(key)

            dataset_row = dataset_by_name.get(name)
            combined_rows.append(
                {
                    "name": name,
                    "type": row_exercise,
                    "split": split,
                    "true_count": safe_float(row.get("true_count", dataset_row["count"] if dataset_row else None)),
                    "raw_pred_count": safe_float(row.get("raw_pred_count")),
                    "eval_pred_count": safe_float(row.get("eval_pred_count")),
                    "abs_error": safe_float(row.get("abs_error")),
                    "source_run_name": run_name,
                    "route_exercise": exercise,
                }
            )
            added_rows += 1
            split_counts[split] += 1

        route_summary_rows.append(
            {
                "exercise": exercise,
                "run_name": run_name,
                "rows_added": added_rows,
                "train_rows": split_counts.get("train", 0),
                "valid_rows": split_counts.get("valid", 0),
            }
        )

    if not combined_rows:
        raise SystemExit("No routed prediction rows were assembled.")

    split_metrics: dict[str, dict[str, float]] = {}
    per_exercise_split_metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    for split in ("train", "valid"):
        split_rows = [row for row in combined_rows if row["split"] == split]
        pred = [float(row["eval_pred_count"]) for row in split_rows]
        true = [float(row["true_count"]) for row in split_rows]
        split_metrics[split] = regression_metrics(pred, true)

        exercise_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in split_rows:
            exercise_groups[str(row["type"])].append(row)
        for exercise, rows in sorted(exercise_groups.items()):
            per_exercise_split_metrics[exercise][split] = regression_metrics(
                [float(row["eval_pred_count"]) for row in rows],
                [float(row["true_count"]) for row in rows],
            )

    output_csv = output_dir / "routed_predictions.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "type",
                "split",
                "true_count",
                "raw_pred_count",
                "eval_pred_count",
                "abs_error",
                "source_run_name",
                "route_exercise",
            ],
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    route_summary_csv = output_dir / "routing_summary.csv"
    with route_summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(route_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(route_summary_rows)

    summary = {
        "index_csv": str(index_csv),
        "routes": routes,
        "rows_total": len(combined_rows),
        "rows_by_split": {
            split: int(sum(1 for row in combined_rows if row["split"] == split))
            for split in ("train", "valid")
        },
        "split_metrics": split_metrics,
        "per_exercise_split_metrics": {
            exercise: metrics for exercise, metrics in sorted(per_exercise_split_metrics.items())
        },
    }
    summary_json = output_dir / "routed_metrics_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lookup_rows = []
    lookup_names = {name.strip() for name in args.lookup_name if name.strip()}
    if lookup_names:
        lookup_rows = [row for row in combined_rows if str(row["name"]) in lookup_names]
        lookup_csv = output_dir / "lookup_rows.csv"
        with lookup_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0].keys()))
            writer.writeheader()
            writer.writerows(lookup_rows)

    print(f"Wrote routed predictions to {output_csv}")
    print(f"Wrote routing summary to {route_summary_csv}")
    print(f"Wrote routed metrics summary to {summary_json}")
    print()
    print(json.dumps(summary["split_metrics"], indent=2))
    if lookup_names:
        print()
        print("Lookup rows:")
        print(json.dumps(lookup_rows, indent=2))


if __name__ == "__main__":
    main()

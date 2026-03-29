#!/usr/bin/env python3
"""
Summarize per-exercise quality statistics for normalized pose sequences.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = PROJECT_DIR / "Data" / "LLSP" / "annotation_cleaned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-exercise pose-sequence quality."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ANNOTATION_DIR / "pose_sequence_summary.csv",
        help="Input pose_sequence_summary.csv file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ANNOTATION_DIR / "pose_sequence_quality_by_exercise.csv",
        help="Output CSV for per-exercise quality summary.",
    )
    parser.add_argument(
        "--worst-csv",
        type=Path,
        default=ANNOTATION_DIR / "pose_sequence_quality_worst_cases.csv",
        help="Output CSV for the lowest-quality individual videos.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of worst individual videos to export.",
    )
    return parser.parse_args()


def safe_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def safe_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "") or 0))
    except ValueError:
        return 0


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "name",
        "type",
        "split",
        "status",
        "frames_total",
        "frames_valid",
        "valid_ratio",
        "mean_conf",
    }
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary_csv.expanduser().resolve())

    ok_rows = [row for row in rows if row.get("status", "").strip() == "ok"]
    if not ok_rows:
        raise SystemExit("No rows with status=ok were found.")

    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ok_rows:
        by_type[row["type"].strip()].append(row)

    summary_records: list[dict[str, object]] = []
    for exercise, ex_rows in sorted(by_type.items()):
        total = len(ex_rows)
        summary_records.append(
            {
                "type": exercise,
                "rows": total,
                "mean_valid_ratio": sum(safe_float(r, "valid_ratio") for r in ex_rows) / total,
                "mean_conf": sum(safe_float(r, "mean_conf") for r in ex_rows) / total,
                "mean_frames_total": sum(safe_int(r, "frames_total") for r in ex_rows) / total,
                "mean_frames_valid": sum(safe_int(r, "frames_valid") for r in ex_rows) / total,
                "min_valid_ratio": min(safe_float(r, "valid_ratio") for r in ex_rows),
                "min_conf": min(safe_float(r, "mean_conf") for r in ex_rows),
            }
        )

    summary_records.sort(key=lambda row: (row["mean_valid_ratio"], row["mean_conf"]))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "type",
                "rows",
                "mean_valid_ratio",
                "mean_conf",
                "mean_frames_total",
                "mean_frames_valid",
                "min_valid_ratio",
                "min_conf",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_records)

    worst_rows = sorted(
        ok_rows,
        key=lambda row: (
            safe_float(row, "valid_ratio"),
            safe_float(row, "mean_conf"),
            safe_int(row, "frames_valid"),
        ),
    )[: max(args.top_k, 1)]

    with args.worst_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "type",
                "split",
                "name",
                "valid_ratio",
                "mean_conf",
                "frames_total",
                "frames_valid",
            ],
        )
        writer.writeheader()
        for row in worst_rows:
            writer.writerow(
                {
                    "type": row["type"].strip(),
                    "split": row["split"].strip(),
                    "name": row["name"].strip(),
                    "valid_ratio": safe_float(row, "valid_ratio"),
                    "mean_conf": safe_float(row, "mean_conf"),
                    "frames_total": safe_int(row, "frames_total"),
                    "frames_valid": safe_int(row, "frames_valid"),
                }
            )

    print(f"Wrote per-exercise summary to {args.output_csv}")
    print(f"Wrote worst-case list to {args.worst_csv}")
    print()
    print("Lowest-quality exercises by mean_valid_ratio:")
    for row in summary_records[: min(10, len(summary_records))]:
        print(
            f"- {row['type']}: rows={row['rows']}, "
            f"mean_valid_ratio={row['mean_valid_ratio']:.4f}, "
            f"mean_conf={row['mean_conf']:.4f}"
        )


if __name__ == "__main__":
    main()

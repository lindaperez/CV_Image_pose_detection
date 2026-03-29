#!/usr/bin/env python3
"""
Build a worklist of videos that still need pose-feature extraction.

This is intended for the widening step after the squat-only prototype. It reads
an existing pose-feature index, checks which target `.npy` files already exist,
and writes:
- a missing-only pose index CSV for extraction
- a per-exercise coverage summary CSV
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANNOTATION_CLEANED_DIR = PROJECT_DIR / "Data" / "LLSP" / "annotation_cleaned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a missing-only pose extraction worklist.")
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=ANNOTATION_CLEANED_DIR / "pose_feature_index.csv",
        help="Full pose feature index CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ANNOTATION_CLEANED_DIR / "pose_feature_index_remaining.csv",
        help="Where to write the missing-only worklist.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ANNOTATION_CLEANED_DIR / "pose_feature_remaining_summary.csv",
        help="Where to write the per-exercise coverage summary.",
    )
    parser.add_argument(
        "--exclude-exercise",
        action="append",
        default=[],
        help="Optional exercise label to exclude from the remaining-worklist output.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"name", "feature_path", "type", "split", "count"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    index_csv = args.index_csv.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    summary_csv = args.summary_csv.expanduser().resolve()
    excluded = {value.strip().lower() for value in args.exclude_exercise}

    rows = load_rows(index_csv)
    remaining_rows: list[dict[str, str]] = []
    summary_counts: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        exercise = row["type"].strip().lower()
        feature_exists = Path(row["feature_path"]).expanduser().exists()
        summary_counts[exercise]["expected"] += 1
        summary_counts[exercise][f"{row['split'].strip().lower()}_expected"] += 1
        if feature_exists:
            summary_counts[exercise]["existing"] += 1
            summary_counts[exercise][f"{row['split'].strip().lower()}_existing"] += 1
            continue
        if exercise not in excluded:
            remaining_rows.append(row)

    summary_rows: list[dict[str, str]] = []
    for exercise in sorted(summary_counts):
        counts = summary_counts[exercise]
        expected = counts["expected"]
        existing = counts["existing"]
        summary_rows.append(
            {
                "type": exercise,
                "expected": str(expected),
                "existing": str(existing),
                "missing": str(expected - existing),
                "train_expected": str(counts["train_expected"]),
                "train_existing": str(counts["train_existing"]),
                "valid_expected": str(counts["valid_expected"]),
                "valid_existing": str(counts["valid_existing"]),
                "test_expected": str(counts["test_expected"]),
                "test_existing": str(counts["test_existing"]),
            }
        )

    write_csv(output_csv, ["name", "feature_path", "type", "split", "count"], remaining_rows)
    write_csv(
        summary_csv,
        [
            "type",
            "expected",
            "existing",
            "missing",
            "train_expected",
            "train_existing",
            "valid_expected",
            "valid_existing",
            "test_expected",
            "test_existing",
        ],
        summary_rows,
    )

    print(f"Wrote {len(remaining_rows)} remaining rows to {output_csv}")
    print(f"Wrote coverage summary for {len(summary_rows)} exercises to {summary_csv}")
    if excluded:
        print(f"Excluded from remaining worklist: {sorted(excluded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

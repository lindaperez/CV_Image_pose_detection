#!/usr/bin/env python3
"""
Build a pose feature index CSV for raw or cleaned LLSP / RepCount annotations.

This script supports both:
- cleaned annotation CSVs with an explicit ``split`` column
- original RepCount split files under ``Data/LLSP/annotation`` where the split
  must be inferred from the filename

It can be used for:
- a squat-only slice of the pipeline
- a full multi-exercise pose-extraction pass
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANNOTATION_CLEANED_DIR = PROJECT_DIR / "Data" / "LLSP" / "annotation_cleaned"
ANNOTATION_DIR = PROJECT_DIR / "Data" / "LLSP" / "annotation"
FEATURE_DIR_DEFAULT = ANNOTATION_CLEANED_DIR / "pose_features"

CANONICAL_LABEL_MAP = {
    "benchpressing": "bench_pressing",
    "frontraise": "front_raise",
    "jump_jack": "jump_jacks",
    "jumpjacks": "jump_jacks",
    "pullups": "pull_up",
    "pushups": "push_up",
    "situp": "sit_up",
    "squant": "squat",
}


def default_annotation_csvs() -> list[Path]:
    cleaned = sorted(ANNOTATION_CLEANED_DIR.glob("*_cleaned.csv"))
    if cleaned:
        return cleaned
    return sorted(ANNOTATION_DIR.glob("*.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a pose feature index CSV from cleaned labels.")
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        nargs="+",
        default=default_annotation_csvs(),
        help=(
            "One or more annotation CSV files. Defaults to cleaned CSVs when they exist, "
            "otherwise falls back to raw split files under Data/LLSP/annotation."
        ),
    )
    parser.add_argument(
        "--exercise",
        type=str,
        default=None,
        help="Optional exercise filter, for example: squat",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=FEATURE_DIR_DEFAULT,
        help="Directory where pose feature .npy files will live.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ANNOTATION_CLEANED_DIR / "pose_feature_index.csv",
        help="Where to write the generated index CSV.",
    )
    return parser.parse_args()


def normalize_label(value: str) -> str:
    normalized = value.strip().lower()
    return CANONICAL_LABEL_MAP.get(normalized, normalized)


def infer_split(annotation_csv: Path, row: Dict[str, str]) -> str:
    split_value = row.get("split", "").strip().lower()
    if split_value:
        return split_value

    stem = annotation_csv.stem.strip().lower()
    if stem in {"train", "valid", "test"}:
        return stem

    raise ValueError(
        f"Could not infer split for row {row.get('name', '<unknown>')} from {annotation_csv}"
    )


def load_rows(annotation_csvs: Iterable[Path], exercise: str | None, feature_dir: Path) -> List[Dict[str, str]]:
    wanted = normalize_label(exercise) if exercise else None
    rows: List[Dict[str, str]] = []
    seen_names = set()
    skipped_missing_count: list[str] = []

    for annotation_csv in annotation_csvs:
        with annotation_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"type", "name", "count"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{annotation_csv} is missing columns: {sorted(missing)}")

            for row in reader:
                exercise_name = normalize_label(row["type"])
                video_name = row["name"].strip()
                split_name = infer_split(annotation_csv, row)
                if wanted and exercise_name != wanted:
                    continue
                if video_name in seen_names:
                    continue
                count = row["count"].strip()
                if not count:
                    skipped_missing_count.append(video_name)
                    continue

                seen_names.add(video_name)
                rows.append(
                    {
                        "name": video_name,
                        "feature_path": str((feature_dir / f"{Path(video_name).stem}.npy").resolve()),
                        "type": exercise_name,
                        "split": split_name,
                        "count": count,
                    }
                )

    if skipped_missing_count:
        preview = ", ".join(skipped_missing_count[:5])
        print(
            f"Skipped {len(skipped_missing_count)} row(s) with missing count"
            f" while building the pose feature index: {preview}"
        )

    return rows


def write_rows(output_csv: Path, rows: List[Dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "feature_path", "type", "split", "count"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.annotation_csv = [path.expanduser().resolve() for path in args.annotation_csv]
    args.feature_dir = args.feature_dir.expanduser().resolve()
    args.output_csv = args.output_csv.expanduser().resolve()

    rows = load_rows(args.annotation_csv, args.exercise, args.feature_dir)
    if not rows:
        print("ERROR: no matching rows found for the requested filter.")
        return 2

    write_rows(args.output_csv, rows)
    exercise_label = args.exercise if args.exercise else "all exercises"
    print(f"Wrote {len(rows)} rows for {exercise_label} to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

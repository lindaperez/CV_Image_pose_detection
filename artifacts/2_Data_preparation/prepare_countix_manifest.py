#!/usr/bin/env python3
"""
Normalize Countix metadata into the repo's annotation contract.

The current LLSP pipeline expects at least:
- name
- type
- split
- count

Countix is kept as a separate benchmark branch, so this script converts an
external Countix CSV into a stable local manifest under Data/Countix without
forcing it into the LLSP cleaned-label files.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
COUNTIX_DIR = PROJECT_DIR / "Data" / "Countix"
COUNTIX_ANNOTATION_DIR = COUNTIX_DIR / "annotation_cleaned"
COUNTIX_VIDEO_DIR = COUNTIX_DIR / "video"

TARGET_CLASSES = {
    "squat",
    "pull_up",
    "push_up",
    "sit_up",
    "bench_pressing",
    "front_raise",
    "jump_jacks",
    "battle_rope",
    "pommelhorse",
}

COUNTIX_TO_YOUR_LABELS = {
    "squats": "squat",
    "squat": "squat",
    "pull ups": "pull_up",
    "pullups": "pull_up",
    "pull up": "pull_up",
    "push ups": "push_up",
    "pushups": "push_up",
    "push up": "push_up",
    "sit ups": "sit_up",
    "situps": "sit_up",
    "sit up": "sit_up",
    "bench press": "bench_pressing",
    "bench pressing": "bench_pressing",
    "front raise": "front_raise",
    "front raises": "front_raise",
    "jumping jacks": "jump_jacks",
    "jumping jack": "jump_jacks",
    "jump jacks": "jump_jacks",
    "jump jack": "jump_jacks",
    "battle ropes": "battle_rope",
    "battle rope": "battle_rope",
    "pommel horse": "pommelhorse",
    "pommelhorse": "pommelhorse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a normalized Countix manifest CSV for the existing pipeline."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Raw Countix metadata CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=COUNTIX_ANNOTATION_DIR / "countix_manifest.csv",
        help="Normalized output manifest CSV.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=COUNTIX_VIDEO_DIR,
        help="Local Countix video directory used to resolve relative paths.",
    )
    parser.add_argument(
        "--name-col",
        type=str,
        default="",
        help="Optional explicit input column for the video name.",
    )
    parser.add_argument(
        "--path-col",
        type=str,
        default="",
        help="Optional explicit input column for the video path.",
    )
    parser.add_argument(
        "--count-col",
        type=str,
        default="",
        help="Optional explicit input column for the repetition count.",
    )
    parser.add_argument(
        "--type-col",
        type=str,
        default="",
        help="Optional explicit input column for the action/category label.",
    )
    parser.add_argument(
        "--split-col",
        type=str,
        default="",
        help="Optional explicit input column for the split name.",
    )
    parser.add_argument(
        "--default-split",
        type=str,
        default="train",
        help="Fallback split if none is provided in the input CSV.",
    )
    parser.add_argument(
        "--default-type",
        type=str,
        default="countix_action",
        help="Fallback type if none is provided in the input CSV.",
    )
    parser.add_argument(
        "--disable-target-filter",
        action="store_true",
        help="Keep all mapped or raw Countix types instead of filtering to the supported exercise set.",
    )
    return parser.parse_args()


def candidate_column(fieldnames: list[str], explicit: str, candidates: list[str]) -> str | None:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"Column '{explicit}' was requested but not found in the input CSV.")
        return explicit
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match:
            return match
    return None


def normalize_split(value: str | None, default: str) -> str:
    text = (value or "").strip().lower()
    return text or default.strip().lower()


def normalize_type(value: str | None, default: str) -> str:
    text = (value or "").strip().lower()
    return text or default.strip().lower()


def normalize_label_key(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[_/-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def map_countix_type(value: str | None, default: str) -> tuple[str, str]:
    raw_type = normalize_type(value, default)
    lookup_key = normalize_label_key(raw_type)
    mapped_type = COUNTIX_TO_YOUR_LABELS.get(lookup_key, raw_type)
    return raw_type, mapped_type


def resolve_name(row: dict[str, str], name_col: str | None, path_col: str | None) -> str:
    if name_col:
        name = row.get(name_col, "").strip()
        if name:
            return Path(name).name
    if path_col:
        path_value = row.get(path_col, "").strip()
        if path_value:
            return Path(path_value).name
    raise ValueError("Could not resolve a video name from the provided row.")


def resolve_video_path(
    row: dict[str, str],
    name: str,
    path_col: str | None,
    video_dir: Path,
) -> str:
    if not path_col:
        return str((video_dir / name).resolve())
    raw_path = row.get(path_col, "").strip()
    if not raw_path:
        return str((video_dir / name).resolve())
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((video_dir / raw_path).resolve())


def parse_count(value: str, row_index: int) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"Row {row_index}: missing count.")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"Row {row_index}: invalid count '{text}'.") from exc
    if parsed.is_integer():
        return str(int(parsed))
    return str(parsed)


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    video_dir = args.video_dir.expanduser().resolve()

    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise SystemExit(f"Input CSV has no header: {input_csv}")

        name_col = candidate_column(
            fieldnames,
            args.name_col,
            ["name", "video_name", "video_id", "id", "basename"],
        )
        path_col = candidate_column(
            fieldnames,
            args.path_col,
            ["video_path", "path", "video_relpath", "relpath", "filepath", "file_path"],
        )
        count_col = candidate_column(
            fieldnames,
            args.count_col,
            ["count", "rep_count", "repetition_count", "num_reps", "reps"],
        )
        type_col = candidate_column(
            fieldnames,
            args.type_col,
            ["type", "class", "action", "label", "category"],
        )
        split_col = candidate_column(
            fieldnames,
            args.split_col,
            ["split", "subset", "partition", "fold"],
        )

        if count_col is None:
            raise SystemExit(
                "Could not infer the count column. Pass --count-col explicitly."
            )
        if name_col is None and path_col is None:
            raise SystemExit(
                "Could not infer either a name column or a path column. "
                "Pass --name-col or --path-col explicitly."
            )

        normalized_rows: list[dict[str, str]] = []
        seen_names: set[str] = set()
        dropped_types: Counter[str] = Counter()

        for row_index, row in enumerate(reader, start=2):
            name = resolve_name(row, name_col, path_col)
            if name in seen_names:
                raise SystemExit(
                    f"Duplicate video name '{name}' detected at row {row_index}. "
                    "Countix rows must resolve to unique names for the current pipeline."
                )
            seen_names.add(name)

            raw_type, mapped_type = map_countix_type(
                row.get(type_col, "") if type_col else "",
                args.default_type,
            )
            if not args.disable_target_filter and mapped_type not in TARGET_CLASSES:
                dropped_types[raw_type] += 1
                continue

            normalized_rows.append(
                {
                    "dataset": "countix",
                    "name": name,
                    "type": mapped_type,
                    "split": normalize_split(row.get(split_col, "") if split_col else "", args.default_split),
                    "count": parse_count(row.get(count_col, ""), row_index),
                    "video_path": resolve_video_path(row, name, path_col, video_dir),
                    "source_id": row.get(name_col or path_col or "", "").strip(),
                    "source_type": raw_type,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "name",
                "type",
                "split",
                "count",
                "video_path",
                "source_id",
                "source_type",
            ],
        )
        writer.writeheader()
        writer.writerows(normalized_rows)

    print(f"Wrote {len(normalized_rows)} Countix rows to {output_csv}")
    print(f"Video root: {video_dir}")
    if args.disable_target_filter:
        print("Target-class filter: disabled")
    else:
        print(f"Target-class filter: kept only {sorted(TARGET_CLASSES)}")
        if dropped_types:
            print("Dropped unmapped / out-of-scope Countix types:")
            for label, count in sorted(dropped_types.items()):
                print(f"- {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

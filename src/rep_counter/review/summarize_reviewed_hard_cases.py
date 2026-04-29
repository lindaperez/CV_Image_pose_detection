#!/usr/bin/env python3
"""
Summarize the reviewed hard-case manifest produced from 7D audit outputs.

The summary focuses on reviewed human-confirmed tags rather than the original
heuristic audit buckets, so it can support a stronger discussion of remaining
data-side versus model-side failure modes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REVIEWED_STATUSES = {"reviewed", "done", "confirmed"}
TRUE_VALUES = {"yes", "y", "true", "1"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize reviewed hard-case manifests.")
    parser.add_argument(
        "--review-csv",
        type=Path,
        action="append",
        required=True,
        help="Path to a reviewed hard_case_review_manifest.csv. Repeat for multiple files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary output path. Defaults beside the first review CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path for per-exercise primary-issue counts. Defaults beside the first review CSV.",
    )
    return parser


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_status(value: str | None) -> str:
    return (value or "").strip().lower()


def truthy(value: str | None) -> bool:
    return normalize_status(value) in TRUE_VALUES


def parse_tags(value: str | None) -> list[str]:
    if not value:
        return []
    return [tag.strip() for tag in value.split(",") if tag.strip()]


PRIMARY_ISSUE_FIELDNAMES = ["exercise", "manual_primary_issue", "count"]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def main() -> None:
    args = build_arg_parser().parse_args()

    review_paths = [path.expanduser().resolve() for path in args.review_csv]
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else review_paths[0].with_name("reviewed_hard_case_summary.json")
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else review_paths[0].with_name("reviewed_hard_case_primary_issues.csv")
    )

    rows: list[dict[str, str]] = []
    for path in review_paths:
        rows.extend(load_csv(path))

    if not rows:
        raise SystemExit("No review rows were loaded.")

    status_counts = Counter(normalize_status(row.get("manual_review_status")) or "pending" for row in rows)
    reviewed_rows = [row for row in rows if normalize_status(row.get("manual_review_status")) in REVIEWED_STATUSES]

    primary_issue_counts = Counter()
    model_outcome_counts = Counter()
    confirmed_tag_counts = Counter()
    per_exercise_primary_issue: dict[str, Counter[str]] = defaultdict(Counter)
    keep_for_report_counts = Counter()

    data_issue_rows = 0
    label_issue_rows = 0
    for row in reviewed_rows:
        exercise = row.get("type", "").strip()
        primary_issue = row.get("manual_primary_issue", "").strip() or "unspecified"
        primary_issue_counts[primary_issue] += 1
        per_exercise_primary_issue[exercise][primary_issue] += 1
        model_outcome_counts[row.get("model_outcome", "").strip() or "unknown"] += 1
        for tag in parse_tags(row.get("manual_issue_tags")):
            confirmed_tag_counts[tag] += 1
        if truthy(row.get("manual_keep_for_report")):
            keep_for_report_counts[exercise or "unknown"] += 1
        if truthy(row.get("manual_rep_definition_ambiguous")) or normalize_status(row.get("manual_count_label_ok")) == "no":
            label_issue_rows += 1
        if (
            truthy(row.get("manual_visibility_issue_confirmed"))
            or normalize_status(row.get("manual_target_person_ok")) == "no"
            or truthy(row.get("manual_pose_failure_confirmed"))
            or truthy(row.get("manual_rgb_context_advantage_confirmed"))
        ):
            data_issue_rows += 1

    exercise_rows: list[dict[str, object]] = []
    for exercise, issue_counts in sorted(per_exercise_primary_issue.items()):
        for issue, count in sorted(issue_counts.items()):
            exercise_rows.append(
                {
                    "exercise": exercise,
                    "manual_primary_issue": issue,
                    "count": count,
                }
            )

    summary = {
        "review_csvs": [str(path) for path in review_paths],
        "rows_total": len(rows),
        "rows_reviewed": len(reviewed_rows),
        "rows_pending": len(rows) - len(reviewed_rows),
        "status_counts": dict(status_counts),
        "primary_issue_counts": dict(primary_issue_counts),
        "confirmed_issue_tag_counts": dict(confirmed_tag_counts),
        "reviewed_model_outcome_counts": dict(model_outcome_counts),
        "keep_for_report_counts": dict(keep_for_report_counts),
        "label_issue_rows": label_issue_rows,
        "data_issue_rows": data_issue_rows,
        "per_exercise_primary_issue_counts": {
            exercise: dict(counter) for exercise, counter in sorted(per_exercise_primary_issue.items())
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_csv, exercise_rows, PRIMARY_ISSUE_FIELDNAMES)

    print(f"Wrote reviewed hard-case summary to {output_json}")
    print(f"Wrote per-exercise issue counts to {output_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build a manual-review manifest from one or more 7D hard-case audit CSVs.

The output preserves the audit-side evidence and appends explicit human-review
columns so the heuristic 7D buckets can be confirmed, corrected, or rejected.
If the output CSV already exists, existing manual annotations are preserved.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MANUAL_REVIEW_FIELDS = [
    "manual_review_status",
    "manual_primary_issue",
    "manual_issue_tags",
    "manual_target_person_ok",
    "manual_count_label_ok",
    "manual_rep_definition_ambiguous",
    "manual_visibility_issue_confirmed",
    "manual_pose_failure_confirmed",
    "manual_rgb_context_advantage_confirmed",
    "manual_keep_for_report",
    "manual_notes",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a review manifest from 7D hard-case audit CSVs.")
    parser.add_argument(
        "--audit-csv",
        type=Path,
        action="append",
        required=True,
        help="Path to a hard_case_audit.csv file. Repeat for multiple exercises/runs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional review-manifest CSV output path. Defaults beside the first audit CSV.",
    )
    parser.add_argument(
        "--top-k-per-audit",
        type=int,
        default=0,
        help="Optional per-audit row cap after existing severity ordering. 0 keeps all rows.",
    )
    return parser


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def classify_model_outcome(pose_abs_error: str | float | int | None, rgb_abs_error: str | float | int | None) -> str:
    try:
        pose_err = float(pose_abs_error)
        rgb_err = float(rgb_abs_error)
    except (TypeError, ValueError):
        return "unknown"
    if pose_err + 1.0 < rgb_err:
        return "pose_clear_win"
    if rgb_err + 1.0 < pose_err:
        return "rgb_clear_win"
    if pose_err == rgb_err:
        return "tie"
    return "close_call"


def make_review_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("source_run_name", "").strip(), row.get("name", "").strip())


def default_review_values() -> dict[str, str]:
    return {
        "manual_review_status": "pending",
        "manual_primary_issue": "",
        "manual_issue_tags": "",
        "manual_target_person_ok": "",
        "manual_count_label_ok": "",
        "manual_rep_definition_ambiguous": "",
        "manual_visibility_issue_confirmed": "",
        "manual_pose_failure_confirmed": "",
        "manual_rgb_context_advantage_confirmed": "",
        "manual_keep_for_report": "",
        "manual_notes": "",
    }


def main() -> None:
    args = build_arg_parser().parse_args()

    audit_paths = [path.expanduser().resolve() for path in args.audit_csv]
    if len(audit_paths) == 1 and args.output_csv is None:
        output_csv = audit_paths[0].with_name("hard_case_review_manifest.csv")
    else:
        output_csv = (
            args.output_csv.expanduser().resolve()
            if args.output_csv
            else audit_paths[0].parent / "hard_case_review_manifest.csv"
        )

    existing_review_rows = load_csv(output_csv) if output_csv.exists() else []
    existing_review_by_key = {make_review_key(row): row for row in existing_review_rows}

    manifest_rows: list[dict[str, object]] = []
    base_fieldnames: list[str] = []

    for audit_path in audit_paths:
        audit_rows = load_csv(audit_path)
        if args.top_k_per_audit > 0:
            audit_rows = audit_rows[: args.top_k_per_audit]
        source_run_name = audit_path.parent.name

        for rank, row in enumerate(audit_rows, start=1):
            if not base_fieldnames:
                base_fieldnames = list(row.keys())

            manifest_row = dict(row)
            manifest_row["source_audit_csv"] = str(audit_path)
            manifest_row["source_run_name"] = source_run_name
            manifest_row["review_priority"] = rank
            manifest_row["model_outcome"] = classify_model_outcome(
                row.get("pose_abs_error"),
                row.get("rgb_abs_error"),
            )

            review_key = (source_run_name, row.get("name", "").strip())
            existing = existing_review_by_key.get(review_key, {})
            for field, default in default_review_values().items():
                manifest_row[field] = existing.get(field, default)

            manifest_rows.append(manifest_row)

    if not manifest_rows:
        raise SystemExit("No hard-case audit rows were loaded.")

    review_meta_fields = ["source_audit_csv", "source_run_name", "review_priority", "model_outcome"]
    fieldnames = base_fieldnames + review_meta_fields + MANUAL_REVIEW_FIELDS
    write_csv(output_csv, manifest_rows, fieldnames)

    print(f"Wrote review manifest to {output_csv}")
    print(f"Rows: {len(manifest_rows)}")
    print(f"Audits merged: {len(audit_paths)}")


if __name__ == "__main__":
    main()

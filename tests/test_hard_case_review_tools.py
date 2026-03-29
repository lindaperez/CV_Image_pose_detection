from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_utils import load_module


build_review_manifest = load_module(
    "build_hard_case_review_manifest",
    "artifacts/3_Modeling/build_hard_case_review_manifest.py",
)
summarize_reviewed = load_module(
    "summarize_reviewed_hard_cases",
    "artifacts/3_Modeling/summarize_reviewed_hard_cases.py",
)


class HardCaseReviewToolTests(unittest.TestCase):
    def test_classify_model_outcome_uses_one_rep_margin(self):
        self.assertEqual(build_review_manifest.classify_model_outcome(2.0, 4.5), "pose_clear_win")
        self.assertEqual(build_review_manifest.classify_model_outcome(4.5, 2.0), "rgb_clear_win")
        self.assertEqual(build_review_manifest.classify_model_outcome(3.0, 3.4), "close_call")

    def test_build_manifest_preserves_existing_manual_annotations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            audit_dir = tmp / "rgb_count_tcn_push_up_seq128"
            audit_dir.mkdir(parents=True)
            audit_csv = audit_dir / "hard_case_audit.csv"
            with audit_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "name",
                        "type",
                        "split",
                        "true_count",
                        "pose_abs_error",
                        "rgb_abs_error",
                        "audit_bucket",
                        "severity",
                        "issue_tags",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "demo.mp4",
                        "type": "push_up",
                        "split": "valid",
                        "true_count": "8",
                        "pose_abs_error": "4.0",
                        "rgb_abs_error": "1.0",
                        "audit_bucket": "rgb_advantage_on_weak_pose",
                        "severity": "medium",
                        "issue_tags": "weak_pose",
                    }
                )

            output_csv = tmp / "hard_case_review_manifest.csv"
            with output_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "name",
                        "source_run_name",
                        "manual_review_status",
                        "manual_primary_issue",
                        "manual_notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "demo.mp4",
                        "source_run_name": "rgb_count_tcn_push_up_seq128",
                        "manual_review_status": "reviewed",
                        "manual_primary_issue": "visibility",
                        "manual_notes": "kept from previous pass",
                    }
                )

            argv = [
                "build_hard_case_review_manifest.py",
                "--audit-csv",
                str(audit_csv),
                "--output-csv",
                str(output_csv),
            ]
            with mock.patch.object(sys, "argv", argv):
                build_review_manifest.main()

            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["manual_review_status"], "reviewed")
            self.assertEqual(rows[0]["manual_primary_issue"], "visibility")
            self.assertEqual(rows[0]["model_outcome"], "rgb_clear_win")
            self.assertEqual(rows[0]["source_run_name"], "rgb_count_tcn_push_up_seq128")

    def test_summarize_reviewed_hard_cases_counts_primary_issues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            review_csv = tmp / "hard_case_review_manifest.csv"
            with review_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "name",
                        "type",
                        "model_outcome",
                        "manual_review_status",
                        "manual_primary_issue",
                        "manual_issue_tags",
                        "manual_keep_for_report",
                        "manual_rep_definition_ambiguous",
                        "manual_count_label_ok",
                        "manual_target_person_ok",
                        "manual_visibility_issue_confirmed",
                        "manual_pose_failure_confirmed",
                        "manual_rgb_context_advantage_confirmed",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "a.mp4",
                        "type": "squat",
                        "model_outcome": "pose_clear_win",
                        "manual_review_status": "reviewed",
                        "manual_primary_issue": "rep_ambiguity",
                        "manual_issue_tags": "partial_rep,depth_unclear",
                        "manual_keep_for_report": "yes",
                        "manual_rep_definition_ambiguous": "yes",
                        "manual_count_label_ok": "no",
                        "manual_target_person_ok": "yes",
                        "manual_visibility_issue_confirmed": "no",
                        "manual_pose_failure_confirmed": "no",
                        "manual_rgb_context_advantage_confirmed": "no",
                    }
                )
                writer.writerow(
                    {
                        "name": "b.mp4",
                        "type": "push_up",
                        "model_outcome": "rgb_clear_win",
                        "manual_review_status": "reviewed",
                        "manual_primary_issue": "visibility",
                        "manual_issue_tags": "off_frame",
                        "manual_keep_for_report": "no",
                        "manual_rep_definition_ambiguous": "no",
                        "manual_count_label_ok": "yes",
                        "manual_target_person_ok": "yes",
                        "manual_visibility_issue_confirmed": "yes",
                        "manual_pose_failure_confirmed": "yes",
                        "manual_rgb_context_advantage_confirmed": "yes",
                    }
                )
                writer.writerow(
                    {
                        "name": "c.mp4",
                        "type": "pull_up",
                        "model_outcome": "close_call",
                        "manual_review_status": "pending",
                        "manual_primary_issue": "",
                        "manual_issue_tags": "",
                        "manual_keep_for_report": "",
                        "manual_rep_definition_ambiguous": "",
                        "manual_count_label_ok": "",
                        "manual_target_person_ok": "",
                        "manual_visibility_issue_confirmed": "",
                        "manual_pose_failure_confirmed": "",
                        "manual_rgb_context_advantage_confirmed": "",
                    }
                )

            summary_json = tmp / "reviewed_summary.json"
            summary_csv = tmp / "reviewed_primary_issues.csv"
            argv = [
                "summarize_reviewed_hard_cases.py",
                "--review-csv",
                str(review_csv),
                "--output-json",
                str(summary_json),
                "--output-csv",
                str(summary_csv),
            ]
            with mock.patch.object(sys, "argv", argv):
                summarize_reviewed.main()

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["rows_total"], 3)
            self.assertEqual(summary["rows_reviewed"], 2)
            self.assertEqual(summary["primary_issue_counts"]["rep_ambiguity"], 1)
            self.assertEqual(summary["primary_issue_counts"]["visibility"], 1)
            self.assertEqual(summary["keep_for_report_counts"]["squat"], 1)
            self.assertEqual(summary["label_issue_rows"], 1)
            self.assertEqual(summary["data_issue_rows"], 1)

    def test_summarize_reviewed_hard_cases_writes_empty_csv_when_all_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            review_csv = tmp / "hard_case_review_manifest.csv"
            with review_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "name",
                        "type",
                        "model_outcome",
                        "manual_review_status",
                        "manual_primary_issue",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "a.mp4",
                        "type": "squat",
                        "model_outcome": "pose_clear_win",
                        "manual_review_status": "pending",
                        "manual_primary_issue": "",
                    }
                )

            summary_json = tmp / "reviewed_summary.json"
            summary_csv = tmp / "reviewed_primary_issues.csv"
            argv = [
                "summarize_reviewed_hard_cases.py",
                "--review-csv",
                str(review_csv),
                "--output-json",
                str(summary_json),
                "--output-csv",
                str(summary_csv),
            ]
            with mock.patch.object(sys, "argv", argv):
                summarize_reviewed.main()

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["rows_total"], 1)
            self.assertEqual(summary["rows_reviewed"], 0)
            self.assertTrue(summary_csv.exists())
            with summary_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()

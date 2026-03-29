from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_utils import load_module


register_experiment = load_module(
    "register_experiment",
    "artifacts/3_Modeling/register_experiment.py",
)
build_routed = load_module(
    "build_routed_count_predictions",
    "artifacts/3_Modeling/build_routed_count_predictions.py",
)


class RegistryAndRoutingTests(unittest.TestCase):
    def test_register_experiment_writes_registry_row_from_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "CV_Image_pose_detection"
            (project_dir / "Data" / "LLSP").mkdir(parents=True)
            (project_dir / "artifacts" / "3_Modeling").mkdir(parents=True)

            metrics_path = project_dir / "artifacts" / "3_Modeling" / "metrics_summary.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "valid_metrics": {
                            "mae": 1.25,
                            "rmse": 2.5,
                            "within_1": 0.75,
                        }
                    }
                ),
                encoding="utf-8",
            )

            registry_path = project_dir / "artifacts" / "3_Modeling" / "experiment_registry.csv"
            argv = [
                "register_experiment.py",
                "--project-dir",
                str(project_dir),
                "--stage",
                "unit_stage",
                "--scope",
                "unit_scope",
                "--question",
                "Does the registry script work?",
                "--decision",
                "Keep",
                "--artifact-reference",
                str(project_dir / "artifacts" / "3_Modeling" / "demo.ipynb"),
                "--metrics-json",
                str(metrics_path),
            ]
            with mock.patch.object(sys, "argv", argv):
                register_experiment.main()

            with registry_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["stage"], "unit_stage")
            self.assertIn("MAE=1.2500", rows[0]["primary_result"])
            self.assertEqual(rows[0]["decision"], "Keep")

    def test_build_routed_count_predictions_creates_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "CV_Image_pose_detection"
            annotation_dir = project_dir / "Data" / "LLSP" / "annotation_cleaned"
            training_dir = project_dir / "artifacts" / "3_Modeling" / "training_outputs"
            annotation_dir.mkdir(parents=True)
            (training_dir / "pose_count_tcn_pull_up_seq192").mkdir(parents=True)
            (training_dir / "rgb_count_tcn_push_up_seq128").mkdir(parents=True)

            index_csv = annotation_dir / "pose_sequence_index.csv"
            with index_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["name", "type", "split", "count"])
                writer.writeheader()
                writer.writerow({"name": "pull1.mp4", "type": "pull_up", "split": "valid", "count": "5"})
                writer.writerow({"name": "push1.mp4", "type": "push_up", "split": "valid", "count": "8"})

            pull_preds = training_dir / "pose_count_tcn_pull_up_seq192" / "predictions.csv"
            with pull_preds.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["name", "type", "split", "true_count", "raw_pred_count", "eval_pred_count", "abs_error"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "pull1.mp4",
                        "type": "pull_up",
                        "split": "valid",
                        "true_count": "5",
                        "raw_pred_count": "4.2",
                        "eval_pred_count": "4.2",
                        "abs_error": "0.8",
                    }
                )

            push_preds = training_dir / "rgb_count_tcn_push_up_seq128" / "predictions.csv"
            with push_preds.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["name", "type", "split", "true_count", "raw_pred_count", "eval_pred_count", "abs_error"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "push1.mp4",
                        "type": "push_up",
                        "split": "valid",
                        "true_count": "8",
                        "raw_pred_count": "8.7",
                        "eval_pred_count": "8.7",
                        "abs_error": "0.7",
                    }
                )

            output_dir = training_dir / "routed_test"
            argv = [
                "build_routed_count_predictions.py",
                "--project-dir",
                str(project_dir),
                "--route",
                "pull_up=pose_count_tcn_pull_up_seq192",
                "--route",
                "push_up=rgb_count_tcn_push_up_seq128",
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", argv):
                build_routed.main()

            summary_path = output_dir / "routed_metrics_summary.json"
            routed_csv = output_dir / "routed_predictions.csv"
            self.assertTrue(summary_path.exists())
            self.assertTrue(routed_csv.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["rows_total"], 2)
            self.assertIn("valid", summary["split_metrics"])
            self.assertAlmostEqual(summary["split_metrics"]["valid"]["mae"], 0.75, places=6)


if __name__ == "__main__":
    unittest.main()

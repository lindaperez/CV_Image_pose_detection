from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_utils import load_module


bootstrap_ci = load_module(
    "bootstrap_count_confidence_intervals",
    "artifacts/3_Modeling/bootstrap_count_confidence_intervals.py",
)


class BootstrapConfidenceIntervalTests(unittest.TestCase):
    def test_bootstrap_script_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions_csv = Path(tmpdir) / "predictions.csv"
            predictions_csv.write_text(
                "\n".join(
                    [
                        "name,type,split,true_count,eval_pred_count",
                        "a.mp4,squat,valid,10,9",
                        "b.mp4,squat,valid,12,13",
                        "c.mp4,squat,valid,8,8",
                        "d.mp4,squat,train,11,10",
                    ]
                ),
                encoding="utf-8",
            )

            output_json = Path(tmpdir) / "bootstrap_summary.json"
            argv = [
                "bootstrap_count_confidence_intervals.py",
                "--predictions-csv",
                str(predictions_csv),
                "--exercise",
                "squat",
                "--split",
                "valid",
                "--bootstrap-samples",
                "200",
                "--seed",
                "11",
                "--output-json",
                str(output_json),
            ]
            with mock.patch.object(sys, "argv", argv):
                bootstrap_ci.main()

            summary = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["split_filter"], "valid")
            self.assertEqual(summary["exercise_filter"], "squat")
            self.assertEqual(summary["bootstrap_samples"], 200)
            self.assertAlmostEqual(summary["metrics"]["mae"]["point_estimate"], 2.0 / 3.0, places=6)
            self.assertAlmostEqual(summary["metrics"]["within_1"]["point_estimate"], 1.0, places=6)
            self.assertLessEqual(
                summary["metrics"]["mae"]["ci_low"],
                summary["metrics"]["mae"]["point_estimate"],
            )
            self.assertGreaterEqual(
                summary["metrics"]["mae"]["ci_high"],
                summary["metrics"]["mae"]["point_estimate"],
            )

    def test_bootstrap_script_tolerates_missing_type_when_exercise_is_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions_csv = Path(tmpdir) / "predictions.csv"
            predictions_csv.write_text(
                "\n".join(
                    [
                        "name,split,true_count,eval_pred_count",
                        "a.mp4,valid,10,9",
                        "b.mp4,valid,12,13",
                        "c.mp4,valid,8,8",
                    ]
                ),
                encoding="utf-8",
            )

            output_json = Path(tmpdir) / "bootstrap_summary.json"
            argv = [
                "bootstrap_count_confidence_intervals.py",
                "--predictions-csv",
                str(predictions_csv),
                "--exercise",
                "squat",
                "--split",
                "valid",
                "--bootstrap-samples",
                "100",
                "--seed",
                "11",
                "--output-json",
                str(output_json),
            ]
            with mock.patch.object(sys, "argv", argv):
                bootstrap_ci.main()

            summary = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["exercise_filter"], "squat")


if __name__ == "__main__":
    unittest.main()

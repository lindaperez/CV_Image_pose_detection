from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from test_utils import load_module


review_server = load_module(
    "hard_case_review_server",
    "src/rep_counter/review/hard_case_review_server.py",
)


class HardCaseReviewServerTests(unittest.TestCase):
    def test_normalize_api_request_path_supports_project_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "ML_System"
            project_dir.mkdir()
            normalized = review_server.normalize_api_request_path(
                "/ML_System/api/health",
                project_dir,
            )
            self.assertEqual(normalized, "/api/health")

    def test_resolve_repo_relative_csv_path_accepts_repo_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            resolved = review_server.resolve_repo_relative_csv_path(
                project_dir,
                "outputs/training_outputs/hard_case_review_manifest.csv",
            )
            self.assertEqual(
                resolved,
                (project_dir / "outputs/training_outputs/hard_case_review_manifest.csv").resolve(),
            )

    def test_resolve_repo_relative_csv_path_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            with self.assertRaises(ValueError):
                review_server.resolve_repo_relative_csv_path(project_dir, "../outside.csv")

    def test_resolve_repo_relative_csv_path_rejects_non_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            with self.assertRaises(ValueError):
                review_server.resolve_repo_relative_csv_path(project_dir, "notes.txt")

    def test_save_and_load_csv_text_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            relative_path = "outputs/training_outputs/hard_case_review_manifest.csv"
            expected_text = "name,manual_review_status\nclip.mp4,reviewed\n"
            saved_path = review_server.save_csv_text(project_dir, relative_path, expected_text)
            self.assertTrue(saved_path.exists())
            loaded_path, loaded_text = review_server.load_csv_text(project_dir, relative_path)
            self.assertEqual(loaded_path, saved_path)
            self.assertEqual(loaded_text, expected_text)


if __name__ == "__main__":
    unittest.main()

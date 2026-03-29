from __future__ import annotations

import unittest
from pathlib import Path

from test_utils import load_module


prepare_countix_manifest = load_module(
    "prepare_countix_manifest",
    "artifacts/2_Data_preparation/prepare_countix_manifest.py",
)


class PrepareCountixManifestTests(unittest.TestCase):
    def test_normalize_label_key(self):
        self.assertEqual(
            prepare_countix_manifest.normalize_label_key("Battle-Rope/Training"),
            "battle rope training",
        )

    def test_map_countix_type_maps_known_label(self):
        raw, mapped = prepare_countix_manifest.map_countix_type("jumping jacks", "fallback")
        self.assertEqual(raw, "jumping jacks")
        self.assertEqual(mapped, "jump_jacks")

    def test_resolve_name_uses_video_id(self):
        row = {"video_id": "abc123.mp4"}
        self.assertEqual(
            prepare_countix_manifest.resolve_name(row, "video_id", None),
            "abc123.mp4",
        )

    def test_resolve_video_path_builds_local_path(self):
        row = {"video_path": "train/example.mp4"}
        path = prepare_countix_manifest.resolve_video_path(
            row=row,
            name="example.mp4",
            path_col="video_path",
            video_dir=Path("/tmp/countix"),
        )
        self.assertTrue(path.endswith("/tmp/countix/train/example.mp4"))

    def test_parse_count_preserves_integer_string(self):
        self.assertEqual(prepare_countix_manifest.parse_count("12", 2), "12")

    def test_parse_count_preserves_float_string(self):
        self.assertEqual(prepare_countix_manifest.parse_count("12.5", 2), "12.5")


if __name__ == "__main__":
    unittest.main()

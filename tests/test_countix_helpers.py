from __future__ import annotations

import sys
import types
import unittest

from test_utils import load_module


sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault(
    "pandas",
    types.SimpleNamespace(
        DataFrame=object,
        errors=types.SimpleNamespace(EmptyDataError=Exception),
    ),
)
countix_download = load_module("countix_download", "Data/countixDownload.py")


class CountixHelperTests(unittest.TestCase):
    def test_normalize_label_key_handles_spacing_and_punctuation(self):
        self.assertEqual(
            countix_download.normalize_label_key("Pull-Ups / Advanced"),
            "pull ups advanced",
        )

    def test_map_countix_label_maps_supported_variant(self):
        raw, mapped = countix_download.map_countix_label("push-ups")
        self.assertEqual(raw, "push-ups")
        self.assertEqual(mapped, "push_up")

    def test_resolve_label_value_falls_back_across_columns(self):
        row = {"class": "battle rope training", "count": "10"}
        self.assertEqual(countix_download.resolve_label_value(row), "battle rope training")

    def test_resolve_video_id_accepts_video_id_column(self):
        row = {"video_id": "abc123"}
        self.assertEqual(countix_download.resolve_video_id(row), "abc123")

    def test_resolve_time_range_prefers_repetition_window(self):
        row = {
            "repetition_start": "2.5",
            "repetition_end": "7.0",
            "kinetics_start": "0",
            "kinetics_end": "10",
        }
        self.assertEqual(countix_download.resolve_time_range(row), (2.5, 7.0))

    def test_build_clip_id_is_clip_specific(self):
        clip_id = countix_download.build_clip_id("abc123", 2.5, 7.0)
        self.assertEqual(clip_id, "abc123_2p5_7")


if __name__ == "__main__":
    unittest.main()

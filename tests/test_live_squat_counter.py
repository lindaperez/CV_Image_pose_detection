from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "3_Modeling" / "run_live_squat_counter.py"
SPEC = importlib.util.spec_from_file_location("live_squat_counter", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load module spec from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["live_squat_counter"] = MODULE
SPEC.loader.exec_module(MODULE)


class LiveSquatCounterTest(unittest.TestCase):
    def test_start_recording_initializes_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_writer = object()
            with mock.patch.object(MODULE, "open_video_writer", return_value=dummy_writer):
                writer, state = MODULE.start_recording(
                    recording_dir=Path(tmpdir),
                    frame_size=(640, 480),
                    target_fps=15.0,
                    started_at=123.0,
                )
        self.assertIs(writer, dummy_writer)
        self.assertTrue(state.active)
        self.assertEqual(state.target_fps, 15.0)
        self.assertEqual(state.started_at, 123.0)
        self.assertTrue(str(state.output_path).endswith(".mp4"))

    def test_compute_recording_frames_to_write_matches_elapsed_time(self) -> None:
        state = MODULE.RecordingState(
            active=True,
            output_path="/tmp/demo.mp4",
            frames_written=3,
            target_fps=10.0,
            started_at=100.0,
        )
        frames_to_write = MODULE.compute_recording_frames_to_write(
            now_sec=100.6,
            recording_state=state,
        )
        self.assertEqual(frames_to_write, 3)

    def test_resolve_live_buffer_sizes_keeps_tcn_window_at_least_warmup(self) -> None:
        movement_window, tcn_window = MODULE.resolve_live_buffer_sizes(
            warmup_frames=40,
            movement_window_frames=12,
            tcn_window_frames=24,
        )
        self.assertEqual(movement_window, 12)
        self.assertEqual(tcn_window, 40)

    def test_build_recording_output_path_uses_expected_name_pattern(self) -> None:
        output_path = MODULE.build_recording_output_path(
            Path("/tmp/live"),
            timestamp=datetime(2026, 3, 28, 17, 5, 42),
        )
        self.assertEqual(str(output_path), "/tmp/live/live_squat_20260328_170542.mp4")

    def test_record_key_helpers_accept_ctrl_s_and_e(self) -> None:
        self.assertTrue(MODULE.is_start_record_key(19))
        self.assertTrue(MODULE.is_start_record_key(ord("s")))
        self.assertTrue(MODULE.is_stop_record_key(ord("e")))
        self.assertFalse(MODULE.is_stop_record_key(ord("q")))

    @unittest.skipIf(MODULE.np is None, "NumPy is required for movement-state feature rows")
    def test_update_live_movement_state_counts_two_cycles(self) -> None:
        np = MODULE.np
        state = MODULE.reset_live_movement_state()
        knee_flex_values = [0.0, 30.0, 50.0, 50.0, 25.0, 10.0, 0.0, 30.0, 50.0, 50.0, 25.0, 10.0]

        for idx, knee_flex in enumerate(knee_flex_values):
            row = np.zeros((len(MODULE.SQUAT_RUNTIME.FEATURE_COLUMNS),), dtype=np.float32)
            row[MODULE.SQUAT_RUNTIME.FEATURE_INDEX["frame_idx"]] = float(idx)
            row[MODULE.SQUAT_RUNTIME.FEATURE_INDEX["knee_flex"]] = knee_flex
            row[MODULE.SQUAT_RUNTIME.FEATURE_INDEX["frame_valid"]] = 1.0
            row[MODULE.SQUAT_RUNTIME.FEATURE_INDEX["mean_conf"]] = 0.95
            state = MODULE.update_live_movement_state(
                state,
                feature_row=row,
                cfg=dict(MODULE.SQUAT_RUNTIME.BEST_FSM_CFG),
            )

        self.assertEqual(state.count, 2)
        self.assertEqual(state.event_frames, [5, 11])

    def test_update_live_prediction_state_anchors_to_session_baseline(self) -> None:
        state = MODULE.reset_live_prediction_state()

        state = MODULE.update_live_prediction_state(
            state,
            raw_pred=11.2,
            eval_pred=11.2,
            movement_count=0,
            feature_frames=40,
            alpha=0.35,
            rise_margin=0.9,
        )
        self.assertEqual(state.display_count, 0)
        self.assertEqual(state.movement_count, 0)
        self.assertAlmostEqual(state.baseline_raw_pred, 11.2)
        self.assertAlmostEqual(state.latest_relative_pred, 0.0)

        state = MODULE.update_live_prediction_state(
            state,
            raw_pred=11.1,
            eval_pred=11.1,
            movement_count=0,
            feature_frames=60,
            alpha=0.35,
            rise_margin=0.9,
        )
        self.assertEqual(state.display_count, 0)

        state = MODULE.update_live_prediction_state(
            state,
            raw_pred=14.0,
            eval_pred=14.0,
            movement_count=1,
            feature_frames=140,
            alpha=1.0,
            rise_margin=0.9,
        )
        self.assertEqual(state.display_count, 1)
        self.assertEqual(state.movement_count, 1)
        self.assertAlmostEqual(state.latest_raw_pred, 14.0)
        self.assertAlmostEqual(state.latest_relative_pred, 2.8)

    def test_update_live_prediction_state_uses_movement_count_for_display(self) -> None:
        state = MODULE.reset_live_prediction_state()
        state = MODULE.update_live_prediction_state(
            state,
            raw_pred=10.0,
            eval_pred=10.0,
            movement_count=0,
            feature_frames=30,
            alpha=1.0,
            rise_margin=0.9,
        )
        state = MODULE.update_live_prediction_state(
            state,
            raw_pred=12.2,
            eval_pred=12.2,
            movement_count=1,
            feature_frames=60,
            alpha=1.0,
            rise_margin=0.9,
        )
        self.assertEqual(state.display_count, 1)

        state = MODULE.update_live_prediction_state(
            state,
            raw_pred=14.9,
            eval_pred=14.9,
            movement_count=1,
            feature_frames=90,
            alpha=1.0,
            rise_margin=0.9,
        )
        self.assertEqual(state.display_count, 1)

    def test_reset_live_prediction_state_clears_counts(self) -> None:
        state = MODULE.LivePredictionState(
            display_count=4,
            movement_count=4,
            baseline_raw_pred=8.1,
            smoothed_raw_pred=4.2,
            latest_raw_pred=4.3,
            latest_eval_pred=4.3,
            latest_relative_pred=3.2,
            inference_calls=12,
            last_feature_frames=200,
        )

        reset = MODULE.reset_live_prediction_state()

        self.assertEqual(reset.display_count, 0)
        self.assertEqual(reset.movement_count, 0)
        self.assertIsNone(reset.baseline_raw_pred)
        self.assertIsNone(reset.smoothed_raw_pred)
        self.assertIsNone(reset.latest_relative_pred)
        self.assertEqual(reset.inference_calls, 0)
        self.assertEqual(reset.last_feature_frames, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "3_Modeling" / "run_squat_counter.py"
SPEC = importlib.util.spec_from_file_location("squat_counter_runtime", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load module spec from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["squat_counter_runtime"] = MODULE
SPEC.loader.exec_module(MODULE)


@unittest.skipIf(MODULE.np is None, "NumPy is not installed in this interpreter")
class SquatCounterRuntimeTest(unittest.TestCase):
    def test_count_squat_reps_counts_two_cycles(self) -> None:
        np = MODULE.np
        rows = []
        knee_flex_values = [0.0, 30.0, 50.0, 50.0, 25.0, 10.0, 0.0, 30.0, 50.0, 50.0, 25.0, 10.0]
        for idx, knee_flex in enumerate(knee_flex_values):
            row = np.zeros((len(MODULE.FEATURE_COLUMNS),), dtype=np.float32)
            row[MODULE.FEATURE_INDEX["frame_idx"]] = float(idx)
            row[MODULE.FEATURE_INDEX["knee_flex"]] = knee_flex
            row[MODULE.FEATURE_INDEX["frame_valid"]] = 1.0
            row[MODULE.FEATURE_INDEX["mean_conf"]] = 0.95
            rows.append(row)
        feature_arr = np.stack(rows, axis=0)

        result = MODULE.count_squat_reps(feature_arr, dict(MODULE.BEST_FSM_CFG))

        self.assertEqual(result.pred_count, 2)
        self.assertEqual(result.event_frames, [5, 11])
        self.assertEqual(result.valid_frames, len(knee_flex_values))

    def test_build_squat_feature_array_from_pose_array_returns_expected_shape(self) -> None:
        np = MODULE.np
        pose_arr = np.zeros((3, 51), dtype=np.float32)

        def set_kpt(frame_idx: int, name: str, x: float, y: float, conf: float = 0.95) -> None:
            base = MODULE.KPT[name] * 3
            pose_arr[frame_idx, base] = x
            pose_arr[frame_idx, base + 1] = y
            pose_arr[frame_idx, base + 2] = conf

        base_frames = [
            {
                "left_shoulder": (0.0, 0.0),
                "right_shoulder": (1.0, 0.0),
                "left_hip": (0.0, 1.0),
                "right_hip": (1.0, 1.0),
                "left_knee": (0.0, 2.0),
                "right_knee": (1.0, 2.0),
                "left_ankle": (0.0, 3.0),
                "right_ankle": (1.0, 3.0),
            },
            {
                "left_shoulder": (0.0, 0.0),
                "right_shoulder": (1.0, 0.0),
                "left_hip": (0.0, 1.0),
                "right_hip": (1.0, 1.0),
                "left_knee": (0.35, 1.85),
                "right_knee": (0.65, 1.85),
                "left_ankle": (0.0, 3.0),
                "right_ankle": (1.0, 3.0),
            },
            {
                "left_shoulder": (0.0, 0.0),
                "right_shoulder": (1.0, 0.0),
                "left_hip": (0.0, 1.0),
                "right_hip": (1.0, 1.0),
                "left_knee": (0.0, 2.0),
                "right_knee": (1.0, 2.0),
                "left_ankle": (0.0, 3.0),
                "right_ankle": (1.0, 3.0),
            },
        ]

        for frame_idx, frame in enumerate(base_frames):
            for name, (x, y) in frame.items():
                set_kpt(frame_idx, name, x, y)

        feature_arr = MODULE.build_squat_feature_array_from_pose_array(pose_arr)

        self.assertEqual(feature_arr.shape, (3, len(MODULE.FEATURE_COLUMNS)))
        self.assertTrue(MODULE.np.isfinite(feature_arr).all())
        self.assertGreaterEqual(float(feature_arr[:, MODULE.FEATURE_INDEX["mean_conf"]].min()), 0.9)

    def test_build_squat_feature_array_from_single_pose_frame(self) -> None:
        np = MODULE.np
        pose_arr = np.zeros((1, 51), dtype=np.float32)

        def set_kpt(name: str, x: float, y: float, conf: float = 0.95) -> None:
            base = MODULE.KPT[name] * 3
            pose_arr[0, base] = x
            pose_arr[0, base + 1] = y
            pose_arr[0, base + 2] = conf

        for name, (x, y) in {
            "left_shoulder": (0.0, 0.0),
            "right_shoulder": (1.0, 0.0),
            "left_hip": (0.0, 1.0),
            "right_hip": (1.0, 1.0),
            "left_knee": (0.0, 2.0),
            "right_knee": (1.0, 2.0),
            "left_ankle": (0.0, 3.0),
            "right_ankle": (1.0, 3.0),
        }.items():
            set_kpt(name, x, y)

        feature_arr = MODULE.build_squat_feature_array_from_pose_array(pose_arr)

        self.assertEqual(feature_arr.shape, (1, len(MODULE.FEATURE_COLUMNS)))
        self.assertTrue(np.isfinite(feature_arr).all())
        self.assertEqual(float(feature_arr[0, MODULE.FEATURE_INDEX["hip_velocity"]]), 0.0)

    def test_prepare_tcn_input_array_drops_frame_idx_and_resamples(self) -> None:
        np = MODULE.np
        feature_arr = np.arange(5 * len(MODULE.FEATURE_COLUMNS), dtype=np.float32).reshape(5, len(MODULE.FEATURE_COLUMNS))
        feature_mean = np.zeros((len(MODULE.FEATURE_COLUMNS) - 1,), dtype=np.float32)
        feature_std = np.ones((len(MODULE.FEATURE_COLUMNS) - 1,), dtype=np.float32)

        prepared = MODULE.prepare_tcn_input_array(
            feature_arr,
            seq_len=8,
            drop_frame_idx=True,
            feature_mean=feature_mean,
            feature_std=feature_std,
            input_dim=len(MODULE.FEATURE_COLUMNS) - 1,
        )

        self.assertEqual(prepared.shape, (8, len(MODULE.FEATURE_COLUMNS) - 1))
        self.assertAlmostEqual(float(prepared[0, 0]), float(feature_arr[0, 1]))
        self.assertAlmostEqual(float(prepared[-1, -1]), float(feature_arr[-1, -1]))

    @unittest.skipIf(MODULE.torch is None, "PyTorch is not installed in this interpreter")
    def test_run_tcn_inference_works_with_local_squat_run(self) -> None:
        project_dir = MODULE.resolve_project_dir()
        run_dir = MODULE.resolve_tcn_run_dir(project_dir, None)
        feature_path = (
            project_dir
            / "Data"
            / "LLSP"
            / "annotation_cleaned"
            / "squat_features"
            / "train3946_squat_features.npy"
        )
        feature_arr = MODULE.load_squat_feature_array(feature_path)

        result = MODULE.run_tcn_inference(
            feature_arr,
            run_dir=run_dir,
            device_name="cpu",
        )

        self.assertEqual(result.run_name, MODULE.DEFAULT_TCN_RUN_NAME)
        self.assertEqual(result.practical_pred_count, 3)
        self.assertAlmostEqual(result.raw_pred_count, 2.955331563949585, places=5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

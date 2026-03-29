#!/usr/bin/env python3
"""
Live squat counting prototype using webcam + YOLO pose + saved squat TCN.

This script is intentionally narrow:
1. Capture frames from a webcam.
2. Run frame-wise YOLO pose with the same tracked-person logic used offline.
3. Accumulate squat pose frames from session start.
4. Periodically run the dedicated squat TCN on the accumulated session buffer.
5. Show the live count on screen.

Controls:
- q: quit
- r: reset the current session buffer and count
"""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import sys
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception as exc:  # pragma: no cover - runtime dependency guard
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime dependency guard
    np = None
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None


def load_squat_runtime_module():
    module_path = Path(__file__).resolve().with_name("run_squat_counter.py")
    module_name = "live_squat_counter_runtime"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load run_squat_counter.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SQUAT_RUNTIME = load_squat_runtime_module()
POSE_RUNTIME = SQUAT_RUNTIME.POSE_EXTRACTOR
YOLO = SQUAT_RUNTIME.YOLO
MODEL_DEFAULT = SQUAT_RUNTIME.MODEL_DEFAULT

KEYPOINT_DRAW_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]
EDGE_INDEX_PAIRS = [(SQUAT_RUNTIME.KPT[a], SQUAT_RUNTIME.KPT[b]) for a, b in KEYPOINT_DRAW_EDGES]


@dataclass
class LivePredictionState:
    display_count: int = 0
    movement_count: int = 0
    baseline_raw_pred: float | None = None
    smoothed_raw_pred: float | None = None
    latest_raw_pred: float | None = None
    latest_eval_pred: float | None = None
    latest_relative_pred: float | None = None
    inference_calls: int = 0
    last_feature_frames: int = 0


@dataclass
class LiveMovementState:
    count: int = 0
    phase: str = "UP"
    bottom_frames: int = 0
    event_frames: list[int] | None = None

    def __post_init__(self) -> None:
        if self.event_frames is None:
            self.event_frames = []


@dataclass
class RecordingState:
    active: bool = False
    output_path: str | None = None
    frames_written: int = 0
    target_fps: float = 0.0
    started_at: float | None = None


def ensure_runtime_dependencies() -> None:
    if np is None:
        raise SystemExit(
            "NumPy is required for live squat counting. Install it in the project environment."
        ) from NUMPY_IMPORT_ERROR
    if cv2 is None:
        raise SystemExit(
            "OpenCV is required for webcam squat counting. Install opencv-python in the project environment."
        ) from CV2_IMPORT_ERROR
    if YOLO is None:
        raise SystemExit(
            "Ultralytics YOLO is required for webcam squat counting. Install ultralytics in the project environment."
        ) from SQUAT_RUNTIME.ULTRALYTICS_IMPORT_ERROR
    SQUAT_RUNTIME.ensure_torch_runtime()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live webcam squat counter.")
    parser.add_argument("--project-dir", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index passed to OpenCV.")
    parser.add_argument("--camera-width", type=int, default=1280, help="Preferred webcam width.")
    parser.add_argument("--camera-height", type=int, default=720, help="Preferred webcam height.")
    parser.add_argument("--mirror", action="store_true", help="Mirror the webcam preview horizontally.")
    parser.add_argument("--model", type=str, default=str(MODEL_DEFAULT), help="YOLO pose model path/name.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", type=str, default=None, help="YOLO inference device, for example cpu, cuda:0, or mps.")
    parser.add_argument("--track-search-expand", type=float, default=1.6, help="Search-box expansion around the tracked person.")
    parser.add_argument("--track-max-misses", type=int, default=8, help="How many missed pose frames before resetting the tracker.")
    parser.add_argument("--tcn-run-dir", type=str, default=None, help="Optional local squat TCN run directory.")
    parser.add_argument("--tcn-device", type=str, default="cpu", help="PyTorch device for TCN inference.")
    parser.add_argument("--warmup-frames", type=int, default=24, help="Minimum detected pose frames before live inference begins.")
    parser.add_argument("--infer-every", type=int, default=5, help="Run the TCN every N detected pose frames.")
    parser.add_argument(
        "--movement-window-frames",
        type=int,
        default=96,
        help="How many recent detected pose frames to use when deriving the live movement feature row.",
    )
    parser.add_argument(
        "--tcn-window-frames",
        type=int,
        default=192,
        help="How many recent detected pose frames to keep for live TCN inference.",
    )
    parser.add_argument("--prediction-alpha", type=float, default=0.35, help="EMA smoothing for the live raw TCN prediction.")
    parser.add_argument("--rise-margin", type=float, default=0.9, help="Minimum smoothed raw margin before the displayed count increments.")
    parser.add_argument("--box-line-thickness", type=int, default=2, help="Bounding box line thickness.")
    parser.add_argument("--skeleton-thickness", type=int, default=2, help="Pose skeleton line thickness.")
    parser.add_argument("--joint-radius", type=int, default=4, help="Pose joint marker radius.")
    parser.add_argument("--overlay-conf-threshold", type=float, default=0.25, help="Minimum keypoint confidence to draw the overlay.")
    parser.add_argument("--window-name", type=str, default="Live Squat Counter", help="OpenCV window title.")
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=None,
        help="Optional directory for recorded live-session videos. Defaults under training_outputs/live_squat_recordings.",
    )
    parser.add_argument(
        "--auto-record",
        action="store_true",
        help="Start recording automatically when the live window starts.",
    )
    parser.add_argument(
        "--recording-fps",
        type=float,
        default=20.0,
        help="Fallback FPS for saved recordings if the camera does not report a usable FPS.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional session summary JSON written on exit.")
    return parser.parse_args()


def update_live_prediction_state(
    state: LivePredictionState,
    *,
    raw_pred: float,
    eval_pred: float,
    movement_count: int,
    feature_frames: int,
    alpha: float,
    rise_margin: float,
) -> LivePredictionState:
    clipped_raw = max(0.0, float(raw_pred))
    if state.smoothed_raw_pred is None:
        smoothed = clipped_raw
    else:
        smoothed = (alpha * clipped_raw) + ((1.0 - alpha) * state.smoothed_raw_pred)

    baseline = smoothed if state.baseline_raw_pred is None else state.baseline_raw_pred
    relative_pred = max(0.0, smoothed - baseline)
    display_count = max(state.display_count, int(movement_count))

    return LivePredictionState(
        display_count=display_count,
        movement_count=int(movement_count),
        baseline_raw_pred=baseline,
        smoothed_raw_pred=smoothed,
        latest_raw_pred=clipped_raw,
        latest_eval_pred=float(eval_pred),
        latest_relative_pred=relative_pred,
        inference_calls=state.inference_calls + 1,
        last_feature_frames=feature_frames,
    )


def reset_live_prediction_state() -> LivePredictionState:
    return LivePredictionState()


def reset_live_movement_state() -> LiveMovementState:
    return LiveMovementState()


def default_recording_dir(project_dir: Path) -> Path:
    return project_dir / "artifacts" / "3_Modeling" / "training_outputs" / "live_squat_recordings"


def resolve_live_buffer_sizes(
    *,
    warmup_frames: int,
    movement_window_frames: int,
    tcn_window_frames: int,
) -> tuple[int, int]:
    movement_window = max(1, int(movement_window_frames))
    tcn_window = max(int(warmup_frames), int(tcn_window_frames), 1)
    return movement_window, tcn_window


def build_recording_output_path(recording_dir: Path, timestamp: datetime | None = None) -> Path:
    ts = timestamp or datetime.now()
    filename = f"live_squat_{ts.strftime('%Y%m%d_%H%M%S')}.mp4"
    return recording_dir / filename


def is_start_record_key(key: int) -> bool:
    return key in {19, ord("s"), ord("S")}


def is_stop_record_key(key: int) -> bool:
    return key in {ord("e"), ord("E")}


def open_video_writer(output_path: Path, frame_size: tuple[int, int], fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(max(fps, 1.0)), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    return writer


def start_recording(
    *,
    recording_dir: Path,
    frame_size: tuple[int, int],
    target_fps: float,
    started_at: float | None = None,
) -> tuple[Any, RecordingState]:
    recording_dir.mkdir(parents=True, exist_ok=True)
    output_path = build_recording_output_path(recording_dir)
    writer = open_video_writer(
        output_path=output_path,
        frame_size=frame_size,
        fps=target_fps,
    )
    state = RecordingState(
        active=True,
        output_path=str(output_path),
        frames_written=0,
        target_fps=float(target_fps),
        started_at=time.time() if started_at is None else float(started_at),
    )
    return writer, state


def compute_recording_frames_to_write(
    *,
    now_sec: float,
    recording_state: RecordingState,
) -> int:
    if (
        not recording_state.active
        or recording_state.started_at is None
        or recording_state.target_fps <= 0.0
    ):
        return 0
    elapsed_sec = max(0.0, now_sec - recording_state.started_at)
    expected_total = max(1, int(round(elapsed_sec * recording_state.target_fps)))
    return max(0, expected_total - recording_state.frames_written)


def update_live_movement_state(
    state: LiveMovementState,
    *,
    feature_row: np.ndarray,
    cfg: dict[str, Any],
) -> LiveMovementState:
    phase = state.phase
    count = state.count
    bottom_frames = state.bottom_frames
    event_frames = list(state.event_frames or [])

    valid_frame = (
        float(feature_row[SQUAT_RUNTIME.FEATURE_INDEX["frame_valid"]]) >= float(cfg["min_valid_ratio"])
        and float(feature_row[SQUAT_RUNTIME.FEATURE_INDEX["mean_conf"]]) >= float(cfg["min_conf"])
    )
    if not valid_frame:
        return LiveMovementState(
            count=count,
            phase=phase,
            bottom_frames=bottom_frames,
            event_frames=event_frames,
        )

    knee_flex = float(feature_row[SQUAT_RUNTIME.FEATURE_INDEX["knee_flex"]])
    frame_idx = int(feature_row[SQUAT_RUNTIME.FEATURE_INDEX["frame_idx"]])

    if phase == "UP":
        bottom_frames = 0
        if knee_flex > float(cfg["enter_down"]):
            phase = "DESCENDING"

    elif phase == "DESCENDING":
        if knee_flex > float(cfg["enter_bottom"]):
            phase = "BOTTOM"
            bottom_frames = 1
        elif knee_flex < float(cfg["back_to_up"]):
            phase = "UP"

    elif phase == "BOTTOM":
        if knee_flex > float(cfg["exit_bottom"]):
            bottom_frames += 1
        else:
            if bottom_frames >= int(cfg["min_bottom_frames"]):
                phase = "ASCENDING"
            else:
                phase = "DESCENDING"

    elif phase == "ASCENDING":
        if knee_flex < float(cfg["back_to_up"]):
            count += 1
            event_frames.append(frame_idx)
            phase = "UP"
            bottom_frames = 0
        elif knee_flex > float(cfg["enter_bottom"]):
            phase = "BOTTOM"
            bottom_frames = 1

    return LiveMovementState(
        count=count,
        phase=phase,
        bottom_frames=bottom_frames,
        event_frames=event_frames,
    )


def open_camera(camera_index: int, width: int, height: int):
    cap = cv2.VideoCapture(camera_index)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def detect_tracked_person(
    *,
    model,
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    device: str | None,
    tracked_bbox: np.ndarray | None,
    misses: int,
    track_search_expand: float,
    track_max_misses: int,
):
    person = None
    frame_h, frame_w = frame.shape[:2]

    if tracked_bbox is not None and misses <= track_max_misses:
        search_box = POSE_RUNTIME.expand_box(tracked_bbox, track_search_expand, frame_w, frame_h)
        crop, offset_xy = POSE_RUNTIME.crop_frame(frame, search_box)
        if crop.size > 0:
            crop_people = POSE_RUNTIME.infer_people(
                model=model,
                frame=crop,
                conf=conf,
                imgsz=imgsz,
                device=device,
                offset_xy=offset_xy,
            )
            person = POSE_RUNTIME.select_person(crop_people, prev_bbox=tracked_bbox)

    if person is None:
        people = POSE_RUNTIME.infer_people(
            model=model,
            frame=frame,
            conf=conf,
            imgsz=imgsz,
            device=device,
        )
        person = POSE_RUNTIME.select_person(people, prev_bbox=tracked_bbox)

    if person is None:
        return None, None

    clipped_bbox = POSE_RUNTIME.clip_box(person.bbox, frame_w, frame_h)
    return person, clipped_bbox


def draw_pose_overlay(
    frame: np.ndarray,
    *,
    person,
    box_line_thickness: int,
    skeleton_thickness: int,
    joint_radius: int,
    conf_threshold: float,
) -> None:
    if person is None:
        return

    bbox = person.bbox.astype(int)
    cv2.rectangle(
        frame,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        (30, 180, 255),
        max(1, box_line_thickness),
    )

    xy = person.keypoints_xy
    conf = person.keypoints_conf
    for idx_a, idx_b in EDGE_INDEX_PAIRS:
        if conf[idx_a] < conf_threshold or conf[idx_b] < conf_threshold:
            continue
        pt_a = tuple(int(v) for v in xy[idx_a])
        pt_b = tuple(int(v) for v in xy[idx_b])
        cv2.line(frame, pt_a, pt_b, (70, 220, 120), max(1, skeleton_thickness), lineType=cv2.LINE_AA)

    for idx in range(xy.shape[0]):
        if conf[idx] < conf_threshold:
            continue
        pt = tuple(int(v) for v in xy[idx])
        cv2.circle(frame, pt, max(1, joint_radius), (255, 235, 100), -1, lineType=cv2.LINE_AA)


def draw_status_panel(
    frame: np.ndarray,
    *,
    state: LivePredictionState,
    recording_state: RecordingState,
    frames_total: int,
    pose_frames_used: int,
    fps_ema: float,
    session_duration_sec: float,
    tracking_status: str,
    run_name: str,
) -> None:
    lines = [
        "Live Squat TCN",
        f"Count: {state.display_count}",
        f"Movement count: {state.movement_count}",
        (
            f"Raw: {state.latest_raw_pred:.2f} | Smooth: {state.smoothed_raw_pred:.2f} | Relative: {state.latest_relative_pred:.2f}"
            if (
                state.latest_raw_pred is not None
                and state.smoothed_raw_pred is not None
                and state.latest_relative_pred is not None
            )
            else "Raw TCN: warming up"
        ),
        f"Detected pose frames: {pose_frames_used}",
        f"All frames: {frames_total}",
        f"Session sec: {session_duration_sec:.1f}",
        f"Preview FPS: {fps_ema:.1f}",
        f"Status: {tracking_status}",
        (
            f"Recording: ON ({recording_state.frames_written} frames)"
            if recording_state.active
            else "Recording: OFF"
        ),
        f"Run: {run_name}",
        "Keys: Ctrl+S record | E stop | R reset | Q quit",
    ]

    pad = 12
    line_h = 28
    panel_w = 420
    panel_h = pad * 2 + (line_h * len(lines))
    overlay = frame.copy()
    cv2.rectangle(overlay, (16, 16), (16 + panel_w, 16 + panel_h), (18, 24, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0.0, frame)

    for idx, line in enumerate(lines):
        y = 16 + pad + ((idx + 1) * line_h) - 8
        font_scale = 0.8 if idx else 0.95
        color = (255, 255, 255) if idx else (150, 240, 255)
        cv2.putText(
            frame,
            line,
            (28, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            lineType=cv2.LINE_AA,
        )


def build_session_summary(
    *,
    args: argparse.Namespace,
    project_dir: Path,
    tcn_run_dir: Path,
    state: LivePredictionState,
    recording_state: RecordingState,
    recording_outputs: list[str],
    frames_total: int,
    pose_frames_used: int,
    session_duration_sec: float,
) -> dict[str, Any]:
    return {
        "task": "live_squat_counting",
        "project_dir": str(project_dir),
        "camera_index": args.camera_index,
        "window_name": args.window_name,
        "backend": "tcn",
        "run_name": tcn_run_dir.name,
        "tcn_run_dir": str(tcn_run_dir),
        "frames_total": frames_total,
        "pose_frames_used": pose_frames_used,
        "session_duration_sec": session_duration_sec,
        "prediction_state": asdict(state),
        "recording_state": asdict(recording_state),
        "recording_outputs": recording_outputs,
        "notes": [
            "This is a live webcam squat-only prototype using the saved dedicated squat TCN.",
            "The displayed live count is movement-gated from the accumulated squat feature buffer.",
            "The TCN prediction is still shown as a supporting estimate on the HUD.",
            "Press Ctrl+S to start recording, E to stop and save, R to reset, and Q to quit.",
            "Use --auto-record if you want recording to start immediately on launch.",
        ],
    }


def main() -> int:
    ensure_runtime_dependencies()
    args = parse_args()
    project_dir = SQUAT_RUNTIME.resolve_project_dir(args.project_dir)
    tcn_run_dir = SQUAT_RUNTIME.resolve_tcn_run_dir(project_dir, args.tcn_run_dir)
    recording_dir = (args.recording_dir.expanduser().resolve() if args.recording_dir else default_recording_dir(project_dir))

    cap = open_camera(args.camera_index, args.camera_width, args.camera_height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index {args.camera_index}.")

    model = YOLO(str(args.model))
    camera_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    writer = None
    recording_state = RecordingState()
    recording_outputs: list[str] = []
    tracked_bbox = None
    misses = 0
    frames_total = 0
    pose_frames_used = 0
    movement_window_size, tcn_window_size = resolve_live_buffer_sizes(
        warmup_frames=args.warmup_frames,
        movement_window_frames=args.movement_window_frames,
        tcn_window_frames=args.tcn_window_frames,
    )
    movement_pose_rows: deque[np.ndarray] = deque(maxlen=movement_window_size)
    tcn_pose_rows: deque[np.ndarray] = deque(maxlen=tcn_window_size)
    live_state = reset_live_prediction_state()
    movement_state = reset_live_movement_state()
    latest_tracking_status = "warming up"
    session_started_at = time.time()
    prev_frame_time = time.time()
    fps_ema = 0.0
    target_recording_fps = camera_fps if camera_fps > 1.0 else args.recording_fps

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the webcam.")

            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.time()
            frame_delta = max(1e-6, now - prev_frame_time)
            current_fps = 1.0 / frame_delta
            fps_ema = current_fps if fps_ema <= 0.0 else (0.15 * current_fps) + (0.85 * fps_ema)
            prev_frame_time = now

            frames_total += 1
            display_frame = frame.copy()
            person, clipped_bbox = detect_tracked_person(
                model=model,
                frame=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                tracked_bbox=tracked_bbox,
                misses=misses,
                track_search_expand=args.track_search_expand,
                track_max_misses=args.track_max_misses,
            )

            if person is None:
                misses += 1
                if misses > args.track_max_misses:
                    tracked_bbox = None
                latest_tracking_status = "no pose target"
            else:
                pose_frame = np.concatenate(
                    [person.keypoints_xy, person.keypoints_conf[:, None]],
                    axis=1,
                ).reshape(-1).astype(np.float32)
                movement_pose_rows.append(pose_frame)
                tcn_pose_rows.append(pose_frame)
                pose_frames_used += 1
                tracked_bbox = clipped_bbox
                misses = 0
                latest_tracking_status = "tracking target"

                movement_pose_arr = np.stack(
                    list(movement_pose_rows),
                    axis=0,
                ).astype(np.float32)
                movement_feature_arr = SQUAT_RUNTIME.build_squat_feature_array_from_pose_array(movement_pose_arr)
                movement_state = update_live_movement_state(
                    movement_state,
                    feature_row=movement_feature_arr[-1],
                    cfg=dict(SQUAT_RUNTIME.BEST_FSM_CFG),
                )
                live_state.display_count = movement_state.count
                live_state.movement_count = movement_state.count

                if len(tcn_pose_rows) >= args.warmup_frames and pose_frames_used % args.infer_every == 0:
                    pose_arr = np.stack(list(tcn_pose_rows), axis=0).astype(np.float32)
                    feature_arr = SQUAT_RUNTIME.build_squat_feature_array_from_pose_array(pose_arr)
                    tcn_result = SQUAT_RUNTIME.run_tcn_inference(
                        feature_arr,
                        run_dir=tcn_run_dir,
                        device_name=args.tcn_device,
                    )
                    live_state = update_live_prediction_state(
                        live_state,
                        raw_pred=tcn_result.raw_pred_count,
                        eval_pred=tcn_result.eval_pred_count,
                        movement_count=movement_state.count,
                        feature_frames=feature_arr.shape[0],
                        alpha=args.prediction_alpha,
                        rise_margin=args.rise_margin,
                    )

            draw_pose_overlay(
                display_frame,
                person=person,
                box_line_thickness=args.box_line_thickness,
                skeleton_thickness=args.skeleton_thickness,
                joint_radius=args.joint_radius,
                conf_threshold=args.overlay_conf_threshold,
            )
            draw_status_panel(
                display_frame,
                state=live_state,
                recording_state=recording_state,
                frames_total=frames_total,
                pose_frames_used=pose_frames_used,
                fps_ema=fps_ema,
                session_duration_sec=time.time() - session_started_at,
                tracking_status=latest_tracking_status,
                run_name=tcn_run_dir.name,
            )

            if args.auto_record and writer is None:
                frame_h, frame_w = display_frame.shape[:2]
                writer, recording_state = start_recording(
                    recording_dir=recording_dir,
                    frame_size=(frame_w, frame_h),
                    target_fps=target_recording_fps,
                    started_at=now,
                )
                latest_tracking_status = "recording"

            if writer is not None and recording_state.active:
                frames_to_write = compute_recording_frames_to_write(
                    now_sec=now,
                    recording_state=recording_state,
                )
                for _ in range(frames_to_write):
                    writer.write(display_frame)
                recording_state.frames_written += frames_to_write

            cv2.imshow(args.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break
            if is_start_record_key(key) and writer is None:
                frame_h, frame_w = display_frame.shape[:2]
                writer, recording_state = start_recording(
                    recording_dir=recording_dir,
                    frame_size=(frame_w, frame_h),
                    target_fps=target_recording_fps,
                    started_at=now,
                )
                latest_tracking_status = "recording"
            if is_stop_record_key(key) and writer is not None:
                writer.release()
                writer = None
                if recording_state.output_path:
                    recording_outputs.append(recording_state.output_path)
                recording_state = RecordingState()
                latest_tracking_status = "recording saved"
            if key == ord("r"):
                tracked_bbox = None
                misses = 0
                frames_total = 0
                pose_frames_used = 0
                movement_pose_rows.clear()
                tcn_pose_rows.clear()
                live_state = reset_live_prediction_state()
                movement_state = reset_live_movement_state()
                session_started_at = time.time()
                latest_tracking_status = "session reset"
    finally:
        if writer is not None:
            writer.release()
            if recording_state.output_path:
                recording_outputs.append(recording_state.output_path)
        cap.release()
        cv2.destroyAllWindows()

    payload = build_session_summary(
        args=args,
        project_dir=project_dir,
        tcn_run_dir=tcn_run_dir,
        state=live_state,
        recording_state=recording_state,
        recording_outputs=recording_outputs,
        frames_total=frames_total,
        pose_frames_used=pose_frames_used,
        session_duration_sec=time.time() - session_started_at,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote live squat summary to {args.output_json.resolve()}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

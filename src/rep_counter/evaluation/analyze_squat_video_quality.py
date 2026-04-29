#!/usr/bin/env python3
"""
Automated first-pass audit for squat videos.

This script joins the squat feature summary with the local video files,
extracts basic video metadata with ffprobe, tags likely failure modes,
and optionally writes contact sheets for quick visual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIDEO_DIR = REPO_ROOT / "Data" / "LLSP" / "video"
DEFAULT_INDEX_CSV = REPO_ROOT / "Data" / "LLSP" / "annotation_cleaned" / "pose_feature_index_squat.csv"


@dataclass
class AuditRow:
    name: str
    split: str
    video_folder: str
    relative_video_path: str
    rep_count: str
    status: str
    frames_total: int
    frames_valid: int
    valid_ratio: float
    mean_conf: float
    width: int
    height: int
    duration_sec: float
    fps: float
    orientation: str
    issue_tags: str
    severity: str
    video_path: str
    contact_sheet: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit squat video quality from local files.")
    parser.add_argument("--summary-csv", type=Path, required=True, help="Path to squat_feature_summary.csv")
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV, help="Path to pose_feature_index_squat.csv")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="Root LLSP video directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "review" / "squat_video_audit",
        help="Directory for audit outputs",
    )
    parser.add_argument(
        "--contact-sheet-limit",
        type=int,
        default=30,
        help="Number of most severe videos for which to write contact sheets. 0 writes none. -1 writes all.",
    )
    parser.add_argument(
        "--contact-width",
        type=int,
        default=320,
        help="Per-tile width for contact sheets.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_video_lookup(video_dir: Path) -> Dict[str, List[Path]]:
    lookup: Dict[str, List[Path]] = {}
    for video_path in video_dir.rglob("*.mp4"):
        lookup.setdefault(video_path.name.lower(), []).append(video_path.resolve())
    return lookup


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def ffprobe_video(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration = parse_float(fmt.get("duration"), 0.0)
    fps = rational_to_float(stream.get("avg_frame_rate")) or rational_to_float(stream.get("r_frame_rate"))
    return {
        "width": width,
        "height": height,
        "duration_sec": duration,
        "fps": fps,
        "orientation": "portrait" if height > width else "landscape",
    }


def rational_to_float(value: Optional[str]) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    if "/" not in value:
        return parse_float(value, 0.0)
    num, den = value.split("/", 1)
    den_f = parse_float(den, 0.0)
    return parse_float(num, 0.0) / den_f if den_f else 0.0


def tag_issues(mean_conf: float, valid_ratio: float, frames_total: int, orientation: str) -> tuple[List[str], str]:
    tags: List[str] = []
    if frames_total < 120:
        tags.append("short_clip")
    if orientation == "portrait":
        tags.append("portrait_framing")

    if valid_ratio == 0.0:
        tags.append("no_valid_lower_body_frames")
    elif valid_ratio < 0.25:
        tags.append("severe_lower_body_visibility_loss")
    elif valid_ratio < 0.75:
        tags.append("partial_lower_body_visibility")
    elif valid_ratio < 0.90:
        tags.append("moderate_lower_body_coverage_loss")

    if mean_conf < 0.25:
        tags.append("extreme_pose_confidence_failure")
    elif mean_conf < 0.40:
        tags.append("very_low_pose_confidence")
    elif mean_conf < 0.50:
        tags.append("low_pose_confidence")
    elif mean_conf < 0.70:
        tags.append("moderate_pose_confidence")

    if valid_ratio < 0.50 and mean_conf < 0.50:
        tags.append("likely_framing_or_occlusion_issue")
    elif valid_ratio < 0.75 and mean_conf >= 0.70:
        tags.append("likely_pose_contract_mismatch")
    elif valid_ratio >= 0.90 and mean_conf < 0.70:
        tags.append("confidence_noise_but_geometry_available")

    if "no_valid_lower_body_frames" in tags or "extreme_pose_confidence_failure" in tags:
        severity = "critical"
    elif (
        "severe_lower_body_visibility_loss" in tags
        or "very_low_pose_confidence" in tags
        or "likely_framing_or_occlusion_issue" in tags
    ):
        severity = "high"
    elif (
        "partial_lower_body_visibility" in tags
        or "low_pose_confidence" in tags
        or "moderate_lower_body_coverage_loss" in tags
    ):
        severity = "medium"
    elif "moderate_pose_confidence" in tags or "short_clip" in tags:
        severity = "review"
    else:
        severity = "ok"
    return tags, severity


def write_contact_sheet(video_path: Path, output_path: Path, duration_sec: float, contact_width: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = 4.0 / duration_sec if duration_sec > 0.0 else 1.0
    vf = f"fps={fps:.6f},scale={contact_width}:-1:force_original_aspect_ratio=decrease,tile=2x2"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        vf,
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def write_csv(path: Path, rows: Iterable[AuditRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "split",
                "video_folder",
                "relative_video_path",
                "rep_count",
                "status",
                "frames_total",
                "frames_valid",
                "valid_ratio",
                "mean_conf",
                "width",
                "height",
                "duration_sec",
                "fps",
                "orientation",
                "issue_tags",
                "severity",
                "video_path",
                "contact_sheet",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.name,
                    row.split,
                    row.video_folder,
                    row.relative_video_path,
                    row.rep_count,
                    row.status,
                    row.frames_total,
                    row.frames_valid,
                    f"{row.valid_ratio:.6f}",
                    f"{row.mean_conf:.6f}",
                    row.width,
                    row.height,
                    f"{row.duration_sec:.3f}",
                    f"{row.fps:.3f}",
                    row.orientation,
                    row.issue_tags,
                    row.severity,
                    row.video_path,
                    row.contact_sheet,
                ]
            )


def main() -> int:
    args = parse_args()
    summary_rows = load_csv(args.summary_csv)
    index_rows = load_csv(args.index_csv)

    index_by_name = {row["name"]: row for row in index_rows}
    video_lookup = build_video_lookup(args.video_dir)

    output_dir = args.output_dir
    contact_dir = output_dir / "contact_sheets"

    audit_rows: List[AuditRow] = []
    missing_videos: List[str] = []

    for row in summary_rows:
        name = row["name"]
        video_candidates = video_lookup.get(name.lower(), [])
        if not video_candidates:
            missing_videos.append(name)
            continue
        video_path = sorted(video_candidates, key=lambda path: str(path))[0]
        relative_video_path = video_path.relative_to(args.video_dir)
        metadata = ffprobe_video(video_path)

        frames_total = parse_int(row.get("frames_total", "0"))
        frames_valid = parse_int(row.get("frames_valid", "0"))
        valid_ratio = (frames_valid / frames_total) if frames_total else 0.0
        mean_conf = parse_float(row.get("mean_conf"), 0.0)
        tags, severity = tag_issues(mean_conf, valid_ratio, frames_total, metadata["orientation"])

        index_row = index_by_name.get(name, {})
        audit_rows.append(
            AuditRow(
                name=name,
                split=index_row.get("split", ""),
                video_folder=relative_video_path.parent.as_posix(),
                relative_video_path=relative_video_path.as_posix(),
                rep_count=index_row.get("count", ""),
                status=row.get("status", ""),
                frames_total=frames_total,
                frames_valid=frames_valid,
                valid_ratio=valid_ratio,
                mean_conf=mean_conf,
                width=metadata["width"],
                height=metadata["height"],
                duration_sec=metadata["duration_sec"],
                fps=metadata["fps"],
                orientation=metadata["orientation"],
                issue_tags=",".join(tags),
                severity=severity,
                video_path=str(video_path),
                contact_sheet="",
            )
        )

    severity_order = {"critical": 0, "high": 1, "medium": 2, "review": 3, "ok": 4}
    audit_rows.sort(key=lambda row: (severity_order.get(row.severity, 9), row.mean_conf, row.valid_ratio, row.name))

    sheet_limit = args.contact_sheet_limit
    if sheet_limit != 0:
        rows_for_sheets = audit_rows if sheet_limit < 0 else audit_rows[:sheet_limit]
        for row in rows_for_sheets:
            sheet_path = contact_dir / f"{Path(row.name).stem}_contact.jpg"
            write_contact_sheet(Path(row.video_path), sheet_path, row.duration_sec, args.contact_width)
            row.contact_sheet = str(sheet_path)

    write_csv(output_dir / "squat_video_audit.csv", audit_rows)

    summary_payload = {
        "total_rows": len(summary_rows),
        "audited_rows": len(audit_rows),
        "missing_videos": missing_videos,
        "severity_counts": {},
        "low_confidence_counts": {
            "mean_conf_lt_0_25": sum(1 for row in audit_rows if row.mean_conf < 0.25),
            "mean_conf_lt_0_40": sum(1 for row in audit_rows if row.mean_conf < 0.40),
            "mean_conf_lt_0_50": sum(1 for row in audit_rows if row.mean_conf < 0.50),
            "mean_conf_lt_0_70": sum(1 for row in audit_rows if row.mean_conf < 0.70),
        },
        "valid_ratio_counts": {
            "valid_ratio_lt_0_25": sum(1 for row in audit_rows if row.valid_ratio < 0.25),
            "valid_ratio_lt_0_50": sum(1 for row in audit_rows if row.valid_ratio < 0.50),
            "valid_ratio_lt_0_75": sum(1 for row in audit_rows if row.valid_ratio < 0.75),
            "valid_ratio_lt_0_90": sum(1 for row in audit_rows if row.valid_ratio < 0.90),
        },
    }
    for severity in severity_order:
        summary_payload["severity_counts"][severity] = sum(1 for row in audit_rows if row.severity == severity)

    (output_dir / "squat_video_audit_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"audited_rows={len(audit_rows)}")
    print(f"missing_videos={len(missing_videos)}")
    print(f"report_csv={output_dir / 'squat_video_audit.csv'}")
    print(f"summary_json={output_dir / 'squat_video_audit_summary.json'}")
    if sheet_limit != 0:
        print(f"contact_sheets_dir={contact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

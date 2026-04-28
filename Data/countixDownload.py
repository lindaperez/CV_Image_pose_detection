# =============================================================================
# countix_download.py
# Countix Exercise Subset — Stream, Filter, Download, Validate
#
# Pipeline:
#   S3 annotations (few MB)
#       ↓  stream + filter in memory
#   Filtered exercise rows (~600-900 clips)
#       ↓  yt-dlp selective clip download
#   OUTPUT_DIR/{split}/{exercise}/{clip_id}.mp4
#
# Usage:
#   Local :  python countix_download.py
#   Colab :  !python countix_download.py
#
# Resume-safe: download_log.csv is updated after every clip.
# Re-run the script after a crash/timeout — completed clips are skipped.
# =============================================================================

import csv
import io
import os
import random
import re
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import cv2
import pandas as pd

# =============================================================================
# CONFIG  — edit these paths before running
# =============================================================================

# Project-native Countix root. In Colab, keep the repo under Drive and this
# resolves to .../CV_Image_pose_detection/Data/Countix.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COUNTIX_ROOT = PROJECT_ROOT / "Data" / "Countix"
VIDEO_DIR = str(COUNTIX_ROOT / "video")
ANNO_DIR = str(COUNTIX_ROOT / "annotation_cleaned")
LOG_PATH = str(COUNTIX_ROOT / "annotation_cleaned" / "download_log.csv")
FILTERED_CSV = str(COUNTIX_ROOT / "annotation_cleaned" / "countix_filtered.csv")
REPCOUNT_CSV = str(COUNTIX_ROOT / "annotation_cleaned" / "countix_repcount_format.csv")

ANNO_URL = "https://s3.amazonaws.com/kinetics/700_2020/annotations/countix.tar.gz"

# Countix class_label  →  your RepCount type column
LABEL_MAP = {
    "squats": "squat",
    "squat": "squat",
    "pull ups": "pull_up",
    "pullups": "pull_up",
    "pull up": "pull_up",
    "push ups": "push_up",
    "pushups": "push_up",
    "push up": "push_up",
    "sit ups": "sit_up",
    "situps": "sit_up",
    "sit up": "sit_up",
    "bench press": "bench_pressing",
    "bench pressing": "bench_pressing",
    "front raise": "front_raise",
    "front raises": "front_raise",
    "jumping jacks": "jump_jacks",
    "jumping jack": "jump_jacks",
    "jump jack": "jump_jacks",
    "jump jacks": "jump_jacks",
    "battle ropes": "battle_rope",
    "battle rope": "battle_rope",
    "pommel horse": "pommelhorse",
    "pommelhorse": "pommelhorse",
}

SPOT_CHECK_N = 10   # number of clips to verify with OpenCV in the final step


def normalize_label_key(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[_/-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def map_countix_label(value: str) -> tuple[str, str | None]:
    raw_label = value.strip().lower()
    mapped_label = LABEL_MAP.get(normalize_label_key(raw_label))
    return raw_label, mapped_label


def resolve_label_value(row: dict[str, str]) -> str:
    for key in ("class_label", "label", "class", "action", "category"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def resolve_video_id(row: dict[str, str]) -> str:
    for key in ("youtube_id", "video_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise KeyError("Countix row is missing a video identifier column.")


def resolve_time_range(row: dict[str, str]) -> tuple[float, float]:
    candidate_pairs = (
        ("repetition_start", "repetition_end"),
        ("time_start", "time_end"),
        ("kinetics_start", "kinetics_end"),
    )
    for start_key, end_key in candidate_pairs:
        start_value = row.get(start_key)
        end_value = row.get(end_key)
        if start_value is None or end_value is None:
            continue
        if not str(start_value).strip() or not str(end_value).strip():
            continue
        return float(start_value), float(end_value)
    raise KeyError("Countix row is missing a supported start/end time pair.")


def format_time_token(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def build_clip_id(youtube_id: str, time_start: float, time_end: float) -> str:
    return f"{youtube_id}_{format_time_token(time_start)}_{format_time_token(time_end)}"

# =============================================================================
# STEP 0 — make output dirs
# =============================================================================

def setup_dirs():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(ANNO_DIR,  exist_ok=True)
    print(f"Countix root: {COUNTIX_ROOT}")
    print(f"Video output: {VIDEO_DIR}")
    print(f"Log file    : {LOG_PATH}\n")


# =============================================================================
# STEP 1 — stream S3 annotations, filter in memory, save filtered CSV
# =============================================================================

def stream_and_filter() -> pd.DataFrame:
    print("Streaming annotations from S3 (few MB)...")
    with urllib.request.urlopen(ANNO_URL) as resp:
        raw_bytes = resp.read()
    print(f"Downloaded annotation bundle : {len(raw_bytes) / 1024:.1f} KB")

    rows = []
    dropped_labels: dict[str, int] = {}
    with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".csv"):
                continue

            split = (
                member.name
                .split("/")[-1]
                .replace("countix_", "")
                .replace(".csv", "")
            )

            f = tar.extractfile(member)
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))

            for row in reader:
                raw_label, mapped_label = map_countix_label(resolve_label_value(row))
                if mapped_label is None:
                    dropped_labels[raw_label] = dropped_labels.get(raw_label, 0) + 1
                    continue

                # handle both 'repetitions' and 'count' column names
                rep_val = row.get("repetitions") or row.get("count") or "0"
                youtube_id = resolve_video_id(row)
                time_start, time_end = resolve_time_range(row)

                rows.append({
                    "clip_id": build_clip_id(youtube_id, time_start, time_end),
                    "youtube_id": youtube_id,
                    "time_start": time_start,
                    "time_end":   time_end,
                    "count":      int(float(rep_val)),
                    "class_label": raw_label,
                    "type":        mapped_label,
                    "split":       split,
                })

    df = pd.DataFrame(
        rows,
        columns=[
            "clip_id",
            "youtube_id",
            "time_start",
            "time_end",
            "count",
            "class_label",
            "type",
            "split",
        ],
    )

    print(f"\nTotal clips matching exercise filter: {len(df)}")
    if len(df):
        print("\nPer-exercise / per-split breakdown:")
        print(df.groupby(["type", "split"]).size().unstack(fill_value=0).to_string())
        print("\nCount distribution (repetitions):")
        print(df["count"].describe().round(2))
    else:
        print("\nNo rows matched the current Countix label filter.")
    if dropped_labels:
        print("\nDropped out-of-scope / unmapped labels:")
        for label, count in sorted(dropped_labels.items(), key=lambda item: (-item[1], item[0]))[:25]:
            print(f"  {label or '<empty>'}: {count}")

    df.to_csv(FILTERED_CSV, index=False)
    print(f"\nFiltered annotations saved → {FILTERED_CSV}")
    return df


# =============================================================================
# STEP 2 — download clips (resume-safe)
# =============================================================================

def download_clips(df_filtered: pd.DataFrame):
    # Load resume log
    if os.path.exists(LOG_PATH):
        df_log   = pd.read_csv(LOG_PATH)
        if "clip_id" not in df_log.columns:
            print("\nExisting download log uses the old youtube_id-only format; starting a new clip-level log.")
            df_log = pd.DataFrame(columns=["clip_id", "youtube_id", "type", "split", "status", "path"])
            done_ids = set()
        else:
            done_ids = set(df_log["clip_id"].tolist())
            print(f"\nResuming — {len(done_ids)} clips already processed, skipping.")
    else:
        df_log   = pd.DataFrame(columns=["clip_id", "youtube_id", "type", "split", "status", "path"])
        done_ids = set()
        print("\nStarting fresh download.")
        df_log.to_csv(LOG_PATH, index=False)

    pending = df_filtered[~df_filtered["clip_id"].isin(done_ids)]
    print(f"Clips to download this session: {len(pending)} / {len(df_filtered)} total\n")

    new_rows = []

    for i, row in enumerate(pending.itertuples(), 1):
        clip_id = row.clip_id
        vid_id  = row.youtube_id
        label   = row.type
        split   = row.split
        t_start = row.time_start
        t_end   = row.time_end

        out_dir  = Path(VIDEO_DIR) / split / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{clip_id}.mp4"

        url = f"https://www.youtube.com/watch?v={vid_id}"

        cmd = [
            "yt-dlp", url,
            "--download-sections", f"*{t_start}-{t_end}",
            "--format", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "-o", str(out_path),
            "--quiet",
            "--no-warnings",
            "--no-playlist",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=90)
            status = "ok" if (result.returncode == 0 and out_path.exists()) else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"

        new_rows.append({
            "clip_id": clip_id,
            "youtube_id": vid_id,
            "type":       label,
            "split":      split,
            "status":     status,
            "path":       str(out_path) if status == "ok" else "",
        })

        # checkpoint after every clip
        df_log = pd.concat(
            [df_log, pd.DataFrame([new_rows[-1]])], ignore_index=True
        )
        df_log.to_csv(LOG_PATH, index=False)

        if i % 10 == 0 or i == len(pending):
            ok_n   = sum(r["status"] == "ok"     for r in new_rows)
            fail_n = sum(r["status"] != "ok"     for r in new_rows)
            print(f"  [{i:>4}/{len(pending)}]  ok={ok_n}  failed/timeout={fail_n}")

        time.sleep(0.3)

    print("\nDownload session complete.")


# =============================================================================
# STEP 3 — summary report
# =============================================================================

def print_summary():
    df_log      = pd.read_csv(LOG_PATH)
    df_filtered = pd.read_csv(FILTERED_CSV)

    ok     = df_log[df_log["status"] == "ok"]
    failed = df_log[df_log["status"] != "ok"]

    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"Total clips targeted     : {len(df_filtered)}")
    print(f"Successfully downloaded  : {len(ok)}")
    print(f"Failed / unavailable     : {len(failed)}")
    print(f"Availability rate        : {len(ok)/max(len(df_filtered),1)*100:.1f}%")

    print("\n--- OK clips per exercise / split ---")
    if len(ok):
        print(ok.groupby(["type", "split"]).size().unstack(fill_value=0).to_string())

    print("\n--- Failed clips per exercise ---")
    if len(failed):
        print(failed["type"].value_counts().to_string())
    else:
        print("None — all clips downloaded successfully.")

    total_bytes = sum(
        os.path.getsize(p)
        for p in ok["path"].dropna()
        if os.path.exists(p)
    )
    print(f"\nTotal disk usage: {total_bytes / 1e9:.2f} GB")


# =============================================================================
# STEP 4 — build RepCount-compatible annotation CSV
# =============================================================================

def build_repcount_csv():
    df_log      = pd.read_csv(LOG_PATH)
    df_filtered = pd.read_csv(FILTERED_CSV)

    ok_ids = set(df_log[df_log["status"] == "ok"]["clip_id"])
    df_ok  = df_filtered[df_filtered["clip_id"].isin(ok_ids)].copy()

    df_repcount = pd.DataFrame({
        "name":       df_ok["clip_id"] + ".mp4",
        "type":       df_ok["type"],
        "count":      df_ok["count"],
        "split":      df_ok["split"],
        "time_start": df_ok["time_start"],
        "time_end":   df_ok["time_end"],
        "source":     "countix",
    })

    df_repcount.to_csv(REPCOUNT_CSV, index=False)
    print(f"\nRepCount-format annotation saved → {REPCOUNT_CSV}")
    print(f"Rows: {len(df_repcount)}")
    print("\nPer-type counts:")
    print(df_repcount.groupby(["type", "split"]).size().unstack(fill_value=0).to_string())
    print("\nSample rows:")
    print(df_repcount.head(5).to_string(index=False))
    return df_repcount


# =============================================================================
# STEP 5 — OpenCV spot-check
# =============================================================================

def spot_check_clips():
    if not os.path.exists(LOG_PATH):
        print("\nNo download log found yet. Run the download step first.")
        return
    df_log   = pd.read_csv(LOG_PATH)
    ok_paths = df_log[df_log["status"] == "ok"]["path"].dropna().tolist()

    if not ok_paths:
        print("\nNo successfully downloaded clips available for spot-checking.")
        return

    sample = random.sample(ok_paths, min(SPOT_CHECK_N, len(ok_paths)))
    print(f"\nSpot-checking {len(sample)} clips...\n")

    results = []
    for path in sample:
        cap      = cv2.VideoCapture(path)
        readable = cap.isOpened()
        fps      = cap.get(cv2.CAP_PROP_FPS)             if readable else 0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)     if readable else 0
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  if readable else 0
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if readable else 0
        cap.release()

        status = "OK" if (readable and n_frames > 0) else "CORRUPT"
        fname  = Path(path).name
        results.append(status)
        print(f"  {status:7s}  {fname:35s}  {w}x{h}  {fps:.1f}fps  {int(n_frames)} frames")

    ok_n = results.count("OK")
    print(f"\n{ok_n}/{len(results)} clips readable.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Colab: mount Drive before running
    # from google.colab import drive; drive.mount('/content/drive')

    setup_dirs()

    # Step 1 — filter annotations (skip if already done)
    if os.path.exists(FILTERED_CSV):
        print(f"Filtered CSV already exists, loading → {FILTERED_CSV}")
        try:
            df_filtered = pd.read_csv(FILTERED_CSV)
        except pd.errors.EmptyDataError:
            print("Filtered CSV is empty; regenerating from Countix annotations.")
            df_filtered = stream_and_filter()
        else:
            if "clip_id" not in df_filtered.columns:
                print("Filtered CSV uses the old youtube_id-only format; regenerating with clip-level IDs.")
                df_filtered = stream_and_filter()
            elif len(df_filtered) == 0:
                print("Filtered CSV has zero rows; regenerating from Countix annotations.")
                df_filtered = stream_and_filter()
    else:
        df_filtered = stream_and_filter()

    # Step 2 — download clips
    download_clips(df_filtered)

    # Step 3 — summary
    print_summary()

    # Step 4 — RepCount-compatible CSV
    build_repcount_csv()

    # Step 5 — verify clips
    spot_check_clips()

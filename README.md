# Pose Detection Repetition Counting

This project explores exercise repetition counting from video using pose detection. The current codebase is centered on an offline pipeline for LLSP exercise videos: clean annotations, extract pose keypoints with YOLO, engineer squat-focused features, and train or evaluate rep-count models in notebooks.

## Current Scope

What is implemented now:

- cleaned training and validation labels under `Data/LLSP/annotation_cleaned`
- YOLO-based pose extraction to per-video `.npy` feature arrays
- pose feature index generation for all exercises or a single exercise such as `squat`
- squat video quality auditing utilities
- exploratory analysis and modeling notebooks under `artifacts`

What is not in this repo yet:

- a production webcam application
- real-time UI overlay
- a finalized multi-exercise rep counter package

## Repository Layout

```text
CV_Image_pose_detection/
├── Data/
│   └── LLSP/
│       ├── annotation/                 # original labels
│       ├── annotation_cleaned/         # cleaned labels and generated pose artifacts
│       ├── original_data/              # source references / download links
│       └── video/                      # train, valid, test videos
├── artifacts/
│   ├── 1_EDA/                          # dataset analysis notebooks and plots
│   ├── 2_Data_preparation/             # preparation notebooks
│   └── 3_Modeling/                     # pose extraction, feature extraction, modeling
├── resources/                          # project notes and study materials
└── requirements-pose.txt               # minimal dependencies for pose extraction
```

## Data Snapshot

The cleaned labels currently used by the pipeline are:

- `train_cleaned.csv`: 732 videos
- `valid_cleaned.csv`: 131 videos

Exercise classes present in the cleaned data include:

- `battle_rope`
- `bench_pressing`
- `front_raise`
- `jump_jacks`
- `pommelhorse`
- `pull_up`
- `push_up`
- `rowing_erg`
- `sit_up`
- `squat`

The largest train classes in the cleaned set are `squat` (102), `pull_up` (94), `bench_pressing` (93), and `sit_up` (93).

## Environment Setup

Create and activate a virtual environment, then install the pose dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r CV_Image_pose_detection/requirements-pose.txt
```

Current pose extraction dependencies:

- `numpy`
- `opencv-python`
- `ultralytics`

Optional tools used by the audit workflow:

- `ffmpeg`
- `ffprobe`

## Model Checkpoint

The pose extraction scripts expect a YOLO pose checkpoint at the repository root:

```text
yolo11n-pose.pt
```

This file is already present in the workspace.

## Main Pipeline

### 1. Build a Pose Feature Index

Generate an index for every cleaned sample:

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/build_pose_feature_index.py
```

Generate a squat-only index:

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/build_pose_feature_index.py \
  --exercise squat \
  --output-csv CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_feature_index_squat.csv
```

The generated CSV maps each video name to a target `.npy` output path and preserves the exercise label, split, and rep count.

### 2. Extract Pose Features with YOLO

Run extraction from an existing index:

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/pose_feature_extraction.py \
  --index-csv CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_feature_index.csv \
  --video-dir CV_Image_pose_detection/Data/LLSP/video \
  --model yolo11n-pose.pt
```

Useful debugging example:

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/pose_feature_extraction.py \
  --index-csv CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_feature_index_squat.csv \
  --video-dir CV_Image_pose_detection/Data/LLSP/video \
  --model yolo11n-pose.pt \
  --max-videos 5 \
  --overwrite
```

What the extractor does for each frame:

1. opens the video with OpenCV
2. runs YOLO pose inference
3. selects the primary person
4. stores 17 keypoints with `x`, `y`, and confidence
5. flattens each frame into a 51-value feature vector

Output format:

- one `.npy` file per video
- array shape: `[T, 51]`
- fallback shape when no pose is found: `[1, 51]` filled with zeros

Generated outputs are written under `Data/LLSP/annotation_cleaned/pose_features` together with:

- `pose_extraction_report.csv`
- `pose_extraction_summary.json`

### 3. Audit Squat Video Quality

The audit script joins summary statistics with local videos and tags common failure modes such as low confidence, poor lower-body visibility, or portrait framing.

It expects the squat feature summary generated in the squat feature extraction workflow, typically:

```text
CV_Image_pose_detection/Data/LLSP/annotation_cleaned/squat_feature_summary.csv
```

Example:

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/analyze_squat_video_quality.py \
  --summary-csv CV_Image_pose_detection/Data/LLSP/annotation_cleaned/squat_feature_summary.csv
```

Audit outputs are written to `artifacts/3_Modeling/squat_video_audit/`.

### 4. Continue in Notebooks

Most downstream experimentation currently lives in notebooks:

- `artifacts/1_EDA/1_EDA_34.ipynb`
- `artifacts/2_Data_preparation/2_Data_Preparation_01.ipynb`
- `artifacts/3_Modeling/3_Model_Training_01.ipynb`
- `artifacts/3_Modeling/4_Squat_Pose_Extraction_Colab.ipynb`
- `artifacts/3_Modeling/5_Squat_Feature_Extraction_Colab.ipynb`
- `artifacts/3_Modeling/6_Squat_Rep_Counting_Colab.ipynb`

## Current Results in the Workspace

The checked-in pose extraction summary shows a small successful run:

- `total_rows`: 20
- `ok`: 20
- `failed`: 0
- `max_videos`: 20

See `Data/LLSP/annotation_cleaned/pose_extraction_summary.json` for the exact run metadata.

## Notes and Caveats

- The repository contains large local assets including videos, a YOLO checkpoint, and intermediate artifacts.
- The workflow is currently notebook-first for modeling and analysis.
- The project documentation in `artifacts/specification.md` describes a broader future direction called `RepCoach`, but the implemented code in this repo is narrower and focused on offline experimentation.

## Useful Files

- `artifacts/specification.md`: target product and system design
- `artifacts/repcount_analysis.md`: dataset notes
- `artifacts/3_Modeling/YOLO_PIPELINE.md`: pose extraction runbook
- `artifacts/3_Modeling/COLAB_SQUAT_POSE.md`: Colab workflow for squat extraction

## Next Steps

Reasonable next improvements for this project are:

- move notebook logic into reusable Python modules
- add a documented training and evaluation script for rep counting
- formalize metrics for per-exercise mean absolute error
- add a webcam inference demo once the offline counter is stable

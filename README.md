# Pose Detection Repetition Counting

This project explores exercise repetition counting from video using pose detection. The current codebase is centered on an offline pipeline for LLSP exercise videos: clean annotations, extract pose keypoints with YOLO, engineer squat-focused features, and train or evaluate rep-count models in notebooks.

In this README, `LLSP` refers to the local exercise-video dataset folder used by the project. In the training context discussed here, this is related to Long Length Partials (`LLP`), also called lengthened partials: a strength-training technique where an exercise is performed only in the most stretched portion of the muscle's range of motion.

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

## Folder Guide

### `artifacts/1_EDA`

This folder contains the exploratory data analysis work used to understand the RepCount / LLSP data before building the pipeline.

Main contents:

- `1_EDA_34.ipynb`: primary EDA notebook
- class distribution plots such as `class_imbalance_train_valid.png`
- repetition and duration plots such as `count_distribution.png` and `cycle_duration.png`
- per-exercise inspection PDFs such as `squat_inspection.pdf`, `push_up_inspection.pdf`, and `pull_up_inspection.pdf`

Purpose:

- inspect the dataset visually
- understand class imbalance
- examine repetition count distributions
- identify data quality issues before modeling

### `artifacts/2_Data_preparation`

This folder contains the notebook used to clean labels and prepare the dataset contract used by later steps.

Main contents:

- `2_Data_Preparation_01.ipynb`: label cleaning, split checks, and preparation workflow

Purpose:

- clean and standardize the annotations
- verify train / validation splits
- produce the cleaned CSVs used by modeling:
  - `Data/LLSP/annotation_cleaned/train_cleaned.csv`
  - `Data/LLSP/annotation_cleaned/valid_cleaned.csv`

### `artifacts/3_Modeling`

This folder contains the executable modeling pipeline and the Colab notebooks used for squat pose extraction, feature engineering, and rep counting.

Main contents:

- `build_pose_feature_index.py`: build `pose_feature_index.csv` or `pose_feature_index_squat.csv`
- `pose_feature_extraction.py`: run YOLO pose extraction and write raw pose `.npy` arrays
- `analyze_squat_video_quality.py`: audit squat feature outputs and tag likely failure modes
- `3_Model_Training_01.ipynb`: baseline temporal training notebook from extracted pose features
- `4_Squat_Pose_Extraction_Colab.ipynb`: Colab stage for squat pose extraction
- `5_Squat_Feature_Extraction_Colab.ipynb`: Colab stage for engineered squat features
- `6_Squat_Rep_Counting_Colab.ipynb`: Colab stage for FSM-based rep counting and evaluation
- `YOLO_PIPELINE.md`, `YOLO_POSE_STAGE.md`, `COLAB_SQUAT_POSE.md`: runbooks and stage documentation

Purpose:

- move from cleaned labels to pose features
- convert raw pose into squat-specific engineered features
- run counting and evaluate the squat baseline
- support alternative training experiments from extracted features

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

## Main Pipeline and Run Order

The project has one main squat-focused path and one optional experimental branch.

### Main Squat Pipeline

Run these in order:

1. `artifacts/1_EDA/1_EDA_34.ipynb`
   Use this first if you want to understand the dataset and class distributions before building features.

2. `artifacts/2_Data_preparation/2_Data_Preparation_01.ipynb`
   Produces the cleaned annotations used by the later stages.

3. `artifacts/3_Modeling/build_pose_feature_index.py`
   Build the squat-only index from the cleaned annotations.

4. `artifacts/3_Modeling/4_Squat_Pose_Extraction_Colab.ipynb`
   Reads videos and writes raw pose arrays in `pose_features/`.

5. `artifacts/3_Modeling/5_Squat_Feature_Extraction_Colab.ipynb`
   Reads `pose_features/` and writes engineered squat features in `squat_features/`, plus:
   - `squat_feature_index.csv`
   - `squat_feature_summary.csv`

6. `artifacts/3_Modeling/6_Squat_Rep_Counting_Colab.ipynb`
   Reads the engineered squat features and produces rep-count predictions and evaluation metrics such as `MAE`, `RMSE`, and `Within-1`.

7. `artifacts/3_Modeling/analyze_squat_video_quality.py`
   Optional audit step after feature extraction when you want to inspect difficult squat videos or diagnose pose/feature quality issues.

### Optional Local Script Path

If you do not want to use Colab for pose extraction, the local script path is:

1. `build_pose_feature_index.py`
2. `pose_feature_extraction.py`
3. downstream Colab or notebook stages for squat features and rep counting

### Optional Experimental Branch

`artifacts/3_Modeling/3_Model_Training_01.ipynb` is a separate experimental branch for training a temporal regressor from extracted pose features. It is not the main squat FSM pipeline and should be treated as an alternative modeling path.

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

## Current Metrics

The project does not yet have a checked-in final rep-count evaluation report, but these metrics are currently available in the workspace:

### Pose Extraction Status

From `Data/LLSP/annotation_cleaned/pose_extraction_summary.json`:

- processed rows: `20`
- successful extractions: `20`
- failed extractions: `0`
- zero-pose outputs: `0`
- run cap used for that check: `20` videos

### Squat Video Quality Audit

From `artifacts/3_Modeling/squat_video_audit/squat_video_audit_summary.json`:

- audited squat videos: `118`
- severity breakdown:
  - `ok`: `90`
  - `review`: `15`
  - `medium`: `11`
  - `high`: `1`
  - `critical`: `1`
- low-confidence counts:
  - mean confidence `< 0.25`: `1`
  - mean confidence `< 0.40`: `2`
  - mean confidence `< 0.50`: `4`
  - mean confidence `< 0.70`: `21`
- lower-body validity counts:
  - valid ratio `< 0.25`: `2`
  - valid ratio `< 0.50`: `2`
  - valid ratio `< 0.75`: `5`
  - valid ratio `< 0.90`: `12`

### Training Alignment Readiness

From `artifacts/3_Modeling/training_outputs/baseline_v2_rebuilt/feature_alignment_report.json`:

- train rows in cleaned labels: `732`
- valid rows in cleaned labels: `131`
- train rows aligned to current feature files: `20`
- valid rows aligned to current feature files: `0`

### Rep Counting Evaluation

The notebooks define these reporting metrics:

- `MAE`
- `RMSE`
- `Within-1 accuracy`

However, there is no checked-in artifact yet with final numeric rep-count results for those metrics. The current notebook contains the evaluation logic, but the saved notebook output does not include persisted values.

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

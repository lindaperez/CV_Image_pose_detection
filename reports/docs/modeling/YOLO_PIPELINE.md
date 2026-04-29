# YOLO Pipeline

This file describes the current YOLO-based pose extraction pipeline in this project.

## Goal

Convert LLSP exercise videos into per-video pose sequences that can be used for:

- pose post-processing
- squat feature engineering
- rep counting
- later exercise modeling

## Current Stage

The current YOLO stage does:

```text
Video -> YOLO pose inference -> main person selection -> keypoint flattening -> .npy pose features
```

It does not yet do:

- pose smoothing
- normalization
- angle computation
- rep counting

## Main Files

- [pose_feature_extraction.py](/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection/artifacts/3_Modeling/pose_feature_extraction.py)
  Runs YOLO pose on each video and writes `.npy` pose features.

- [build_pose_feature_index.py](/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection/artifacts/3_Modeling/build_pose_feature_index.py)
  Builds index CSV files such as `pose_feature_index.csv` and `pose_feature_index_squat.csv`.

- [YOLO_POSE_STAGE.md](/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection/artifacts/3_Modeling/YOLO_POSE_STAGE.md)
  Runbook for the extraction stage.

- [Squat_Pose_Extraction_Colab.ipynb](/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection/artifacts/3_Modeling/Squat_Pose_Extraction_Colab.ipynb)
  Colab notebook for squat-only GPU extraction.

## Inputs

### 1. Video data

Expected video root:

```text
CV_Image_pose_detection/Data/LLSP/video
```

The extractor scans this directory recursively for `.mp4` files.

### 2. Cleaned annotations

Expected cleaned annotation files:

```text
CV_Image_pose_detection/Data/LLSP/annotation_cleaned/train_cleaned.csv
CV_Image_pose_detection/Data/LLSP/annotation_cleaned/valid_cleaned.csv
```

These are used to build the pose index files.

### 3. YOLO pose model

Expected checkpoint:

```text
yolo11n-pose.pt
```

## Index Files

The extractor can run from an index CSV.

Example:

```csv
name,feature_path,type,split,count
stu4_66.mp4,/.../pose_features/stu4_66.npy,squat,train,27.0
```

Why the index matters:

- defines which videos to process
- defines where each output file should be saved
- lets us create subset runs like squat-only extraction

Important index files in this repo:

- `pose_feature_index.csv`
- `pose_feature_index_squat.csv`

## Extraction Logic

For each video:

1. Open video with OpenCV
2. Read frames sequentially
3. Run YOLO pose on each frame
4. Choose one subject using the largest bounding box
5. Extract 17 keypoints
6. Save `x`, `y`, and `confidence` for each keypoint
7. Flatten frame pose into a 51-dimensional vector
8. Stack all valid pose frames into a `[T, 51]` array
9. Save the array as `.npy`

Feature shape:

```text
17 keypoints * 3 values = 51 features per frame
```

Output shape:

```text
[num_pose_frames, 51]
```

If no pose is found in a video, the extractor writes:

```text
[1, 51]
```

filled with zeros so downstream code can still load the sample.

## Outputs

Main output directory:

```text
CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_features
```

Main output artifacts:

- per-video `.npy` pose files
- `pose_extraction_report.csv`
- `pose_extraction_summary.json`

## Standard Workflow

### A. Build a squat-only index

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/build_pose_feature_index.py \
  --exercise squat \
  --output-csv CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_feature_index_squat.csv
```

### B. Run pose extraction

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/pose_feature_extraction.py \
  --index-csv CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_feature_index_squat.csv \
  --video-dir CV_Image_pose_detection/Data/LLSP/video
```

### C. Colab version with videos in Drive

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/build_pose_feature_index.py \
  --exercise squat \
  --feature-dir /content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_features \
  --output-csv /content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_feature_index_squat.csv

python3 /content/CV_Image_pose_detection/artifacts/3_Modeling/pose_feature_extraction.py \
  --index-csv /content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_feature_index_squat.csv \
  --video-dir /content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/Data/LLSP/video \
  --model /content/CV_Image_pose_detection/artifacts/3_Modeling/yolo11n-pose.pt \
  --report-path /content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_extraction_report.csv \
  --summary-path /content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_extraction_summary.json \
  --device cuda:0
```

## Current Squat Flow

Current squat pipeline:

```text
train_cleaned.csv + valid_cleaned.csv
    ->
build_pose_feature_index.py --exercise squat
    ->
pose_feature_index_squat.csv
    ->
pose_feature_extraction.py
    ->
pose_features/*.npy
    ->
next stage: pose post-processing and squat feature extraction
```

## Next Stage After YOLO

After raw pose extraction, the next stage should be:

```text
raw pose .npy
    ->
confidence filtering
    ->
smoothing
    ->
normalization
    ->
knee / hip / depth features
    ->
squat rep-counting logic
```

## Common Failure Modes

### Missing video errors

Cause:
- the index references a video that is not present under `--video-dir`

Symptom:

```text
FAILED (missing video): <video_name>
```

### Index mismatch

Cause:
- `pose_feature_index_squat.csv` was built from an older cleaned dataset snapshot

Fix:
- regenerate the index from the current cleaned CSVs

### Mixed storage locations

Cause:
- code and annotations are in one location, videos are in another

Fix:
- pass every path explicitly with:
  - `--index-csv`
  - `--video-dir`
  - `--model`
  - `--report-path`
  - `--summary-path`

## Current Definition of Done for This Stage

The YOLO stage is complete when:

- the correct index file is generated
- the videos are reachable from `--video-dir`
- pose extraction finishes without missing-video failures
- `.npy` pose files are created for the intended subset
- report and summary files are written successfully

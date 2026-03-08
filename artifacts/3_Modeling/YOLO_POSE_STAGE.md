# YOLO Pose Extraction Stage

This stage converts each LLSP video into a temporal pose feature array using a YOLO pose checkpoint.

## Inputs

- Videos under `CV_Image_pose_detection/Data/LLSP/video`
- YOLO pose checkpoint at `yolo11n-pose.pt`
- Optional existing index CSV at `CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_feature_index.csv`

## Outputs

- Per-video `.npy` pose feature files in `CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_features`
- Extraction report CSV
- Extraction summary JSON
- Optional generated index CSV

## Install

```bash
python3 -m pip install -r CV_Image_pose_detection/requirements-pose.txt
```

## Run with Existing Index

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/pose_feature_extraction.py \
  --index-csv CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_feature_index.csv \
  --video-dir CV_Image_pose_detection/Data/LLSP/video \
  --model yolo11n-pose.pt
```

## Run by Discovering Videos

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/pose_feature_extraction.py \
  --discover-from-videos \
  --video-dir CV_Image_pose_detection/Data/LLSP/video \
  --feature-dir CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_features \
  --write-index-csv CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_feature_index.csv
```

## Useful Debug Flags

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/pose_feature_extraction.py \
  --discover-from-videos \
  --max-videos 5 \
  --overwrite \
  --conf 0.25 \
  --imgsz 640
```

## Feature Format

Each frame is flattened into 51 values:

- 17 keypoints
- For each keypoint: `x`, `y`, `confidence`

Saved array shape:

```text
[num_pose_frames, 51]
```

If YOLO finds no pose in a video, the stage writes a zero array with shape `[1, 51]` so downstream steps can still load the sample.

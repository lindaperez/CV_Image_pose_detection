# Google Colab: Squat-Only Pose Extraction

Use this when you want a small end-to-end run on only the squat videos with a Colab GPU.

## 1. Runtime

In Colab, set:

- `Runtime` -> `Change runtime type`
- `Hardware accelerator` -> `GPU`

## 2. Get the Project into Colab

If the repo is on GitHub:

```bash
!git clone <YOUR_REPO_URL>
%cd personal-git
```

If the repo is already in Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/<YOUR_PATH>/personal-git
```

For the commands below, assume:

```python
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection"
DRIVE_VIDEO_DIR = f"{DRIVE_PROJECT_ROOT}/Data/LLSP/video"
DRIVE_ANNOTATION_DIR = f"{DRIVE_PROJECT_ROOT}/Data/LLSP/annotation_cleaned"
DRIVE_POSE_FEATURE_DIR = f"{DRIVE_ANNOTATION_DIR}/pose_features"
```

## 3. Install Dependencies

```bash
!python3 -m pip install -r CV_Image_pose_detection/requirements-pose.txt
```

## 4. Confirm GPU

```python
import torch
print("cuda_available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())
print("device_name =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
```

## 5. Build a Squat-Only Index

```bash
!python3 CV_Image_pose_detection/artifacts/3_Modeling/build_pose_feature_index.py \
  --exercise squat \
  --feature-dir $DRIVE_POSE_FEATURE_DIR \
  --output-csv $DRIVE_ANNOTATION_DIR/pose_feature_index_squat.csv
```

## 6. Smoke Test on 5 Squat Videos

```bash
!python3 CV_Image_pose_detection/artifacts/3_Modeling/pose_feature_extraction.py \
  --index-csv $DRIVE_ANNOTATION_DIR/pose_feature_index_squat.csv \
  --video-dir $DRIVE_VIDEO_DIR \
  --model /content/CV_Image_pose_detection/artifacts/3_Modeling/yolo11n-pose.pt \
  --report-path $DRIVE_ANNOTATION_DIR/pose_extraction_report.csv \
  --summary-path $DRIVE_ANNOTATION_DIR/pose_extraction_summary.json \
  --device cuda:0 \
  --max-videos 5 \
  --overwrite
```

## 7. Run the Full Squat Set

```bash
!python3 CV_Image_pose_detection/artifacts/3_Modeling/pose_feature_extraction.py \
  --index-csv $DRIVE_ANNOTATION_DIR/pose_feature_index_squat.csv \
  --video-dir $DRIVE_VIDEO_DIR \
  --model /content/CV_Image_pose_detection/artifacts/3_Modeling/yolo11n-pose.pt \
  --report-path $DRIVE_ANNOTATION_DIR/pose_extraction_report.csv \
  --summary-path $DRIVE_ANNOTATION_DIR/pose_extraction_summary.json \
  --device cuda:0
```

## 8. Inspect Results

```python
import pandas as pd

report = pd.read_csv(f"{DRIVE_ANNOTATION_DIR}/pose_extraction_report.csv")
report.head()
```

```python
import json
from pathlib import Path

summary = json.loads(Path(f"{DRIVE_ANNOTATION_DIR}/pose_extraction_summary.json").read_text())
summary
```

## Notes

- Start with `--max-videos 5` before launching the full squat set.
- Build the index with `--feature-dir $DRIVE_POSE_FEATURE_DIR` so the `.npy` pose files are written to Drive, not temporary `/content`.
- If Colab disconnects, rerun the same command without `--overwrite` to skip completed files.
- If your repo path already contains `yolo11n-pose.pt`, the extractor will find it automatically.

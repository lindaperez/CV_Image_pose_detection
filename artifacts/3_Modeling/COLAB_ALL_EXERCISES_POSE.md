# Colab All-Exercises Pose Extraction

This runbook is the widening step after the squat-only prototype.

It assumes:

- the full multi-exercise pose index exists at  
  `Data/LLSP/annotation_cleaned/pose_feature_index.csv`
- the missing-only worklist exists at  
  `Data/LLSP/annotation_cleaned/pose_feature_index_remaining.csv`
- `others` has been excluded from the remaining-worklist because it is a
  heterogeneous catch-all category rather than a coherent exercise class

## Goal

Extract YOLO pose features only for the videos that do not already have a
corresponding `.npy` artifact under `Data/LLSP/annotation_cleaned/pose_features`.

## Inputs

- videos under `Data/LLSP/video/{train,valid,test}`
- missing-only index: `pose_feature_index_remaining.csv`

## Outputs

- additional `.npy` pose feature files under `Data/LLSP/annotation_cleaned/pose_features`
- extraction report CSV
- extraction summary JSON

## Recommended Colab command

```bash
python /content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/artifacts/3_Modeling/pose_feature_extraction.py \
  --index-csv /content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_feature_index_remaining.csv \
  --video-dir /content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/Data/LLSP/video \
  --report-path /content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_extraction_report_remaining.csv \
  --summary-path /content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/Data/LLSP/annotation_cleaned/pose_extraction_summary_remaining.json \
  --device cuda
```

## After extraction

Rebuild the remaining-worklist to confirm coverage:

```bash
python /content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/artifacts/3_Modeling/build_remaining_pose_worklist.py \
  --exclude-exercise others
```

If the widening step completed correctly, the updated
`pose_feature_index_remaining.csv` should shrink substantially or reach zero for
the exercises you have decided to include.

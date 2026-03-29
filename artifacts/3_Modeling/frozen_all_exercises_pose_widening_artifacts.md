# Frozen All-Exercises Pose Widening Artifacts

This manifest records the artifacts that must be preserved for the widened
Stage 4 and Stage 5 pipeline.

- Frozen result record:
  - `artifacts/3_Modeling/frozen_all_exercises_pose_widening_result.json`

## Stage 4: Remaining-Exercise Pose Extraction

The widening pose-extraction run was completed in:

- `artifacts/3_Modeling/4_All_Exercises_Pose_Extraction_Colab.ipynb`

Required output artifacts:

- `Data/LLSP/annotation_cleaned/pose_feature_index.csv`
- `Data/LLSP/annotation_cleaned/pose_feature_index_remaining.csv`
- `Data/LLSP/annotation_cleaned/pose_feature_remaining_summary.csv`
- `Data/LLSP/annotation_cleaned/pose_extraction_report_remaining.csv`
- `Data/LLSP/annotation_cleaned/pose_extraction_summary_remaining.json`
- `Data/LLSP/annotation_cleaned/pose_features/`

## Stage 5: Generic Pose-Sequence Preparation

The generic all-exercises sequence-preparation stage is defined by:

- `artifacts/3_Modeling/build_pose_sequence_dataset.py`
- `artifacts/3_Modeling/5_All_Exercises_Pose_Sequence_Preparation_Colab.ipynb`

Required output artifacts:

- `Data/LLSP/annotation_cleaned/pose_sequence_index.csv`
- `Data/LLSP/annotation_cleaned/pose_sequence_summary.csv`
- `Data/LLSP/annotation_cleaned/pose_sequences/`

## Drive / Colab Location

If these stages were produced in Colab, the expected Drive root is:

`/content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/`

The outputs above should therefore exist under:

- `/content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/Data/LLSP/annotation_cleaned/`

## Preservation Goal

Preserving these artifacts freezes the widened pose-first data pipeline before
Stage 6 changes the downstream counting setup. This keeps the all-exercises
pose extraction and generic sequence-preparation results reproducible and fixed
as the reference point for later multi-exercise counting experiments.

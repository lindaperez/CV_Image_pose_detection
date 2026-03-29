# Frozen Shared Stage 6 Baseline Artifacts

This manifest records the artifacts that should be preserved for the first
shared multi-exercise counting baseline on normalized pose sequences.

## Frozen Baseline Identity

- Status: official current shared baseline
- Task: multi-exercise repetition counting from normalized pose sequences
- Notebook: `artifacts/3_Modeling/6_All_Exercises_Counting_Baseline_Colab.ipynb`
- Trainer: `artifacts/3_Modeling/train_pose_count_tcn.py`
- Comparison script: `artifacts/3_Modeling/compare_count_run_to_baseline.py`
- Frozen summary: `artifacts/3_Modeling/frozen_all_exercises_counting_baseline_result.json`

## Upstream Data Contract To Preserve

These files define the Stage 5 full sequence dataset used by the frozen Stage 6 baseline:

- `Data/LLSP/annotation_cleaned/pose_feature_index.csv`
- `Data/LLSP/annotation_cleaned/pose_sequence_index.csv`
- `Data/LLSP/annotation_cleaned/pose_sequence_summary.csv`
- `Data/LLSP/annotation_cleaned/pose_sequences/`

Note:
- The full Stage 5 rebuild attempted `1041` rows and wrote `1003` `ok` sequence rows.
- The remaining `38` failed rows correspond to the heterogeneous `others` bucket and are not written into `pose_sequence_index.csv`.

## Per-Exercise Run Folders To Preserve

Each of these run folders should keep:

- `config.json`
- `history.csv`
- `predictions.csv`
- `metrics_summary.json`
- `baseline_comparison_summary.json`
- `baseline_comparison_rows.csv`
- `best_model.pt`
- `feature_mean.npy`
- `feature_std.npy`

Expected run folders:

- `artifacts/3_Modeling/training_outputs/pose_count_tcn_battle_rope/`
- `artifacts/3_Modeling/training_outputs/pose_count_tcn_bench_pressing/`
- `artifacts/3_Modeling/training_outputs/pose_count_tcn_front_raise/`
- `artifacts/3_Modeling/training_outputs/pose_count_tcn_jump_jacks/`
- `artifacts/3_Modeling/training_outputs/pose_count_tcn_pommelhorse/`
- `artifacts/3_Modeling/training_outputs/pose_count_tcn_pull_up/`
- `artifacts/3_Modeling/training_outputs/pose_count_tcn_push_up/`
- `artifacts/3_Modeling/training_outputs/pose_count_tcn_sit_up/`
- `artifacts/3_Modeling/training_outputs/pose_count_tcn_squat/`

## Drive Path Reminder

In Colab / Drive, the expected root is:

- `/content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/`

So the full run artifacts should live under:

- `/content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/artifacts/3_Modeling/training_outputs/`

## Follow-Up Path

This frozen baseline is the reference point for:

- exercise-by-exercise improvement
- first tuning axis: sequence length
- recommended starting exercises:
  - `pull_up`
  - `bench_pressing`
  - `pommelhorse`
  - `push_up`
  - `squat`

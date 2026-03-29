# Frozen Best Squat Artifacts

This manifest records the artifacts that must be preserved for the current frozen squat baseline:

- Run name: `squat_tcn_l1_channels96`
- Frozen result record:
  - `artifacts/3_Modeling/frozen_best_squat_result.json`

## Required Run Artifacts

These files should be preserved under the run directory:

`artifacts/3_Modeling/training_outputs/squat_tcn_l1_channels96/`

- `config.json`
- `metrics_summary.json`
- `history.csv`
- `predictions.csv`
- `best_model.pt`
- `feature_mean.npy`
- `feature_std.npy`

## Required Post-Review Artifacts

These files should also be preserved for the same run after applying the validation-review policy:

- `policy_filtered_metrics_summary.json`
- `policy_filtered_valid_predictions.csv`

## Review Policy References

These supporting artifacts define the filtered evaluation policy:

- `artifacts/3_Modeling/validation_failure_review.csv`
- `artifacts/3_Modeling/validation_failure_review.md`
- `artifacts/3_Modeling/apply_validation_review_policy.py`

## Drive / Colab Location

If the run was produced in Colab, the expected Drive location is:

`/content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/artifacts/3_Modeling/training_outputs/squat_tcn_l1_channels96/`

## Preservation Goal

Preserving these artifacts ensures that the current squat baseline remains reproducible, reviewable, and fixed as the official reference point for later multi-exercise work.

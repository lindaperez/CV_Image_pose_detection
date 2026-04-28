# Final Report Reproducibility Addendum

This text is intended to be pasted into the final report, ideally after Section 3.4 or as a short appendix.

## Reproducibility Checklist

The complete codebase is available at:

```text
https://github.khoury.northeastern.edu/khouryquanxing/CS5330_SP26_Group1/tree/main/CV_Image_pose_detection-main
```

All commands below assume the current working directory is the project subfolder, i.e. `CV_Image_pose_detection-main`, the folder containing `README.md`.

### Environment Setup

```bash
git clone https://github.khoury.northeastern.edu/khouryquanxing/CS5330_SP26_Group1.git
cd CS5330_SP26_Group1/CV_Image_pose_detection-main
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-pose.txt
```

The dependency file includes the Python packages used by the runnable scripts:

```text
numpy, opencv-python, pandas, pillow, torch, torchvision, ultralytics
```

### Basic Runnable Check

To verify the repository after setup:

```bash
python tests/run_tests.py all
```

The static project pages can be served locally with:

```bash
python3 -m http.server 8000
```

and opened at:

```text
http://localhost:8000/index.html
```

### Runtime Inference Check

The offline squat counter can be run from an input feature sequence:

```bash
python3 artifacts/3_Modeling/run_squat_counter.py \
  --feature-path Data/LLSP/annotation_cleaned/squat_features/train3946_squat_features.npy \
  --counter-backend tcn \
  --pretty
```

Running from raw video uses YOLO pose extraction first:

```bash
python3 artifacts/3_Modeling/run_squat_counter.py \
  --video-path Data/LLSP/video/valid/train3946.mp4 \
  --output-json artifacts/3_Modeling/training_outputs/train3946_squat_runtime.json \
  --pretty
```

These inference examples require the local LLSP video/features and the saved squat TCN checkpoint files documented in the README. If these files are not included directly in the Git repository, the README provides the dataset link and artifact paths needed to place or regenerate them.

### Experiment Reproduction Path

The reportable experiments were produced through the staged notebooks and scripts documented in `README.md`:

1. Exploratory data analysis: `artifacts/1_EDA/1_EDA_34.ipynb`
2. Data preparation: `artifacts/2_Data_preparation/2_Data_Preparation_01.ipynb`
3. Pose feature indexing: `artifacts/3_Modeling/build_pose_feature_index.py`
4. Pose extraction: `artifacts/3_Modeling/pose_feature_extraction.py` and `4_All_Exercises_Pose_Extraction_Colab.ipynb`
5. Pose sequence preparation: `artifacts/3_Modeling/build_pose_sequence_dataset.py` and `5_All_Exercises_Pose_Sequence_Preparation_Colab.ipynb`
6. Pose TCN baselines: `artifacts/3_Modeling/train_pose_count_tcn.py` and `6_All_Exercises_Counting_Baseline_Colab.ipynb`
7. RGB baselines: `artifacts/3_Modeling/extract_rgb_frame_features.py`, `train_rgb_count_tcn.py`, and `7_RGB_Counting_Baseline_Colab.ipynb`
8. Routed predictions: `artifacts/3_Modeling/build_routed_count_predictions.py`
9. Confidence intervals: `artifacts/3_Modeling/bootstrap_count_confidence_intervals.py`

The main result artifacts are summarized in:

```text
artifacts/3_Modeling/ARCHITECTURE_RESULTS_MATRIX.md
artifacts/3_Modeling/EXPERIMENT_SHOWCASE.md
artifacts/3_Modeling/architecture_results_dashboard.html
```

## Small Consistency Fixes for the Report

Replace the routing sentence in Section 2.5 with:

> After evaluating the branches, we adopt a deterministic exercise-dependent routing strategy based on validation evidence rather than a single universal rule. For squats, the dedicated 16-D pose-feature TCN is selected because it has the strongest MAE and matches the tuned FSM on Within-1. For pull-ups, we select the shared pose TCN because it has the best practical Within-1 tradeoff, even though a stronger RGB branch has slightly lower MAE. For push-ups, we select the RGB branch because it is the strongest practical branch among the tested options.

Replace the dependency sentence in Section 3.3 with:

> Core Python dependencies are listed in `requirements-pose.txt`: numpy, opencv-python, pandas, pillow, torch, torchvision, and ultralytics.

Replace the sentence “All results include 95% bootstrap confidence intervals” with:

> The final reportable branches include 95% bootstrap confidence intervals with 5,000 resamples; exploratory baselines are reported as point estimates.

Fix the pull-up RGB row in Table 4:

```text
pull-up | Stronger RGB TCN | ResNet50 | 4.20 | 0.36 | —
```

or, if using the original ResNet18 branch:

```text
pull-up | RGB TCN | ResNet18 | 4.87 | 0.14 | —
```

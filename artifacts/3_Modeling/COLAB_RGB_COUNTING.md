# RGB Counting Guide

## Purpose

This guide explains the RGB branch introduced in Stage 7 of the project. The goal is not to replace the pose pipeline blindly. The RGB branch exists to answer a specific research question:

> Does a visual representation built from raw video frames outperform the current pose-only counting setup on selected exercises?

The RGB branch was added after the following pose-first results:

- the shared pose TCN baseline worked as a real baseline, but absolute error remained high
- sequence-length tuning (`6B`) helped only modestly
- manual keypoint weighting (`6C`) was mostly negative
- pseudo-density counting (`6D`) was also mostly negative

At that point, the next defensible step was to test a stronger visual representation.

Stage `7B` is the direct follow-up to this baseline. It keeps the same counting task and controlled exercise subset, but upgrades the frozen RGB backbone so the project can test whether the remaining weakness is RGB representation strength rather than RGB as a modality.

## What This Stage Is Testing

The RGB stage is a controlled comparison, not a full pipeline replacement.

It tests:

- `squat`
- `pull_up`
- `push_up`

These three exercises were chosen intentionally:

- `squat`: strong control case from the pose branch
- `pull_up`: one of the more promising pose-only classes
- `push_up`: unresolved hard case where extra visual context may matter

This keeps the experiment small enough to interpret.

## High-Level Pipeline

The RGB branch has two main steps:

1. extract frozen RGB feature sequences from raw videos
2. train a counting-only TCN on those RGB feature sequences

The flow is:

```text
raw video (.mp4)
  ->
uniform frame sampling
  ->
frozen ResNet18 frame encoder
  ->
RGB feature sequence (.npy)
  ->
RGB TCN count regressor
  ->
count prediction and evaluation
```

This mirrors the pose branch structurally, but changes the representation:

- pose branch: normalized pose sequences
- RGB branch: frozen visual frame features

## Files Involved

Main Stage 7 files:

- `artifacts/3_Modeling/7_RGB_Counting_Baseline_Colab.ipynb`
- `artifacts/3_Modeling/7C_Representation_Fit_Analysis_Colab.ipynb`
- `artifacts/3_Modeling/7B_Stronger_RGB_Backbone_Colab.ipynb`
- `artifacts/3_Modeling/7D_Hard_Case_Data_Audit_Colab.ipynb`
- `artifacts/3_Modeling/extract_rgb_frame_features.py`
- `artifacts/3_Modeling/train_rgb_count_tcn.py`
- `artifacts/3_Modeling/analyze_representation_fit.py`
- `artifacts/3_Modeling/audit_counting_hard_cases.py`
- `artifacts/3_Modeling/compare_count_run_to_baseline.py`

Relevant upstream files:

- `Data/LLSP/annotation_cleaned/pose_feature_index.csv`
- `Data/LLSP/video/`
- `Data/LLSP/annotation_cleaned/pose_sequence_index.csv`

Relevant pose baselines for comparison:

- `pose_count_tcn_squat_seq256`
- `pose_count_tcn_pull_up_seq192`
- `pose_count_tcn_push_up_seq128`

## Why The RGB Stage Still Uses `pose_feature_index.csv`

The RGB extractor needs metadata:

- video name
- exercise type
- split
- repetition count

That metadata already exists in `pose_feature_index.csv`, so the RGB branch reuses it as the stage input index.

Important clarification:

- RGB extraction does **not** use the pose arrays themselves
- it only uses the index as metadata to locate and label the videos

## Environment Assumptions

The intended environment is Colab with Drive mounted.

Drive project root:

```text
/content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection
```

Main data locations:

- project root: `/content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection`
- videos: `/content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/Data/LLSP/video`
- annotation dir: `/content/drive/MyDrive/FinalProjectCV/CV_Image_pose_detection/Data/LLSP/annotation_cleaned`

## Stage 7 Notebook Structure

### 1. Environment Setup

The notebook defines:

- repo paths
- data paths
- Stage 7 script paths

It also synchronizes selected files from `/content/CV_Image_pose_detection` into Drive.

Important implementation detail:

- the notebook now avoids blindly overwriting Drive copies
- it compares timestamps first
- this prevents stale `/content` files from silently replacing newer Drive scripts

This matters because a stale RGB extractor can remove progress logging and make debugging much harder.

### 2. Controlled Exercise Subset

The notebook limits the experiment to:

- `squat`
- `pull_up`
- `push_up`

It also reuses the best `seq_len` settings found in `6B`:

- `squat = 256`
- `pull_up = 192`
- `push_up = 128`

This keeps the RGB comparison fair.

### 3. RGB Feature Extraction

This stage runs:

```bash
python -u .../extract_rgb_frame_features.py ...
```

with arguments similar to:

- `--index-csv pose_feature_index.csv`
- `--video-dir Data/LLSP/video`
- `--feature-dir Data/LLSP/annotation_cleaned/rgb_resnet18_features`
- `--output-index-csv rgb_feature_index_selected.csv`
- `--output-summary-csv rgb_feature_summary_selected.csv`
- `--max-frames 256`
- `--batch-size 32`
- `--device cuda`
- `--overwrite`
- `--log-every 1`
- `--save-progress-every 5`
- repeated `--exercise` filters

#### What the extractor does

For each selected video:

1. read metadata from `pose_feature_index.csv`
2. locate the matching `.mp4`
3. sample up to `max_frames` frames uniformly across the video
4. apply a frozen ResNet18 image encoder to each sampled frame
5. stack the frame embeddings into one sequence array
6. save the sequence as:

```text
rgb_resnet18_features/<video_stem>_rgb_resnet18.npy
```

#### Why ResNet18 is frozen

The first RGB stage is meant to isolate representation, not full end-to-end video learning.

Using a frozen backbone:

- makes the experiment simpler
- makes it cheaper than training a full video model
- keeps the comparison focused on whether raw visual features carry more useful counting information than pose alone

### 4. RGB Feature Quality Review

After extraction, the notebook reads:

- `rgb_feature_summary_selected.csv`

This is used to check:

- how many rows succeeded
- how many failed
- mean `frames_total`
- mean `frames_used`
- feature dimension consistency

If many rows fail here, the problem is usually:

- video decoding
- Drive I/O
- corrupted or missing videos

not the RGB TCN itself.

### 5. RGB TCN Training

This stage runs:

- `train_rgb_count_tcn.py`

once per selected exercise.

The current Stage 7 notebook runs:

- `rgb_count_tcn_squat_seq256`
- `rgb_count_tcn_pull_up_seq192`
- `rgb_count_tcn_push_up_seq128`

with the same general TCN settings used in the pose branch:

- `epochs = 80`
- `batch_size = 16`
- `channels = 96`
- `kernel_size = 3`
- `num_blocks = 4`
- `dropout = 0.2`
- `loss = l1`
- `selection_metric = mae`

This design choice is deliberate:

- it keeps the architecture family fixed
- it changes the representation
- it makes pose-vs-RGB comparison easier to interpret

### 6. Pose vs RGB Metric Review

The notebook then compares:

- RGB run metrics
- best pose `6B` run metrics

using:

- `valid_mae`
- `valid_rmse`
- `valid_within_1`

This answers the first main research question:

> Does RGB beat the best pose baseline on the same exercise?

### 7. RGB Baseline Comparison Review

The notebook also compares RGB against the trivial train-split baseline using:

- `compare_count_run_to_baseline.py`

This matters because a model can look interesting while still not beating a naive baseline.

The baseline review asks:

- does RGB beat the trivial baseline on MAE?
- does RGB improve Within-1?
- how does RGB compare against pose on the same exercise?

## Output Artifacts

### RGB Extraction Outputs

Written under:

- `Data/LLSP/annotation_cleaned/rgb_resnet18_features/`
- `Data/LLSP/annotation_cleaned/rgb_feature_index_selected.csv`
- `Data/LLSP/annotation_cleaned/rgb_feature_summary_selected.csv`

#### `rgb_feature_index_selected.csv`

This is the training-ready RGB index. It contains only rows that succeeded in extraction.

Each row points to:

- video metadata
- split
- exercise
- count
- RGB feature path

#### `rgb_feature_summary_selected.csv`

This is the extraction audit file. It includes:

- success/failure status
- frame counts
- messages for debugging

### RGB Training Outputs

Written under:

```text
artifacts/3_Modeling/training_outputs/<run_name>/
```

Each run writes:

- `history.csv`
- `predictions.csv`
- `metrics_summary.json`
- `best_model.pt`

These mirror the artifact contract of the pose trainers.

## Runtime Expectations

RGB extraction is usually cheaper than full pose extraction, but it can still take significant time.

Why:

- video decoding is expensive
- reading `.mp4` files from Drive is slow
- uniform sampling still requires scanning the videos

Important nuance:

- the RGB backbone is not usually the main bottleneck
- video I/O and decoding are often the real bottlenecks

Recent implementation improvements already added:

- sequential frame reading instead of repeated random seeks
- progress logging
- periodic CSV writes during extraction
- unbuffered Python launch in the notebook (`python -u`)

## Common Failure Modes And Fixes

### 1. No progress logs appear

Possible causes:

- stale script in Drive
- notebook copied an older `/content` file into Drive
- child Python process is buffered

Current fixes already in place:

- the notebook uses `python -u`
- the extractor logs each processed row
- the sync cell avoids blindly overwriting newer Drive copies

What to check:

- the sync/setup cell should report that the Drive extractor contains logging flags
- the extraction cell should print:

```text
Starting RGB extraction for ...
```

followed by per-video progress lines.

### 2. Extraction feels too slow

Most likely cause:

- Drive-backed video decoding

Practical levers:

- reduce `--max-frames`
- keep the exercise subset small
- avoid rerunning extraction unnecessarily

### 3. Many rows fail in extraction

Likely causes:

- missing videos
- unreadable videos
- path mismatches
- frame decoding problems

Check:

- `rgb_feature_summary_selected.csv`

### 4. RGB training runs but does not beat pose

Interpretation:

- raw RGB frame features may not yet provide enough advantage
- the current TCN may still be too weak
- the next step may need a stronger video representation or RGB+pose fusion

This is still a valid result. It means the project learned something real about representation choice.

## How To Interpret The RGB Experiment

### If RGB wins only on `push_up`

This suggests:

- pose is already adequate for the cleaner exercises
- RGB mainly helps when context and appearance matter more

### If RGB wins broadly

This suggests:

- the research should move toward RGB-first or multimodal modeling
- pose-only is likely too restrictive for the broader counting task

### If RGB fails to beat pose

This suggests:

- the problem may not be representation alone
- the current counting architecture may still be the main bottleneck
- more powerful temporal modeling or a different video backbone may be needed

## Current Research Position

The RGB branch is the next comparison stage, not the final answer.

At this point the research sequence is:

1. validate the pose-first baseline
2. widen the pose pipeline
3. test cheap pose-side improvements
4. observe that those improvements are not enough
5. compare pose against RGB on a controlled subset

This is the right progression because it avoids switching representations without evidence.

## Recommended Usage Pattern

Use Stage 7 as:

- a controlled representation comparison
- a subset experiment
- a decision point for whether the next branch should be:
  - stronger RGB
  - RGB+pose multimodal
  - or a different counting architecture

Do **not** use Stage 7 yet as:

- proof that RGB is better for every exercise
- a full replacement for the pose pipeline
- a production design

## Minimal Stage 7 Checklist

Before running:

- Drive notebook is up to date
- Drive scripts are up to date
- `pose_feature_index.csv` exists
- videos exist in `Data/LLSP/video`

During extraction:

- confirm live progress appears
- confirm `.npy` RGB features are being written
- confirm summary CSV is updating

After extraction:

- inspect `rgb_feature_summary_selected.csv`
- verify selected exercises succeeded

After training:

- inspect `metrics_summary.json`
- compare RGB to the pose `6B` runs, and for `squat` also compare against the frozen dedicated squat baseline `squat_tcn_l1_channels96`
- compare RGB to the trivial baseline

## Short Summary

Stage 7 applies a controlled RGB counting experiment on `squat`, `pull_up`, and `push_up` by extracting frozen ResNet18 frame-feature sequences from raw videos, training exercise-specific TCN count regressors on those RGB features, and comparing the resulting metrics against the best shared pose `6B` baselines, the frozen dedicated squat baseline for `squat`, and the trivial train-split count baseline. The value of this stage is not only in whether RGB wins, but in whether the project now has enough evidence to move beyond a pose-only representation.

## Stage 7E Multimodal Follow-Up

Stage `7E` is the first pose+RGB fusion follow-up after the Stage `7`, `7B`, `7C`, and `7D` findings. It keeps the exercise subset narrow and uses a simple late-fusion TCN so the project can test whether pose and RGB combine productively before committing to a larger multimodal or video-native architecture.

The multimodal stage reuses the strongest RGB source per exercise rather than forcing one backbone everywhere:

- `squat`: `ResNet50` RGB reference from `7B`
- `pull_up`: `ResNet50` RGB reference from `7B`
- `push_up`: `ResNet18` RGB reference from Stage `7`

Use Stage `7E` as:

- the first direct test of whether simple pose+RGB fusion can beat both single-modality baselines
- a controlled bridge from the current exercise-dependent representation evidence to any larger multimodal architecture

If Stage `7E` still fails to justify a shared fusion model, the next practical step is not another generic architecture tweak. The next practical step is `8_Exercise_Dependent_Counting_Colab.ipynb`, which routes each supported exercise to the strongest current branch so the project can count some videos reliably even while the broader multi-exercise research question remains unresolved.

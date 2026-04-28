# Exercise Repetition Counting From Video
## Final Implementation Document

This document summarizes the final implementation, architecture, model branches, evaluation results, and limitations of the computer vision project. It is written as the cleaned final version of the implementation record. The earlier working file, `implementation_document_draft.md`, is intentionally preserved and should remain available as the historical development log.

---

## Document Status

- Document type: final implementation document
- Project area: computer vision, human pose estimation, temporal repetition counting
- Primary task: estimate exercise repetition counts from video
- Final system interpretation: exercise-dependent routed counting system
- Runtime implementation scope: squat-focused prototype
- Broader research scope: offline comparison across squat, pull-up, and push-up branches

The project should not be described as a finished production multi-exercise app. It is best described as an end-to-end research and engineering pipeline that validates an exercise-dependent counting architecture, with a packaged squat runtime prototype and offline routed evaluation for the supported exercise branches.

---

## Project Team

- Linda Perez Penaranda
- Kunyi Shi
- Peihan Wang
- Quanxing Lu

---

## 1. Problem And Motivation

Exercise repetition counting from video is a practical computer vision problem with applications in fitness coaching, rehabilitation tracking, remote training, and workout analytics. The goal is to estimate how many repetitions a person completes in a video without requiring wearable sensors or manual counting.

The task is difficult because exercise videos are not controlled sensor streams. Camera viewpoint, scale, framing, lighting, occlusion, body size, and exercise style all affect the visual signal. A squat is mostly described by lower-body vertical motion, while a push-up is horizontal and can be harder for pose estimation. Pull-ups introduce additional ambiguity from camera viewpoint, body-bar interaction, and target selection.

The project therefore investigates a system-level question:

> Which representation and temporal model are most defensible for counting repetitions for each exercise?

The final conclusion is that the strongest implementation is not one universal model. The best result is an exercise-dependent routed architecture:

- `squat` is best supported by dedicated engineered pose features and a TCN.
- `pull_up` is best served by the shared normalized pose TCN when prioritizing exact-count reliability.
- `push_up` is best supported by RGB appearance features from a frozen ResNet18 backbone and a TCN.

---

## 2. Final System Architecture

The implemented project contains a staged pipeline. The system starts with raw exercise videos and cleaned annotations, extracts visual representations, trains multiple temporal counting models, compares their results, and selects an exercise-specific route.

```text
Input Videos + Raw Annotations
            |
            v
Exploratory Data Analysis
            |
            v
Data Cleaning / Relabeling
            |
            v
Cleaned Annotation Contracts
            |
            +----------------------------+
            |                            |
            v                            v
YOLO11n-pose Extraction              RGB Frame Sampling
            |                            |
            v                            v
Raw Pose Arrays [T, 51]              Frozen CNN Features
            |                            |
     +------+----------+                 |
     |                 |                 |
     v                 v                 v
Generic Pose       Squat-Specific     RGB Feature
Sequences          Pose Features      Sequences
     |                 |                 |
     v                 v                 v
Shared Pose TCN    FSM / Squat TCN    RGB TCN
Transformer        Dedicated Branch   ResNet18 / ResNet50
Ablations
     |                 |                 |
     +-----------------+-----------------+
                       |
                       v
         Architecture Comparison + Hard-Case Review
                       |
                       v
          Exercise-Dependent Routed Counting
                       |
                       v
              Predictions + Evaluation
```

### Final Routed Branches

| Exercise | Final selected branch | Representation | Temporal model | Reason selected |
|---|---|---|---|---|
| `squat` | `squat_tcn_l1_channels96` | engineered squat pose features | TCN regressor | best squat MAE and RMSE; strongest single project result |
| `pull_up` | `pose_count_tcn_pull_up_seq192` | normalized pose sequence | shared pose TCN | best practical `Within-1` result among compared branches |
| `push_up` | `rgb_count_tcn_push_up_seq128` | frozen ResNet18 RGB features | RGB TCN | strongest practical push-up branch; pose-only was weak |

The routing assumes that the exercise label is known at inference time. A production-ready exercise-recognition layer is not yet implemented.

---

## 3. Dataset Preparation

### Dataset

The project uses a RepCount / LLSP-style exercise video dataset.

Confirmed Part-A totals:

- `758` train videos
- `131` validation videos
- `152` test videos
- `1041` total videos

### EDA Findings

The exploratory analysis identified issues that needed to be resolved before modeling:

- inconsistent exercise labels and spelling variants;
- a heterogeneous `others` class;
- class imbalance across exercises;
- right-skewed repetition counts;
- sparse temporal annotation columns;
- one row with missing count in the train/validation analysis surface;
- a need for leakage checks between train and validation.

### Cleaning And Relabeling

The data-preparation stage normalized typo variants and label names, including examples such as:

- `squant` -> `squat`
- `frontraise` -> `front_raise`
- `benchpressing` -> `bench_pressing`
- `jumpjacks` / `jump_jack` -> `jump_jacks`
- `situp` -> `sit_up`
- `pullups` -> `pull_up`
- `pushups` -> `push_up`

The `others` class was manually reviewed. It was not a coherent class and contained multiple movement groups, including rowing-related motions, soccer juggling / ball control, and one squat sample. The project relabeled `11` train videos to `rowing_erg` and removed `26` ambiguous or unusable train videos.

### Data Contracts

The project created stable data contracts for downstream modeling:

- cleaned train and validation annotations;
- decision manifest for relabel/remove rules;
- pose feature index files;
- pose sequence index files;
- squat feature index files;
- RGB feature index files;
- saved CSV, JSON, and NPY artifacts for reproducibility.

The test split was intentionally not used for iterative cleaning or model selection.

---

## 4. Pose Extraction And Normalization

### Pose Extraction

The pose branch uses Ultralytics `YOLO11n-pose`. For each frame, the extractor produces `17` COCO-format body keypoints. Each keypoint contains:

- `x` coordinate
- `y` coordinate
- confidence score

This gives a raw pose array of shape `[T, 51]` per video, where `T` is the number of frames and `51 = 17 keypoints x 3 values`.

Implementation details:

- model: `yolo11n-pose.pt`
- confidence threshold: `0.25`
- image size: `640`
- IoU-based temporal tracking with search expansion: `1.6`
- tracker reset after `8` consecutive missed frames
- if no person is tracked, the extractor writes a fallback zero pose array so downstream stages have a stable file contract

### Generic Pose Normalization

The generic pose sequence branch normalizes raw pose arrays before training shared pose models:

1. Mask keypoints with confidence below `0.25`.
2. Forward-fill and backward-fill missing values over time.
3. Apply exponential moving average smoothing with `alpha = 0.2`.
4. Normalize coordinates relative to the torso using shoulders and hips.
5. Resample temporal sequences to fixed model lengths.

The normalized pose branch preserves the `51`-dimensional sequence contract.

### Squat-Specific Pose Features

The squat branch computes engineered biomechanical features from pose, including:

- left and right knee angles;
- average knee angle;
- knee flexion;
- hip angle;
- hip, knee, and ankle vertical positions;
- hip drop;
- leg extension;
- hip velocity;
- validity and confidence support metrics.

The generated feature table contains `16` columns including `frame_idx`. The selected trained squat TCN drops `frame_idx`, so the model input dimension is `15`.

---

## 5. Model Branches Implemented

### 5.1 Squat FSM Baseline

The first squat counting backend was an interpretable finite-state machine over engineered squat features. It tracks state transitions such as:

- standing / up;
- descending;
- bottom position;
- ascending;
- return to up.

The FSM counts a repetition when the movement completes a valid transition from descending through bottom and back to the up state. It remains useful as a transparent baseline, but it was surpassed by the learned squat TCN on average error.

### 5.2 Dedicated Squat TCN

The strongest squat branch is a Temporal Convolutional Network trained on squat-specific engineered pose features.

Final selected run:

- run name: `squat_tcn_l1_channels96`
- input dimension: `15`
- sequence length: `128`
- channels: `96`
- kernel size: `3`
- number of blocks: `4`
- dropout: `0.2`
- loss: L1
- optimizer: AdamW
- learning rate: `0.001`
- weight decay: `0.0001`
- early stopping patience: `15`
- best epoch: `53`

The implementation uses residual temporal convolution blocks, adaptive average pooling, and a regression head. The implementation is not a causal TCN with layer normalization; it uses standard padded 1D convolutions, ReLU, dropout, residual connections, adaptive pooling, and linear layers.

### 5.3 Shared Pose TCN

The shared pose branch trains TCN regressors on normalized `[T, 51]` pose sequences. This branch acts as the main pose-only baseline for multiple exercises.

Selected routed pull-up branch:

- run name: `pose_count_tcn_pull_up_seq192`
- exercise: `pull_up`
- input dimension: `51`
- sequence length: `192`
- balanced count sampling: enabled
- temporal stretching: `+/-12%`
- feature noise: `0.02`
- frame dropout: `0.03`

The shared pose branch did not become a universal winner, but it was the best practical route for pull-up when prioritizing `Within-1` accuracy.

### 5.4 RGB TCN Branch

The RGB branch was added to test whether visual appearance can recover signal that pose misses.

Implementation:

- sample frames from each video;
- feed frames through a frozen CNN backbone;
- train a TCN on the per-frame feature sequence.

Backbones tested:

- frozen ResNet18, producing `512`-dimensional features;
- frozen ResNet50, producing `2048`-dimensional features.

Selected routed push-up branch:

- run name: `rgb_count_tcn_push_up_seq128`
- backbone: ResNet18
- input dimension: `512`
- sequence length: `128`
- balanced count sampling: enabled
- temporal stretching: `+/-12%`
- feature noise: `0.02`
- frame dropout: `0.03`

The ResNet18 RGB branch became the selected practical branch for push-up. ResNet50 was useful as a comparison, especially for pull-up MAE, but it was not the final push-up route.

### 5.5 Pose Transformer Branch

The project tested a pose Transformer over the same normalized pose sequence contract used by the shared pose TCN.

The Transformer branch includes:

- linear input projection;
- learned positional embedding;
- PyTorch Transformer encoder layers;
- GELU activation;
- mean pooling;
- regression head.

The Transformer did not replace the selected routed branches. It improved some pose-only push-up MAE relative to shared pose TCN, but it remained behind the RGB branch and did not provide a robust practical gain.

### 5.6 Keypoint Weighting, Density, And Multimodal Fusion

Additional ablations included:

- keypoint-weighted pose TCN;
- weak pseudo-density pose TCN;
- multimodal late fusion of pose and RGB features.

These were negative or mixed results. Multimodal late fusion lowered MAE in some cases, but it degraded `Within-1` enough that it did not justify the added complexity.

---

## 6. Evaluation Metrics

The project reports:

- `MAE`: mean absolute repetition-count error; lower is better.
- `RMSE`: root mean squared repetition-count error; lower is better.
- `Within-1`: fraction of videos where the predicted count is within one repetition of the true count; higher is better.

The final reportable branches also include `95%` bootstrap confidence intervals.

Metric interpretation matters. Some branches had lower `MAE` but weaker `Within-1`. Therefore, final routing should not be described as simply choosing the lowest-MAE model for every exercise.

---

## 7. Final Results

### Reportable Routed Results

| Exercise | Selected branch | n | MAE | RMSE | Within-1 |
|---|---|---:|---:|---:|---:|
| `squat` | dedicated squat pose TCN | 16 | 2.1405 | 3.1016 | 0.5625 |
| `pull_up` | shared pose TCN | 14 | 4.6088 | 7.0169 | 0.4286 |
| `push_up` | RGB ResNet18 TCN | 18 | 6.6018 | 10.2865 | 0.2778 |

### Bootstrap Confidence Intervals

| Exercise | MAE 95% CI | RMSE 95% CI | Within-1 95% CI |
|---|---:|---:|---:|
| `squat` | `[1.1266, 3.3313]` | `[1.6982, 4.2837]` | `[0.3125, 0.8125]` |
| `pull_up` | `[2.0863, 7.5386]` | `[3.5909, 9.7687]` | `[0.2143, 0.7143]` |
| `push_up` | `[3.3063, 10.4238]` | `[5.1748, 14.8974]` | `[0.0556, 0.5000]` |

### Core Architecture Comparison

| Exercise | Architecture | MAE | Within-1 | Interpretation |
|---|---|---:|---:|---|
| `squat` | FSM tuned baseline | 3.0625 | 0.5625 | interpretable historical baseline |
| `squat` | dedicated squat TCN | 2.1405 | 0.5625 | best current squat solution |
| `squat` | shared pose TCN | 8.0430 | 0.2500 | generic pose underfits squat |
| `squat` | RGB ResNet18 TCN | 6.5446 | 0.0625 | not competitive with dedicated pose |
| `pull_up` | shared pose TCN | 4.6088 | 0.4286 | best `Within-1` route |
| `pull_up` | RGB ResNet50 TCN | 4.1992 | 0.3571 | lower MAE but not best `Within-1` |
| `pull_up` | dedicated pull-up pose | 3.5463 | 0.2857 | best MAE but weaker exact-count reliability |
| `push_up` | shared pose TCN | 8.8724 | 0.0000 | pose-only weak |
| `push_up` | RGB ResNet18 TCN | 6.6018 | 0.2778 | selected practical branch |
| `push_up` | multimodal late fusion | 6.1691 | 0.1111 | lower MAE but weaker `Within-1` |

### Results Interpretation

The strongest outcome is not a single universal model. The strongest outcome is a representation map:

- Squat is best handled by a dedicated pose branch.
- Push-up is best handled by an RGB branch.
- Pull-up remains mixed, with shared pose selected for exact-count reliability.
- Transformer and simple multimodal fusion were useful ablations but not final winners.

The confidence intervals are wide because the validation subsets are small. The results support scoped research conclusions, not production-level claims.

---

## 8. Hard-Case And Failure Analysis

The project included manual hard-case review to understand why models fail.

Confirmed failure categories included:

- pose failure;
- camera viewpoint;
- repetition ambiguity;
- model failure;
- target selection;
- label mismatch;
- visibility issues;
- camera motion;
- side-view framing;
- pose jitter.

Exercise-specific patterns:

- `squat`: errors were often related to repetition-boundary ambiguity, pose failure, or label mismatch.
- `pull_up`: errors were often related to viewpoint and target-selection ambiguity.
- `push_up`: errors were more often related to pose failure and weak pose representation, supporting the RGB branch.

This analysis reinforces the main architecture decision: the residual failure modes differ by exercise, so a routed system is more defensible than a forced generic counter.

---

## 9. Demo And Runtime Implementation

The packaged runtime path is intentionally squat-focused.

Runtime flow:

```text
Input squat video
        |
        v
YOLO11n-pose extraction
        |
        v
Squat-specific feature computation
        |
        +------------------+
        |                  |
        v                  v
FSM baseline           Squat TCN
        |                  |
        +--------+---------+
                 |
                 v
          Predicted count
```

The demo is useful because it shows the project running from video through pose extraction, feature computation, and repetition-count prediction. The runtime does not yet package a production multi-exercise router. The broader routed system is validated offline through saved artifacts and notebooks.

Presentation assets are stored separately under:

- `artifacts/5_presentation`

Result visualizations are stored under:

- `artifacts/4_results`

---

## 10. Main Artifacts

Key implementation and result artifacts include:

- `artifacts/implementation_document_draft.md`
- `artifacts/implementation_document_final.md`
- `artifacts/final_report_methodology_replacement.md`
- `artifacts/problem_statement_cvpr.md`
- `artifacts/3_Modeling/ARCHITECTURE_RESULTS_MATRIX.md`
- `artifacts/3_Modeling/EXPERIMENT_SHOWCASE.md`
- `artifacts/3_Modeling/architecture_results_long.csv`
- `artifacts/3_Modeling/frozen_best_squat_result.json`
- `artifacts/3_Modeling/pose_feature_extraction.py`
- `artifacts/3_Modeling/build_pose_sequence_dataset.py`
- `artifacts/3_Modeling/run_squat_counter.py`
- `artifacts/3_Modeling/train_pose_count_tcn.py`
- `artifacts/3_Modeling/train_squat_tcn.py`
- `artifacts/3_Modeling/train_rgb_count_tcn.py`
- `artifacts/3_Modeling/train_pose_count_transformer.py`
- `artifacts/3_Modeling/extract_rgb_frame_features.py`
- `artifacts/4_results/figure_1_routed_performance_ci.svg`
- `artifacts/4_results/figure_2_architecture_mae_heatmap.svg`
- `artifacts/4_results/figure_3_mae_within1_tradeoff.svg`
- `artifacts/4_results/figure_4_per_exercise_mae_comparison.svg`
- `artifacts/4_results/figure_5_routed_architecture.svg`
- `artifacts/5_presentation/exercise_counting_hiring_manager_presentation.pptx`

---

## 11. Tools And Technologies

Core tools:

- Python
- NumPy
- Pandas
- OpenCV
- Ultralytics YOLO
- PyTorch
- Torchvision
- Google Colab
- CSV, JSON, and NPY artifact contracts

Reasons for tool selection:

- YOLO11n-pose provides a practical pretrained pose frontend.
- Python and NumPy/Pandas support reproducible artifact processing.
- PyTorch supports custom TCN, Transformer, and RGB-feature training loops.
- Colab made GPU-heavy extraction and training practical.
- Saved intermediate artifacts made the pipeline inspectable and debuggable.

---

## 12. Limitations

The current implementation has important limitations:

- The live/runtime path is squat-only.
- The routed multi-exercise system is validated offline, not packaged as a single production inference endpoint.
- The system assumes the exercise label is known at inference time.
- There is no production-ready exercise-recognition layer yet.
- Validation subsets are small:
  - `n = 16` for squat
  - `n = 14` for pull-up
  - `n = 18` for push-up
- Confidence intervals are wide.
- Most training workflows remain notebook-first because GPU-heavy stages were executed in Colab.
- Large videos and model/data artifacts are not all committed directly to the repository.
- The results should not be presented as state-of-the-art benchmark claims.

---

## 13. Future Work

Recommended next steps:

1. Package the routed multi-exercise counter behind a single inference entry point.
2. Add an exercise-recognition stage so the system no longer requires the exercise label as input.
3. Add target-person selection and tracking for scenes with multiple people.
4. Improve push-up and pull-up robustness with additional exercise-specific analysis.
5. Expand confidence-interval reporting to any additional final branches.
6. Reduce notebook dependency by centralizing experiment execution.
7. Evaluate on a held-out test surface only after the routed pipeline is frozen.
8. Consider stronger pose-native models such as ST-GCN, BlockGCN, or SkeleTR after the current routed baseline is stable.
9. Consider direct video-counting methods such as RepNet, TransRAC, ESCounts, or CountLLM only if the project intentionally shifts from an interpretable pose/RGB branch design to a video-native counting architecture.

The recommended long-term architecture is:

```text
video
  -> target person selection / tracking
  -> exercise recognition
  -> branch selection
  -> repetition counting
```

This follows the main lesson of the project: counting should happen after the system identifies the relevant person and exercise context.

---

## 14. Final Conclusion

This project implemented an end-to-end computer vision pipeline for exercise repetition counting. It moved from raw videos and noisy annotations through EDA, cleaning, pose extraction, feature engineering, temporal modeling, RGB comparison, Transformer and fusion ablations, hard-case review, routed evaluation, result visualization, and presentation artifacts.

The final engineering conclusion is that repetition counting is exercise-dependent. A dedicated pose branch is strongest for squat, RGB appearance features are most useful for push-up, and pull-up remains a tradeoff where shared pose provides the best exact-count reliability. The routed architecture is therefore the most defensible current design.

The project contributes a reproducible, inspectable implementation pipeline and a clear architecture decision supported by experiments. It does not claim a new state-of-the-art counting model or a production-ready multi-exercise product. Its value is in the evidence-driven system design and the practical foundation it creates for future exercise-aware repetition counting.

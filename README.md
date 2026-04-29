# ML System: Exercise Repetition Counting

We investigate whether pose-based or appearance-based visual representations
produce more reliable repetition counts from exercise video. We build an end-to-end
pipeline spanning pose extraction, feature engineering, and temporal modeling, and
conduct a structured comparison across squat, pull-up, and push-up exercises on
the RepCount / LLSP Part-A dataset.

Our analysis reveals that the RepCount validation split contains 14–18 videos per
exercise — below the minimum threshold for reliable model comparison (see Data
Viability Assessment). We therefore present pipeline outputs as a documented basis
for future investigation, not as research conclusions. The primary contribution of
this work is the pipeline, the viability framework, and the failure-mode hypotheses
it generates.

**Contributions:**
- End-to-end pipeline for comparing pose-based and appearance-based representations
  for repetition counting, covering feature extraction, temporal modeling, and evaluation
- Data viability framework with explicit minimum criteria for dataset quality before
  model comparison is meaningful
- Structured failure-mode analysis on RepCount generating hypotheses for future work
- Squat runtime prototype demonstrating the full pipeline from raw video to predicted count

**Scope:** the live runtime is squat-only. Multi-exercise routing is validated offline.
The exercise label is supplied at inference time; exercise recognition is not implemented.

---

## Pipeline Outputs

The following tables record the pipeline's outputs on the RepCount validation split.
Per-exercise validation sets contain 14–18 videos, below the minimum threshold for
reliable model comparison (see Data Viability Assessment in the project specification).
These are records of what the system produced under these conditions, not research
conclusions about which representation or architecture is superior.

### Selected Branch Per Exercise

| Exercise | Branch | Representation | Model | MAE | RMSE | Within-1 |
|---|---|---|---|---:|---:|---:|
| `squat` | dedicated squat TCN | engineered pose features (15-dim) | TCN | 2.1405 | 3.1016 | 0.5625 |
| `pull_up` | shared pose TCN | normalized pose sequence (51-dim) | TCN | 4.6088 | 7.0169 | 0.4286 |
| `push_up` | RGB ResNet18 TCN | frozen CNN features (512-dim) | TCN | 6.6018 | 10.2865 | 0.2778 |

### Confidence Intervals (95%)

The wide intervals reflect the small validation set sizes and confirm that the
per-exercise outputs cannot be used to rank architectures reliably.

| Exercise | MAE CI | RMSE CI | Within-1 CI |
|---|---|---|---|
| `squat` | `[1.13, 3.33]` | `[1.70, 4.28]` | `[0.31, 0.81]` |
| `pull_up` | `[2.09, 7.54]` | `[3.59, 9.77]` | `[0.21, 0.71]` |
| `push_up` | `[3.31, 10.42]` | `[5.17, 14.90]` | `[0.06, 0.50]` |

### Architecture Comparison

| Exercise | Architecture | MAE | Within-1 | Notes |
|---|---|---:|---:|---|
| `squat` | dedicated squat TCN | 2.1405 | 0.5625 | lowest recorded MAE on this dataset |
| `squat` | FSM baseline | 3.0625 | 0.5625 | interpretable reference |
| `squat` | shared pose TCN | 8.0430 | 0.2500 | higher error with generic representation |
| `pull_up` | shared pose TCN | 4.6088 | 0.4286 | highest recorded Within-1 on this dataset |
| `pull_up` | RGB ResNet50 TCN | 4.1992 | 0.3571 | lower MAE, lower Within-1 |
| `push_up` | RGB ResNet18 TCN | 6.6018 | 0.2778 | lowest recorded error on this dataset |
| `push_up` | multimodal late fusion | 6.1691 | 0.1111 | lower MAE, lower Within-1 |
| `push_up` | shared pose TCN | 8.8724 | 0.0000 | Within-1 = 0 on this dataset |

---

## System Architecture

```text
Input Videos + Cleaned Annotations
            |
            v
Exploratory Data Analysis + Cleaning
            |
            +-----------------------------+
            |                             |
            v                             v
YOLO11n-pose Extraction              RGB Frame Sampling
[T, 51] → Generic Pose [T,51]        Frozen CNN features [T,512]
          → Squat Features [T,15]
            |                             |
            +-------------+---------------+
                          |
                          v
              Exploratory TCN / FSM Runs
              (behavioral observations — Section 7 of PROJECT_SPEC)
                          |
                          v
                  Hard-Case Review
                          |
                          v
          Failure-Mode Hypothesis Generation
                          |
                          v
              Runtime Prototype (squat)
```

### Pose Extraction Details

- Model: `yolo11n-pose.pt`, 17 COCO keypoints per frame → `[T, 51]` arrays
- Confidence threshold: 0.25; IoU-based temporal tracking with search expansion 1.6
- Tracker reset after 8 consecutive missed frames; zero-array fallback for untracked subjects
- Normalization: confidence masking → forward/backward fill → EMA smoothing (α=0.2)
  → torso-relative coordinate normalization

### Squat Feature Engineering

16-column engineered feature table including knee angles (left/right/avg), knee
flexion, hip angle, vertical joint positions, hip drop, leg extension, hip velocity,
validity and confidence metrics. Model input dimension is 15 (frame_idx dropped).

---

## Dataset

RepCount / LLSP Part-A exercise video dataset:

- 758 train videos, 131 validation videos, 152 test videos (1041 total)
- Exercises: squat, pull-up, push-up (plus others, removed or relabeled)
- Annotations: per-video rep count + per-rep start/end frame timestamps
- Test split not yet used for final evaluation (reserved for frozen pipeline)

Data cleaning applied: label normalization (typos, variants), `others` class review,
11 train videos relabeled to `rowing_erg`, 26 ambiguous videos removed.

---

## Scope and Limitations

- Live runtime is squat-only; the routed multi-exercise system is offline only
- Exercise label is assumed known at inference time — no recognition layer exists
- Validation subsets are small; confidence intervals are wide
- Most GPU-heavy training stages were executed in Colab notebooks
- Test split is reserved and has not been used for any iterative decision
- Results should not be presented as state-of-the-art benchmark claims

---

## Repository Layout

```text
ML_System/
├── src/rep_counter/          # runnable Python source
│   ├── data/                 # dataset manifest and Countix helpers
│   ├── features/             # pose/RGB feature extraction and indexes
│   ├── modeling/             # TCN/Transformer training scripts
│   ├── runtime/              # offline and live squat counters
│   ├── review/               # hard-case review tools and server
│   └── evaluation/           # metrics, routing, audit, and report builders
├── notebooks/                # exploratory and Colab notebooks
│   ├── 01_eda/
│   ├── 02_data_preparation/
│   └── 03_modeling/
├── datasets/metadata/        # small tracked CSV/JSON/XLSX metadata
├── reports/                  # dashboards, figures, docs, results, presentations
├── Data/                     # local/raw/generated dataset files, mostly ignored
├── outputs/                  # local training/runtime outputs, ignored
├── scripts/                  # repository maintenance helpers
├── tests/                    # lightweight regression tests
├── index.html                # static project landing page
└── requirements-pose.txt
```

---

## Setup

```bash
cd /Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/ML_System
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-pose.txt
```

## Tests

```bash
python3 tests/run_tests.py --list
python3 tests/run_tests.py all
```

---

## Runtime Examples

Offline squat counter (from saved feature file):

```bash
python3 src/rep_counter/runtime/run_squat_counter.py \
  --feature-path Data/LLSP/annotation_cleaned/squat_features/train3946_squat_features.npy \
  --pretty
```

Live webcam squat counter:

```bash
python3 src/rep_counter/runtime/run_live_squat_counter.py \
  --camera-index 0
```

---

## Key Entry Points

Feature extraction and runtime:

```text
src/rep_counter/features/pose_feature_extraction.py     # YOLO11n-pose extraction
src/rep_counter/features/build_pose_feature_index.py    # index pose feature files
src/rep_counter/features/build_pose_sequence_dataset.py # build sequence datasets
src/rep_counter/runtime/run_squat_counter.py            # offline squat counter
src/rep_counter/runtime/run_live_squat_counter.py       # live webcam counter
src/rep_counter/review/hard_case_review_server.py       # hard-case review tools
```

Exploratory pipeline scripts (generated the behavioral data in PROJECT_SPEC §7):

```text
src/rep_counter/data/prepare_countix_manifest.py        # dataset manifest
src/rep_counter/modeling/train_pose_count_tcn.py        # shared pose TCN
src/rep_counter/modeling/train_squat_tcn.py             # dedicated squat TCN
src/rep_counter/modeling/train_rgb_count_tcn.py         # RGB TCN
src/rep_counter/evaluation/build_routed_count_predictions.py  # routed eval
```

---

## Data Policy

Tracked:

- small metadata under `datasets/metadata/`
- source code under `src/`
- notebooks under `notebooks/`
- reports and figures under `reports/`
- tests and scripts

Ignored:

- `.venv/`
- `.DS_Store`
- `.ipynb_checkpoints/`
- `__pycache__/`
- raw videos
- generated arrays such as `*.npy` and `*.npz`
- model/checkpoint files such as `*.pt`, `*.pth`, and `*.onnx`
- archives such as `*.zip`, `*.tar`, and `*.tgz`
- local `outputs/`

---

## Commit Helper

Preview what would be committed:

```bash
scripts/commit_code.py --dry-run
```

Commit allowed source-like changes:

```bash
scripts/commit_code.py -m "your message"
```

The helper performs a filtered `git add` and avoids committing local environments,
raw data, generated arrays, videos, checkpoints, caches, and training outputs.

# Exercise Recognition and Repetition Counting
## Working Draft

This document is a step-by-step working draft for the project. It is intended to be updated incrementally as the implementation, evaluation, and project framing evolve.

---

## Document Status

- Project goal: multi-exercise system for exercise recognition and repetition counting
- Current implementation scope: first iteration focused on squat videos only
- Current role of this iteration: validate the pose-based architecture, preprocessing pipeline, feature extraction strategy, and repetition-counting logic before scaling

---

## 1. Introduction

### Draft

Repetitive action counting is an important problem in computer vision and video understanding because many real-world human activities involve repeated motion over time. Examples include fitness exercises, rehabilitation routines, and sports drills, where accurate repetition counts are useful for progress tracking, adherence monitoring, and performance assessment.

The long-term goal of this project is to build a multi-exercise system capable of both recognizing the exercise being performed and estimating repetition counts automatically from video. The current implementation is the first iteration of that system and focuses specifically on squat videos. This narrower prototype is used to validate the proposed pose-based architecture, preprocessing strategy, feature extraction pipeline, and repetition-counting approach before extending the system to additional exercises and broader deployment settings.

### Notes to Update

- Add final problem statement wording
- Add final motivation paragraph
- Add final scope sentence after the multi-exercise direction is finalized

---

## 2. System Design / Architecture

### High-Level Pipeline

```text
Input Videos + Raw Annotations
            |
            v
Exploratory Data Analysis (EDA)
            |
            v
Data Cleaning / Relabeling
            |
            v
Prepared Cleaned Annotations
            |
            v
Pose Index Generation
            |
            v
Pose Estimation (YOLO Pose)
            |
            v
Raw Temporal Pose Features [T, 51]
            |
            v
Pose Postprocessing + Feature Engineering
            |
            v
Squat-Specific Temporal Features
            |
            v
Repetition Counting (FSM / future alternatives)
            |
            v
Predictions + Evaluation
```

### Stage-by-Stage Description

#### Stage 1. Exploratory Data Analysis (EDA)

**Purpose**
- Understand the dataset structure and identify quality issues before modeling.

**What was done**
- Verified the downloaded Part-A total against the expected dataset size.
- Confirmed the split sizes:
  - `758` train
  - `131` valid
  - `152` test
  - `1041` total videos
- Inspected class distribution, repetition-count distribution, duration variation, label consistency, and temporal annotation structure through the `L1-L302` columns.
- Excluded the test split from analytical decisions and used it only for size verification and typo cleaning.
- Reviewed videos manually to identify ambiguous or suspicious labels, especially inside the `others` class.
- In particular, manually reviewed all `37` videos in the `others` class.
- The manual review found that `others` included multiple distinct motion groups, including:
  - soccer juggling / ball control
  - indoor rowing / rowing erg
  - on-water rowing
  - one squat sample
- Generated visual inspection artifacts such as `others_inspection.pdf` and per-class inspection PDFs.

**Current conclusion**
- The raw dataset was not ready to use directly.
- EDA revealed:
  - annotation inconsistencies
  - typo and naming issues
  - ambiguous classes such as `others`
  - strong class imbalance
  - high repetition-count variability and outliers
  - one row with missing `count` in `train + valid`
- EDA confirmed that a cleaning and relabeling stage was necessary before modeling.
- A key insight from inspecting `others` was that the videos were not exact duplicates, but they formed same-type clusters. This showed that `others` was a noisy and heterogeneous category, so keeping all of those videos inside one generic class would likely hurt future exercise classification quality.

**To update**
- Add selected EDA figures if needed.

#### Stage 2. Data Cleaning and Relabeling

**Purpose**
- Correct label issues before building the pipeline.

**What was done**
- Fixed known typo and naming variants in the `type` column.
- Corrected examples such as:
  - `squant` -> `squat`
  - `frontraise` -> `front_raise`
  - `benchpressing` -> `bench_pressing`
  - `jumpjacks` / `jump_jack` -> `jump_jacks`
  - `situp` -> `sit_up`
  - `pullups` -> `pull_up`
  - `pushups` -> `push_up`
- Reduced the apparent raw label space from `16` action names to `10` clean classes.
- Reviewed the `others` class manually after generating `others_inspection.pdf`.
- The `others` category originally contained `37` train videos and `0` validation videos, so all manual review decisions were concentrated in the train split.
- The review showed that `others` was not a single coherent class. Instead, it mixed multiple semantically different repeated actions, including rowing-related motions, soccer juggling / ball control, and one squat.
- Checked for duplicates inside `others` and found:
  - no duplicate filenames
  - no duplicate file contents
- Applied manual decisions to `others`:
  - relabeled `11` train videos to `rowing_erg`
  - removed `26` ambiguous or otherwise unusable train videos
- After decisions, `others` was reduced to `0` videos in `train + valid`.

**Current conclusion**
- Clean labels were necessary for reliable downstream feature extraction and evaluation.
- The dataset required both automatic typo normalization and manual semantic review.
- The EDA process did not only clean spelling, it materially changed the usable class structure.
- The `others` review demonstrated that a single catch-all class can hide meaningful clusters of exercises or motion patterns. Treating such a heterogeneous group as one class would likely degrade both exercise recognition and any later multi-exercise extension of the system.

**To update**
- Add a pointer to the decisions manifest if needed.

#### Stage 3. Data Preparation

**Purpose**
- Produce a stable cleaned dataset contract for modeling.

**What was done**
- Separated data preparation from EDA so that diagnosis and deterministic fixes were handled in different notebooks.
- Processed `train + valid` only in this stage; the test split was intentionally excluded from curation decisions to avoid evaluation leakage.
- Preserved split identity and raw temporal `L*` columns during loading.
- Dropped validation columns that were entirely null because they provided no validation signal.
- Applied a fixed typo-normalization policy through `TYPO_MAP`.
- Applied explicit manual curation through:
  - `RELABEL_MAP`
  - `REMOVE_LIST`
- Included a confirmed manual correction:
  - `stu6_11.mp4 -> squat`
- Kept class-exclusion hooks available but empty by default:
  - `DOMAIN_EXCLUDE_TYPES = []`
  - `POLICY_EXCLUDE_TYPES = []`
- Removed the single row with missing `count` from count-based analysis.
- Added leakage checks between `train` and `valid`:
  - exact overlap by video name
  - exact overlap by (`type`, `name`, `count`)
  - near-duplicate base-name overlap
  - validation hits from relabel/remove lists
- Exported reproducibility artifacts:
  - `train_cleaned.csv`
  - `valid_cleaned.csv`
  - `decisions_manifest.json`

**Current conclusion**
- The project now has a reproducible dataset interface for modeling.
- Data preparation converted a noisy raw annotation set into a stable modeling contract.
- The cleaned dataset is the reason downstream stages can be run consistently.
- This stage formalized the curation policy instead of leaving cleaning decisions implicit inside analysis code.
- It also added guardrails against split leakage before modeling.

**To update**
- Final cleaned counts from the EDA summary:
  - `732` train
  - `131` valid
  - `152` test
  - `10` action classes

#### Stage 4. Pose Index Generation

**Purpose**
- Define which videos will be processed and where outputs will be written.

**What was done**
- Built pose index CSV files from the cleaned annotations, including a squat-only index for the first implementation.
- Prepared the squat-only worklist used by **Colab 4**, which performs the pose-extraction stage.
- In Colab 4, generated `pose_feature_index_squat.csv` and pointed the output feature paths to the persistent Google Drive artifact location rather than ephemeral Colab storage.
- Verified the planned workload before extraction by checking the row count of the squat index.
- Confirmed that the final squat-only index contained `118` videos.
- Used the following Drive-backed contract in Colab 4:
  - `Data/LLSP/annotation_cleaned/pose_feature_index_squat.csv`
  - `Data/LLSP/annotation_cleaned/pose_features/`

**Current conclusion**
- The squat subset can be processed as a clean first-iteration problem.
- This stage created the contract between cleaned annotations and the GPU-based pose-extraction run.
- Using a Drive-backed index and output path made the pose stage reproducible across Colab sessions and established the handoff contract for later stages.

#### Stage 5. Pose Estimation

**Purpose**
- Convert raw videos into temporal human-pose sequences.

**What was done**
- Used a pretrained YOLO Pose model to extract 17 keypoints per frame.
- Stored each frame as 51 values: `x`, `y`, and confidence for each keypoint.
- Ran this stage in **Colab 4**.
- Used **GPU inference** because pose extraction requires running the pretrained YOLO model on every frame of every video, which is computationally expensive even though the model is not being trained.
- GPU was used to accelerate frame-by-frame pose inference and make full squat-subset extraction feasible within the project timeline.
- Verified in Colab 4 that CUDA was available and ran inference on `cuda:0`.
- Started with a smoke test on `5` videos before launching the full extraction run.
- After the smoke test, ran the full extraction on the squat subset indexed from the cleaned annotations.
- The full run processed `118` squat videos and saved per-video pose arrays to the Drive-backed `pose_features` directory.
- The extraction report showed `status = ok` for all `118` rows.
- The saved report structure confirmed:
  - `feat_dim = 51` for all videos
  - `frames_total` ranged from `100` to `2550`
  - `frames_used` ranged from `100` to `2550`
  - the report `message` field was blank for successful rows, which appears as `NaN` when loaded with pandas defaults
- The synced local summary now matches the Colab 4 full run:
  - `total_rows = 118`
  - `ok = 118`
  - `skipped_exists = 0`
  - `failed = 0`
  - `ok_with_zero_pose_frames = 0`
- From the synced local report:
  - mean `frames_total` = `888.6`
  - mean `frames_used` = `885.09`
  - `16` videos had `frames_used < frames_total`, indicating minor frame-level pose misses but not extraction failure

**Current conclusion**
- The pose-based architecture is viable for the squat subset.
- Using Colab 4 with GPU was an implementation choice for efficiency, not because a new pose model was being trained.
- The smoke test reduced execution risk before the full run.
- The full Colab 4 result showed that pose extraction was not the blocking part of the current pipeline: the squat subset completed successfully with complete per-video feature generation and no failed videos.

**Current evidence**
- Pose extraction summary and report exist under `Data/LLSP/annotation_cleaned`.
- The primary artifacts created or updated by Colab 4 are:
  - `pose_feature_index_squat.csv`
  - `pose_extraction_report.csv`
  - `pose_extraction_summary.json`
  - `pose_features/*.npy`

#### Stage 6. Pose Postprocessing and Feature Engineering

**Purpose**
- Turn raw keypoints into more stable and interpretable squat signals.

**What was done**
- Ran this stage in **Colab 5** after Colab 4 pose extraction had completed.
- Read the Drive-backed squat pose index from `pose_feature_index_squat.csv` and the raw pose arrays from `pose_features/*.npy`.
- Wrote the processed outputs back to Drive so the next notebook could consume persistent feature artifacts across sessions.
- Generated squat-specific temporal features such as knee flexion, hip drop, confidence-related support, and other lower-body signals.
- Performed a single-video inspection before the full batch run to verify that the feature helpers behaved sensibly before scaling to all squat videos.
- Processed the full squat subset and saved:
  - `squat_features/*.npy`
  - `squat_feature_index.csv`
  - `squat_feature_summary.csv`
- The batch run completed for all `118` rows, with `status = ok` for every processed video.
- The saved summary confirms:
  - `118` rows
  - `118` `ok`
  - mean `frames_valid` = `872.042372881356`
  - mean `mean_conf` = `0.9408578704726898`
  - `frames_valid` range = `77` to `2550`
  - `mean_conf` range = `0.23204709589481354` to `0.9987007975578308`
- In this summary:
  - `frames_valid` means the number of frames in a video whose lower-body pose signal remained usable after the feature-stage validity checks.
  - mean `frames_valid = 872.042372881356` means that, on average, each squat video contributed about `872` usable frames to downstream repetition counting.
  - `mean_conf` is the average lower-body confidence score used by the squat feature pipeline, not a generic whole-body confidence.
  - mean `mean_conf = 0.9408578704726898` means that the extracted squat features were built from strong lower-body detections in most videos.
- The local synced workspace now contains:
  - `118` engineered squat feature `.npy` files under `Data/LLSP/annotation_cleaned/squat_features`
  - `squat_feature_index.csv`
  - `squat_feature_summary.csv`

**Current conclusion**
- Raw pose alone is not the most usable representation for counting.
- Engineered squat features provide a clearer signal for downstream repetition logic.
- Colab 5 completed successfully for the full squat subset, so feature engineering is no longer a blocking stage.
- The high mean lower-body confidence and large average valid-support signal indicate that the counting stage is operating on mostly strong downstream features rather than severely degraded pose inputs.
- This stage also established the contract for Colab 6, which reads `squat_feature_index.csv` and `squat_features/*.npy` for repetition counting.

#### Stage 7. Repetition Counting

**Purpose**
- Estimate the number of squat repetitions from engineered temporal features.

**What was done**
- Ran this stage in **Colab 6** using the engineered squat features produced by Colab 5.
- Used an FSM-based counter over squat movement phases instead of a learned end-to-end repetition-count model.
- Counted repetitions from the temporal squat features and saved the baseline per-video predictions to:
  - `squat_rep_count_results.csv`
- Computed the baseline rep-count evaluation metrics from those saved predictions.
- Performed threshold tuning on the `train` split only so that `valid` could remain the main honest check of generalization.
- Saved the tuning search output to:
  - `squat_rep_tuning_results.csv`
- Reran the counter with the best train-selected thresholds and saved the tuned per-video predictions to:
  - `squat_rep_count_results_tuned.csv`
- The synced local tuning table currently contains `144` candidate configurations.
- The best saved train-selected threshold set in the synced local outputs is:
  - `min_conf = 0.25`
  - `min_valid_ratio = 0.5`
  - `enter_down = 25.0`
  - `enter_bottom = 45.0`
  - `exit_bottom = 30.0`
  - `back_to_up = 20.0`
  - `min_bottom_frames = 2`
- The synced baseline results currently show:
  - overall rows = `118`
  - overall `MAE = 3.9745762711864407`
  - overall `RMSE = 6.9581070872248105`
  - overall `Within-1 = 0.559322033898305`
- The synced tuned results currently show:
  - `valid` rows = `16`
  - `valid MAE = 3.0625`
  - `valid RMSE = 4.9180788932265`
  - `valid Within-1 = 0.5625`
  - `train` rows = `102`
  - `train MAE = 2.303921568627451`
  - `train RMSE = 4.7309991482958935`
  - `train Within-1 = 0.6764705882352942`

**Current conclusion**
- Pose-derived repetition counting is feasible.
- The main remaining challenge is counting quality and calibration, not raw pose extraction.
- The current FSM backend is working end to end, but it still requires calibration and does not yet remove the need for exercise-specific tuning.
- The tuned results are better than the untuned baseline in the synced local artifacts, which supports the decision to keep validation separate from train during tuning.
- The main bottleneck has now moved from data generation to counting robustness and eventual scaling beyond squat.

#### Stage 8. Output and Evaluation

**Purpose**
- Measure whether the first-iteration system produces useful results.

**What was done**
- Generated per-stage outputs, saved per-video prediction files, and computed rep-count evaluation metrics.
- Persisted the main Colab 6 result artifacts locally under `Data/LLSP/annotation_cleaned`:
  - `squat_rep_count_results.csv`
  - `squat_rep_tuning_results.csv`
  - `squat_rep_count_results_tuned.csv`
- The notebook also includes logic to write:
  - `squat_rep_metrics_summary.csv`
  but that summary CSV is not yet present in the synced local workspace, which suggests the final metrics-export cell may not have been rerun after the notebook change.

**Current conclusion**
- The current implementation is a functional validation prototype, not only a conceptual design.
- The project now has saved evidence for both an untuned baseline and a tuned FSM result, which is enough to support a first-iteration evaluation narrative.
- The remaining documentation task is mainly synchronization and presentation rather than a missing counting implementation.

---

## 3. Tools and Technologies

### Core Stack

- Python
- Ultralytics YOLO Pose
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Google Colab

### Development Setup

- Local repository for scripts and documentation
- Colab notebooks for stages 4, 5, and 6
- CSV / JSON / NPY artifacts passed between stages

### Why These Tools Were Chosen

- Fast to prototype
- Compatible with pretrained pose inference
- Good fit for a staged, inspectable pipeline
- Practical for a first validation iteration

### To Update

- Add exact package versions if needed
- Add any final Colab / hardware notes

---

## 4. Initial Implementation Progress

### Implemented Components

- Data loading from cleaned annotations
- Squat-only index generation
- Pose extraction pipeline
- Squat feature-extraction pipeline
- FSM-based repetition-counting baseline
- Saved reports and metric artifacts

### Current Progress Summary

- Upstream data preparation: working
- Pose extraction: working
- Squat feature extraction: working
- Rep-count evaluation: working
- Multi-exercise recognition: not yet implemented in the current iteration
- Cleaned artifacts are versioned through a preparation-stage manifest and exported CSV contract

### To Update

- Add the latest final status after each rerun
- Add screenshots or result tables if needed

---

## 5. Dataset Preparation

### Dataset Used

- RepCount / LLSP-style exercise-video setup
- Current implementation begins with squat as the first validated class

### Important Findings From EDA

- The expected total for Part-A was confirmed: `1041` videos.
- Raw labels were inconsistent and contained multiple typo variants.
- The class `others` was not modeling-ready and required manual inspection.
- `others` appeared only in the train split during EDA review (`37` train, `0` valid), so the ambiguity problem was localized to train rather than validation.
- The `others` inspection showed that the videos were not duplicates but naturally grouped into same-type clusters, confirming that `others` was heterogeneous rather than coherent.
- The key clusters found in `others` included:
  - soccer juggling / ball control
  - indoor rowing / rowing erg
  - on-water rowing
  - one squat sample
- A new clean class, `rowing_erg`, emerged from relabeling videos originally grouped under `others`.
- Class imbalance was substantial:
  - before final `others` decisions, train imbalance was about `7.77x`
  - after cleaning decisions, train imbalance was about `9.2x`
- Repetition counts were right-skewed with large outliers:
  - `14` outlier videos were flagged
  - the outlier threshold was approximately `59` repetitions
- Temporal annotation consistency was strong overall:
  - train mismatches: `1`
  - valid mismatches: `0`
- `L` columns were highly sparse, but that sparsity was structural by design rather than a corruption issue.

### Data Structure

- `Data/LLSP/video`
- `Data/LLSP/annotation`
- `Data/LLSP/annotation_cleaned`

### Preprocessing Completed So Far

- EDA
- label inspection
- typo fixes
- relabeling of ambiguous samples
- removal of unusable ambiguous samples
- conversion of `others` videos into explicit decisions: relabel or remove
- cleaned train/validation annotation generation
- deterministic data-preparation policy exported through `decisions_manifest.json`
- leakage checks between `train` and `valid`
- preservation of configurable class-exclusion hooks for later experiments
- pose index generation

### To Update

- Add any final per-class cleaned counts if needed
- Add final explanation of `LLSP` naming if needed

---

## 6. Existing Work and Research Gap

### Overview

Prior work relevant to this project falls into two main groups:

- direct video repetition-counting models
- pose / skeleton sequence models that are strong candidates for the next multi-exercise iteration

The current project does not aim to reproduce state-of-the-art counting immediately. Instead, it uses a practical first iteration to validate a pose-based system design before scaling. The references below are divided into:

- **classic references** that are still foundational
- **recent references** from `2023` onward that better reflect the current research landscape

### Classic References

| Method | Type | Pros | Cons | Best use | Feasibility for this project |
|---|---|---|---|---|---|
| `RepNet (CVPR 2020)` | Direct video repetition counting | Class-agnostic counting, direct video-to-count pipeline, no handcrafted exercise rules | Less interpretable, different data/evaluation setup from current repo, weaker fit for future exercise recognition | Pure video-based repetition-count prediction | Good benchmark reference, low practicality for current implementation |
| `Context-Aware and Scale-Insensitive Temporal Repetition Counting (CVPR 2020)` | Direct video repetition counting | Addresses changing repetition speed, strong historical counting reference | Older architecture family, not well aligned with the broader pose-based multi-exercise plan | Historical and methodological reference | Useful for literature review, low implementation priority |
| `TransRAC (CVPR 2022)` | Direct video repetition counting with transformers and density regression | Strong RepCount relevance, better handling of long and realistic videos, stronger than early counting baselines | Higher implementation complexity, less interpretable, different assumptions from current repo | Research-style direct counting on RepCount-like data | High relevance as prior work, moderate to low implementation feasibility here |
| `ST-GCN / 2s-AGCN` | Skeleton-based action recognition | Natural fit for pose data, good bridge from YOLO to learned temporal modeling, better scaling than FSM | Older than newer skeleton literature, more engineering than TCN | Learned action recognition and count prediction from pose sequences | Realistic next-step family, but heavier than a simple baseline |

### Recent References (2023-Present)

| Method | Type | Pros | Cons | Best use | Feasibility for this project |
|---|---|---|---|---|---|
| `SkeleTR (ICCV 2023)` | Skeleton-based action recognition | More modern pose-sequence backbone, better aligned with shared multi-exercise learning | More research-oriented, more complex than TCN | Multi-exercise pose-sequence learning in realistic settings | Moderate feasibility, good next-iteration reference |
| `Skeleton-in-Context (CVPR 2024)` | Unified skeleton sequence modeling | Supports one shared representation across tasks, conceptually strong for multi-exercise learning | Experimental, less straightforward to reproduce quickly | Research-oriented shared skeleton representation learning | Good inspiration, lower immediate feasibility |
| `BlockGCN (CVPR 2024)` | Skeleton-based action recognition | Newer graph-based pose modeling, more up to date than only citing classic graph baselines | More engineering effort than TCN, still heavier than needed for the fastest next step | Strong modern graph-based skeleton modeling | Reasonable future option, not the fastest first build |
| `Motion Feature Learning (WACV 2024)` | Direct video repetition counting | Strong modern counting reference, improves robustness by modeling motion explicitly | Still RGB-centric, less aligned with the current pose-first repo | Modern direct video-counting benchmark reference | Useful literature reference, low direct implementation fit |
| `Every Shot Counts / ESCounts (ACCV 2024)` | Direct video repetition counting | Strong recent performance, good reference for direct counting capability | More complex, less interpretable, weaker fit for pose-first multi-exercise plans | Research comparison for direct counting from video | Strong comparison point, but not the most practical implementation target |
| `CountLLM (CVPR 2025)` | Direct repetitive-action counting with multimodal / LLM-style architecture | Very recent, strong generalization motivation, cutting-edge reference | Highest complexity, poor fit for a compact engineering iteration | Research frontier reference | Low near-term feasibility, high literature value |

### Practical Interpretation For This Project

- The **current squat prototype** was best served by a practical and inspectable pose-based pipeline, not by jumping immediately to the most complex direct video-counting model.
- For the **next iteration**, the most suitable direction is still likely:
  - keep YOLO Pose
  - stop handcrafting exercise-specific logic
  - train one shared temporal model across exercises
- In that context, the most feasible implementation choices are:
  - `TCN / 1D CNN` as the fastest learned baseline
  - `ST-GCN / 2s-AGCN` or newer skeleton models as stronger longer-term pose-native options
- Direct video methods such as `RepNet`, `TransRAC`, `ESCounts`, and `CountLLM` remain important references, but they are better treated as:
  - literature context
  - evaluation baselines for comparison
  - possible future alternatives if the project later shifts away from pose-based modeling

### Why RepNet / TransRAC Were Not Used In The First Iteration

- Direct video-counting methods such as **RepNet** and **TransRAC** were considered as relevant prior work, but they were not selected for the first implementation stage.
- Although these methods are attractive because they predict repetition counts directly from video, they are not drop-in replacements for the current repository structure.
- Adopting them would have required a different data contract, different training and evaluation setup, and additional integration work around annotation formatting, model execution, and result interpretation.
- The first iteration of this project was designed to answer a different engineering question: whether a **pose-based architecture** could provide a reliable and interpretable foundation for a broader multi-exercise system.
- For that reason, the project prioritized:
  - a staged and inspectable pipeline
  - reusable intermediate artifacts
  - easier debugging of failures in data cleaning, pose extraction, feature engineering, and counting
- This made the YOLO-plus-feature-plus-FSM pipeline more appropriate for the first iteration, even if it was not the most direct research-style route to video-only count prediction.
- In other words, RepNet or TransRAC may have been conceptually simpler for pure repetition-count prediction, but they were not necessarily faster or lower risk for this specific project once dataset adaptation, evaluation alignment, and future multi-exercise extensibility were taken into account.
- The current implementation should therefore be understood as an architecture-validation prototype rather than an attempt to compete immediately with research-grade direct video-counting models.

### Current conclusion

- The classic references remain useful for grounding the project, but the next-iteration planning should be informed by newer `2023-2025` work.
- The most adequate model for this project is not necessarily the newest one; however, checking newer work is still important to confirm that the chosen next-step architecture is technically defensible.
- For this project, the best balance of adequacy and feasibility still appears to be a shared learned pose-sequence model rather than either:
  - continuing with handcrafted FSM logic across many exercises
  - or jumping immediately to a heavy direct RGB counting architecture

---

## 7. Current Metrics

### Baseline / Current Known Metrics

- The synced local Colab 6 baseline outputs currently indicate:
  - overall rows = `118`
  - `MAE = 3.9745762711864407`
  - `RMSE = 6.9581070872248105`
  - `Within-1 = 0.559322033898305`

### Current Tuned Metrics

- Primary reportable split: `valid`
  - rows = `16`
  - `MAE = 3.0625`
  - `RMSE = 4.9180788932265`
  - `Within-1 = 0.5625`
- Diagnostic split: `train`
  - rows = `102`
  - `MAE = 2.303921568627451`
  - `RMSE = 4.7309991482958935`
  - `Within-1 = 0.6764705882352942`
- These values were recomputed from the synced local `squat_rep_count_results_tuned.csv` artifact and should be treated as the current saved-state metrics unless a newer Colab rerun overwrites them.

### To Update

- Sync `squat_rep_metrics_summary.csv` after rerunning the final Colab 6 export cell if a single-file metrics summary is still desired.
- Keep this section synchronized with any newer saved Colab 6 reruns.

---

## 8. Next Steps

### Immediate Next Steps

- Finalize the current squat counting baseline
- Save and verify all result CSV artifacts
- Inspect the hardest validation examples

### Mid-Term Next Steps

- Extend to one or two additional exercise classes
- Add exercise recognition before exercise-specific counting
- Compare FSM against a learned temporal pose model if still within scope

### Long-Term Next Steps

- Generalize to a multi-exercise system
- Reduce notebook dependency
- Move toward broader deployment settings

### Candidate Models For The Next Iteration

The next iteration is expected to move away from handcrafted exercise-specific rules and toward a shared learned model that can support multiple exercises at once. If YOLO Pose is retained as the front end, the main design question becomes which temporal model should map pose sequences to:

- exercise classification
- repetition-count prediction

The target architecture would be:

`video -> YOLO pose -> pose sequence -> shared temporal model -> {exercise class, rep count}`

The candidate models listed below are restricted to approaches that are well aligned with a **YOLO-pose-first pipeline**. Architectures such as `CNN + LSTM` were intentionally not prioritized here because they are more naturally suited to raw RGB frame sequences than to already-extracted pose/keypoint sequences.

#### Candidate Model Summary

| Candidate model | Best use | Feasibility for this project |
|---|---|---|
| `TCN / 1D CNN` | Best practical first learned baseline | High |
| `ST-GCN / 2s-AGCN` | Best pose-native long-term direction | Moderate |
| `GRU / LSTM` | Straightforward sequence baseline | Moderate |
| `SkeleTR` | Modern research-oriented skeleton model | Moderate |
| `BlockGCN` | Modern graph-based skeleton model | Moderate |
| `PoseConv3D` | Stronger advanced skeleton representation | Low |
| `Transformer encoder on pose tokens` | Flexible temporal modeling, but high tuning risk | Low |

#### Option 1. TCN / 1D CNN on Pose Sequences

**Description**
- Flatten or normalize the pose sequence and feed it to a temporal convolutional model.

**Pros**
- Fastest learned baseline to implement
- Efficient training and inference
- Lower engineering risk than graph-based models
- Good practical choice for a first multi-exercise learned prototype

**Cons**
- Does not explicitly model the body-joint graph structure
- May be less expressive than skeleton-native models for fine joint relationships

**Fit for this project**
- Best choice for the fastest next iteration
- Strong recommendation if implementation speed is important

#### Option 2. ST-GCN / 2s-AGCN

**Description**
- Use a graph-based temporal model where joints are graph nodes and body connections define the skeleton structure over time.

**Pros**
- More natural representation for human-pose sequences
- Designed specifically for skeleton-based action understanding
- Better long-term fit for multi-exercise learning than handcrafted rules

**Cons**
- More engineering complexity than TCN
- Harder to implement and debug in a short time window
- Requires careful data formatting for skeleton graphs

**Fit for this project**
- Best pose-native long-term direction
- Strong option if the next iteration prioritizes scalability over speed of implementation

#### Option 3. GRU / LSTM

**Description**
- Feed pose or feature sequences into a recurrent temporal model and predict exercise class and repetition count from the learned hidden representation.

**Pros**
- Straightforward sequence-modeling baseline
- Easier to understand conceptually
- Works directly on temporal pose features

**Cons**
- Slower and less parallel than TCN-style models
- Often less attractive than TCN for practical implementation today
- Can be less stable on longer sequences

**Fit for this project**
- Reasonable baseline, but not the strongest recommendation compared with TCN or graph models

#### Option 4. PoseConv3D

**Description**
- Use a skeleton-based model that represents pose as a spatiotemporal volume rather than relying only on graph convolutions.

**Pros**
- Stronger skeleton-modeling direction in the literature
- Reported robustness to pose noise
- More expressive than simpler temporal baselines

**Cons**
- Heavier engineering effort
- Less suitable for the fastest next iteration
- Higher implementation complexity than TCN and standard recurrent baselines

**Fit for this project**
- Promising advanced option, but not the first model to build next

#### Option 5. Transformer Encoder on Pose Tokens

**Description**
- Encode pose frames or joint tokens with a transformer and predict both exercise type and repetition count.

**Pros**
- Flexible temporal modeling
- Modern architecture family with strong representational capacity

**Cons**
- Higher tuning cost
- Greater overfitting risk on a modest dataset
- Not the best first replacement for the current FSM pipeline

**Fit for this project**
- Better as a later research experiment than as the immediate next iteration

### Count-Head Design Options

Regardless of the temporal backbone, the repetition-count output can be modeled in different ways:

#### Direct Regression
- Predict one scalar repetition count for the whole video.
- Simplest design and easiest to implement.
- Recommended as the first learned count head.

#### Count Classification
- Predict the repetition count as a class from a bounded count range.
- Can be stable when counts are limited, but becomes awkward when the range is wide.

#### Density / Event Prediction
- Predict a temporal event or density signal and integrate it into a final count.
- More aligned with some research counting methods, but more complex to implement.

### Models Not Prioritized For This Pipeline

| Model family | Why it is not prioritized here |
|---|---|
| `CNN + LSTM` | Better suited to raw RGB frame pipelines than to YOLO pose sequences; adds complexity without matching the current pose-first design |
| Direct RGB video counters (`RepNet`, `TransRAC`, `ESCounts`, `CountLLM`) | Important as related work, but not the best-aligned next implementation step for a YOLO-pose backend |

### Current Recommendation

- Keep the current squat FSM pipeline as the baseline prototype.
- For the next iteration, retain YOLO Pose and replace handcrafted exercise logic with a shared multi-exercise temporal model.
- Recommended order of exploration:
  - `TCN / 1D CNN` as the fastest practical next model
  - `ST-GCN / 2s-AGCN` as the strongest pose-native long-term direction
  - `direct regression` as the first count-output design

This recommendation supports the broader project goal of moving from a squat-only validation prototype toward a shared multi-exercise repetition-counting system without relying on handcrafted per-exercise rules.

---

## 9. References

### Project References

- `README.md`
- `artifacts/specification.md`
- `artifacts/3_Modeling/YOLO_PIPELINE.md`
- `artifacts/3_Modeling/YOLO_POSE_STAGE.md`
- `artifacts/3_Modeling/4_Squat_Pose_Extraction_Colab.ipynb`
- `artifacts/3_Modeling/5_Squat_Feature_Extraction_Colab.ipynb`
- `artifacts/3_Modeling/6_Squat_Rep_Counting_Colab.ipynb`

### External References To Keep

- RepCount dataset page: https://svip-lab.github.io/dataset/RepCount_dataset.html
- RepNet (CVPR 2020): https://openaccess.thecvf.com/content_CVPR_2020/html/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.html
- Context-Aware and Scale-Insensitive Temporal Repetition Counting (CVPR 2020): https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Context-Aware_and_Scale-Insensitive_Temporal_Repetition_Counting_CVPR_2020_paper.html
- TransRAC (CVPR 2022): https://openaccess.thecvf.com/content/CVPR2022/html/Hu_TransRAC_Encoding_Multi-Scale_Temporal_Correlation_With_Transformers_for_Repetitive_Action_CVPR_2022_paper.html
- SkeleTR (ICCV 2023): https://openaccess.thecvf.com/content/ICCV2023/html/Duan_SkeleTR_Towards_Skeleton-based_Action_Recognition_in_the_Wild_ICCV_2023_paper.html
- Skeleton-in-Context (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Skeleton-in-Context_Unified_Skeleton_Sequence_Modeling_with_In-Context_Learning_CVPR_2024_paper.html
- BlockGCN (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_BlockGCN_Redefine_Topology_Awareness_for_Skeleton-Based_Action_Recognition_CVPR_2024_paper.html
- Motion Feature Learning for Repetitive Action Counting (WACV 2024): https://openaccess.thecvf.com/content/WACV2024/html/Li_Repetitive_Action_Counting_With_Motion_Feature_Learning_WACV_2024_paper.html
- Every Shot Counts / ESCounts (ACCV 2024): https://openaccess.thecvf.com/content/ACCV2024/html/Sinha_Every_Shot_Counts_Using_Exemplars_for_Repetition_Counting_in_Videos_ACCV_2024_paper.html
- CountLLM (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Yao_CountLLM_Towards_Generalizable_Repetitive_Action_Counting_via_Large_Language_Model_CVPR_2025_paper.html
- Bai et al. Temporal Convolutional Networks: https://vladlen.info/publications/tcn/
- ST-GCN: https://aaai.org/papers/12328-spatial-temporal-graph-convolutional-networks-for-skeleton-based-action-recognition/
- 2s-AGCN: https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html
- PoseConv3D: https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html
- Ultralytics docs: https://docs.ultralytics.com/
- OpenCV docs: https://docs.opencv.org/
- NumPy docs: https://numpy.org/doc/
- Pandas docs: https://pandas.pydata.org/docs/

---

## Update Log

Use this section to keep a quick running history of changes.

- `2026-03-13`: Working draft created.
- `2026-03-13`: Updated with findings and actions from `1_EDA_34.ipynb`, including typo cleaning, `others` inspection, relabeling to `rowing_erg`, removals, split verification, imbalance, and outlier observations.
- `2026-03-13`: Updated with deterministic cleaning and export details from `2_Data_Preparation_01.ipynb`, including typo normalization policy, relabel/remove maps, split-leakage checks, and cleaned artifact exports.
- `2026-03-13`: Updated Stage 4 and Stage 5 from `4_Squat_Pose_Extraction_Colab.ipynb`, including the Drive-backed squat index, Colab 4 smoke test, GPU-based YOLO inference, and the successful `118/118` squat pose-extraction result.
- `2026-03-13`: Updated Stage 6 from `5_Squat_Feature_Extraction_Colab.ipynb`, including the Drive-based feature pipeline, saved squat feature artifacts, and the successful `118/118` engineered-feature run with high lower-body confidence.
- `2026-03-13`: Updated Stage 7, Stage 8, and the metrics section from `6_Squat_Rep_Counting_Colab.ipynb` and synced Colab 6 CSV outputs, including baseline results, tuning artifacts, and the current saved tuned train/valid metrics.

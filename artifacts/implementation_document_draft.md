# Exercise Recognition and Repetition Counting
## Working Draft

This document records the implementation, evaluation, and architectural decisions for the project. It has been updated incrementally as the pipeline and evidence base matured from a squat-only baseline into a multi-exercise comparative study.

---

## Document Status

- Project goal: multi-exercise system for exercise recognition and repetition counting
- Current implementation scope: squat-specific pose branch plus widened multi-exercise comparison stages
- Current role of this iteration: validate the pose-based architecture, preprocessing pipeline, feature extraction strategy, repetition-counting logic, and the first exercise-dependent follow-up stages before scaling

## Project Team

- Linda Perez Penaranda
- Kunyi Shi
- Peihan Wang
- Quanxing Lu

---

## 1. Introduction

### Draft

Repetitive action counting is an important problem in computer vision and video understanding because many real-world human activities involve repeated motion over time. Examples include fitness exercises, rehabilitation routines, and sports drills, where accurate repetition counts are useful for progress tracking, adherence monitoring, and performance assessment.

The long-term goal of this project is to build a multi-exercise system capable of both recognizing the exercise being performed and estimating repetition counts automatically from video. The current implementation is no longer only a squat prototype. It now includes a frozen squat-specific pose branch, a widened shared pose-sequence branch across multiple exercises, a controlled RGB comparison branch on `squat`, `pull_up`, and `push_up`, a negative multimodal fusion result, and an exercise-dependent routed counting surface. The role of the current iteration is therefore to validate the upstream pose-first pipeline end to end and to determine which representation and counting strategy are most defensible for each supported exercise under realistic video difficulty.

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
Pose / Video Index Contracts
            |
            v
Pose Estimation (YOLO Pose) ------------------------------+
            |
            v                                            |
Raw Temporal Pose Features [T, 51]                       |
            |                                            |
   +--------+-------------------------+                  |
   |                                  |                  |
   v                                  v                  v
Generic Pose Sequence Preparation   Squat-Specific    RGB Frame Sampling
and Normalization                   Feature Engineering + Frozen CNN Features
   |                                  |                  |
   v                                  v                  v
Shared Pose TCN / Transformer       Squat FSM /       RGB TCN / Stronger RGB
Per-Exercise Tuning                 Squat TCN         Comparison Branch
   |                                  |                  |
   +--------------------+-------------+---------+--------+
                        |                       |
                        v                       v
             Representation Audit / Hard-Case Analysis
                        |
                        v
             Optional Multimodal Fusion Comparison
                        |
                        v
             Exercise-Dependent Routed Counting Output
                        |
                        v
                 Predictions + Evaluation
```

### Stage-by-Stage Description

#### Stage 1. Exploratory Data Analysis (EDA)

**Purpose**
- Understand the dataset structure and identify issues that would invalidate direct modeling.

**What was done**
- Verified the expected Part-A total and split sizes:
  - `758` train
  - `131` valid
  - `152` test
  - `1041` total videos
- Reviewed class distribution, count distribution, duration variability, sparse temporal annotation structure, and obvious label noise.
- Manually inspected the heterogeneous `others` class and generated inspection artifacts.
- Identified one row with missing `count` in `train + valid`, strong count skew, and broad exercise imbalance in the cleaned train split.

**Current conclusion**
- The raw annotations were not ready to use directly.
- The main issues were label inconsistency, ambiguous classes, long-tailed repetition counts, and a small number of missing or questionable labels.

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

#### Stage 4. Pose Extraction and Indexing

**Purpose**
- Convert videos into persistent per-video pose arrays and define reusable indexing contracts.

**What was done**
- Built pose index contracts for:
  - the initial squat-only branch
  - the widened all-exercises branch
- Used YOLO Pose in Colab with GPU inference to extract 17 keypoints per frame and save raw pose arrays.
- Produced persistent Drive-backed artifacts such as:
  - `pose_feature_index_squat.csv`
  - `pose_feature_index.csv`
  - `pose_features/*.npy`
  - extraction summaries and reports

**Current conclusion**
- Pose extraction is operational and not the main project bottleneck.
- The pose front end is now stable enough to support multiple downstream branches.

#### Stage 5. Pose Representation Preparation

**Purpose**
- Build the two pose representations used by the project:
  - squat-specific engineered features
  - generic normalized pose sequences

**What was done**
- For the squat branch:
  - engineered lower-body features such as knee flexion, hip drop, and validity / confidence support
  - exported `squat_features/*.npy`, `squat_feature_index.csv`, and `squat_feature_summary.csv`
- For the widened branch:
  - normalized and resampled raw pose arrays into a shared temporal sequence contract
  - exported `pose_sequence_index.csv` and `pose_sequence_summary.csv`

**Current conclusion**
- The project now supports both:
  - a high-performance exercise-specific pose branch for squat
  - a reusable generic pose representation for shared and per-exercise learned models

#### Stage 6. Pose Counting Baselines and Ablations

**Purpose**
- Establish what pose-only counting can achieve before moving to RGB or multimodal comparisons.

**What was done**
- Built the first squat counting branch:
  - FSM baseline over engineered squat features
  - tuned FSM
  - learned squat TCN replacements
- Froze the strongest learned squat baseline:
  - `squat_tcn_l1_channels96`
- Widened counting to all supported exercises with a shared normalized-pose TCN baseline.
- Ran exercise-by-exercise ablations:
  - `6B` sequence-length sweep
  - `6C` keypoint weighting
  - `6D` density counting

**Current conclusion**
- Pose-only counting is viable, but the results are strongly exercise-dependent.
- The dedicated squat pose branch is clearly strong.
- The widened shared pose branch is useful as a baseline, not as a final generic solution.

#### Stage 7. RGB Branch, Representation Analysis, and Multimodal Check

**Purpose**
- Test whether raw visual information recovers counting signal that pose misses.

**What was done**
- Extracted frozen RGB frame-feature sequences and trained RGB TCN baselines on:
  - `squat`
  - `pull_up`
  - `push_up`
- Strengthened the RGB branch with a frozen `ResNet50` backbone in `7B`.
- Ran `7C` representation-fit analysis and `7D` hard-case audit to study when RGB helps.
- Tested simple late fusion in `7E`.

**Current conclusion**
- RGB is useful, but not uniformly.
- The evidence is now:
  - `squat`: pose-first
  - `push_up`: RGB-first
  - `pull_up`: mixed
- Simple late fusion did not beat the better single-modality branch consistently, so it is treated as a negative-but-informative result.

#### Stage 8. Exercise-Dependent Routed Counting

**Purpose**
- Build a practical counting surface without forcing a single shared architecture to win every exercise.

**What was done**
- Routed each supported exercise to the strongest current branch:
  - `squat -> squat_tcn_l1_channels96`
  - `pull_up -> pose_count_tcn_pull_up_seq192`
  - `push_up -> rgb_count_tcn_push_up_seq128`
- Exported routed predictions and routed metrics summaries.

**Current conclusion**
- The most defensible practical output of the project so far is an exercise-dependent counter, not a universal shared model.

#### Stage 9. Pose Transformer and Augmentation Ablation

**Purpose**
- Test whether a transformer encoder over generic normalized pose sequences improves on the shared pose TCN, and whether the current pose augmentation helps.

**What was done**
- Added a pose-transformer branch on the same generic pose-sequence contract used by Stage 6.
- Added `9B` to compare augmentation on versus off under otherwise matched transformer settings.
- Fixed the trainer contract so saved train metrics now come from deterministic full-train evaluation loaders rather than sampled or augmented training loaders.

**Current conclusion**
- This stage closed as a negative-but-informative architectural check on the generic pose branch.
- The transformer did not beat the shared pose TCN on the practical tradeoff for the supported exercises.
- The `9B` augmentation ablation showed no robust validation benefit:
  - `pull_up` got slightly better `MAE` but much worse `Within-1`
  - `push_up` was effectively unchanged
  - `squat` was effectively unchanged
- This branch does not replace the frozen dedicated squat baseline or the exercise-dependent routed conclusion.

#### Stage 10. Dedicated Pull-Up Pose Follow-Up

**Purpose**
- Start the first exercise-specific pose follow-up beyond squat.

**What was done**
- Added a dedicated `pull_up` tuning stage on the shared Stage 5 pose sequences.
- Compared dedicated pull-up pose candidates directly against:
  - the saved shared pose baseline
  - the saved RGB baselines for `pull_up`

**Current conclusion**
- This stage was informative, but not decisive.
- The best dedicated `pull_up` candidate reduced `MAE` to `3.5463`, but its `Within-1` dropped to `0.2857`.
- No dedicated `pull_up` candidate beat the current practical tradeoff established by:
  - shared pose on `Within-1`
  - stronger RGB on `MAE`
- So `pull_up` remains mixed rather than becoming a clear pose-specialized win.

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
- Colab notebooks for the staged branches from pose extraction through routed counting and dedicated follow-up experiments
- CSV / JSON / NPY artifacts passed between stages

### Why These Tools Were Chosen

- Fast to prototype
- Compatible with pretrained pose inference
- Good fit for a staged, inspectable pipeline
- Practical for a first validation iteration

---

## 4. Initial Implementation Progress

### Implemented Components

- Data loading from cleaned annotations
- Pose index generation for both squat-specific and widened branches
- Pose extraction pipeline
- Squat feature-extraction pipeline
- Generic pose-sequence preparation pipeline
- FSM-based repetition-counting baseline
- Shared pose TCN baseline and ablations
- RGB comparison branch and stronger RGB backbone branch
- Representation-fit and hard-case audit branches
- Multimodal late-fusion comparison
- Exercise-dependent routed counting surface
- Pose-transformer and augmentation-ablation evaluation
- Dedicated pull-up pose-tuning stage
- Experiment showcase and registry artifacts
- Saved reports and metric artifacts

### Current Progress Summary

- Upstream data preparation: working
- Pose extraction: working
- Squat feature extraction: working
- Generic pose-sequence preparation: working
- Rep-count evaluation: working
- Pose vs RGB comparison branch: working
- Representation audits: working
- Exercise-dependent counting comparisons: working
- Routed counting prototype: working
- Dedicated `pull_up` follow-up stage: completed, informative, but not decisive
- Exercise recognition as a separate deployed stage: not yet implemented
- Cleaned artifacts are versioned through a preparation-stage manifest and exported CSV contract

---

## 5. Dataset Preparation

### Dataset Used

- RepCount / LLSP-style exercise-video setup
- LLSP dataset access link:
  - https://drive.google.com/drive/folders/1NUiY4bCTy_zGmJ8AECBcAIpqee5g8F_g?usp=sharing
- Current implementation includes a frozen squat-specific branch plus widened multi-exercise pose and RGB comparison stages
- Countix onboarding scaffolding has now been added as a separate benchmark branch under `Data/Countix`; it is currently deferred and is not part of the active training results in this draft

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

| Method | Type | Dataset / scale | Pros | Cons | Best use | Feasibility for this project |
|---|---|---|---|---|---|---|
| `RepNet (CVPR 2020)` | Direct video repetition counting | `Countix`: about `8.8k` videos / `422` classes; also evaluated on `QUVA Repetition` (`100` videos) and periodicity benchmarks | Class-agnostic counting, direct video-to-count pipeline, no handcrafted exercise rules | Less interpretable, different data/evaluation setup from current repo, weaker fit for future exercise recognition | Pure video-based repetition-count prediction | Good benchmark reference, low practicality for current implementation |
| `Context-Aware and Scale-Insensitive Temporal Repetition Counting (CVPR 2020)` | Direct video repetition counting | `UCFRep`: `526` videos; also evaluated on older small benchmarks such as `YTSeg` (`100`) and `QUVA Repetition` (`100`) | Addresses changing repetition speed, strong historical counting reference | Older architecture family, not well aligned with the broader pose-based multi-exercise plan | Historical and methodological reference | Useful for literature review, low implementation priority |
| `TransRAC (CVPR 2022)` | Direct video repetition counting with transformers and density regression | `RepCount`: `1,451` videos with about `20k` cycle annotations (`Part-A 1,041`, `Part-B 410`) | Strong RepCount relevance, better handling of long and realistic videos, stronger than early counting baselines | Higher implementation complexity, less interpretable, different assumptions from current repo | Research-style direct counting on RepCount-like data | High relevance as prior work, moderate to low implementation feasibility here |
| `ST-GCN / 2s-AGCN` | Skeleton-based action recognition | Typically reported on skeleton action-recognition sets such as `NTU RGB+D 60` (`56,880` sequences / `60` classes) and `Kinetics-Skeleton` (`240,436` train + `19,796` val / `400` classes) | Natural fit for pose data, good bridge from YOLO to learned temporal modeling, better scaling than FSM | Older than newer skeleton literature, more engineering than TCN | Learned action recognition and count prediction from pose sequences | Realistic next-step family, but heavier than a simple baseline |

`ST-GCN / 2s-AGCN` are primarily skeleton action-recognition references rather than repetition-counting papers, so their dataset scale is not directly comparable to `Countix`, `UCFRep`, or `RepCount`.

### Recent References (2023-Present)

| Method | Type | Dataset / scale | Pros | Cons | Best use | Feasibility for this project |
|---|---|---|---|---|---|---|
| `SkeleTR (ICCV 2023)` | Skeleton-based action recognition | Mixed wild skeleton benchmarks including `Kinetics-Skeleton` (`240,436` train + `19,796` val / `400` classes), `AVA` (`430` 15-minute videos / `80` actions), and `Volleyball` (`55` videos) | More modern pose-sequence backbone, better aligned with shared multi-exercise learning | More research-oriented, more complex than TCN | Multi-exercise pose-sequence learning in realistic settings | Moderate feasibility, good next-iteration reference |
| `Skeleton-in-Context (CVPR 2024)` | Unified skeleton sequence modeling | Multi-task skeleton benchmarks including `Human3.6M` (`3.6M` poses / `15` actions), `AMASS` (`40+` hours, `300+` subjects, `11k+` motions), and `3DPW` (`60` sequences, `51k+` frames) | Supports one shared representation across tasks, conceptually strong for multi-exercise learning | Experimental, less straightforward to reproduce quickly | Research-oriented shared skeleton representation learning | Good inspiration, lower immediate feasibility |
| `BlockGCN (CVPR 2024)` | Skeleton-based action recognition | Large skeleton action-recognition sets such as `NTU RGB+D 60` (`56,880` sequences / `60` classes) and `NTU RGB+D 120` (`114,480` sequences / `120` classes) | Newer graph-based pose modeling, more up to date than only citing classic graph baselines | More engineering effort than TCN, still heavier than needed for the fastest next step | Strong modern graph-based skeleton modeling | Reasonable future option, not the fastest first build |
| `Motion Feature Learning (WACV 2024)` | Direct video repetition counting | `RepCount` (`1,451` videos) and `UCFRep` (`526` videos), plus cross-dataset evaluation | Strong modern counting reference, improves robustness by modeling motion explicitly | Still RGB-centric, less aligned with the current pose-first repo | Modern direct video-counting benchmark reference | Useful literature reference, low direct implementation fit |
| `Every Shot Counts / ESCounts (ACCV 2024)` | Direct video repetition counting | `RepCount` (`1,451` videos), `UCFRep` (`526` videos), and `Countix` (about `8.8k` videos / `422` classes) | Strong recent performance, good reference for direct counting capability | More complex, less interpretable, weaker fit for pose-first multi-exercise plans | Research comparison for direct counting from video | Strong comparison point, but not the most practical implementation target |
| `CountLLM (CVPR 2025)` | Direct repetitive-action counting with multimodal / LLM-style architecture | Pretraining on `WebVid-10M`, then RAC benchmarks `RepCount` (`1,451` videos), `UCFRep` (`526` videos), and `Countix` (about `8.8k` videos / `422` classes) | Very recent, strong generalization motivation, cutting-edge reference | Highest complexity, poor fit for a compact engineering iteration | Research frontier reference | Low near-term feasibility, high literature value |

The first three recent references are pose- or skeleton-modeling papers rather than repetition-counting papers, so their benchmark scale is shown for representation-learning context rather than as a directly comparable counting dataset.

### Practical Interpretation For This Project

- The project is no longer best described as a single squat prototype or as a search for one generic counter.
- The completed experiments support an **exercise-dependent interpretation**:
  - `squat` is best handled by the dedicated pose branch
  - `push_up` is best handled by the RGB branch
  - `pull_up` remains mixed even after the dedicated pose follow-up
- In that context, the most defensible practical output is now:
  - keep YOLO Pose as the reusable structured frontend
  - keep RGB as the complementary branch where pose underperforms
  - route each supported exercise to the branch that actually earned its place experimentally
- Direct video methods such as `RepNet`, `TransRAC`, `ESCounts`, and `CountLLM` remain important references, but they are better treated as:
  - literature context
  - upper-bound style comparison points
  - possible future alternatives if the project later shifts toward video-native counting

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
- For this project, the strongest current result is not one universal model but a **representation map**:
  - dedicated pose for `squat`
  - RGB for `push_up`
  - mixed evidence for `pull_up`
- That makes an exercise-dependent routed system more defensible than either:
  - forcing one shared generic model across all exercises
  - or jumping immediately to a heavy end-to-end direct video architecture

---

## 6.5. Novelty and Contribution

### What is novel here

- The project does **not** claim novelty in the sense of inventing a new state-of-the-art repetition-counting model.
- The novelty is instead in the **system-level design and validation strategy** used for this specific exercise-analysis problem.

### Project-specific contribution

- The project starts from a noisy RepCount / LLSP-style dataset and performs explicit **EDA-driven cleaning, relabeling, and preparation** before modeling.
- It validates a **pose-first architecture** for repetition counting rather than moving directly to a raw RGB video-counting model.
- It uses a **staged architecture comparison strategy**:
  - dedicated squat pose branch
  - shared pose TCN baseline and ablations
  - RGB comparison branch
  - representation-fit and hard-case audits
  - negative multimodal fusion check
  - practical routed counting prototype
- It compares two different counting backends on top of the same upstream pipeline:
  - an interpretable **FSM-based counter**
  - a learned **TCN-based temporal regressor**
- It shows that the counting backend can be replaced without changing the upstream data-preparation, pose-extraction, and feature-engineering stages.
- It also adds failure analysis showing that the remaining learned-model errors are not strongly explained by pose-quality summary metrics alone.
- It demonstrates that the useful representation is **exercise-dependent** under realistic video difficulty rather than universally pose-only or universally RGB-only.

### Why this matters

- This makes the work a meaningful **engineering and methodology contribution**, even without proposing a brand-new counting architecture.
- The project demonstrates an end-to-end, inspectable pipeline that goes from:
  - raw exercise videos
  - cleaned annotations
  - YOLO-based pose extraction
  - engineered temporal features
  - alternative counting backends
  - saved evaluation artifacts
- That combination is useful because it creates a practical foundation for the intended next step: an exercise-aware system that can decide whether pose, RGB, or a dedicated branch should be used for a supported exercise.

### Contribution statement

> The main contribution of this project is the design and validation of a modular repetition-counting pipeline for exercise videos that moves from dataset cleaning and pose extraction through comparative architecture evaluation. Rather than introducing a new foundational counting model, the project contributes a reproducible workflow that integrates a dedicated squat pose branch, shared pose baselines, RGB comparison, hard-case audits, and an exercise-dependent routed prototype, thereby establishing a practical foundation for future exercise recognition and counting.

---

## 7. Current Metrics

### Project-Level Architecture Summary

The project should now be summarized as a set of completed architecture results rather than as one squat-only metric block. The strongest measured results are:

| Architecture family | Exercise | MAE | Within-1 | Interpretation |
|---|---|---:|---:|---|
| FSM tuned baseline | `squat` | 3.0625 | 0.5625 | Historical interpretable baseline |
| Dedicated squat TCN | `squat` | 2.1405 | 0.5625 | Strongest single result in the project |
| Shared pose TCN (`6B`) | `pull_up` | 4.6088 | 0.4286 | Best shared pose-only result among the compared exercises |
| Pose transformer (`9/9B`) | `push_up` | 7.4561 | 0.0556 | Better than the shared pose TCN on `MAE`, but not competitive with the RGB winner and not improved meaningfully by augmentation |
| RGB TCN Stage `7` | `push_up` | 6.6018 | 0.2778 | Strongest current push-up branch |
| Stronger RGB TCN (`7B`) | `pull_up` | 4.1992 | 0.3571 | Best RGB pull-up `MAE`, but not best `Within-1` |
| Multimodal late fusion (`7E`) | `push_up` | 6.1691 | 0.1111 | Lower `MAE`, but not the best practical branch because `Within-1` degraded |

### Architecture Comparison: `squat`, `pull_up`, `push_up`

| Exercise | Architecture / stage | MAE | RMSE | Within-1 | Verdict |
|---|---|---:|---:|---:|---|
| `squat` | FSM tuned baseline | 3.0625 | 4.9181 | 0.5625 | Good interpretable baseline |
| `squat` | Dedicated squat TCN | 2.1405 | 3.1016 | 0.5625 | Best current squat solution |
| `squat` | Shared pose TCN (`6B`, `seq_len=256`) | 8.0430 | 11.1896 | 0.2500 | Generic pose underfits squat relative to the dedicated branch |
| `squat` | Pose transformer (`9B`, best variant) | 9.1502 | 13.1290 | 0.1250 | Worse than the shared pose TCN; augmentation did not recover squat performance |
| `squat` | RGB TCN Stage `7` (`ResNet18`) | 6.5446 | 8.2765 | 0.0625 | Better `MAE` than shared pose, but far below the dedicated squat branch |
| `squat` | Stronger RGB TCN (`7B`, `ResNet50`) | 5.4245 | 6.8711 | 0.1875 | Best generic non-squat-specific alternative, still far below the dedicated pose branch |
| `squat` | Multimodal late fusion (`7E`) | 6.6988 | 8.2765 | 0.0000 | Negative result |
| `pull_up` | Shared pose TCN (`6B`, `seq_len=192`) | 4.6088 | 7.0169 | 0.4286 | Best `Within-1` result so far |
| `pull_up` | Pose transformer (`9B`, aug_on) | 5.0210 | 6.5912 | 0.0714 | Slightly better `MAE` than transformer `aug_off`, but much worse `Within-1` than shared pose |
| `pull_up` | RGB TCN Stage `7` (`ResNet18`) | 4.8686 | - | 0.1429 | RGB baseline weaker than pose |
| `pull_up` | Stronger RGB TCN (`7B`, `ResNet50`) | 4.1992 | 5.8931 | 0.3571 | Best RGB `MAE`, but not the best practical branch on both metrics |
| `pull_up` | Multimodal late fusion (`7E`) | 4.1193 | 5.2182 | 0.1429 | `MAE` gain, but `Within-1` collapsed too much |
| `pull_up` | Dedicated `pull_up` pose (`10`, `channels128`) | 3.5463 | 5.5040 | 0.2857 | Best `MAE`, but still not the best practical branch on both metrics |
| `push_up` | Shared pose TCN (`6B`, `seq_len=128`) | 8.8724 | 11.2798 | 0.0000 | Pose-only remains weak |
| `push_up` | Pose transformer (`9B`, aug_off) | 7.4561 | 9.9192 | 0.0556 | Better than shared pose, but still behind the RGB branch |
| `push_up` | Keypoint-weighted pose TCN (`6C`) | 8.5139 | - | 0.1111 | Small isolated pose-side improvement |
| `push_up` | Density pose TCN (`6D`) | 8.6100 | - | 0.0000 | No meaningful recovery |
| `push_up` | RGB TCN Stage `7` (`ResNet18`) | 6.6018 | 10.2865 | 0.2778 | Best current practical push-up branch |
| `push_up` | Stronger RGB TCN (`7B`, `ResNet50`) | 7.3768 | - | 0.0556 | Worse than Stage `7` RGB |
| `push_up` | Multimodal late fusion (`7E`) | 6.1691 | 9.8993 | 0.1111 | Better `MAE`, but still worse than Stage `7` RGB on `Within-1` |

### Squat-Only TCN Experiment

- A squat-only `TCN / 1D CNN` regressor was trained as a learned alternative to the FSM backend while keeping the same upstream Colab 4 and Colab 5 pipeline.
- The initial TCN baseline was then followed by controlled runs on:
  - `loss = l1`
  - `seq_len = 192`
  - metric-aware checkpointing
  - wider channel settings
  - targeted dropout and learning-rate changes
- Variable-length videos were handled by temporal resampling rather than by random window crops:
  - each full feature sequence was resampled to the fixed target length used by the TCN
  - the model therefore received a compressed whole-video representation rather than only a short partial clip
  - this preserves whole-video coverage for count prediction, but it also means that long videos are temporally compressed, so the resampling strategy is itself an important methodological choice
- The strongest final learned run from `6_TCN_Training_Colab.ipynb` is:
  - `run_name = squat_tcn_l1_channels96`
  - `channels = 96`
  - `lr = 0.001`
  - `dropout = 0.2`
  - `loss = l1`
  - `best_epoch = 53`
  - `valid rows = 16`
  - `valid MAE = 2.140498`
  - `valid RMSE = 3.101630`
  - `valid Within-1 = 0.5625`
- This run outperformed the other targeted TCN variants in overall balance:
  - best `MAE` in the final targeted comparison
  - near-identical `RMSE` to the previous best
  - stronger `Within-1` than the lower-learning-rate, lower-dropout, and larger-width alternatives

### FSM vs TCN Comparison

| Model | Split | Rows | MAE | RMSE | Within-1 |
|---|---|---:|---:|---:|---:|
| `FSM (tuned)` | `valid` | 16 | 3.0625 | 4.9181 | 0.5625 |
| `TCN (best current)` | `valid` | 16 | 2.1405 | 3.1016 | 0.5625 |

### Interpretation

- The best tuned squat-only TCN improved validation `MAE` and `RMSE` relative to the tuned FSM baseline, which suggests better average count estimation.
- In the latest run, the TCN also matched the tuned FSM on `Within-1`, which means it no longer gives up strict near-exact accuracy to achieve the lower average error.
- Both `MAE` and `Within-1` are error-oriented quality measures, but they emphasize different behavior:
  - `MAE` measures the average size of the counting error
  - `Within-1` measures how often the prediction lands within one repetition of the true count
- Therefore, it is possible for `MAE` to improve while `Within-1` decreases if the model reduces large mistakes overall but produces fewer near-exact predictions.
- In practical terms, this means the learned TCN now provides clearly better average count quality while also matching the FSM on strict tolerance-based performance in the current validation split.
- Additional failure analysis in `6_TCN_Training_Colab.ipynb` also showed that the remaining TCN errors are only weakly correlated with `frames_valid`, `mean_conf`, and `valid_ratio`, which suggests that the remaining misses are not strongly driven by poor pose extraction alone.
- This comparison supports the conclusion that a learned backend is viable, but metric selection still matters because different models optimize different aspects of counting quality.

### Failure Analysis Summary

The reviewed hard-case layer is now complete for the selected `7D` subset: `48 / 48` hard cases were manually reviewed and summarized in `reviewed_hard_case_summary.json`. The confirmed failure taxonomy is more precise than the earlier heuristic pass. Across the reviewed cases, the largest buckets were `pose_failure = 11`, `camera_viewpoint = 8`, `rep_ambiguity = 6`, `model_failure = 6`, `target_selection = 5`, `label_mismatch = 3`, and `visibility = 3`, with `6` additional rows left as broadly difficult but not assigned a narrower primary issue. The dominant confirmed tags were `pose_jitter = 32`, `no_clear_issue = 32`, `camera_motion = 16`, `reframe = 9`, and `side_view = 9`, which makes the residual error surface much more concrete than a simple “bad videos” explanation.

The exercise-specific pattern is also now clearer. For `squat`, the reviewed hard cases concentrated on `rep_ambiguity`, `pose_failure`, and a smaller number of clear `label_mismatch` rows, which supports the interpretation that squat remains pose-friendly but still sensitive to borderline rep definitions and partial clip boundaries. For `pull_up`, the dominant reviewed issues were `camera_viewpoint` and `target_selection`, which is consistent with the earlier “mixed” conclusion: the difficulty is not simply weak pose confidence, but long repetitive sequences filmed from harder viewpoints and sometimes with ambiguous target tracking. For `push_up`, the reviewed rows leaned toward `pose_failure`, `model_failure`, and a block of broadly low-quality cases, which reinforces the earlier finding that push-up is the exercise where raw RGB context carries the strongest practical value.

Taken together, the completed reviewed audit shows that the remaining counting errors are driven by a mixture of viewpoint/framing difficulty, true pose-tracking failures, and rep-definition ambiguity rather than by a single uniform model weakness. This strengthens the project’s main architectural conclusion: a routed, exercise-dependent system is more defensible than a forced generic counter, because the residual failure modes differ materially by exercise and by representation.

The reviewed cases and keep/flag/exclude decisions are preserved in `artifacts/3_Modeling/validation_failure_review.md` so that future reruns can separate true upstream failures from valid but harder benchmark videos.

### Results Summary

The current squat-only prototype can now be treated as a consolidated first-iteration result rather than an open-ended tuning exercise. The tuned FSM remains the interpretable rule-based baseline, while the tuned TCN provides the strongest learned alternative on the same upstream pose and feature pipeline.

At the validation level, the tuned FSM achieved:
- `MAE = 3.0625`
- `RMSE = 4.9181`
- `Within-1 = 0.5625`

The best learned TCN configuration, `squat_tcn_l1_channels96`, achieved:
- `MAE = 2.1405`
- `RMSE = 3.1016`
- `Within-1 = 0.5625`

After applying the reviewed validation policy and excluding the confirmed unusable upstream failure (`train3898.mp4`), the filtered validation view of the same frozen TCN baseline became:
- `rows = 15`
- `MAE = 2.0893`
- `RMSE = 3.1141`
- `Within-1 = 0.6000`

Taken together, these results show that the learned TCN substantially improves average counting accuracy relative to the FSM baseline while also matching the FSM on the stricter near-exact tolerance metric in the raw validation split. The policy-filtered view further suggests that at least part of the remaining error was driven by a confirmed unusable upstream case rather than by the learned counting model itself. This means the project has already validated two important points:
- the upstream pose-based architecture is workable end to end
- the counting backend can be replaced by a learned temporal model without changing the upstream pipeline

Therefore, the present squat-only system should be reported as a validated prototype with both an interpretable baseline and a learned temporal baseline, rather than as an unfinished exploratory branch.

### Shared Stage 6 Widened Counting Baseline

After rebuilding the full generic pose-sequence dataset from `pose_feature_index.csv`, the first shared Stage 6 baseline trained a separate pose-sequence TCN regressor for each exercise using the same common configuration (`seq_len = 192`, `channels = 96`, `L1` loss, balanced count sampling, and light augmentation). The resulting shared baseline is weaker than the frozen squat-specific branch in absolute terms, but it still provides a meaningful reference point for the widened project. Across the nine supported exercises (`battle_rope`, `bench_pressing`, `front_raise`, `jump_jacks`, `pommelhorse`, `pull_up`, `push_up`, `sit_up`, and `squat`), the shared baseline achieved a lower macro-average `MAE` than the trivial train-split mean-count baseline (`7.97` versus `10.33`) and a higher macro-average `Within-1` score (`0.155` versus `0.064`). It beat the trivial baseline on `MAE` for seven of the nine exercises and improved `Within-1` for six of the nine.

The per-exercise results were strongly heterogeneous. `pull_up`, `bench_pressing`, and `pommelhorse` showed the clearest positive signal, while `jump_jacks`, `sit_up`, and the generic shared `squat` run remained weak in absolute terms. The shared `squat` baseline still improved on the trivial squat baseline, but it remained far worse than the dedicated frozen squat branch, which indicates that a shared generic pose-sequence setup loses important exercise-specific structure. This widened Stage 6 result should therefore be treated as the official shared baseline for the multi-exercise branch, not as a final counting solution. Its main value is to identify which exercises appear learnable from pose alone and to define the next experiment path: targeted, exercise-by-exercise improvement starting with sequence-length sweeps rather than broader untargeted retuning.

### RGB, Transformer, Audit, and Multimodal Results

The Stage `7` branch changed the project from “pose-only tuning” to “representation comparison.” The main results were:

- `squat`:
  - RGB improved over the weak shared pose baseline in `MAE`
  - but stayed far below the frozen dedicated squat pose branch
  - `7C` and `7D` showed that squat remains pose-first even when pose quality is already strong
- `pull_up`:
  - RGB became competitive under the stronger `ResNet50` backbone
  - but the best practical branch still depends on whether `MAE` or `Within-1` is prioritized
- `push_up`:
  - RGB clearly outperformed the shared pose branch
  - `7C` showed that RGB was not only rescuing weak pose, but also contributing information beyond pose quality alone

The simple late-fusion multimodal experiment (`7E`) did not justify itself. It improved `MAE` in some cases, but degraded `Within-1` enough that it did not beat the better single-modality branch consistently. This means the project should not present multimodal late fusion as a successful next architecture.

The generic pose-transformer check (`9/9B`) also closed without changing the overall architecture decision. The best transformer variants:

- did not beat the shared pose TCN on the practical `pull_up` tradeoff
- remained far below the dedicated squat pose branch
- improved `push_up` relative to the shared pose TCN, but still stayed well behind the RGB branch

The augmentation ablation inside `9B` was also not a success case. It showed:

- `pull_up`: slightly lower `MAE`, but `Within-1` fell from `0.2143` to `0.0714`
- `push_up`: no meaningful change
- `squat`: no meaningful change

So `9B` should be treated as a completed negative-but-informative result rather than an open branch.

### Current Practical Counting Surface

The most defensible practical artifact in the repo is now the Stage `8` routed system:

- `squat -> squat_tcn_l1_channels96`
- `pull_up -> pose_count_tcn_pull_up_seq192`
- `push_up -> rgb_count_tcn_push_up_seq128`

This routed system is important because it reflects the experimental evidence directly instead of forcing one architecture family to win across all supported exercises.

### Stage 11 Bootstrap Confidence Intervals

Stage `11` added bootstrap confidence-interval reporting for the final reportable prediction artifacts. The completed CI outputs were:

- `squat` dedicated pose control (`squat_tcn_l1_channels96`, `n = 16`)
  - `MAE = 2.1405`, `95% CI [1.1266, 3.3313]`
  - `RMSE = 3.1016`, `95% CI [1.6982, 4.2837]`
  - `Within-1 = 0.5625`, `95% CI [0.3125, 0.8125]`

- `pull_up` routed pose branch (`pose_count_tcn_pull_up_seq192`, `n = 14`)
  - `MAE = 4.6088`, `95% CI [2.0863, 7.5386]`
  - `RMSE = 7.0169`, `95% CI [3.5909, 9.7687]`
  - `Within-1 = 0.4286`, `95% CI [0.2143, 0.7143]`
- `push_up` routed RGB branch (`rgb_count_tcn_push_up_seq128`, `n = 18`)
  - `MAE = 6.6018`, `95% CI [3.3063, 10.4238]`
  - `RMSE = 10.2865`, `95% CI [5.1748, 14.8974]`
  - `Within-1 = 0.2778`, `95% CI [0.0556, 0.5000]`

These intervals reinforce the same interpretation already reached from the point estimates: the reported branches are directionally meaningful, but the small validation surfaces produce wide uncertainty ranges and therefore do not support over-precise external claims.

### Methodological Concerns and Responses

- **Concern: the squat validation split is very small.**  
  **Response:** This is a real limitation of the current evaluation. The primary validation split contains only `16` videos, so the reported `MAE`, `RMSE`, and `Within-1` values are fragile and can shift noticeably when one or two unusual videos are present. This was demonstrated directly when excluding one confirmed upstream failure (`train3898.mp4`), which changed `Within-1` from `0.5625` to `0.6000`.

- **Concern: the report currently gives point estimates but not uncertainty ranges.**  
  **Response:** This criticism has now been addressed for the main reportable branches. A reusable bootstrap utility was added and Stage `11` produced confidence intervals for the dedicated `squat` control plus the routed `pull_up` and `push_up` branches. Those intervals are wide, which supports the document's caution that these evaluation slices are useful for scoped conclusions but not for narrow external claims.

- **Concern: would cross-validation over train plus valid be more reliable?**  
  **Response:** Yes, cross-validation over the current `train + valid` pool, while keeping `test` fully held out, would likely produce more stable project-level estimates than a single `16`-video validation split. However, that would become a project-specific evaluation protocol rather than a direct reproduction of the current fixed-split benchmark setup.

- **Concern: the TCN sequence length may create a temporal mismatch with whole-video count targets.**  
  **Response:** This should be documented explicitly. The TCN uses `seq_len = 192`, while the source squat videos are often much longer. In the current trainer, variable-length videos are not randomly cropped to `192` frames; instead, the full sequence is temporally resampled to a fixed length before training and evaluation. That design preserves whole-video coverage, but it also compresses longer temporal structure, so the resampling choice has real implications for count prediction and should be treated as a methodological decision rather than an implementation detail.

- **Concern: `Within-1 = 0.5625` needs stronger contextual interpretation.**  
  **Response:** This is correct. A raw `Within-1` value of `0.5625` means the model lands within one repetition of the true count in only about `56%` of the raw validation videos, which would likely be insufficient for a finished fitness product. At the same time, this number should not be over-interpreted as a direct benchmark comparison because published RepCount papers typically report dataset-level Part-A or RepCount-pose results, not the same `16`-video squat-only validation slice used here. In the RepCount literature, `OBO` is effectively the same idea as `Within-1`, so a separate `OBO` value is not an additional metric in this setting. Because the dataset also contains broad count variability and outliers, `Within-1` is best treated as a supplementary strict-tolerance metric, while `MAE` and `RMSE` remain important for understanding the magnitude of residual counting error.

- **Concern: class imbalance and count outliers may be distorting the squat-only evaluation.**  
  **Response:** This needs to be interpreted carefully. The large class-imbalance ratio discussed in EDA is mainly a multi-exercise concern and does not directly affect the current squat-only evaluation, because the current model is not classifying across exercises. The count-outlier observation is more relevant, but the earlier flagged outliers were identified at the broader dataset level rather than in the current squat validation slice. In the frozen `16`-video squat validation set, the highest counts are `44` and `39`, so the current valid split does not contain the most extreme count outliers described in EDA. However, these relatively high-count squat videos can still increase `MAE` and `RMSE` sensitivity, because the same relative counting mistake produces a larger absolute error when the true repetition count is high.

### Reporting Improvement Still Open

- If stronger external rigor is needed later, extend the same CI reporting to any additional final branches or aggregate routed summaries beyond the current reportable control/routed set.
- If the reviewed hard-case taxonomy is to be quoted externally as a final quantitative result, it is still worth normalizing the small `unspecified` bucket into narrower issue labels for maximum consistency.

---

## 8. Next Steps

### Immediate Next Steps

- Keep Stage `9B` closed as a completed negative result; do not reopen the generic pose-transformer branch unless a new representation question justifies it.
- Keep the frozen squat branch as the control baseline for any future squat comparison.
- Keep the routed Stage `8` system as the practical surface while the remaining research questions are resolved.
- Use the completed reviewed hard-case summary from `7D` when discussing remaining failures, instead of relying only on heuristic buckets or anecdotal examples.
- Only after that, decide whether a dedicated `push_up` pose branch is worth building as a falsification test against the current RGB winner.
- Keep Countix deferred until a later external-validation question justifies the added scope.

### Error Analysis Checklist

To interpret the current TCN results correctly, the next evaluation pass should inspect `predictions.csv` and focus on the `valid` examples:

1. sort the `valid` rows by `abs_error`
2. identify the worst-counted validation videos
3. compare `pred_count` against `true_count`
4. determine whether the TCN errors are mostly moderate misses or a few large failures
5. manually inspect whether hard videos include contextual variation such as occlusion, exercise assistance, or atypical squat execution
6. separate the remaining errors into:
   - pose / feature quality issues
   - contextual or biomechanical variation
   - model counting behavior or target ambiguity

### Mid-Term Next Steps

- Stop broad untargeted tuning on the already-completed negative branches (`6C`, `6D`, `7E`) unless a new failure pattern justifies revisiting them.
- Keep the squat branch as the validated prototype and baseline for comparison.
- Improve the supported exercises one by one rather than forcing a generic winner too early.
- Finish the `pull_up` decision first, then decide whether `push_up` deserves one dedicated pose-side falsification run.
- Keep the representation choice explicit in the documentation:
  - pose-first for `squat`
  - RGB-first for `push_up`
  - unresolved / comparison-sensitive for `pull_up`

### Evaluation Plan For The Widened Stage

- Keep the original dataset split semantics:
  - `train` for fitting the widened model
  - `valid` for model selection, failure analysis, and iterative comparison
  - `test` held out during development
- Use the widened `train + valid` data only for internal development until the multi-exercise counting pipeline is stable enough to justify a final held-out test evaluation.
- If cross-validation is later introduced for more stable internal estimates, treat that as a project-specific secondary analysis and keep the official `test` split untouched until the final reporting stage.
- Report widened results at two levels once Stage 6 exists:
  - aggregate multi-exercise metrics across the included classes
  - per-exercise metrics where the sample size is large enough to be interpretable
- Reuse the same discipline established in the squat branch:
  - keep raw evaluation outputs
  - document any manual keep/flag/exclude policy separately from the raw benchmark view

### Long-Term Next Steps

- Generalize from the routed prototype toward an exercise-aware system that can first identify the exercise and then call the right counting branch.
- Build an explicit exercise-recognition stage on top of the existing pose or RGB front ends.
- Revisit stronger pose-native models such as `ST-GCN`, `BlockGCN`, or `SkeleTR` only after the current exercise-dependent baseline is frozen.
- Revisit stronger video-native RGB models only if the project intentionally shifts away from the current frozen-feature RGB branch.
- Reduce notebook dependency and centralize experiment execution and registration.
- Move toward broader deployment settings only after the exercise-dependent branch map is stable.

### Proposed Next Architecture

The current practical architecture is already exercise-dependent at the counting stage. The next architecture should therefore not undo that evidence by forcing a single shared counter too early. Instead, it should explicitly separate three decisions that are currently entangled in the video: identifying the target person, identifying the exercise or movement being performed, and selecting or applying the right repetition-counting branch. The recommended long-term direction is therefore a two-stage design in which the system first selects the relevant exercising person and recognizes the exercise, and only then applies the appropriate counting model to that chosen target.

In practical terms, the intended next architecture is:

`video -> target person selection / tracking -> exercise recognition -> branch selection -> repetition counting`

The pose-based version of this architecture would keep the validated YOLO frontend, use a temporal classifier to identify the exercise from the selected person's pose sequence, and then either route to the strongest saved branch or apply a future exercise-specific counter to the same selected target. This design is better aligned with the real project goal than applying one generic counting model directly to the whole scene, especially in videos that contain multiple people, unrelated motion, occlusion, or obstacles.

This also establishes a clear design principle for future work: counting should be applied only after the system has identified the correct subject and the correct exercise context. In other words, the next-stage system is expected to be a detect-first-then-count pipeline rather than a single counting model applied to the full video without subject or activity disambiguation.

### Candidate Models For The Next Iteration

The next modeling iteration is no longer best framed as “find one generic counter.” Instead, the open design question is which temporal models should be used for:

- exercise classification
- branch strengthening where the current routed system is still weak or unresolved

If YOLO Pose is retained as the front end, the shared recognition target architecture would be:

`video -> YOLO pose -> pose sequence -> shared temporal model -> exercise class`

The candidate models listed below are therefore restricted to approaches that are well aligned with a **YOLO-pose-first pipeline** and are most relevant either for future exercise recognition or for stronger pose-native branch development. Architectures such as `CNN + LSTM` were intentionally not prioritized here because they are more naturally suited to raw RGB frame sequences than to already-extracted pose/keypoint sequences.

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
- Better as a controlled follow-up after the pose TCN and RGB branches, using the same Stage 5 pose-sequence contract and augmentation path

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

- Keep the current dedicated squat TCN as the squat control baseline.
- Keep the shared pose TCN as the generic pose reference, not as the final universal model.
- Use the Stage `8` routed system as the presentable practical counting surface.
- Recommended order of exploration now:
  - freeze the dedicated `pull_up` result as informative but mixed
  - freeze the transformer augmentation evidence from `9B` as a negative result
  - only then decide whether a dedicated `push_up` pose branch is worth building
  - defer heavier skeleton models and video-native RGB models until the exercise-dependent baseline is frozen

This recommendation supports the broader project goal of moving from architecture exploration toward a stable exercise-dependent counting system that can later be expanded with exercise recognition.

---

## 9. References

### Project References

- `README.md`
- `artifacts/specification.md`
- `artifacts/2_Data_preparation/COUNTIX_INTEGRATION.md`
- `artifacts/2_Data_preparation/prepare_countix_manifest.py`
- `artifacts/3_Modeling/YOLO_PIPELINE.md`
- `artifacts/3_Modeling/YOLO_POSE_STAGE.md`
- `artifacts/3_Modeling/validation_failure_review.md`
- `artifacts/3_Modeling/EXPERIMENT_SHOWCASE.md`
- `artifacts/3_Modeling/ARCHITECTURE_RESULTS_MATRIX.md`
- `artifacts/3_Modeling/experiment_registry.csv`
- `artifacts/3_Modeling/4_All_Exercises_Pose_Extraction_Colab.ipynb`
- `artifacts/3_Modeling/5_All_Exercises_Pose_Sequence_Preparation_Colab.ipynb`
- `artifacts/3_Modeling/6_All_Exercises_Counting_Baseline_Colab.ipynb`
- `artifacts/3_Modeling/6_Squat_Rep_Counting_Colab.ipynb`
- `artifacts/3_Modeling/7_RGB_Counting_Baseline_Colab.ipynb`
- `artifacts/3_Modeling/7B_Stronger_RGB_Backbone_Colab.ipynb`
- `artifacts/3_Modeling/7C_Representation_Fit_Analysis_Colab.ipynb`
- `artifacts/3_Modeling/7D_Hard_Case_Data_Audit_Colab.ipynb`
- `artifacts/3_Modeling/7E_Multimodal_Pose_RGB_Fusion_Colab.ipynb`
- `artifacts/3_Modeling/8_Exercise_Dependent_Counting_Colab.ipynb`
- `artifacts/3_Modeling/9_Pose_Transformer_Colab.ipynb`
- `artifacts/3_Modeling/9B_Pose_Transformer_Augmentation_Ablation_Colab.ipynb`
- `artifacts/3_Modeling/10_PullUp_Dedicated_Pose_Colab.ipynb`

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
- `2026-03-13`: Updated Stage 4 and Stage 5 from the original `4_Squat_Pose_Extraction_Colab.ipynb` workflow, including the Drive-backed squat index, Colab 4 smoke test, GPU-based YOLO inference, and the successful `118/118` squat pose-extraction result.
- `2026-03-13`: Updated Stage 6 from `5_Squat_Feature_Extraction_Colab.ipynb`, including the Drive-based feature pipeline, saved squat feature artifacts, and the successful `118/118` engineered-feature run with high lower-body confidence.
- `2026-03-13`: Updated Stage 7, Stage 8, and the metrics section from `6_Squat_Rep_Counting_Colab.ipynb` and synced Colab 6 CSV outputs, including baseline results, tuning artifacts, and the current saved tuned train/valid metrics.
- `2026-03-13`: Added the squat-only TCN training branch, including the Colab notebook, trainer script, TCN metrics, and an FSM-versus-TCN comparison in the draft.
- `2026-03-13`: Updated the draft with the then-current `6_TCN_Training_Colab.ipynb` tuning results, promoted `squat_tcn_l1_channels96_dropout01` as the best learned squat baseline at that time, and added the failure-analysis conclusion that remaining TCN errors are only weakly correlated with pose-quality summary metrics.
- `2026-03-23`: Updated the draft after the latest Colab 6 rerun, promoted `squat_tcn_l1_channels96` as the strongest learned squat baseline (`MAE = 2.1405`, `RMSE = 3.1016`, `Within-1 = 0.5625`), and added the manual failure-analysis summary for the hardest validation videos.
- `2026-03-26`: Added the Stage 9 pose-transformer experiment scaffolding, reusing the Stage 6 augmentation path and the same artifact contract as the pose TCN runs for direct comparison.
- `2026-03-26`: Added `register_experiment.py` so new stages can append or update `experiment_registry.csv` without editing the registry manually.
- `2026-03-26`: Added the Stage 9B augmentation-ablation notebook so the transformer branch can be evaluated with the current pose augmentation on versus off under otherwise matched conditions.
- `2026-03-27`: Updated the draft around the completed architecture study, including the RGB, stronger RGB, audit, multimodal, and routed-counting results, and rewrote the metrics and next-steps sections around the current exercise-dependent conclusion.
- `2026-03-27`: Added the new project references and updated Stage `10` as the dedicated `pull_up` pose follow-up, with the result that `pull_up` remains mixed rather than producing a clear new pose-specialized winner.
- `2026-03-27`: Finalized the Stage `9B` augmentation-ablation result: the generic pose-transformer branch did not produce a robust validation gain, and augmentation did not improve the branch consistently enough to change the project conclusion.
- `2026-03-27`: Added `bootstrap_count_confidence_intervals.py` plus a lightweight regression test so small validation splits can be reported with bootstrap confidence intervals instead of point estimates alone.
- `2026-03-27`: Added Stage `11` as a Colab surface for running bootstrap confidence intervals on the final reportable prediction artifacts in Drive.
- `2026-03-27`: Added the completed Stage `11` CI results to the draft for the dedicated `squat` control and the routed `pull_up` and `push_up` branches.
- `2026-03-27`: Added `build_hard_case_review_manifest.py` and `summarize_reviewed_hard_cases.py` so the heuristic `7D` audit can be converted into a reviewed hard-case annotation layer.
- `2026-03-27`: Added Countix onboarding scaffolding as a separate benchmark branch under `Data/Countix`, then deferred it from the active experiment flow so the current LLSP conclusions remain the main project surface.

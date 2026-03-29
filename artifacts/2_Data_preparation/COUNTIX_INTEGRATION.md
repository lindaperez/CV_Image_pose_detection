# Countix Integration

This repo is currently centered on the `LLSP` exercise-video pipeline, but the
core extraction scripts already support explicit CSV and path arguments. That
means `Countix` can be added as a **separate benchmark branch** without forcing
it into the cleaned `LLSP` annotation files.

Current status:

- Countix is prepared as a benchmark-extension branch
- Countix is **not** required for the current LLSP conclusions
- Countix is currently deferred until a later external-validation or transfer question justifies the added scope

## Local layout

Recommended folder structure:

```text
CV_Image_pose_detection/Data/Countix/
├── annotation_cleaned/
│   └── countix_manifest.csv
└── video/
```

## Step 1. Normalize Countix metadata

Use this only when you intentionally activate the deferred Countix branch.

Convert the raw Countix metadata CSV into the repo's manifest contract:

```bash
python3 CV_Image_pose_detection/artifacts/2_Data_preparation/prepare_countix_manifest.py \
  --input-csv CV_Image_pose_detection/Data/Countix/raw/countix_metadata.csv \
  --output-csv CV_Image_pose_detection/Data/Countix/annotation_cleaned/countix_manifest.csv \
  --video-dir CV_Image_pose_detection/Data/Countix/video
```

By default, this step now:

- maps Countix / Kinetics-style action names into the repo label space
- keeps only the currently supported exercise set:
  - `squat`
  - `pull_up`
  - `push_up`
  - `sit_up`
  - `bench_pressing`
  - `front_raise`
  - `jump_jacks`
  - `battle_rope`
  - `pommelhorse`
- prints a summary of any dropped Countix classes

If you want to keep all Countix classes for a broader benchmark manifest, add:

```bash
  --disable-target-filter
```

Expected normalized columns:

- `dataset`
- `name`
- `type`
- `split`
- `count`
- `video_path`
- `source_id`
- `source_type`

Only `name`, `type`, `split`, and `count` are required by the current pose
indexing script. `video_path` is preserved for future video-native work.

## Step 2. Build the pose feature index

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/build_pose_feature_index.py \
  --annotation-csv CV_Image_pose_detection/Data/Countix/annotation_cleaned/countix_manifest.csv \
  --feature-dir CV_Image_pose_detection/Data/Countix/annotation_cleaned/pose_features \
  --output-csv CV_Image_pose_detection/Data/Countix/annotation_cleaned/pose_feature_index.csv
```

## Step 3. Extract pose features

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/pose_feature_extraction.py \
  --index-csv CV_Image_pose_detection/Data/Countix/annotation_cleaned/pose_feature_index.csv \
  --video-dir CV_Image_pose_detection/Data/Countix/video \
  --report-path CV_Image_pose_detection/Data/Countix/annotation_cleaned/pose_extraction_report.csv \
  --summary-path CV_Image_pose_detection/Data/Countix/annotation_cleaned/pose_extraction_summary.json \
  --device cuda
```

## Step 4. Build normalized pose sequences

```bash
python3 CV_Image_pose_detection/artifacts/3_Modeling/build_pose_sequence_dataset.py \
  --index-csv CV_Image_pose_detection/Data/Countix/annotation_cleaned/pose_feature_index.csv \
  --sequence-dir CV_Image_pose_detection/Data/Countix/annotation_cleaned/pose_sequences \
  --output-index-csv CV_Image_pose_detection/Data/Countix/annotation_cleaned/pose_sequence_index.csv \
  --output-summary-csv CV_Image_pose_detection/Data/Countix/annotation_cleaned/pose_sequence_summary.csv
```

## Notes

- Keep `Countix` separate from `LLSP`.
- Do not merge Countix rows into `train_cleaned.csv` / `valid_cleaned.csv`.
- Treat Countix as a benchmark-expansion branch for stronger external evidence,
  not as a silent replacement for the current exercise-focused pipeline.
- Do not download or process Countix by default; keep it off the critical path unless a later scoped benchmark question requires it.

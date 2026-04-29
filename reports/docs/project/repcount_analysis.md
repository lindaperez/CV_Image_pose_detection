# RepCount Dataset — Description & Analysis

## Overview
  
RepCount is a **repetition action counting dataset** developed by Gao's Lab at ShanghaiTech University. It contains videos with significant variations in length and supports multiple anomaly cases, with fine-grained annotations marking the beginning and end of each action cycle. In total, it provides **1,451 videos** with **19,280 annotations**, and videos average **39.36 seconds** in length.

- **Official Page:** https://svip-lab.github.io/dataset/RepCount_dataset.html
- **GitHub:** https://github.com/SvipRepetitionCounting/TransRAC

---

## Dataset Structure

The dataset is divided into **two subsets**:

| Subset | Source | Split |
|---|---|---|
| **Part-A** | YouTube (1,041 videos) | Train / Validation / Test |
| **Part-B** | Recorded volunteers (junior school students & teachers) | Test only |

Part-A includes workout activities (squatting, pull-ups, front raising, etc.), athletic events (rowing, pommel horse), and other repetitive actions like soccer juggling. Part-B records exercises such as sit-ups and pull-ups done by volunteers, and is intended specifically to **test model generalization**.

---

## Annotation Files (CSV/TXT Structure)

The dataset ships with annotation files per split: `train.csv`, `valid.csv`, `test.csv`.

### Column Descriptions

| Column | Name | Description |
|---|---|---|
| 1 | `name` / `video_id` | Filename or unique identifier of the video clip |
| 2 | `url` | Original YouTube URL (for Part-A) |
| 3 | `type` / `action` | Action category label (e.g., `squat`, `pullup`, `rowing`) |
| 4 | `num_frames` | Total number of frames in the video |
| 5 | `count` | Ground truth **total number of repetition cycles** in the video |
| 6 | `start_X` / `end_X` | Frame-level timestamps of the **start and end of each individual cycle** (multiple column pairs, one per repetition) |

> **Note:** The start/end columns are dynamic — a video with 10 repetitions will have 10 `start`/`end` column pairs. Videos with fewer repetitions will have empty/null values in the remaining columns.

### Row Description

Each **row** = one video clip, containing:
- Its metadata (source URL, action type, frame count)
- Its **global count** (total repetitions)
- Its **temporal localization** (where each repetition begins and ends, in frame numbers)

---

## Annotation Types

The labels are composed of two main parts:

1. **Count** — the total number of repetition cycles in the video
2. **Location** — the temporal position of each cycle on the time axis (start and end frame of every individual repetition)

This dual annotation supports two levels of tasks:
- **Counting tasks** — predict how many repetitions occurred
- **Localization tasks** — predict *when* each repetition happened

---

## Key Research Considerations

| Aspect | Implication for Research |
|---|---|
| **Variable-length videos** | Models must handle temporal scale variation |
| **Anomaly cases** | Incomplete or irregular repetitions are included — models must be robust |
| **Two-part structure** | Part-B enables cross-domain generalization testing |
| **Fine-grained temporal labels** | Enables both counting and segmentation/localization benchmarks |

---

## Citation

```
@article{hu2022transrac,
  title={TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting},
  author={Hu, Huazhang and Dong, Sixun and Zhao, Yiqun and Lian, Dongze and Li, Zhengxin and Gao, Shenghua},
  journal={arXiv preprint arXiv:2204.01018},
  year={2022}
}
```

---

## Recommended Next Steps

1. **Download the dataset** and run a statistical analysis on the `count` column to understand the distribution of repetition counts.
2. **Inspect the action type distribution** to assess class imbalance.
3. **Review the TransRAC paper** as the primary baseline method for this dataset.
4. **Compare with related datasets** (e.g., UCFRep, Countix) to contextualize RepCount's contributions.

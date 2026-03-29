## Experiment Showcase

Last updated: March 27, 2026

This artifact summarizes the main repetition-counting experiments completed so far, the question each stage answered, the strongest observed result, and the decision taken afterward. It is intended as the shortest stable record of the project's experimental arc.

### Current high-level conclusion

The project did not converge to one strong generic counter across exercises. The evidence now supports an exercise-dependent view:

- `squat`: dedicated pose branch remains clearly best
- `push_up`: RGB branch is strongest so far
- `pull_up`: mixed, with no decisive shared winner

The next practical direction is therefore an exercise-dependent routed system rather than another generic shared model.

The most recent targeted modeling follow-up was a dedicated `pull_up` pose-tuning stage on the shared pose sequences:
- [10_PullUp_Dedicated_Pose_Colab.ipynb](/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection/artifacts/3_Modeling/10_PullUp_Dedicated_Pose_Colab.ipynb)

### Experiment timeline

| Stage | Question | Setup | Key result | Decision |
| --- | --- | --- | --- | --- |
| Frozen squat baseline | Can a dedicated single-exercise pose branch count squat repetitions reliably on the project validation slice? | Squat-specific pose features + squat TCN | `MAE = 2.1405`, `RMSE = 3.1016`, `Within-1 = 0.5625` on the frozen squat validation surface | Keep as control baseline for all later squat comparisons |
| `6_ALL` shared pose baseline | Can a generic pose-sequence TCN scale beyond squat across exercises? | Stage 5 normalized pose sequences + per-exercise TCN runs | Beat the trivial baseline on `MAE` for `7 / 9` exercises, but absolute errors remained high | Freeze as the shared pose baseline, not the final answer |
| `6B` seq-length sweep | Is temporal compression the main bottleneck? | Per-exercise `seq_len` sweep on the strongest shared pose setup | `seq_len` mattered, but gains were modest and exercise-dependent | Treat `seq_len` as a secondary lever, not the main fix |
| `6C` keypoint weighting | Can hand-crafted exercise-specific pose emphasis materially improve counting? | Same pose TCN with weighted keypoint profiles | Mostly negative; only `push_up` showed a small isolated gain | Stop broad manual weighting work |
| `6D` density counting | Does explicit temporal density prediction fix the scalar-regression weakness? | Weakly supervised pseudo-density TCN over pose sequences | Mostly negative; pseudo-density targets were too weak to beat `6B` | Do not adopt weak density supervision as the new baseline |
| Stage `7` RGB baseline | Does RGB recover counting signal that pose alone misses? | Frozen `ResNet18` RGB features + RGB TCN on `squat`, `pull_up`, `push_up` | `push_up` improved clearly, `pull_up` stayed pose-friendly, `squat` improved `MAE` vs shared pose but stayed far below the dedicated squat baseline | Continue RGB branch, but only as an exercise-dependent comparison |
| `7B` stronger RGB backbone | Was Stage 7 limited by weak RGB representation rather than RGB as a modality? | Frozen `ResNet50` RGB features + RGB TCN | Helped `pull_up` and `squat`, but `push_up` regressed relative to Stage `7` | Stronger RGB matters, but still does not produce a universal winner |
| `7C` representation fit | Is RGB only helping when pose quality is weak? | Row-level pose-vs-RGB join with pose-quality buckets | No. `push_up` showed RGB gains even in stronger pose-quality buckets; `squat` remained pose-first even when pose quality was strong | Treat RGB as complementary information, not just a weak-pose fallback |
| `7D` hard-case audit | Are the remaining errors mainly visibility issues, semantic ambiguity, or representation mismatch? | Pose/RGB error audit with pose-quality tags and video metadata, followed by a completed 48-case manual review layer | Reviewed hard cases were dominated by `pose_failure (11)`, `camera_viewpoint (8)`, and `rep_ambiguity (6)`. `squat` clustered around rep ambiguity and pose failure, `pull_up` around viewpoint and target selection, and `push_up` around pose/model difficulty | Keep exercise-dependent interpretation; use the reviewed hard-case taxonomy as the final error-analysis layer |
| `7E` multimodal fusion | Can simple late fusion beat both pose-only and RGB-only branches? | Generic pose sequences + RGB features + late-fusion TCN | No clear win. `pull_up` and `push_up` improved `MAE` in places, but `Within-1` degraded; `squat` stayed clearly worse than the dedicated pose baseline | Freeze as a negative-but-informative result; do not adopt simple late fusion |
| Stage `8` routed counting | Can the project produce a practical counting surface for supported exercises without forcing a generic model? | Exercise-dependent routing across best current branches | New practical direction: `squat -> squat_tcn_l1_channels96`, `pull_up -> pose_count_tcn_pull_up_seq192`, `push_up -> rgb_count_tcn_push_up_seq128` | Build and evaluate the first routed counting artifact |
| Stage `9/9B` pose transformer | Can a generic pose transformer or the current pose augmentation beat the shared pose TCN? | Transformer encoder over generic pose sequences with augmentation-on/off ablation | No robust gain. `pull_up` saw slightly better `MAE` but worse `Within-1`; `push_up` and `squat` were effectively unchanged by augmentation; the branch did not beat the routed choices | Freeze as a completed negative result; do not promote the transformer branch |
| Stage `10` dedicated `pull_up` pose | Does exercise-specific pose tuning beyond squat produce a clear new `pull_up` winner? | Dedicated `pull_up` TCN candidates on shared Stage 5 pose sequences | Best `MAE` candidate (`channels128`) improved to `3.5463`, but `Within-1` dropped to `0.2857`; no candidate beat the current practical tradeoff | Keep `pull_up` as mixed and retain the routed pose choice based on `Within-1` |
| Stage `11` bootstrap CIs | How stable are the final reportable metrics on these small validation splits? | Bootstrap confidence intervals over final `predictions.csv` artifacts | Wide `95%` CIs for the dedicated `squat` control and the routed `pull_up` / `push_up` branches, confirming that the conclusions are directional but statistically fragile on small `n` | Use CIs in the final report and avoid over-precise claims |

### Supported practical route

The first routed counting surface is intentionally narrow and honest. It only uses branches that already earned their place experimentally.

| Exercise | Selected branch | Reason |
| --- | --- | --- |
| `squat` | `squat_tcn_l1_channels96` | Best result in the project by a large margin |
| `pull_up` | `pose_count_tcn_pull_up_seq192` | Slightly worse `MAE` than stronger RGB, but materially better `Within-1` |
| `push_up` | `rgb_count_tcn_push_up_seq128` | Best current push-up branch |

### Deferred branch

Countix has been scaffolded as a separate benchmark branch under `Data/Countix`, but it is currently deferred. The main project conclusions do not depend on downloading or integrating Countix.

### Key artifact references

- [frozen_best_squat_result.json](/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection/artifacts/3_Modeling/frozen_best_squat_result.json)
- [frozen_all_exercises_counting_baseline_result.json](/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection/artifacts/3_Modeling/frozen_all_exercises_counting_baseline_result.json)
- [7B_Stronger_RGB_Backbone_Colab.ipynb](/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection/artifacts/3_Modeling/7B_Stronger_RGB_Backbone_Colab.ipynb)
- [7C_Representation_Fit_Analysis_Colab.ipynb](/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection/artifacts/3_Modeling/7C_Representation_Fit_Analysis_Colab.ipynb)
- [7D_Hard_Case_Data_Audit_Colab.ipynb](/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection/artifacts/3_Modeling/7D_Hard_Case_Data_Audit_Colab.ipynb)
- [7E_Multimodal_Pose_RGB_Fusion_Colab.ipynb](/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection/artifacts/3_Modeling/7E_Multimodal_Pose_RGB_Fusion_Colab.ipynb)
- [8_Exercise_Dependent_Counting_Colab.ipynb](/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection/artifacts/3_Modeling/8_Exercise_Dependent_Counting_Colab.ipynb)

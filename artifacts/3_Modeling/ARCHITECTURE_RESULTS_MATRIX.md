## Architecture Results Matrix

Last updated: March 27, 2026

This artifact showcases the measured outcome of each major architecture or experiment family used in the project. It is complementary to [EXPERIMENT_SHOWCASE.md](/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection/artifacts/3_Modeling/EXPERIMENT_SHOWCASE.md): the showcase explains the arc of the work, while this matrix focuses on the results themselves.

### Reading guide

- `MAE`: lower is better
- `Within-1`: higher is better
- the same architecture can look good on one exercise and weak on another
- for `squat`, the correct control is the frozen dedicated squat baseline `squat_tcn_l1_channels96`, not only the weaker shared pose-sequence branch

### Strongest result by architecture family

| Architecture family | Representation | Best measured result in project | Interpretation |
| --- | --- | --- | --- |
| FSM squat baseline | Squat-specific engineered pose features | `squat`: `MAE = 3.0625`, `Within-1 = 0.5625` | Useful interpretable baseline, but surpassed by the squat TCN |
| Dedicated squat TCN | Squat-specific engineered pose features | `squat`: `MAE = 2.1405`, `Within-1 = 0.5625` | Strongest single result in the project |
| Shared pose TCN (`6B`) | Generic normalized pose sequences | `pull_up`: `MAE = 4.6088`, `Within-1 = 0.4286` | Best shared pose-only result among the compared exercises |
| Pose transformer (`9/9B`) | Generic normalized pose sequences with transformer encoder | `push_up`: `MAE = 7.4561`, `Within-1 = 0.0556` | Better than the shared pose TCN on `push_up` `MAE`, but not a competitive replacement and not improved meaningfully by augmentation |
| Keypoint-weighted pose TCN (`6C`) | Generic pose + manual weighting | `push_up`: `MAE = 8.5139`, `Within-1 = 0.1111` | Only isolated gain; broad strategy failed |
| Density pose TCN (`6D`) | Generic pose + weak pseudo-density | `push_up`: `MAE = 8.61`, `Within-1 = 0.0` | Did not produce a new baseline |
| RGB TCN Stage `7` | Frozen `ResNet18` RGB features | `push_up`: `MAE = 6.6018`, `Within-1 = 0.2778` | Strongest current push-up branch |
| Stronger RGB TCN (`7B`) | Frozen `ResNet50` RGB features | `pull_up`: `MAE = 4.1992`, `Within-1 = 0.3571` | Best RGB result for pull-up, but still not dominant on both metrics |
| Multimodal late fusion (`7E`) | Generic pose + RGB | `push_up`: `MAE = 6.1691`, `Within-1 = 0.1111` | Improved `MAE` in places, but did not justify fusion complexity |
| Dedicated `pull_up` pose tuning (`10`) | Shared Stage 5 pose sequences with dedicated per-exercise tuning | `pull_up`: `MAE = 3.5463`, `Within-1 = 0.2857` | Lowered `MAE`, but did not beat the best practical `Within-1` tradeoff |

### Core comparison: `squat`, `pull_up`, `push_up`

| Exercise | Architecture / stage | MAE | Within-1 | Verdict |
| --- | --- | ---: | ---: | --- |
| `squat` | FSM tuned baseline | 3.0625 | 0.5625 | Good historical baseline |
| `squat` | Dedicated squat TCN | 2.1405 | 0.5625 | Best current squat solution |
| `squat` | Shared pose TCN (`6B`, `seq_len=256`) | 8.0430 | 0.2500 | Generic pose underfits squat relative to the dedicated branch |
| `squat` | Pose transformer (`9B`, best variant) | 9.1502 | 0.1250 | Worse than the shared pose TCN; augmentation did not change the conclusion |
| `squat` | RGB TCN Stage `7` (`ResNet18`) | 6.5446 | 0.0625 | Better `MAE` than shared pose, but poor discrete accuracy |
| `squat` | Stronger RGB TCN (`7B`, `ResNet50`) | 5.4245 | 0.1875 | Best generic non-squat-specific alternative, still far below the dedicated squat baseline |
| `squat` | Multimodal late fusion (`7E`) | 6.6988 | 0.0000 | Negative result |
| `pull_up` | Shared pose TCN (`6B`, `seq_len=192`) | 4.6088 | 0.4286 | Best `Within-1` result |
| `pull_up` | Pose transformer (`9B`, aug_on) | 5.0210 | 0.0714 | Slight `MAE` gain over transformer `aug_off`, but not a practical win |
| `pull_up` | RGB TCN Stage `7` (`ResNet18`) | 4.8686 | 0.1429 | RGB baseline weaker than pose |
| `pull_up` | Stronger RGB TCN (`7B`, `ResNet50`) | 4.1992 | 0.3571 | Best `MAE`, but not best `Within-1` |
| `pull_up` | Multimodal late fusion (`7E`) | 4.1193 | 0.1429 | Best `MAE`, but `Within-1` collapsed too much |
| `pull_up` | Dedicated `pull_up` pose (`10`, `channels128`) | 3.5463 | 0.2857 | Best `MAE`, but still not the best practical branch on both metrics |
| `push_up` | Shared pose TCN (`6B`, `seq_len=128`) | 8.8724 | 0.0000 | Pose-only remains weak |
| `push_up` | Pose transformer (`9B`, aug_off) | 7.4561 | 0.0556 | Better than shared pose, but still behind the RGB branch |
| `push_up` | Keypoint-weighted pose TCN (`6C`) | 8.5139 | 0.1111 | Small isolated pose-side improvement |
| `push_up` | Density pose TCN (`6D`) | 8.61 | 0.0000 | No meaningful temporal-formulation recovery |
| `push_up` | RGB TCN Stage `7` (`ResNet18`) | 6.6018 | 0.2778 | Best current practical push-up branch |
| `push_up` | Stronger RGB TCN (`7B`, `ResNet50`) | 7.3768 | 0.0556 | Worse than Stage `7` RGB |
| `push_up` | Multimodal late fusion (`7E`) | 6.1691 | 0.1111 | Best `MAE`, but still worse than Stage `7` RGB on `Within-1` |

### Pose ablation track: selected exercises from `6B`, `6C`, `6D`

This view shows what happened when the project kept the same generic pose-sequence setup and changed only the pose-side architecture idea.

| Exercise | `6B` best shared pose | `6C` weighted pose | `6D` density pose | Result |
| --- | --- | --- | --- | --- |
| `bench_pressing` | `MAE = 5.06`, `Within-1 = 0.231` | `MAE = 5.59`, `Within-1 = 0.154` | `MAE = 5.63`, `Within-1 = 0.154` | Both follow-ups were worse |
| `pommelhorse` | `MAE = 5.09`, `Within-1 = 0.067` | `MAE = 5.89`, `Within-1 = 0.133` | `MAE = 5.48`, `Within-1 = 0.067` | No clear improvement |
| `pull_up` | `MAE = 4.61`, `Within-1 = 0.429` | `MAE = 5.39`, `Within-1 = 0.286` | `MAE = 5.94`, `Within-1 = 0.071` | `6B` remained strongest |
| `push_up` | `MAE = 8.87`, `Within-1 = 0.000` | `MAE = 8.51`, `Within-1 = 0.111` | `MAE = 8.61`, `Within-1 = 0.000` | `6C` gave a small isolated gain |
| `squat` | `MAE = 8.04`, `Within-1 = 0.250` | `MAE = 9.38`, `Within-1 = 0.125` | `MAE = 11.29`, `Within-1 = 0.000` | Both follow-ups were clearly worse |

### Architecture decisions

| Architecture / experiment | Keep as active baseline? | Why |
| --- | --- | --- |
| Dedicated squat TCN | Yes | Best project result and correct squat reference |
| Shared pose TCN (`6B`) | Yes, as shared pose baseline | Best generic pose reference |
| Pose transformer (`9/9B`) | No | No robust validation gain and no change to the routed decision |
| Keypoint-weighted pose (`6C`) | No | Mostly negative |
| Density pose (`6D`) | No | Mostly negative under weak supervision |
| RGB TCN Stage `7` | Yes, for `push_up` | Best current push-up branch |
| Stronger RGB TCN (`7B`) | Yes, for comparison on `pull_up` and `squat` | Useful representation-strength result |
| Multimodal late fusion (`7E`) | No | Complexity did not earn itself |
| Exercise-dependent routing (`8`) | Yes, as practical next step | Matches the evidence instead of forcing a generic winner |

### Presentable one-line conclusion

The strongest outcome is not a single universal architecture. The strongest outcome is a defensible architecture map: `squat` remains best with a dedicated pose branch, `push_up` is best with the RGB branch, `pull_up` remains mixed, and neither the generic pose transformer nor simple multimodal late fusion outperformed the better single-modality choice.

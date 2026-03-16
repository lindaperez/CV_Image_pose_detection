# Presentation Q&A

## Why did you start with squat only?

Squat was used as a controlled first iteration to validate the architecture before scaling. The goal was to prove that the pipeline from video to pose to counting worked end to end on one exercise before adding multi-exercise complexity.

## Is this a counting project or an exercise-recognition project?

The long-term project goal is both exercise recognition and repetition counting. The current implementation is the first stage of that goal and focuses only on squat repetition counting to validate the architecture.

## How does the current squat prototype connect to the long-term multi-exercise goal?

The squat prototype validates the most important parts of the system in a controlled setting: cleaned annotations, pose extraction, feature generation, and temporal counting. Once those stages are validated on one exercise, the same architecture can be extended toward multi-exercise recognition and counting.

## Why did you choose RepCount?

RepCount is a recognized repetitive-action benchmark and provides realistic video variability. It was useful for validating the pipeline under harder conditions than a small custom controlled dataset.

## How clean was the dataset?

It was not clean enough to use directly. We found typo issues, heterogeneous labels such as `others`, and ambiguous samples. That is why EDA, relabeling, and cleaning were necessary before modeling.

## Why did you choose YOLO Pose?

YOLO Pose gave a practical pretrained 2D pose-estimation frontend that was fast to integrate, worked well in Colab, and supported a modular, interpretable pipeline. It let us inspect keypoints and feature quality before counting.

## Why not use OpenPose, MediaPipe, or a 3D CNN?

OpenPose and MediaPipe were reasonable alternatives, but YOLO Pose fit the current Colab-based workflow more directly and integrated better with the existing Python pipeline. A 3D CNN would have required a heavier raw-video architecture and a much larger design shift, which was not ideal for a staged first iteration.

## Why not direct video counting from the start?

Direct video counting is heavier and less interpretable. For a first iteration, the pose-first pipeline was lower risk, easier to debug, and more aligned with the broader goal of exercise analysis rather than count prediction alone.

## Why use FSM first?

The FSM provided an interpretable baseline. It let us count squats using explicit movement phases and gave a clear reference point before moving to a learned temporal model.

## Why did you later use TCN?

The FSM does not scale well and is sensitive to handcrafted thresholds. TCN was introduced as a learned temporal alternative that could model the feature sequence directly and serve as a better foundation for future multi-exercise learning.

## Which model performed best?

The tuned FSM remained the best interpretable baseline. The best learned model was `squat_tcn_l1_channels96_dropout01`, which improved MAE and RMSE over the FSM while remaining close on Within-1.

## What do the TCN results mean in practice?

The best TCN achieved a validation MAE of about `2.13`, meaning the predicted repetition count was off by about two reps on average. Its Within-1 accuracy of `0.50` means about half of the validation videos were counted within one repetition of the true count.

## Is 50% Within-1 accuracy good enough?

Not as a final production-level result. It is acceptable for a first validated prototype because it shows the system is learning meaningful temporal structure, but it is not strong enough to claim that the counting problem is solved.

## Why does the TCN improve MAE but not completely dominate Within-1?

MAE measures average count error, while Within-1 measures how often the prediction lands inside a strict `±1` tolerance band. A model can reduce large mistakes and improve MAE while still failing to produce enough near-exact predictions to dominate Within-1.

## How do your results compare to papers like TransRAC, PoseRAC, or ESCounts?

Those papers report stronger benchmark performance on RepCount-style evaluations, especially PoseRAC and ESCounts. However, they optimize directly for repetitive-action counting benchmarks, while this project validates a broader pose-based exercise-analysis pipeline and is not a one-to-one reproduction of those methods.

## Why is Within-1 still limited?

Because the task is strict and real exercise videos contain ambiguity. The model may be close on average while still failing the exact `±1` tolerance on many videos.

## If lower-body confidence is high, why do errors still remain?

High confidence shows pose extraction was generally successful, but it does not eliminate contextual or semantic difficulty. Hard videos can still include chair assistance, obstacles, occlusion, unusual movement style, or ambiguous rep boundaries.

## What kinds of videos still fail?

The hardest failures are associated with contextual variation, including occlusion, assisted movement, multiple people, obstacles, and ambiguous annotations, rather than broad failure of pose extraction.

## How much do occlusion, obstacles, or multiple people affect the system?

They can affect the system significantly. They create ambiguity about which person should be tracked, which motion belongs to the target exercise, and whether the body joints remain reliable enough for counting.

## How do your results compare to the literature?

They are below strong benchmark methods such as PoseRAC or ESCounts on RepCount, but the comparison is not one-to-one because those papers are optimized specifically for counting benchmarks, while this project validates a broader staged pose-based system.

## Are the failures caused by the model, the data, or the labels?

At this stage, the answer is likely all three to some degree. The model is still limited, the videos contain real-world contextual difficulty, and some cases remain ambiguous at the annotation level. The failure analysis suggests the problem is not explained mainly by low pose confidence alone.

## If the accuracy is still limited, what has the project demonstrated?

It demonstrated that the proposed pose-based architecture works end to end, that the counting backend can be replaced by a learned temporal model, and that the main bottlenecks are now in robustness and ambiguity rather than raw pose extraction.

## Is this already a usable system or only a prototype?

It should be described as a validated prototype, not a finished high-accuracy system. The system is complete enough to demonstrate the architecture and produce measurable results, but not accurate enough to claim robust deployment readiness.

## What is the real contribution if no single paper matches this exact architecture?

The contribution is the system-level design and validation of a pose-based exercise-counting pipeline, including dataset cleaning, YOLO-based pose extraction, engineered squat features, a rule-based baseline, and a learned temporal alternative. It is an applied architecture-validation contribution rather than a new foundational model claim.

## Why not just use PoseRAC next?

PoseRAC is an important reference, but it is not a direct drop-in continuation of the current repo. The next intended step is to build on the validated YOLO pipeline and move toward a detect-first-then-count architecture.

## What is the next concrete step?

The next step is to preserve the squat prototype as the validated baseline, complete failure analysis on the hardest cases, and then move toward a multi-exercise pose-based architecture that first identifies the target person and exercise before counting.

## Why is the next step not simply train on all exercises?

Because the remaining squat errors show that the current branch still has unresolved robustness issues. Moving to all exercises immediately would make the problem harder and could hide whether the real issue comes from the architecture, the data, or the video conditions.

## Would a multitask model be better?

Potentially yes. A shared temporal model with one head for exercise classification and another for repetition counting is the most natural next architecture if the current pose-first direction is retained.

## If you move to all exercises, will accuracy drop even more?

Possibly yes. A multi-exercise setting adds class confusion and more movement variation on top of the same contextual issues already seen in the squat branch. That is why the current prototype is being treated as a validation stage rather than the final system.

## What if the current architecture still does not improve enough?

Then a redesign becomes justified. The next alternative would be to compare the current pose-based direction against stronger counting architectures such as PoseRAC or other benchmark-driven methods.

## Why is the accuracy still low?

The problem is difficult because the videos contain ambiguity in movement boundaries, contextual interference, assisted or atypical execution, and possible annotation uncertainty. The failure analysis suggests the remaining error is not caused mainly by pose-confidence failure, but by the broader complexity of the task.

## Why should this architecture scale to more exercises?

It should only scale if the pose-first representation continues to work after robustness improvements. The architecture is promising because it is modular and already supports a learned temporal backend, but true scalability still needs to be tested in a multi-exercise setting.

## What would you do next if you had more time?

I would complete the hard-case review, improve target-person handling, add exercise recognition before counting, and then train a shared multi-exercise temporal model on top of the validated YOLO pose frontend. If that still did not improve performance enough, I would compare against stronger benchmark architectures such as PoseRAC or ESCounts.

## Why is Within-1 only 50% — isn't that low?

Yes, it is low for a final solution. It is acceptable only as a first-iteration prototype result. It shows the model is learning the motion structure, but not with enough precision to call the counting problem solved.

## How do you explain the gap between MAE improving but Within-1 dropping from FSM to TCN?

MAE measures average error size, while Within-1 measures how often the prediction stays inside a strict `±1` rep band. A model can reduce large mistakes and improve MAE while still not producing enough near-exact predictions to improve Within-1. In other words, it can be closer on average without being more exact.

## You only have 16 validation videos — how reliable are these numbers?

They are useful for comparing prototypes, but they are not highly stable. With only 16 validation videos, the metrics should be interpreted as indicative rather than definitive. Small changes in a few videos can noticeably change the reported scores.

## Did you evaluate on the test set?

No, not in the current validated prototype stage. The work focused on training and validation behavior for architecture validation. The test split was intentionally kept out of the iterative development loop.

## Why did you only use squats and not the full dataset?

Squat was used as a controlled first iteration. The goal was to validate the architecture, data pipeline, and counting logic on one exercise before moving to the harder multi-exercise setting.

## How does your cleaning affect comparability with published results on RepCount?

It reduces strict comparability. Published RepCount papers usually evaluate on the benchmark as defined in their setup, while this project performed EDA-driven cleaning and relabeling to make the data usable for the current architecture. So the results should be presented as project-specific prototype results, not as direct benchmark claims.

## Your validation set is very small — could the results be due to chance?

Partly, yes. With a small validation split, variance is higher. That is why the results are useful for architecture validation and comparison inside the project, but not strong enough to make broad claims of state-of-the-art performance.

## Why did you use a TCN instead of a Transformer or LSTM?

TCN was chosen because it is simpler, lighter, and easier to integrate into the current pose pipeline than a transformer, while still modeling temporal structure effectively. It was also a practical next step after the FSM baseline. LSTM was possible, but TCN offered a cleaner and more efficient first learned temporal model.

## Why not use RepNet or TransRAC directly?

They are important references, but they are not clean drop-in continuations of the current repo. They also focus on direct repetitive-action counting rather than the broader detect-first-then-count exercise-analysis goal. The project needed a practical pose-based architecture first.

## Why pose-based instead of raw RGB — wouldn't RGB give more information?

RGB gives more visual information, but it is also heavier, less interpretable, and harder to debug. Pose-based modeling was chosen because it provides a structured intermediate representation that is easier to inspect and more aligned with exercise-analysis goals.

## Why YOLO Pose specifically?

YOLO Pose was practical, pretrained, fast to integrate, compatible with Colab, and already fit the Python pipeline. It provided a usable 2D pose frontend without requiring custom pose-model training.

## Did you tune thresholds on validation?

No. The FSM tuning was done on the training split, and validation was kept as the main honest check of generalization. That was done specifically to avoid leakage from validation into tuning.

## How did you prevent overfitting with only 102 training videos?

The project limited scope to a compact temporal model, used validation monitoring, compared multiple simple configurations, and avoided treating the current results as final proof of generalization. The prototype should still be considered at some risk of overfitting because the training set is small.

## Why FSM first — wasn't it obvious a learned model would be better?

Not necessarily. FSM was useful as an interpretable baseline and as a way to validate the feature pipeline quickly. It also gave a clear reference point to test whether a learned temporal model truly improved the results.

## If errors aren't from pose quality, what exactly is causing them?

The likely causes are contextual and semantic difficulty: occlusion, obstacles, chair assistance, unusual movement execution, multiple people, ambiguous repetition boundaries, and annotation uncertainty. The correlation analysis suggests the remaining error is not explained mainly by poor pose confidence.

## How confident are you that the same pipeline will generalize to other exercises?

Not highly confident yet. The squat branch validates the architecture direction, but true generalization still has to be tested. A multi-exercise setting will introduce new class confusion and more motion variation, so the current result should be seen as a promising first stage, not proof of full scalability.

## Why not just use PoseRAC since it already achieves 0.56 on RepCount?

PoseRAC is a strong benchmark reference, but it is not a direct continuation of the current implementation. The current project was built as a pose-first architecture-validation pipeline using YOLO Pose, engineered features, FSM, and TCN. PoseRAC is better treated as a comparative next branch if the current architecture does not improve enough, rather than as something that could simply replace the current repo with no redesign.

# Hard-Case Review Guide

Use this guide when filling `hard_case_review_manifest.csv`. The goal is to
tag the *main reason* a row is hard, plus any secondary tags that help explain
it. Review only the selected `7D` rows, not the full dataset.

## Review Fields

### Manual Review Field Definitions

| Field | What it means | How to fill it |
| --- | --- | --- |
| `manual_review_status` | Review progress for the row. | Use `pending` before inspection, `reviewed` after one reviewer has filled the row, and `confirmed` if you want to mark it as especially settled or double-checked. |
| `manual_primary_issue` | The single main reason the row is difficult, questionable, or informative. | Pick one dominant explanation only, even if multiple things are wrong. Use `manual_issue_tags` for the secondary details. |
| `manual_issue_tags` | Extra descriptive tags that refine the primary issue. | Use short comma-separated tags such as `off_frame`, `partial_rep`, or `pose_jitter`. Prefer the controlled vocabulary below instead of inventing new phrases. |
| `manual_target_person_ok` | Whether the correct person to count is clearly identifiable. | Mark `yes` when the target subject is obvious for the whole clip. Mark `no` when multiple people, target switches, or ambiguous framing make the counted person uncertain. |
| `manual_count_label_ok` | Whether the annotated repetition count itself looks correct. | Mark `yes` when the stored count seems reasonable. Mark `no` when the number of reps in the label looks wrong or clearly debatable. This is about the rep count, not the exercise class. |
| `manual_rep_definition_ambiguous` | Whether the definition of a valid repetition is unclear in the clip. | Mark `yes` for partial reps, shallow depth, unclear extension, holds, pauses, or borderline cycles where a human could reasonably disagree about what counts as one rep. |
| `manual_visibility_issue_confirmed` | Whether observability problems are truly present in the video. | Mark `yes` when occlusion, off-frame motion, cropping, unstable framing, or viewpoint makes the action genuinely harder to see. |
| `manual_pose_failure_confirmed` | Whether the pose estimate itself visibly fails. | Mark `yes` when the overlay is noisy, incomplete, jittery, swapped, or too unstable to support confident counting on its own. Do not use this for general video difficulty unless the pose output itself is clearly bad. |
| `manual_rgb_context_advantage_confirmed` | Whether the raw RGB video contains useful counting signal beyond what pose is representing well. | Mark `yes` when the raw video makes the motion understandable but the pose overlay is weak, incomplete, or semantically poor. This is about a real RGB information advantage, not just a better numeric RGB result. |
| `manual_keep_for_report` | Whether the row is worth citing in the final writeup or slides. | Mark `yes` for especially representative, surprising, or easy-to-explain cases that support the project conclusions. |
| `manual_notes` | Short reviewer note about what was seen. | Write 1 to 2 precise sentences, for example: `Annotated as 19, but visual review suggests about 16 valid reps; last 3 look partial.` |

### Quick Yes/No Distinctions

- `manual_count_label_ok`
  This asks whether the stored number of reps looks correct.

- `manual_rep_definition_ambiguous`
  This asks whether the meaning of "one valid rep" is itself unclear in the clip.

- `manual_target_person_ok`
  This asks whether the correct subject is obvious.

- `manual_pose_failure_confirmed`
  This asks whether the pose overlay itself breaks, not just whether the clip is hard.

- `manual_rgb_context_advantage_confirmed`
  This asks whether the raw video gives useful cues that pose does not capture well.

## Primary Issue Table

| What you encounter | What it means in review | `manual_primary_issue` | Useful `manual_issue_tags` | Usually mark these fields |
| --- | --- | --- | --- | --- |
| Occlusion or body parts hidden | Important motion is blocked by self-occlusion, equipment, furniture, or framing | `visibility` | `occlusion`, `equipment_obstruction`, `self_occlusion` | `manual_visibility_issue_confirmed=yes` |
| Person partly off-frame | Head, arms, torso, or legs leave the frame during key phases | `visibility` | `off_frame`, `cropped_body`, `portrait_framing` | `manual_visibility_issue_confirmed=yes` |
| Camera moving or unstable framing | Handheld motion or reframing changes observability | `camera_viewpoint` | `camera_motion`, `reframe`, `zoom_shift` | `manual_visibility_issue_confirmed=yes` if it hurts observability |
| Extreme viewpoint | Overhead, very oblique, or unusual side angle changes what is visible | `camera_viewpoint` | `viewpoint_extreme`, `depth_unclear`, `side_view`, `top_view` | `manual_visibility_issue_confirmed=yes` if it causes ambiguity |
| Multiple people or unclear target | Not obvious which person should be counted | `target_selection` | `multi_person`, `target_switch`, `background_person` | `manual_target_person_ok=no` |
| Partial repetitions | Motion looks incomplete or borderline for counting | `rep_ambiguity` | `partial_rep`, `shallow_depth`, `not_full_extension` | `manual_rep_definition_ambiguous=yes` |
| Pauses, holds, or irregular cycle boundaries | Hard to decide where one repetition ends and the next starts | `rep_ambiguity` | `pause_or_hold`, `tempo_change`, `boundary_unclear` | `manual_rep_definition_ambiguous=yes` |
| Assistance or unusual execution style | Assisted movement, momentum, or nonstandard technique changes semantics | `execution_variation` | `assisted_motion`, `kipping`, `unusual_technique`, `modified_form` | Often `manual_rep_definition_ambiguous=yes` |
| Ground-truth count seems wrong or debatable | The label itself may be incorrect or at least arguable | `label_mismatch` | `count_label_suspect`, `annotation_disagreement` | `manual_count_label_ok=no` |
| Pose estimate clearly breaks | Keypoints jitter, disappear, swap sides, or fail to track the motion | `pose_failure` | `pose_jitter`, `missing_keypoints`, `joint_swap`, `low_confidence_pose` | `manual_pose_failure_confirmed=yes` |
| RGB clearly has extra context signal | Scene, object interaction, or appearance explains why RGB wins | `rgb_context_advantage` | `scene_context`, `equipment_context`, `body_orientation`, `appearance_cue` | `manual_rgb_context_advantage_confirmed=yes` |
| No obvious video/data issue | The clip looks countable and the failure is mainly model-side | `model_failure` | `no_clear_issue` | leave the confirm flags blank unless something is clear |

## Recommended Tag Vocabulary

Use short, reusable tags instead of writing a new phrase every time.

Suggested tags:

- `occlusion`
- `self_occlusion`
- `equipment_obstruction`
- `off_frame`
- `cropped_body`
- `portrait_framing`
- `camera_motion`
- `reframe`
- `zoom_shift`
- `viewpoint_extreme`
- `depth_unclear`
- `side_view`
- `top_view`
- `multi_person`
- `target_switch`
- `background_person`
- `partial_rep`
- `shallow_depth`
- `not_full_extension`
- `pause_or_hold`
- `tempo_change`
- `boundary_unclear`
- `assisted_motion`
- `kipping`
- `unusual_technique`
- `modified_form`
- `count_label_suspect`
- `annotation_disagreement`
- `pose_jitter`
- `missing_keypoints`
- `joint_swap`
- `low_confidence_pose`
- `scene_context`
- `equipment_context`
- `body_orientation`
- `appearance_cue`
- `no_clear_issue`

## Tag Definitions By Primary Issue

Use these definitions when a tag feels close to another one. They are grouped
by the recommended `manual_primary_issue` so reviewers can stay consistent.

### `visibility`

| Tag | Definition |
| --- | --- |
| `occlusion` | The important body part or motion is blocked by something in the scene. |
| `self_occlusion` | The body blocks itself, for example arms covering the torso or legs overlapping. |
| `equipment_obstruction` | Gym equipment, furniture, or another object hides part of the motion. |
| `off_frame` | The important body part leaves the visible frame. |
| `cropped_body` | The subject is cut off by the frame even without obvious camera motion. |
| `portrait_framing` | Narrow vertical framing reduces visibility of the full body or range of motion. |

### `camera_viewpoint`

| Tag | Definition |
| --- | --- |
| `camera_motion` | The camera itself moves or shakes during the clip. |
| `reframe` | The camera composition changes during the clip, so the subject is shown differently over time. |
| `zoom_shift` | The apparent scale changes because the camera zooms or moves closer or farther. |
| `viewpoint_extreme` | The camera angle is unusually high, low, oblique, or otherwise hard to interpret. |
| `depth_unclear` | The motion is hard to judge because depth is compressed or visually ambiguous. |
| `side_view` | The clip is dominated by a side profile view. Use when that view itself matters for the difficulty. |
| `top_view` | The clip is dominated by an overhead or top-down view. |

### `target_selection`

| Tag | Definition |
| --- | --- |
| `multi_person` | More than one person is visible and this could affect who should be counted. |
| `target_switch` | The viewer could reasonably switch from one person to another while watching the clip. |
| `background_person` | A non-target person in the background creates confusion. |

### `rep_ambiguity`

| Tag | Definition |
| --- | --- |
| `partial_rep` | A movement looks incomplete but may still be close enough to be debated. |
| `shallow_depth` | The range of motion does not go deep enough for a clear full repetition. |
| `not_full_extension` | The movement does not return to the expected end position. |
| `pause_or_hold` | The performer stops or holds a position long enough to blur repetition boundaries. |
| `tempo_change` | The speed changes within the clip enough to make boundaries harder to interpret. |
| `boundary_unclear` | It is hard to decide where one repetition ends and the next begins. |

### `execution_variation`

| Tag | Definition |
| --- | --- |
| `assisted_motion` | Another force, support, or person appears to help complete the movement. |
| `kipping` | Momentum-driven pull-up style that changes the usual repetition semantics. |
| `unusual_technique` | The exercise is performed in a nonstandard way that may confuse the counting logic. |
| `modified_form` | A deliberate variant or adaptation of the usual movement is being used. |

### `label_mismatch`

| Tag | Definition |
| --- | --- |
| `count_label_suspect` | The stored count label seems questionable after watching the clip. |
| `annotation_disagreement` | A reviewer would likely disagree with the existing label or count definition. |

### `pose_failure`

| Tag | Definition |
| --- | --- |
| `pose_jitter` | Keypoints move erratically instead of tracking the body smoothly. |
| `missing_keypoints` | Important joints disappear for part of the clip. |
| `joint_swap` | Left-right or body-part assignments appear to flip incorrectly. |
| `low_confidence_pose` | The pose estimate is visibly weak or unstable overall. |

### `rgb_context_advantage`

| Tag | Definition |
| --- | --- |
| `scene_context` | The wider scene helps explain the action better than pose alone. |
| `equipment_context` | Equipment interaction provides useful information that pose misses. |
| `body_orientation` | The way the body is oriented in the image explains why RGB helps. |
| `appearance_cue` | Clothing, silhouette, or other visual appearance cues help explain the motion. |

### `model_failure`

| Tag | Definition |
| --- | --- |
| `no_clear_issue` | No obvious data, label, or visibility problem stands out; the miss looks mostly model-side. |

### Common Distinctions

- `camera_motion` vs `reframe`
  `camera_motion` means the camera physically moves or shakes.
  `reframe` means that motion changes how the subject is composed in the image over time.

- `reframe` vs `off_frame`
  `reframe` is the changing composition.
  `off_frame` is the resulting visibility problem when part of the body leaves the image.

- `cropped_body` vs `off_frame`
  `cropped_body` is a static framing problem.
  `off_frame` usually means the key body part moves out of view during the clip.

- `partial_rep` vs `boundary_unclear`
  `partial_rep` questions whether the movement was complete enough.
  `boundary_unclear` questions where repetitions should be separated.

## Practical Rules

- Pick one `manual_primary_issue` per row, even if multiple things are wrong.
- Use `manual_issue_tags` for the secondary details.
- If the count label itself is doubtful, prefer `label_mismatch` or `rep_ambiguity` over `model_failure`.
- If the clip is visually poor, do not automatically mark `pose_failure`; only do that when the pose output itself clearly fails.
- Use `manual_keep_for_report=yes` for cases that are especially representative, surprising, or easy to explain in writing.

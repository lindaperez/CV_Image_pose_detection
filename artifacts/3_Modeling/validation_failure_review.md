# Validation Failure Review

This note captures the manual review decisions for selected validation videos from the current best squat TCN run, `squat_tcn_l1_channels96`.

## Reviewed Cases

| Name | True Count | Pred Count | Abs Error | Audit Severity | Classification | Action | Notes |
|---|---:|---:|---:|---|---|---|---|
| `train3898.mp4` | 2.0 | 4.908228 | 2.908228 | `critical` | Unusable upstream failure | Exclude from main squat-count evaluation | Severe framing and occlusion. No valid lower-body frames. |
| `stu7_65.mp4` | 9.0 | 9.132257 | 0.132257 | `medium` | Valid harder variation | Keep and flag as hard case | Appears to be a Zercher squat / squat-style variation with partial lower-body visibility. |
| `stu10_63.mp4` | 16.0 | 15.041590 | 0.958410 | `review` | Valid framed/noisy case | Keep and flag as framed/noisy | Portrait gym scene. Confidence noise, but usable motion geometry. |
| `test2349.mp4` | 6.0 | 2.633353 | 3.366647 | `review` | Valid low-quality case | Keep and flag as low-quality | Low-resolution, portrait, low-FPS clip with moderate confidence noise. |
| `train3921.mp4` | 4.0 | 3.844428 | 0.155572 | `ok` | Valid contextual hard case | Keep and flag as assisted/context-heavy | Chair or box-assisted setup, surrounding equipment, and another visible person add semantic ambiguity. |

## Current Policy

- Exclude true upstream failures from the main squat-count benchmark if lower-body visibility is unusable.
- Keep valid but harder examples in the dataset, but flag them as:
  - squat variation
  - framed/noisy
  - low-quality
  - assisted/context-heavy
- Use these flags to separate model limitations from data-quality limitations in the report.

## Main Conclusion

The remaining validation errors are not explained primarily by pose extraction failure. The dominant residual issues are now video semantics, exercise variation, framing quality, and contextual ambiguity.

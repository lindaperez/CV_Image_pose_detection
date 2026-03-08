from pathlib import Path

md = """# RepCoach — V1 Specification & Architecture

**Project:** Real-time Exercise Identification + Rep Counting (Webcam)  
**Primary goal:** Build an interview-ready, production-minded computer vision app that (1) identifies the exercise, (2) counts reps robustly, and (3) optionally provides simple form feedback.

---

## 1) Problem statement

People exercising without a trainer often miscount reps and can’t easily detect poor form. RepCoach is a real-time webcam application that uses human pose estimation to:

1. **Identify** the exercise being performed (Squat / Push-up / Bicep Curl)
2. **Count** repetitions reliably under noisy pose estimates
3. **Optionally** flag common form issues (interpretable rules)

---

## 2) Users & use cases

### Primary user
- A person exercising at home with a laptop webcam.

### Use cases
- Start webcam → app auto-detects exercise type
- Live overlay of skeleton + exercise label + rep count + current phase/state
- Optional: “no-rep” or warnings (e.g., shallow squat)

---

## 3) Success metrics (measurable)

### Realtime
- **FPS:** ≥ 20 FPS average on laptop webcam (target)
- **Latency:** < 100 ms/frame (target)

### Counting quality
- **RepCount dataset:** report **MAE** (mean absolute error) per exercise
- **Webcam manual test:** < 1 false rep/minute in “normal” conditions

### Robustness
- When pose confidence is low / user partially out of frame: **pause counting** (do not hallucinate reps)

---

## 4) Scope (V1) and non-goals

### V1 scope (portfolio-ready)
- Exercises: **Squat, Push-up, Bicep Curl**
- Single-person behavior, with deterministic primary subject selection if multiple appear
- Pipeline: Pose → Postprocess → Features → Classifier → FSM Counter → UI
- Offline evaluation: RepCount **MAE** (Event-F1 is V1.1 stretch)

### Non-goals (V1)
- Training large deep models end-to-end
- Medical-grade biomechanics accuracy
- Multi-camera / 3D pose

---

## 5) System architecture (dataflow)

### High-level pipeline
```text
Webcam Frames
   ↓
Pose Estimation (MediaPipe / YOLOv8-Pose)
   ↓
Pose Postprocess (confidence gating, smoothing, normalization)
   ↓
Feature Extraction (angles, velocities, ROM proxies)
   ↓
Exercise Classifier (rules or small ML model)
   ↓
Rep Counter (FSM per exercise) + Form Checker
   ↓
UI Overlay + Logging + Metrics


# RepCoach — Week 1 Plan (Mon Mar 2 – Sun Mar 8, 2026)

**Week theme:** Turn the pose stream into a *stable, countable signal* and ship a robust **Squat V1** (FSM + hysteresis).  
**Time budget:** ~15 hours (≈ 2–3 hrs/day)  
**Milestone M1:** Squat rep counting is reliable on webcam + unit-tested core logic.

---

## Definition of Done (Week 1)

By **Sun, Mar 8** you can:

- Run `make run` and see:
  - Pose overlay + FPS
  - **Exercise = SQUAT**
  - **Rep count increments correctly**
  - **Current FSM state** displayed (e.g., `UP/DESCENDING/BOTTOM/ASCENDING`)
- The counter is **robust to jitter** (no rapid double-counting)
- If pose confidence is low / joints missing: **state freezes and no rep events are emitted**
- Unit tests cover:
  - angle computation
  - FSM transitions (happy path + jitter edge cases)

---

## Architecture focus (what we implement this week)

```text
PoseEstimator -> PosePostprocess -> FeatureExtractor -> (Squat rule) -> SquatCounter(FSM) -> UI overlay
```

This week we harden:
- **PosePostprocess:** confidence gating + smoothing + normalization
- **FeatureExtractor:** angles + depth signal for squats
- **SquatCounter:** finite state machine with hysteresis

---

## File-level deliverables (what exists in the repo)

Create these modules (names are suggestions; keep consistent with `docs/design.md`):

```text
repcoach/processing/
  smoothing.py
  gating.py
  normalize.py

repcoach/features/
  angles.py
  squat_features.py

repcoach/counting/
  fsm_base.py
  squat_counter.py

tests/
  test_angles.py
  test_squat_fsm.py
```

---

## Day-by-day plan (activities + estimates + acceptance criteria)

### Mon (Mar 2) — Confidence gating + PoseFrame contract hardening (2–3h)
**Tasks**
1) Define `PoseFrame` (dataclass) in `repcoach/pose/types.py`
2) Implement confidence gating:
   - `overall_conf` computed from required joints (e.g., hips + knees + ankles)
   - If below threshold `τ`, mark `pose.valid = False`

**Acceptance criteria**
- When you cover the camera / leave frame, overlay indicates **“LOW CONF”** and counter does not move.

---

### Tue (Mar 3) — Smoothing + missing joint handling (2h)
**Tasks**
1) Implement simple EMA smoothing (start here; OneEuro optional later)
2) Missing joint strategy:
   - If joint missing for current frame, keep previous value for up to `N` frames OR mark invalid

**Acceptance criteria**
- Jitter visibly reduced (angles stop jumping wildly).
- If a joint disappears briefly, app degrades gracefully (no sudden rep spikes).

---

### Wed (Mar 4) — Normalization (2h)
**Tasks**
1) Choose coordinate convention (recommended: normalized [0..1] image coords)
2) Implement normalization:
   - translate so hip-center is origin
   - scale by torso length (shoulder-center to hip-center) or hip-to-shoulder distance

**Acceptance criteria**
- Features are more consistent when you move closer/farther from camera.

---

### Thu (Mar 5) — Squat feature extraction (2–3h)
**Tasks**
1) Implement angle helpers in `repcoach/features/angles.py`
   - knee angle (hip-knee-ankle)
   - hip angle (shoulder-hip-knee) or alternative stable proxy
2) Implement squat signals:
   - `knee_flex = 180 - knee_angle` (higher means deeper)
   - `depth_proxy` (hip y relative to knee y) if stable in your camera setup
3) Add debug overlay in app: show `knee_angle`, `depth_proxy`

**Acceptance criteria**
- Debug text updates smoothly and correlates with you squatting down/up.

---

### Fri (Mar 6) — Squat FSM + hysteresis (3h)
**Tasks**
1) Implement `SquatCounter` FSM states:
   - `UP`, `DESCENDING`, `BOTTOM`, `ASCENDING`
2) Define thresholds with hysteresis (example):
   - `enter_down`: knee_flex > 25
   - `enter_bottom`: knee_flex > 55
   - `exit_bottom`: knee_flex < 45
   - `back_to_up`: knee_flex < 20  -> **rep++**
3) Enforce confidence gating: if `pose.valid=False`, do **no transitions**

**Acceptance criteria**
- One squat produces exactly **one rep increment**.
- No rapid double counts if you hover near bottom.

---

### Sat (Mar 7) — Unit tests (2–3h)
**Tasks**
1) `test_angles.py`: verify angle computation on known triangles
2) `test_squat_fsm.py`:
   - feed a synthetic sequence of `knee_flex` values
   - ensure expected state transitions and exactly 1 rep event

**Acceptance criteria**
- `pytest -q` passes locally.
- A tiny jittery sequence does **not** produce extra counts.

---

### Sun (Mar 8) — Polish + internal demo script (1–2h)
**Tasks**
1) Add a “demo mode” overlay:
   - Exercise label (SQUAT)
   - Rep count
   - Current state
   - Confidence status
2) Write a short demo script in `docs/demo_week1.md` (30 sec walkthrough)

**Acceptance criteria**
- You can screen-record a clean 30–60 sec clip of squat counting.

---

## Parameters to tune (keep in config)
Create a config (YAML/JSON or plain dict) for:
- `CONF_THRESHOLD` (e.g., 0.5)
- `EMA_ALPHA` (e.g., 0.2)
- Squat FSM thresholds (`enter_down`, `enter_bottom`, `exit_bottom`, `back_to_up`)

This makes interviews easier: you can explain tuning as controlled configuration, not magic constants.

---

## If something breaks (fast recovery checklist)
- Webcam not opening → try `VideoCapture(1)` or close other camera apps
- MediaPipe lagging → reduce frame size (e.g., 640x480)
- Jitter too high → increase EMA alpha slightly or add minimum dwell time per state

---

## End-of-week output checklist
- [ ] Squat rep counting demo works on webcam
- [ ] Confidence gating prevents hallucinated reps
- [ ] Core logic unit tests exist and pass
- [ ] `docs/demo_week1.md` exists (demo script)


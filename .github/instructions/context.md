# Pose Gaze Context

## Scope

Active Holistic pipeline lives under `backend/ai_services/pose_gaze/pose_gaze`.
It has two entrypoints:

- `main`: realtime webcam processing.
- `holistic/batch_dataset`: offline processing from the canonical manifest and original videos.

Stage 1 and Stage 2 are historical references only. They are not runtime dependencies, and their artifacts are not required inputs for the new pipeline.

## Runtime Flow

```text
OpenCV BGR frame
  -> person detector
  -> duplicate suppression
  -> foreground policy and tracking
  -> TrackingManager session and student assignment
  -> Holistic landmarks for visible assigned tracks
  -> frame features or rolling temporal window
  -> XGBoost inference
```

Batch mode processes video frames sequentially, emits frame-track rows, then builds temporal windows. Realtime mode processes frames sequentially and maintains rolling windows. Both modes share landmark, tracking, and schema contracts.

## Tracking Contract

Shared contracts are defined in `pose_gaze/tracking/schemas/schemas.py`:

- `BoundingBox`: image-space `[x1, y1, x2, y2]` pixel coordinates.
- `PersonDetection`: detector bbox, confidence, and class name.
- `TrackedPerson`: track state, bbox, confidence, age, missed-frame count, presence, and optional `student_id`.
- `TrackPacket`: session ID, frame ID, timestamp, and tracked people.

`track_id` is temporary identity inside one tracking session. `student_id` is assigned by the session/user layer. They are not interchangeable.

Normalized landmark coordinates must stay finite. Invalid values use a validity mask and finite fallback values; never write NaN or Infinity.

## Current Tracking Behavior

Current runtime components:

- `tracking/detectors/detectors.py`: filters detector output to class `person`; Ultralytics uses confidence threshold and currently does not pass explicit `iou` or `max_det` settings.
- `tracking/tracker/tracker.py`: `IoUPersonTracker` uses greedy IoU association, `max_tracks=2`, and a grace period for missed detections.
- `tracking/manager/manager.py`: owns per-session tracker state, student assignment, persistence, restore, remap, and pose/gaze handoff.
- `tracking/webcam/webcam.py`: runs detector and manager once per frame.
- `holistic/landmark/landmarks.py`: keeps Holistic processor state per `track_id`.

Defaults in `pose_gaze/settings/settings.py`:

- `DEFAULT_PERSON_CONFIDENCE = 0.50`
- `DEFAULT_MIN_IOU = 0.30`
- `DEFAULT_MAX_MISSED_FRAMES = 30`
- `max_tracks = 2`

`TrackingManager.get_pose_gaze_input()` sends only visible tracks with assigned student IDs to landmark extraction. Handoff is ready only when all configured student slots are visible and assigned.

## Foreground Policy

Stage 2 contains a historical export selector, `_select_front_tracks()`, with these rules:

- confidence at least `0.5`;
- bbox area ratio at least `0.01`;
- sort by bbox area;
- keep at most two tracks.

This selector does not currently run as a runtime foreground gate. Preserve its intent, but implement runtime policy inside `pose_gaze/pose_gaze/tracking` rather than importing Stage 2 code.

Runtime foreground policy target:

1. Filter detector class and confidence.
2. Suppress duplicate or contained detections using overlap plus center and size evidence. Do not use IoU alone, because two nearby students can overlap.
3. Associate existing assigned/present tracks before considering new tracks.
4. Rank new candidates with confidence, bbox area ratio, and configured seat/ROI plausibility when available.
5. Keep at most two active student tracks.
6. Pass only canonical visible assigned tracks to Holistic.

A high-confidence background person must not displace an existing student. Two real students with moderate bbox overlap must remain separate.

## Tracking Invariants

- At most two active student tracks per frame/session.
- Existing student assignments receive association priority.
- One physical person must not create two canonical tracks from duplicate detections.
- Short detection loss preserves the same track/student identity during grace period.
- `track_id`, `student_id`, `TrackedPerson`, and `TrackPacket` contracts stay stable.
- Holistic processor state stays keyed to canonical `track_id`.
- Raw detection bbox, tracked bbox, crop bbox, and landmark coordinates remain distinguishable during debugging.
- Coordinate transforms must pass synthetic round-trip tests before letterbox code changes.

## Planned Tracking Update

Implement in this order:

1. Add raw detection, deduplicated detection, track, crop, and landmark telemetry.
2. Pass explicit Ultralytics `iou` and `max_det` settings without changing public detector contracts.
3. Add deterministic duplicate suppression before track creation.
4. Strengthen `IoUPersonTracker` association with existing-track priority and bounded center, area, and motion gates.
5. Add runtime foreground ranking and two-track cap.
6. Verify crop and landmark overlay stability after identity behavior passes.
7. Regenerate offline frame/window data only after runtime tracking tests pass.

Do not increase confidence or area thresholds only to hide duplicate bugs. Any new threshold needs fixtures containing two real students and a background outsider.

## ByteTrack Boundary

ByteTrack is optional. It may replace only the association backend after baseline `IoUPersonTracker` tests pass.

A ByteTrack adapter must:

- preserve `TrackedPerson` and `TrackPacket` contracts;
- enforce foreground policy and `max_tracks=2` outside or around the backend;
- preserve student assignment and remap semantics;
- avoid arbitrary background tracks;
- be disabled by default until benchmark results show better identity continuity without new swaps.

The YOLO paper does not define two-student foreground selection. It is relevant to robustness risks such as viewpoint, occlusion, scale, illumination, clutter, camera motion, and brightness, not as the source of this project-specific selector.

## Data and Model Boundary

The canonical manifest is input for batch processing, not webcam runtime. Original videos are preferred. Existing `data/processed/train`, `val`, and `test` artifacts are Stage 1 outputs and must not be mixed into the new contract without proving matching preprocessing, labels, FPS semantics, and split policy.

CSV contracts:

- `frame_track.csv`: one row per `frame_id` and canonical `track_id`, with timestamp, student identity, quality flags, landmark values, and validity masks.
- `window_features.csv`: one row per temporal window and directional source/peer relation, with frame counts, valid-frame counts, temporal features, and label.

XGBoost target is temporal-window features. Current frame-level `MODEL_FEATURE_COLUMNS` must be versioned or wrapped explicitly; runtime must never silently load a frame model with a window vector or vice versa.

Keep all seven classes: `c1` cellular device, `c2` exchange paper, `c3` looking friend, `c4` cheatsheet, `c5` no cheating, `c6` talking friend, and `c7` giving sign code. Keep `c1` and `c4` separate. Pair actions retain source/peer direction.

## Verification

Focused tracking tests must run before data regeneration or model retraining:

```powershell
python -m pytest backend/ai_services/pose_gaze/pose_gaze/tracking/tests/test_tracking/test_tracking.py
```

Required cases:

- duplicate boxes for one person produce one canonical track;
- two students with moderate overlap survive suppression;
- high-confidence outsider does not displace existing students;
- initial foreground selection respects configured seat/ROI policy;
- short disappearance restores track and student identity;
- crossing or near-contact does not swap student assignment;
- Holistic receives one processor key per canonical track;
- raw bbox, crop bbox, and landmark overlay identify the same person;
- letterbox inverse mapping passes synthetic pixel round-trip tolerance;
- detector parsing and explicit NMS settings are covered.

After tracking passes, validate manifest rules, frame/window CSV contracts, split leakage, model metadata, and batch extraction. Record unresolved benchmark risks instead of presenting unverified behavior as completed.

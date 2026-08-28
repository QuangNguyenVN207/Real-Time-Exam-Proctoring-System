# Pose/Gaze Dataset Blueprint v1

Scope: `pose_gaze` only. No audio, LSTM text, object detection labels, or face identity model.

## Decision

Use one relational, temporal dataset:

`session -> clip -> window -> frame + track -> landmarks -> temporal features -> label`

Keep two students in the same scene. A relation is `source_student -> peer_student`; never reduce it to a single crop label.

## Source claims

| Claim | Source | Dataset consequence |
|---|---|---|
| A gesture is a chunk; interaction uses two gesture sequences. | [R4...pdf](R4_CheatingVideoDescriptionBasedonSequencesofGestures_rev2%20(2).pdf), pp. 1-2 | Store `window_id`, source/peer IDs, frame range, and directional relation. |
| The R4 system chunks streams by 30 frames; its example input uses five gesture symbols. | [R4...pdf](R4_CheatingVideoDescriptionBasedonSequencesofGestures_rev2%20(2).pdf), p. 4 | Use fixed temporal windows. Repo v1 should use time-based windows with frame count recorded, not assume FPS. |
| XGBoost classifies gesture features; interaction must retain both subjects. | [R4...pdf](R4_CheatingVideoDescriptionBasedonSequencesofGestures_rev2%20(2).pdf), pp. 2, 5 | Export one tabular row per directional window plus aggregate quality features. |
| Classroom data collection needs consent and controlled camera views; the Exploring paper uses three views and records the scene setup. | [Exploring...pdf](Exploring_the_potential_of_skeleton_and_machine_le%20(5).pdf), pp. 1-2 | Manifest must record session, layout, camera/view, subject group, and consent/data-use status. |
| Cheating-action recognition is affected by viewpoint, occlusion, scale, illumination, clutter, camera motion, and brightness. | [data-07...pdf](data-07-00122%20(2).pdf), pp. 2-3 | Store quality fields separately; do not convert bad landmarks into guessed labels. |
| The Data paper treats short-range action as a few-second sequence and evaluates current action labels from temporally unfinished sequences. | [data-07...pdf](data-07-00122%20(2).pdf), p. 2 | Use a minimum temporal window and preserve `window_phase`; reject clips too short for the window. |

PDF page numbers refer to the PDF pages printed by `pdftotext`; paper page numbers can differ.

## Existing repo evidence

- [pose_gaze/spec.md](backend/ai_services/pose_gaze/spec.md): relational A->B/B->A reasoning, peer target zones, quality-gated gaze, baseline-relative head pose, temporal state machine.
- [CONTEXT_1.md](CONTEXT_1.md): one CSV row is `frame + track_id`; required metadata includes `clip_id`, `window_id`, `source_frame_index`, `timestamp_ms`, `track_id`, `student_id`, `class_code`, `window_phase`; confidence fields stay separate.
- [tracking/schemas.py](backend/ai_services/pose_gaze/tracking/schemas.py): `BoundingBox`, `TrackedPerson`, and `TrackPacket(session_id, frame_id, timestamp_ms, tracks)` are the current handoff contract.
- [tracking/tracker.py](backend/ai_services/pose_gaze/tracking/tracker.py): IoU is association only; track state keeps `age_frames`, `missed_frames`, `is_present`.
- [tracking/manager.py](backend/ai_services/pose_gaze/tracking/manager.py): manual `track_id -> student_id` assignment is persisted per session; pose input is ready only when required visible students are assigned.
- [stage0_manifest.csv](data/processed/stage0_manifest.csv): current labels include class/layout/action intervals, actors, split group, review, quality, and exclusion status.
- [pose_gaze_input.json](test_data_tracking/media_v_c1_s1_s2_02a29f21/pose_gaze_input.json): current runtime sample contains session/frame/time and assigned track bboxes/confidence.

## Canonical schema v1

### Frame-track table

One row per `frame_id + track_id`.

Required identity/time fields:

`schema_version, clip_id, window_id, session_id, source_frame_index, frame_id, timestamp_ms, track_id, student_id, peer_student_id, class_code, window_phase, split_group, split`

Tracking fields:

`bbox_x1, bbox_y1, bbox_x2, bbox_y2, track_confidence, track_age_frames, missed_frames, is_present`

Quality fields, never merged:

`frame_quality_score, pose_quality_score, face_quality_score, kp_conf, face_valid, pose_valid, hands_valid, landmark_missing_mask`

Landmark representation:

- normalized crop/frame coordinates in `[0, 1]` where applicable;
- `x`, `y`, `z`, `visibility/presence` per selected keypoint;
- missing values are `0.0` with a mask, never NaN/Infinity or fabricated points;
- keep upper-body/head, mouth, and left/right hands per repo policy; do not export face oval as training features.

### Window feature table

One row per directional relation and window:

`window_id, source_student_id, peer_student_id, start_ms, end_ms, frame_count, valid_frame_count, valid_ratio, track_switch_count, mean_track_confidence, mean_pose_quality, mean_face_quality, delta_yaw_mean/std, delta_pitch_mean/std, delta_roll_mean/std, torso_turn_mean/std, head_toward_peer_ratio, eye_toward_peer_ratio, peer_zone_angle_mean/std, pair_score_mean/max, label, label_confidence, label_source`

`eye_toward_peer_ratio=0` only means no positive eye signal; preserve `face_valid` and `gaze_unknown_ratio` so this is not confused with observed center gaze.

### Labels

Directional event labels:

- `normal`
- `looking_toward_peer`
- `looking_down_toward_peer`
- `body_turn_toward_peer`
- `standing_or_leaving_seat`
- `face_missing`
- `unknown_quality`

Rules:

- `c1` phone cheating and `c4` cheat-sheet cheating remain separate.
- Two-person actions keep source and peer IDs; do not flatten to one actor label.
- Mixed or ambiguous c1+c4 clips are excluded from v1 training until taxonomy is resolved.
- Clip around 1 second is excluded or re-shot; it cannot support the required temporal window.
- `unknown_quality` is a quality outcome, not a cheating label.

## Window and split policy

- Default window: 2 seconds, configurable; record exact start/end milliseconds and actual frame count.
- Overlap may be used for inference, but split by `split_group`/recording session before window generation to prevent leakage.
- Train/validation/test split by session and subject group, never random frames from one video.
- Keep the original clip and action intervals in the manifest; windows inherit labels only when temporal overlap and review rules agree.

## Pipeline contract

`person detector -> tracker -> foreground selector -> manual student assignment -> crop -> MediaPipe Holistic -> quality gate -> frame-track CSV -> temporal feature table -> XGBoost-ready classifier input`

YOLO/person detection supplies bbox and confidence. Tracker supplies stable `track_id`; it does not identify students. Holistic supplies landmarks. Feature extraction supplies relational temporal features. The classifier receives features, not raw guessed labels.

## Verification checklist

- [ ] Every row has `schema_version`, `session_id`, `clip_id`, `window_id`, `frame_id`, `timestamp_ms`, `track_id`.
- [ ] Every visible assigned track has a stable `student_id`; unassigned tracks are not model-ready.
- [ ] No NaN/Infinity; missing landmarks have mask + zero/default value.
- [ ] `track_confidence`, `kp_conf`, face/pose/frame quality stay separate.
- [ ] c1/c4 separation and short-clip rule are enforced.
- [ ] Directional pair rows preserve source and peer IDs.
- [ ] Enforce `split-before-window`: split by session/subject group before overlapping windows are emitted.
- [ ] Existing Stage 1 artifacts are reused; no expensive video regeneration is required for this blueprint.

## Known gaps

- Current repository has tracking/pose input contracts but no single canonical landmark CSV and no finalized feature-table writer.
- Thresholds such as face pixel size, pair score, and valid-frame ratio remain hypotheses until validation clips are labeled.
- Current IoU tracker is a deterministic foundation, not YOLO + ByteTrack production tracking.
- PDF claims are research context; repository spec and artifacts control implementation decisions.

# Landmark Pipeline Phase 1-4 Report

## Scope

Only `backend/ai_services/pose_gaze/pose_gaze/**` was changed. Temporal-window inference, scheduling, and benchmark work remain deferred.

## Implemented

- Bounded EMA crop/bbox stabilization remains enabled per track.
- Removed per-frame Holistic processor reset when face landmarks are empty.
- Added bounded face hold (`face_hold_frames`, default `3`). Held coordinates are explicitly marked with `face_predicted=true`; they are not counted as fresh face observations.
- Added additive quality fields to `TrackHolisticResult` and streamed JSON:
  - `face_valid`
  - `mouth_valid`
  - `face_predicted`
- Landmark JSON schema is now `format_version=3` and declares `quality_fields`.
- Added dependency-light JSON validator: `holistic/test_media/validate_landmarks.py`.
- Added mouth landmark index set from the existing lip connections.
- Added an opt-in standalone Face Landmarker fallback. It runs only after repeated
  Holistic face loss and at a configurable interval; normal frames pay no fallback
  inference cost. Enable it with `--face-fallback-model`.
- Defined sparse feature groups for mouth, both eyes, nose, and both ears. These
  groups intentionally do not include the face oval/contour.

## Validation

Generated artifact:

`holistic/test_media/outputs/sample_v3.json`

Command used: 120-frame extraction from `VID20260804140845.mp4`, with YOLO `yolov8n.pt`, bbox alpha `0.5`, crop alpha `0.8`, face hold `3`.

Validator result:

- `valid: true`
- `format_version: 3`
- `frame_count: 120`
- `frame_track_count: 240`
- face-valid tracks: `179 / 240` (`74.6%`)
- mouth-valid tracks: `188 / 240` (`78.3%`)
- predicted face tracks: `9 / 240` (`3.8%`)
- structural, finite-value, timestamp, duplicate-key, and bbox checks: passed

The annotated sample video was generated successfully at `holistic/test_media/outputs/sample_v3.mp4`.

## Feature benchmark on previously failing video

Input: `VID20260804140845.mp4` (the prior full demo had face loss and legacy
schema). A new 120-frame demo was generated at
`holistic/test_media/outputs/sample_features.mp4` with JSON at
`holistic/test_media/outputs/sample_features.json`.

- Wall time: `22.04 s`
- Frame-track rows: `240`
- Face-valid rows: `179/240`
- Mouth-valid rows: `188/240`
- Predicted face rows: `9/240`
- JSON validator: **PASS**
- No face-outline connections are included in the feature groups.

The same configuration was then run on the complete video:

- Output video: `holistic/test_media/outputs/full_features.mp4`
- Output JSON: `holistic/test_media/outputs/full_features.json`
- Wall time: `50.48 s`
- Frames: `455`
- Frame-track rows: `910`
- Face-valid rows: `527/910` (`57.9%`)
- Mouth-valid rows: `567/910` (`62.3%`)
- Predicted face rows: `40/910` (`4.4%`)
- JSON validator: **PASS**

The current workspace does not contain `weights/mediapipe/face_landmarker.task`,
so the fallback was compile-validated but not benchmarked in this run. When the
CLI option is supplied, the model is resolved/downloaded lazily through the
existing MediaPipe model helper.

## Standalone fallback benchmark

The fallback model was downloaded to `weights/mediapipe/face_landmarker.task`
and tested on the complete previously failing video. It now runs in independent
IMAGE mode per ROI, because a shared VIDEO-mode graph caused cross-track
timestamp/state conflicts.

- Output video: `holistic/test_media/outputs/full_features_fallback.mp4`
- Output JSON: `holistic/test_media/outputs/full_features_fallback.json`
- Wall time: `54.56 s`
- Face-valid rows: `646/910` (`70.9%`)
- Mouth-valid rows: `790/910` (`86.8%`)
- Predicted face rows: `144/910`
- JSON validator: **PASS**
- Original 5-9 second interval, track 2: `18/120` face-valid rows

The fallback improves overall and mouth coverage, but does not fully recover
track 2 during 5-9 seconds. The remaining issue is detector visibility/quality
for that ROI, not JSON serialization or timestamp handling. Full continuous
face recovery is therefore still not claimed.

## Dedicated face ROI experiment

The fallback was additionally tested on a dedicated upper `55%` ROI of each
person crop, while Holistic continued using its fixed-size full crop. The
experiment completed successfully in `53.68 s`:

- Output: `holistic/test_media/outputs/full_features_face_roi.json`
- Face-valid rows: `651/910` (`71.5%`)
- Mouth-valid rows: `797/910` (`87.6%`)
- Original 5-9 second interval, track 2: `18/120` face-valid rows
- JSON validator: **PASS**

The dedicated ROI improved aggregate metrics only marginally and did not solve
the target interval. It should not be used as the dataset quality gate yet.

## Existing JSON audit

The previous full demo JSON was intentionally not rewritten. It is legacy data:

- `format_version=2`
- `455` frames and `910` frame-track rows
- missing `face_valid`, `mouth_valid`, `face_predicted`, and `quality_fields`
- validator result: **invalid legacy artifact**

This is a schema failure, not evidence that its landmark coordinates are correct.

## Test notes

`py_compile` and VS Code diagnostics passed for all touched Python files. The focused unittest command was attempted with two import roots; both were blocked by existing package-layout imports (`backend...` versus the nested `pose_gaze` package), not by an assertion failure. The new JSON validator and real 120-frame extraction are executable checks and passed.

MediaPipe emitted its existing `NORM_RECT without IMAGE_DIMENSIONS` warning during extraction; the process completed successfully.

## Remaining limitation

The sample proves that face/mouth quality is now observable and continuity is bounded, but it does not prove full face recovery for every frame. A separate face detector/Face Landmarker ROI fallback is still required if the acceptance target is near-continuous face coverage. The current pipeline correctly reports missing quality instead of fabricating valid landmarks.

## Error log retained

1. Per-frame processor reset/reacquire was an incorrect attempted fix and was removed.
2. EMA smoothing improved geometry stability but did not solve face detection loss.
3. Confidence-threshold changes were attempted before proving whether the model emitted face landmarks.
4. An initial test invocation used the wrong `PYTHONPATH`; a second invocation exposed the repository's mixed package roots. No unrelated wrapper code was changed.
5. Demo flags were added before the quality/schema contract was proven.

## Status

Phases 1-4 are implemented at the contract, bounded-continuity, validator, and sample-extraction level. Full face-recovery acceptance remains **not claimed** because the current MediaPipe output still has missing face observations. Phases 5-7 are deferred as requested by the current scope.

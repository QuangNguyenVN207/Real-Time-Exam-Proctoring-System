# Pose/Gaze causal actor demo

This package classifies each actor, not a whole video. The promoted realtime
artifact is
`tmp/benchmark_face_mesh_restored_cuda_snapshot_verify_final_20260820/` and
emits `suspicious_activity`, `c2`, `c3`, or temporary `c5` for each actor.
It uses current and past frames only. The historical three-class
`benchmark goc` remains a separate reference and is not the deployed profile.

## Setup

Use the repository virtual environment.  It must provide `opencv-python`,
`ultralytics`, `mediapipe`, `numpy`, `xgboost`, and `scikit-learn`.
The repository tracks the YOLO person-detector weights at `weights/yolov8n.pt`.

## Webcam

Run from the repository root:

```powershell
$env:PYTHONPATH="backend\ai_services\pose_gaze"

& ".\.venv\Scripts\python.exe" `
  -m pose_gaze.holistic.test_webcam `
  --target-fps 8 `
  --device 0 `
  --xgboost-device cuda:0 `
  --actions c2,c3,suspicious_activity `
  --causal-model-dir "tmp\benchmark_face_mesh_restored_cuda_snapshot_verify_final_20260820" `
  --live-pair student_01:student_02
```

Run the one-person head-turn control demo:

```powershell
$env:PYTHONPATH="backend\ai_services\pose_gaze"

& ".\.venv\Scripts\python.exe" `
  -m pose_gaze.holistic.test_webcam.test_webcam_one_person `
  --target-fps 8 `
  --device 0
```

The one-person command verifies actor tracking and head-turn extraction only;
it does not classify directed two-actor C3.

- `X`: reset classifier evidence and start a new 15-frame readiness period; camera and
  tracker continue running.
- `Q`: quit.
- C2 requires the configured explicit pair and midpoint evidence.
- The temporary one-person C3 demo uses screen-right as the turn direction.
  It is not an official benchmark rule.

## Video demo

```powershell
$env:PYTHONPATH="C:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze"
$video = "C:\path\to\your-video.mp4"

& "C:\Real-Time-Exam-Proctoring-System\.venv\Scripts\python.exe" `
  -m pose_gaze.holistic.test_media `
  $video `
  --model "weights\yolov8n.pt" `
  --xgboost-model-dir "tmp\benchmark_face_mesh_restored_cuda_snapshot_verify_final_20260820" `
  --causal-live `
  --live-pair student_01:student_02 `
  --target-fps 30
```

A tracked sample is available at
`backend/ai_services/pose_gaze/demo/pose_gaze_sample.mp4`. Any local MP4 is
processed causally, frame by frame.

## Artifact contents

Inference reads these eight model/contract files:

```text
causal_actor_metrics.json
causal_c2_feature_names.json
causal_c2_specialist.ubj
causal_c3_feature_names.json
causal_c3_specialist.ubj
causal_suspicious_activity_feature_names.json
causal_suspicious_activity_specialist.ubj
causal_specialist_predictions.csv
```

The promoted actor-level result is macro-F1 `0.8264485221` over
`[suspicious_activity,c2,c3,c5]`. The confusion matrix is
`[[15,0,0,3],[1,11,0,0],[0,0,5,3],[3,0,0,15]]` in that order.

# Pose/Gaze causal C2/C3 demo

This package classifies each actor, not a whole video.  The committed
`benchmark goc` artifact is the causal C2/C3/C5 model under
`tmp/behavior_actor_causal_pose_only_20260812/`; it uses current and past
frames only.

## Setup

Use the repository virtual environment.  It must provide `opencv-python`,
`ultralytics`, `mediapipe`, `numpy`, `xgboost`, and `scikit-learn`.
The repository tracks the YOLO person-detector weights at `weights/yolov8n.pt`.

## Webcam

Run from the repository root:

```powershell
$env:PYTHONPATH="C:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze"

& "C:\Real-Time-Exam-Proctoring-System\.venv\Scripts\python.exe" `
  -m pose_gaze.holistic.test_webcam `
  --actions c2,c3 `
  --causal-model-dir "C:\Real-Time-Exam-Proctoring-System\tmp\behavior_actor_causal_pose_only_20260812" `
  --live-pair student_01:student_02 `
  --target-fps 10
```

- `X`: reset classifier evidence and start a new 30-frame warmup; camera and
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
  --xgboost-model-dir "tmp\behavior_actor_causal_pose_only_20260812" `
  --causal-live `
  --live-pair student_01:student_02 `
  --target-fps 10
```

You may supply any local MP4; it is processed causally, frame by frame.

An optional tracked sample is available at
`backend/ai_services/pose_gaze/demo/pose_gaze_sample.mp4`.

## Artifact contents

Inference reads only these five files:

```text
causal_actor_metrics.json
causal_c2_feature_names.json
causal_c2_specialist.ubj
causal_c3_feature_names.json
causal_c3_specialist.ubj
```

The locked actor-level result is macro-F1 `0.8730158730` over `[c2,c3,c5]`.
`causal_specialist_predictions.csv` is audit output and is intentionally not
committed.

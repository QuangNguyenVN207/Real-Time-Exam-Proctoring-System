# Pose/Gaze causal actor demo

This package classifies each actor, not a whole video.  The committed
The promoted realtime artifact is
`tmp/behavior_actor_extended_suspicious_current_geometry_20260815/` and
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
$env:PYTHONPATH="C:\Real-Time-Exam-Proctoring-System\backend\ai_services\pose_gaze"

& "C:\Real-Time-Exam-Proctoring-System\.venv\Scripts\python.exe" `
  -m pose_gaze.holistic.test_webcam `
  --actions c2,c3,c5,suspicious_activity `
  --causal-model-dir "C:\Real-Time-Exam-Proctoring-System\tmp\behavior_actor_extended_suspicious_current_geometry_20260815" `
  --live-pair student_01:student_02 `
  --target-fps 30
```

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
  --xgboost-model-dir "tmp\behavior_actor_extended_suspicious_current_geometry_20260815" `
  --causal-live `
  --live-pair student_01:student_02 `
  --target-fps 30
```

No sample video is committed. Supply a local MP4; it is processed causally,
frame by frame.

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

The promoted actor-level result is macro-F1 `0.7284946237` over
`[suspicious_activity,c2,c3,c5]`. The confusion matrix is
`[[9,2,0,9],[1,10,0,1],[0,0,5,3],[1,0,0,19]]` in that order.

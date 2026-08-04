# Test webcam tracking

## Ý tưởng

CLI kiểm tra realtime riêng YOLO + IoU tracking + assignment trước khi ghép với
Holistic.

## Công năng và nhiệm vụ

Parse tham số webcam/model/tracker, tạo fresh hoặc restore session, chạy preview
và lưu handoff JSON khi thoát.

## Input

Webcam index, YOLO model/device, confidence, FPS, `max_tracks`, IoU, grace
period và kích thước capture.

## Output

Preview annotated, tracking state và `pose_gaze_input.json`.

## Chạy

```powershell
python -m backend.ai_services.pose_gaze.tracking.test_webcam --target-fps 10
```

# Test webcam Holistic

## Ý tưởng

CLI realtime ghép webcam person tracking với một Holistic context riêng cho mỗi
track đang hiện diện.

## Công năng và nhiệm vụ

Parse cấu hình, chạy detect/track/landmark, vẽ overlay FPS và hỗ trợ assignment
không blocking.

## Input

Webcam, YOLO/tracker config, Holistic model/input size/confidence và crop
padding.

## Output

Preview annotated realtime và tracking handoff JSON khi thoát.

## Chạy

```powershell
python -m backend.ai_services.pose_gaze.holistic.test_webcam --target-fps 8
```

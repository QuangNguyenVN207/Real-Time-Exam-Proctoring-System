# Test media

## Ý tưởng

Chạy toàn pipeline tracking + Holistic trên một ảnh hoặc video, đồng thời stream
JSON theo frame để không giữ toàn bộ video trong RAM.

## Công năng và nhiệm vụ

Nhận diện loại media, sampling FPS, auto-assign student ID, annotate, ghi media,
tracking JSON và landmarks schema v2.

## Input

Đường dẫn ảnh/video cùng các tùy chọn output, display, sampling, YOLO, tracker,
Holistic confidence/model/input size.

## Output

Annotated image/video, `*_landmarks.json` và session tracking JSON.

## Chạy

```powershell
python -m backend.ai_services.pose_gaze.holistic.test_media input.mp4 --target-fps 8 --no-display
```

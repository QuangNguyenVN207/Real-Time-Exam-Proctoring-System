# Webcam tracking module

## Ý tưởng

Đóng gói detector, manager và vòng lặp webcam thành một module có thể tái sử
dụng cho tracking độc lập hoặc làm đầu vào của Holistic.

## Công năng và nhiệm vụ

- `PersonTrackingConfig`: cấu hình model/session/tracker.
- `PersonTrackingModule`: xử lý từng frame, vẽ track và mở webcam.
- `ProcessingRateController`: giới hạn và đo FPS toàn vòng lặp.

## Input

OpenCV BGR frame hoặc webcam index cùng `PersonTrackingConfig`.

## Output

`TrackPacket`, annotated frame và tracking/session JSON khi kết thúc CLI.

## Import

```python
from backend.ai_services.pose_gaze.tracking.webcam import PersonTrackingModule
```

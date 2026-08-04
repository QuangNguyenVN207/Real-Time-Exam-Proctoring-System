# Schemas

## Ý tưởng

Định nghĩa data contract nhẹ, không phụ thuộc OpenCV/YOLO/MediaPipe, dùng chung
giữa detector, tracker, API và Holistic.

## Công năng và nhiệm vụ

Chứa `BoundingBox`, `PersonDetection`, `TrackedPerson`, `TrackPacket` và hàm
chuyển payload dictionary thành detection đã validate.

## Input

Tọa độ bbox, confidence, track metadata hoặc payload JSON-compatible.

## Output

Dataclass bất biến và các dictionary tuần tự hóa ổn định.

## Import

```python
from backend.ai_services.pose_gaze.tracking.schemas import BoundingBox, TrackPacket
```

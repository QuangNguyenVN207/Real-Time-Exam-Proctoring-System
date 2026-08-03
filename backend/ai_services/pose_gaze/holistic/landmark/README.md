# Landmark extractor

## Ý tưởng

Mỗi `track_id` có một MediaPipe Holistic processor và timestamp riêng. Crop
người được letterbox về canvas ổn định, lọc index sử dụng rồi map tọa độ về frame
gốc.

## Công năng và nhiệm vụ

- Quản lý processor theo track và backend legacy/Tasks.
- Chỉ tạo pose `0–24`, hand `0–20` và 53 face point đã chọn.
- Loại bỏ thành phần độ sâu khỏi object/JSON.
- Áp dụng confidence chuẩn `0.50` và ngưỡng mềm `0.20`.
- Vẽ pose, tay và selected face lên frame.

## Input

OpenCV BGR frame, `TrackPacket`, model/input size, crop padding và confidence.

## Output

Tuple `TrackHolisticResult` có mapping track/student/bbox và landmark hai chiều.

## Import

```python
from backend.ai_services.pose_gaze.holistic.landmark import HolisticLandmarkExtractor
```

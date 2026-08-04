# Detectors

## Ý tưởng

Chuẩn hóa nhiều detector người về cùng hợp đồng `PersonDetector` để tracker
không phụ thuộc trực tiếp vào OpenCV HOG hay Ultralytics YOLO.

## Công năng và nhiệm vụ

- `OpenCVHOGPersonDetector`: fallback CPU.
- `UltralyticsPersonDetector`: chạy YOLO, chỉ giữ class `person` đạt confidence.

## Input

Một OpenCV BGR frame và cấu hình model/confidence/device khi khởi tạo.

## Output

Danh sách `PersonDetection`, mỗi phần tử có bbox, confidence và class name.

## Import

```python
from backend.ai_services.pose_gaze.tracking.detectors import UltralyticsPersonDetector
```

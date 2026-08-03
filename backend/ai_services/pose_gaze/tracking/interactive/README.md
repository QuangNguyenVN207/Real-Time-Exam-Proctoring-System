# Interactive controls

## Ý tưởng

Nhận phím không blocking để điều khiển webcam trong lúc pipeline vẫn duy trì
capture và inference đều đặn.

## Công năng và nhiệm vụ

Gán/bỏ student ID, retry thao tác, tăng giảm FPS, yêu cầu save/finalize/quit và
tạo nội dung overlay hướng dẫn.

## Input

Phím từ cửa sổ OpenCV hoặc terminal Windows, `TrackingManager`, session ID và
`ProcessingRateController`.

## Output

Trạng thái interaction, `TrackPacket` cập nhật và các cờ điều khiển vòng lặp.

## Import

```python
from backend.ai_services.pose_gaze.tracking.interactive import WebcamInteractionController
```

# Manager

## Ý tưởng

Tách vòng đời session và assignment khỏi thuật toán IoU để trạng thái theo dõi
có thể lưu, khôi phục và dùng qua API/CLI.

## Công năng và nhiệm vụ

- Tạo, restore và đóng tracking session.
- Nhận detection theo frame và gọi tracker.
- Gán/remap/bỏ `student_id`.
- Persist `tracking_state.json` và sinh `pose_gaze_input.json`.

## Input

Session ID, frame/timestamp và danh sách `PersonDetection`.

## Output

`TrackPacket`, trạng thái session trên đĩa và handoff JSON cho Holistic.

## Import

```python
from backend.ai_services.pose_gaze.tracking.manager import TrackingManager
```

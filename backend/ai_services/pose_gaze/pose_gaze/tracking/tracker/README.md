# Tracker

## Ý tưởng

Duy trì ID người qua các frame bằng IoU association và grace period ngắn khi
detection mất tạm thời.

## Công năng và nhiệm vụ

- Ghép detection mới với track còn hiệu lực.
- Tạo track khi còn slot và xóa track quá grace period.
- Remap ID, kiểm tra trạng thái hiện diện và export/restore tracker state.

## Input

Danh sách `PersonDetection`, `max_tracks`, `min_iou` và
`max_missed_frames`.

## Output

Tuple `TrackedPerson` có ID, bbox, confidence, tuổi và trạng thái hiện diện.

## Import

```python
from backend.ai_services.pose_gaze.tracking.tracker import IoUPersonTracker
```

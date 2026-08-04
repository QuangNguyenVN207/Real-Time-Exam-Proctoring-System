# Tracking

## Ý tưởng và nhiệm vụ

Package này biến detection `person` theo từng frame thành các track có ID ổn
định. Nó chịu trách nhiệm detect người, ghép detection với track cũ bằng IoU,
quản lý grace period khi mất dấu, gán `student_id`, lưu/khôi phục session và tạo
handoff cho bước Holistic.

Pipeline:

```text
OpenCV BGR frame -> person detector -> IoU tracker -> TrackPacket
                 -> assignment/session manager -> pose_gaze_input.json
```

## Cấu trúc file

| File | Công năng |
|---|---|
| `detectors/detectors.py` | Adapter OpenCV HOG và Ultralytics YOLO; chỉ trả class `person` đạt confidence |
| `tracker/tracker.py` | Ghép IoU, tạo/xóa track, grace period, remap và export/restore state |
| `manager/manager.py` | Quản lý session, assignment, persistence và JSON handoff |
| `schemas/schemas.py` | Hợp đồng dữ liệu `BoundingBox`, `PersonDetection`, `TrackedPerson`, `TrackPacket` |
| `webcam/webcam.py` | `PersonTrackingModule`, xử lý một frame và vòng lặp webcam tái sử dụng |
| `interactive/interactive.py` | Phím gán/bỏ ID, điều chỉnh FPS, lưu và thoát |
| `test_webcam/test_webcam.py` | CLI kiểm tra tracking realtime |
| `test_image/test_image.py` | Kiểm tra detector trên một ảnh; `image.png` nằm cùng folder |
| `tests/test_tracking/test_tracking.py` | Unit/regression test của tracker và session |

Các default không nằm trong package này mà dùng chung từ
`backend.ai_services.pose_gaze.settings`; file thật nằm trong
`settings/settings.py`.

## Input

`PersonTrackingModule.process_frame()` nhận:

- `frame`: ảnh OpenCV BGR, shape `(height, width, 3)`.
- `timestamp_ms`: timestamp tùy chọn; nếu bỏ trống sẽ dùng thời gian hệ thống.
- Config detector/tracker: model path, device, person confidence, `max_tracks`,
  `min_iou`, `max_missed_frames`, session và storage root.

## Output

Output trong bộ nhớ là `TrackPacket`:

```json
{
  "session_id": "room_01",
  "frame_id": 12,
  "timestamp_ms": 1234,
  "tracks": [
    {
      "track_id": 1,
      "student_id": "student_01",
      "bbox_xyxy": [100, 80, 400, 700],
      "track_confidence": 0.91,
      "age_frames": 12,
      "missed_frames": 0,
      "is_present": true
    }
  ]
}
```

Session được lưu tại `test_data_tracking/<session_id>/tracking_state.json`; handoff
cuối dùng cho pose/gaze là `pose_gaze_input.json`.

## Chạy kiểm tra

```powershell
python -m backend.ai_services.pose_gaze.tracking.test_webcam --target-fps 10
python -m unittest discover -s backend/ai_services/pose_gaze/tracking/tests -v
```

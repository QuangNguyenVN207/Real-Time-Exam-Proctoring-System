# Pose/Gaze Tracking Foundation

Phần này hoàn thành P0–P1 cho Module 1: theo dõi tối đa hai người, gán `track_id` cho `student_id`, lưu ánh xạ theo session và xuất packet sẵn sàng cho pose/gaze.

## Luồng API

Khởi động backend sau khi cài dependencies:

```powershell
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

1. Tạo session:

```http
POST /api/pose-gaze/sessions
{"session_id":"exam-room-01"}
```

2. Detector (YOLO/HOG hoặc service dùng chung) gửi person detections của từng frame:

```http
POST /api/pose-gaze/sessions/exam-room-01/detections
{
  "frame_id": 1,
  "detections": [
    {"bbox_xyxy":[90,120,310,690],"confidence":0.96},
    {"bbox_xyxy":[430,125,650,690],"confidence":0.94}
  ]
}
```

3. Dashboard lấy `track_id` trả về, rồi gán thủ công:

```http
PUT /api/pose-gaze/sessions/exam-room-01/tracks/1/assignment
{"student_id":"SV_A"}
```

4. Pose/gaze lấy input chỉ gồm track đang thấy và đã gán ID:

```http
GET /api/pose-gaze/sessions/exam-room-01/pose-gaze-input
```

Trường `ready=true` chỉ khi đủ hai track đang hiển thị và mỗi track có `student_id`. Module pose/gaze ở bước tiếp theo chỉ cần nhận `tracks` từ endpoint này cùng frame cùng `frame_id`.

## Detector adapters

`detectors.py` có `OpenCVHOGPersonDetector` (fallback CPU) và `UltralyticsPersonDetector` (YOLO). Cả hai là adapter tùy chọn; tracker không phụ thuộc vào thư viện detector.

Model YOLO dùng cho adapter phải có class `person`. File `weights/yolov8_finetuned.pt` hiện cần được kiểm tra class trước khi dùng, vì model object-detection tùy biến có thể không chứa class này.

## Tests

```powershell
python -m unittest discover -s backend/ai_services/pose_gaze/tests -v
```

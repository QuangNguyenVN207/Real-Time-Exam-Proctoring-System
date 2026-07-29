# Pose/Gaze Tracking Foundation

Phần này hoàn thành P0–P1 cho Module 1: theo dõi tối đa hai người, gán
`person_id` ổn định do giám thị chọn vào `track_id` tạm, lưu ánh xạ theo session
và xuất packet sẵn sàng cho pose/gaze.

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

3. Dashboard lấy `track_id` tạm trả về, rồi gán `person_id` thủ công:

```http
PUT /api/pose-gaze/sessions/exam-room-01/tracks/1/assignment
{"person_id":"SV_A"}
```

Response vẫn chứa `student_id` để tương thích client cũ, nhưng `person_id` là tên
field nên dùng. Sau khi gán, pipeline lưu fingerprint vùng mặt/quần áo; nếu người
bị mất track lâu rồi xuất hiện lại, fingerprint khớp sẽ tự khôi phục numeric
`track_id`, `person_id` và paper ownership. Người khác ở cùng vị trí nhận track
tạm mới. Khi hình ảnh không đủ rõ, gán cùng `person_id` vào track tạm là fallback
thủ công. Nếu track cũ vẫn đang hiện, hệ thống từ chối gán trùng ID.

4. Pose/gaze lấy input chỉ gồm track đang thấy và đã gán ID:

```http
GET /api/pose-gaze/sessions/exam-room-01/pose-gaze-input
```

Trường `ready=true` chỉ khi đủ hai track đang hiển thị và mỗi track có
`person_id`. Module pose/gaze ở bước tiếp theo chỉ cần nhận `tracks` từ endpoint
này cùng frame cùng `frame_id`.

## Detector adapters

`detectors.py` có `OpenCVHOGPersonDetector` (fallback CPU) và `UltralyticsPersonDetector` (YOLO). Cả hai là adapter tùy chọn; tracker không phụ thuộc vào thư viện detector.

Model YOLO dùng cho adapter phải có class `person`. File `weights/yolov8_finetuned.pt` hiện cần được kiểm tra class trước khi dùng, vì model object-detection tùy biến có thể không chứa class này.

## Paper tracking

Pipeline mới tận dụng `track_id` của người để gắn `owner_track_id` cho từng
`paper_id`. Giấy thi hợp lệ được đăng ký trong giai đoạn setup; giấy vật lý khác
được theo dõi và xác nhận qua nhiều lần inference trước khi cảnh báo. Xem hướng
dẫn đầy đủ tại [PAPER_TRACKING.md](PAPER_TRACKING.md).

`paper_id` hiển thị chính thức do giám thị nhập. Paper vừa detect chỉ có ID tạm
và response trả `paper_id_assigned=false`; webcam sẽ hỏi ID số dương. Sau khi
đăng ký, pipeline lưu appearance fingerprint của đề thi theo owner. Khi giấy bị
re-track, fingerprint khớp sẽ tự phục hồi ID, authorization và owner; giấy khác
tạo ID mới và alert kèm `owner_person_id`.

## Tests

```powershell
python -m unittest discover -s backend/ai_services/pose_gaze/tracking/tests -v
python -m unittest discover -s backend/ai_services/object_detect/tests -v
```

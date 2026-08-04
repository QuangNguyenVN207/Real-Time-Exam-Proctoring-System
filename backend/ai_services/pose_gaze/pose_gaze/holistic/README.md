# Holistic landmarks

## Ý tưởng và nhiệm vụ

Package này nhận `TrackPacket` từ tracking, crop riêng từng người đang xuất hiện
và chạy một MediaPipe Holistic processor độc lập cho mỗi `track_id`. Crop thay
đổi kích thước được letterbox về canvas vuông cố định rồi tọa độ được map ngược
về crop và frame gốc.

Pipeline:

```text
OpenCV BGR frame + TrackPacket
  -> crop từng bbox -> letterbox -> MediaPipe Holistic theo track_id
  -> lọc index ngay khi tạo point -> confidence gate -> JSON/drawing
```

Mỗi track có processor và timestamp riêng để landmark của hai sinh viên không bị
gán chéo.

## Cấu trúc file

| File | Công năng |
|---|---|
| `landmark/landmarks.py` | Crop/letterbox, quản lý processor, lọc và map landmark, confidence gate, serializer và drawing |
| `feature_csv/feature_csv.py` | Flatten landmark thành schema CSV/XGBoost cố định, không có `z` |
| `batch_dataset/batch_dataset.py` | Đọc hàng nghìn ảnh và stream `train.csv`, `val.csv`, `test.csv` |
| `test_webcam/test_webcam.py` | CLI realtime kết hợp YOLO tracking và Holistic |
| `test_media/test_media.py` | CLI ảnh/video, annotated output và JSON landmarks dạng stream |
| `tests/test_landmarks/test_landmarks.py` | Test lọc index, map letterbox, timestamp và ba mức confidence |
| `tests/test_feature_csv/test_feature_csv.py` | Test schema feature/CSV |
| `tests/test_batch_dataset/test_batch_dataset.py` | Test discovery và deterministic split |

## Landmarks thực sự được giữ

Landmark không dùng bị bỏ trước khi tạo `LandmarkPoint`, không phải chỉ ẩn khi
vẽ:

| Nhóm | Index được tạo/xuất |
|---|---|
| Pose normalized/world | `0–24` (mặt, vai, tay và hông); không tạo `25–32` của chân |
| Left hand normalized/world | `0–20` |
| Right hand normalized/world | `0–20` |
| Selected face | 53 index thuộc môi, trán và trục đầu; không xuất face oval/tessellation dư |

Selected face indices:

```text
0, 1, 10, 13, 14, 17, 37, 39, 40, 54, 61, 67, 78, 80, 81, 82,
84, 87, 88, 91, 95, 103, 109, 146, 152, 178, 181, 185, 191, 234,
267, 269, 270, 284, 291, 297, 308, 310, 311, 312, 314, 317, 318,
321, 324, 332, 338, 375, 402, 405, 409, 415, 454
```

## Confidence, visibility và presence

Default chung được import từ `backend.ai_services.pose_gaze.settings`; file thật
nằm tại `settings/settings.py`:

- Holistic confidence chuẩn: `0.50`.
- Soft landmark confidence: `0.20`.

`visibility` là độ tin cậy landmark đang nhìn thấy/không bị che. `presence` là
độ tin cậy landmark thực sự hiện diện trong ảnh. Hai giá trị này do MediaPipe
trả về; chúng không phải cờ dùng để ẩn landmark. Khi cả hai tồn tại, gate dùng
giá trị nhỏ hơn để tránh giữ một điểm có một tín hiệu chất lượng thấp.

| Score của point | Output |
|---|---|
| `>= 0.50` | Giữ `x`, `y`, `frame_x`, `frame_y`, `visibility`, `presence` |
| `>= 0.20` và `< 0.50` | Giữ tọa độ dự đoán; đặt `visibility=null`; giữ `presence` nếu model có trả |
| `< 0.20` | Giữ `index` để nhận diện point; mọi giá trị đo còn lại là `null` |
| Model không có score point | Giữ tọa độ; `visibility`/`presence` tiếp tục là `null` |

Không có trường tọa độ chiều sâu trong data class hoặc JSON.

## Input

`HolisticLandmarkExtractor.process_packet(frame, packet)` nhận:

- `frame`: ảnh OpenCV BGR gốc.
- `packet`: `TrackPacket`; chỉ track `is_present=true` được xử lý.
- Config: confidence, soft confidence, crop padding, model Tasks và input size.

## Output

Mỗi track trả một `TrackHolisticResult`. Ví dụ point chất lượng thấp hơn ngưỡng
mềm:

```json
{
  "track_id": 1,
  "student_id": "student_01",
  "pose_landmarks": [
    {
      "index": 0,
      "x": null,
      "y": null,
      "frame_x": null,
      "frame_y": null,
      "visibility": null,
      "presence": null
    }
  ]
}
```

`test_media/test_media.py` ghi `format_version=2` và stream từng frame xuống
`*_annotated_landmarks.json`, nên không giữ toàn bộ video trong RAM. Version 2
đánh dấu schema chỉ còn tọa độ hai chiều và lọc landmark theo index sử dụng.

Batch ảnh ghi mặc định vào `batch_dataset/data/train.csv`, `val.csv`,
`test.csv`. Một hàng tương ứng một người trong một ảnh/frame; `status` giữ lại
cả ảnh lỗi, không thấy người và thiếu landmark để audit dataset. `class_code`
là target chính, còn `label` và các cột annotation là metadata.

## Chạy kiểm tra

```powershell
python -m backend.ai_services.pose_gaze.holistic.test_webcam --target-fps 8
python -m backend.ai_services.pose_gaze.holistic.test_media input.mp4 --target-fps 8 --no-display
python -m backend.ai_services.pose_gaze.holistic.batch_dataset D:\dataset
python -m unittest discover -s backend/ai_services/pose_gaze/holistic/tests -v
```

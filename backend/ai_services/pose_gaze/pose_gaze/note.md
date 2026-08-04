# Ghi chú commit Pose/Gaze — 03/08/2026

Commit message đề xuất:

```text
feat(pose-gaze): package landmark datasets and realtime XGBoost inference
```

## Mục tiêu commit

Hoàn thiện việc đóng gói `pose_gaze` thành các package độc lập, thống nhất cấu
hình tracking/Holistic, thu gọn landmark output và bổ sung luồng dữ liệu từ ảnh
đến CSV cùng entry point realtime sẵn sàng nạp model XGBoost đã train.

## Nội dung hoàn thành

### 1. Sắp xếp lại package

- Mỗi file chức năng của tracking được đặt trong folder cùng tên:
  `detectors/`, `manager/`, `schemas/`, `tracker/`, `webcam/`, `interactive/`,
  `test_image/`, `test_webcam/`.
- Holistic được tách thành `landmark/`, `feature_csv/`, `batch_dataset/`,
  `test_media/`, `test_webcam/` và các folder test tương ứng.
- Thêm `__init__.py` re-export để các import công khai của tracking tiếp tục
  hoạt động sau khi move file.
- Thêm `__main__.py` cho các CLI dùng với `python -m`.
- Mỗi folder chứa code đều có `README.md` mô tả ý tưởng, nhiệm vụ, input,
  output và cách chạy/import.
- Di chuyển ảnh mẫu vào `tracking/test_image/image.png` và thêm ngoại lệ
  `.gitignore` đúng phạm vi để Git theo dõi folder này.

### 2. Settings dùng chung

- Chuyển cấu hình vào `settings/settings.py` và export qua
  `backend.ai_services.pose_gaze.settings`.
- Tracking dùng chung:
  - person confidence: `0.50`;
  - minimum IoU: `0.30`;
  - maximum missed frames: `30`.
- Holistic dùng chung:
  - confidence chuẩn: `0.50`;
  - soft landmark confidence: `0.20`.
- Bổ sung cấu hình giảm log native MediaPipe/TFLite nhưng vẫn giữ exception và
  lỗi nghiêm trọng.

### 3. Thu gọn landmark và JSON

- Chỉ tạo/xuất các index thực sự được sử dụng:
  - pose normalized/world: `0–24`;
  - left/right hand normalized/world: `0–20`;
  - selected face: 53 điểm môi, trán và trục hỗ trợ hướng đầu.
- Landmark dư bị loại ngay khi materialize, không chỉ bị ẩn ở bước vẽ.
- Loại bỏ hoàn toàn tọa độ `z` khỏi data class, JSON và schema CSV.
- Giữ `visibility` và `presence` vì đây là score chất lượng của MediaPipe,
  không phải cờ dùng để ẩn landmark.
- Áp dụng ba mức confidence:
  - `>= 0.50`: giữ tọa độ và score;
  - `0.20–0.50`: giữ tọa độ dự đoán, `visibility=null`;
  - `< 0.20`: giữ index nhưng toàn bộ giá trị đo là `null`.
- Landmarks JSON của media test sử dụng `format_version=2` và stream theo frame.

### 4. Batch ảnh thành CSV train/val/test

- Thêm `holistic/batch_dataset/batch_dataset.py`.
- Pipeline batch:

```text
image -> YOLO person -> IoU tracking -> per-track Holistic -> fixed CSV row
```

- Detector và MediaPipe được khởi tạo một lần, ảnh được đọc tuần tự nên có thể
  xử lý hàng nghìn ảnh mà không giữ toàn bộ dataset trong RAM.
- Hỗ trợ ba kiểu input:
  1. `train/val/test/<class>/...` đã chia sẵn;
  2. `<class>/...` chưa chia, split ổn định bằng hash;
  3. manifest CSV/TSV có `filename`, `class_code`, `label`, `split`, actor,
     action actor, `observed_*`, quality và note.
- Hỗ trợ `sequence_id` để giữ `track_id` xuyên các frame tách từ cùng video.
- Toàn bộ frame của một sequence được giữ trong cùng split để hạn chế leakage.
- Mỗi người trong mỗi frame tạo một hàng CSV.
- Ảnh lỗi hoặc không có kết quả vẫn được ghi với `status`:
  `read_error`, `no_person`, `no_landmarks`, `inference_error`.
- CSV mặc định được stream vào:

```text
holistic/batch_dataset/data/train.csv
holistic/batch_dataset/data/val.csv
holistic/batch_dataset/data/test.csv
```

- File `.part` được dùng trong lúc chạy; chỉ publish CSV cuối khi run hoàn tất.
- `--overwrite` là bắt buộc nếu muốn thay output cũ.

### 5. Feature contract dùng chung

- Thêm `holistic/feature_csv/feature_csv.py` làm nguồn schema duy nhất cho batch
  CSV và realtime XGBoost.
- Một hàng chứa metadata source/split/annotation/tracking cùng feature landmark
  cố định.
- `MODEL_FEATURE_COLUMNS` chỉ chứa landmark `x`, `y`, `visibility`, `presence`
  và valid ratio; không đưa `class_code`, `label`, actor hoặc `observed_*` vào
  model input.
- Missing landmark được để trống trong CSV và chuyển thành `NaN` khi tạo
  `DMatrix`, phù hợp với cách XGBoost xử lý missing value.
- Có valid ratio cho từng nhóm pose, hand, face và toàn bộ landmark.

### 6. Realtime XGBoost demo

- Thêm `main/main.py` và root `pose_gaze/__main__.py`.
- Pipeline realtime:

```text
webcam -> YOLO -> IoU tracking -> Holistic
       -> shared feature vector -> XGBoost -> class_code/confidence
```

- Nạp model native `.json` hoặc `.ubj`, không dùng pickle/joblib.
- Metadata model giữ thứ tự `class_codes`, mapping `class_labels` và tùy chọn
  `feature_columns`.
- Runtime từ chối model nếu feature names/order không khớp schema hiện tại.
- Hỗ trợ binary probability, multiclass soft probability và multiclass softmax.
- Prediction được làm mượt EMA riêng theo từng `track_id`.
- Không classification khi toàn bộ landmark của track đều thiếu.
- Hiển thị class code, label và confidence trực tiếp trên webcam.

### 7. Dataset note và taxonomy

- Thêm `manifest.example.tsv` theo cấu trúc note hiện tại.
- `class_code` được giữ làm target chính; `label` là mô tả.
- Header manifest được đọc không phân biệt hoa/thường và output được chuẩn hóa
  snake_case.
- Chưa tự map `action_actor_ids` sang `track_id`; mapping này cần calibration,
  actor crop hoặc annotation bổ sung trước khi train frame có nhiều người.
- Các điểm cần chốt ở vòng dữ liệu tiếp theo:
  - `c1` và `c4` đang cùng mô tả `looking_down_suspiscous`;
  - `hand_read_toward_friend` và `hand_reach_toward_friend` chưa thống nhất;
  - nên sửa `suspiscous` thành `suspicious`;
  - `Observed_sign_code` có cách viết hoa khác các cột còn lại;
  - manifest hiện tại là video-level, batch ảnh cần manifest frame-level sau
    bước extract frame/interval.

### 8. Dependency và model artifact

- Pin `mediapipe==0.10.33` để dùng Tasks Holistic API hiện tại.
- Bổ sung `xgboost>=2.1,<4.0`.
- Model thật, metadata thật và CSV sinh ra được Git ignore; README và file mẫu
  vẫn được theo dõi.

## Lệnh sử dụng chính

### Batch ảnh

```powershell
python -m backend.ai_services.pose_gaze.holistic.batch_dataset D:\dataset `
  --model .\weights\yolov8n.pt `
  --device 0
```

### Batch bằng manifest

```powershell
python -m backend.ai_services.pose_gaze.holistic.batch_dataset D:\dataset `
  --manifest D:\dataset\annotations.tsv `
  --overwrite
```

### Realtime classification sau khi đã đặt model

```powershell
python -m backend.ai_services.pose_gaze --device 0
```

### Unit tests

```powershell
python -m unittest discover -s backend/ai_services/pose_gaze/tracking/tests -v
python -m unittest discover -s backend/ai_services/pose_gaze/holistic/tests -v
python -m unittest discover -s backend/ai_services/pose_gaze/main/tests -v
```

## Trạng thái xác minh

- Python compile: pass.
- Tracking: `10/10` test pass.
- Holistic/CSV/batch: `22/22` test pass.
- Realtime main: `4/4` test pass.
- Tổng: `36/36` test pass.
- Batch CLI `--help`: pass.
- Root realtime CLI `--help`: pass.
- Mọi folder chứa code đều có README.
- Không còn import Python dùng đường dẫn Holistic cũ.
- `git diff --check`: không có whitespace error.

## Chưa nằm trong commit này

- Chưa có model XGBoost đã train hoặc metric đánh giá thật.
- Chưa có script extract frame từ video theo `action_start_s/action_end_s`.
- Chưa map actor identity với `track_id` tự động.
- Chưa có feature engineering hình học như head pose, shoulder/body angle,
  hand-to-face distance hoặc pairwise feature.
- Chưa có temporal window aggregation cho XGBoost.
- Chưa có event hysteresis, calibration threshold hoặc API classification.

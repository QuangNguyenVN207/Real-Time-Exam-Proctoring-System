# Holistic Training Onboarding

## Mục tiêu

Dự án nhận diện 7 lớp hành vi thi cử `c1`–`c7` từ landmark MediaPipe Holistic theo chuỗi thời gian. Mục tiêu đánh giá chính là khả năng tổng quát hóa sang video chưa từng thấy, vì vậy metric quan trọng nhất là `video_macro_f1` trên validation và test.

## Bắt đầu nhanh

Workspace root:

```powershell
C:\Real-Time-Exam-Proctoring-System
```

Python environment:

```powershell
.\.venv\Scripts\python.exe
```

Code root:

```text
backend/ai_services/pose_gaze/pose_gaze/holistic
```

Các file cần xem đầu tiên:

```text
backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/export_json_features.py
backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/build_temporal_dataset.py
backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/train_baseline.py
backend/ai_services/pose_gaze/pose_gaze/holistic/temporal_windows.py
```

Canonical inputs:

```text
data/raw_video/processed/holistic_manifest.csv
data/raw_video/processed/holistic_outputs/
```

Canonical derived data:

```text
data/processed/holistic_features/
data/processed/holistic_temporal/
```

## Pipeline tổng thể

```text
raw_video/Dataset
        |
        v
MediaPipe Holistic extraction
        |
        v
data/raw_video/processed/holistic_outputs/*.json
        |
        |  export_json_features.py
        v
data/processed/holistic_features/features_<variant>.csv
        |
        |  build_temporal_dataset.py
        v
data/processed/holistic_temporal/windows_<variant>.npz
        |
        |  train_baseline.py
        v
XGBoost model + metadata + W&B metrics
```

### 1. Manifest và JSON canonical

Manifest chính:

```text
data/raw_video/processed/holistic_manifest.csv
```

Manifest có 93 video, đủ 7 lớp:

```text
c1: 18, c2: 10, c3: 14, c4: 14, c5: 15, c6: 9, c7: 13
```

JSON canonical nằm trong:

```text
data/raw_video/processed/holistic_outputs/
```

Đã kiểm tra: 93/93 JSON parse được, không rỗng, finite và khớp manifest.

### 2. Frame feature CSV

Exporter đọc JSON canonical, loại rows có `face_predicted=True`, flatten pose/hand/face features và ghi CSV.

Các variant gốc:

```text
2d
2d_quality
2d_world
full
```

Variant geometry:

```text
2d_geometry
```

Geometry hiện gồm shoulder-centered normalization, shoulder scale, elbow angles, torso angle, orientation sin/cos và khoảng cách wrist-nose/wrist-shoulder.

Quan trọng: exporter phải sort theo `split -> split_group -> source_filename -> track_id -> frame_id`. Nếu không sort, các rows của cùng video bị xen kẽ và temporal builder có thể làm mất nhiều windows. Fix này nằm trong `export_json_features.py`.

Lệnh export geometry:

```powershell
Push-Location backend/ai_services/pose_gaze
$env:PYTHONPATH = "../../../;."
python -m pose_gaze.holistic.feature_csv.export_json_features `
  --manifest ../../../data/raw_video/processed/holistic_manifest.csv `
  --json-dir ../../../data/raw_video/processed/holistic_outputs `
  --output-dir ../../../data/processed/holistic_features `
  --variant 2d_geometry
Pop-Location
```

### 3. Temporal windows

Policy chuẩn:

```text
window size: 16 frames
stride: 4 frames
max timestamp gap: 250 ms
```

Rows được nhóm theo:

```text
split + split_group + source_filename + track_id
```

Scaler chỉ fit trên train rows. Sau đó dữ liệu được lưu dạng compressed NPZ, không dùng JSONL.

Lệnh build chuẩn:

```powershell
Push-Location backend/ai_services/pose_gaze
$env:PYTHONPATH = "../../../;."
python -m pose_gaze.holistic.feature_csv.build_temporal_dataset `
  --input-dir ../../../data/processed/holistic_features `
  --output-dir ../../../data/processed/holistic_temporal `
  --variant 2d
Pop-Location
```

Stride có thể cấu hình bằng `--stride`. Dataset stride 8 đã được tạo riêng ở:

```text
data/processed/holistic_temporal_stride8/
```

## Dataset chuẩn đã xác nhận

File:

```text
data/processed/holistic_temporal/windows_2d.npz
```

Shape:

```text
X: (23911, 16, 300)
```

Phân chia:

```text
train: 15771 windows, 47 videos
val:    4124 windows, 23 videos
test:   4016 windows, 23 videos
```

Kiểm tra integrity đã đạt:

```text
finite X: True
cross-split videos: 0
cross-split groups: 0
mixed-label videos: 0
```

NPZ có các field chính:

```text
X
split
split_group
source_filename
track_id
label
feature_columns
```

## Trainer

File trainer:

```text
backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/train_baseline.py
```

Model hiện tại là XGBoost `multi:softprob`, chạy CPU với:

```text
max_depth=3
min_child_weight=5
learning_rate=0.05
subsample=0.8
colsample_bytree=0.5
reg_lambda=15
reg_alpha=0.1
tree_method=hist
early stopping=30
```

Trainer có các feature mode:

```text
flatten
summary
temporal_pooling
```

### flatten

16 frames x 300 features = 4,800 model features.

### summary

Gồm:

```text
frame mean
frame std
velocity mean
acceleration mean
```

Với `2d`: 1,200 model features.

### temporal_pooling

Gồm 11 nhóm thống kê cho mỗi frame feature:

```text
first frame
last frame
last - first displacement
minimum
maximum
median
percentile 10
percentile 90
velocity std
acceleration std
temporal slope
```

Với `2d`: 3,300 model features. Đây là mode giàu thông tin thời gian nhưng vẫn nhỏ hơn flatten 4,800.

## Sample weights

Trainer hỗ trợ:

```text
none
class
video
class_video
```

Video weight hiện tại:

$$
w_{video} = \frac{1}{\sqrt{n_{windows\_in\_video}}}
$$

Sau đó weights được chuẩn hóa để mean gần bằng 1. `class_video` nhân class balancing với sqrt video balancing.

## Cách train

Train baseline 2D cũ:

```powershell
.\.venv\Scripts\python.exe `
  backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/train_baseline.py `
  --variant 2d `
  --rounds 800 `
  --early-stopping-rounds 30 `
  --checkpoint-every 10 `
  --output-dir weights/baseline `
  --wandb-project real-time-exam-proctoring `
  --wandb-run baseline-2d `
  --wandb-mode online
```

Train model đang tốt nhất theo temporal pooling:

```powershell
.\.venv\Scripts\python.exe `
  backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/train_baseline.py `
  --variant 2d `
  --feature-mode temporal_pooling `
  --weight-mode class_video `
  --rounds 800 `
  --early-stopping-rounds 30 `
  --checkpoint-every 10 `
  --output-dir weights/baseline_2d_temporal_pooling_sqrt_video `
  --wandb-project real-time-exam-proctoring `
  --wandb-run baseline-2d-temporal-pooling-sqrt-video `
  --wandb-mode online
```

Train trên stride 8:

```powershell
.\.venv\Scripts\python.exe `
  backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/train_baseline.py `
  --data-root data/processed/holistic_temporal_stride8 `
  --variant 2d `
  --feature-mode temporal_pooling `
  --weight-mode class_video `
  --rounds 800 `
  --early-stopping-rounds 30 `
  --checkpoint-every 10 `
  --output-dir weights/baseline_2d_stride8_pooling_sqrt_video `
  --wandb-project real-time-exam-proctoring `
  --wandb-run baseline-2d-stride8-pooling-sqrt-video `
  --wandb-mode online
```

W&B credentials đã được cấu hình trong máy. Không in, copy hoặc yêu cầu API key trong chat.

## Artifact model

Mỗi output model thường có:

```text
baseline.ubj          best iteration model
baseline_full.ubj     full final-round model
metadata.json         config, feature count, best iteration, metrics
checkpoints/          periodic UBJ checkpoints
```

## Kết quả experiments

### Summary + class/video weight, stride 4

Artifact:

```text
weights/baseline_2d_summary_class_video/
```

Kết quả:

```text
best iteration: 144
val macro-F1: 0.3879
val video macro-F1: 0.3718
val balanced accuracy: 0.4274
val video accuracy: 0.4783
test macro-F1: 0.2776
test video macro-F1: 0.1726
test balanced accuracy: 0.2756
test video accuracy: 0.2609
```

### Geometry + summary + class/video weight

Artifact:

```text
weights/baseline_2d_geometry/
```

Kết quả:

```text
best iteration: 103
val video macro-F1: 0.1644
test video macro-F1: 0.2046
test video accuracy: 0.3043
```

### Temporal pooling + sqrt video weight, stride 4

Artifact:

```text
weights/baseline_2d_temporal_pooling_sqrt_video/
```

W&B:

```text
https://wandb.ai/nghiepphat4-ho-chi-minh-city-university-of-technology/real-time-exam-proctoring/runs/4jph649r
```

Kết quả:

```text
feature count: 3300
best iteration: 77
val macro-F1: 0.3244
val video macro-F1: 0.2880
val balanced accuracy: 0.3931
val video accuracy: 0.3913
test macro-F1: 0.2107
test video macro-F1: 0.2377
test balanced accuracy: 0.2215
test video accuracy: 0.3043
```

### Temporal pooling + sqrt video weight, stride 8

Artifact:

```text
weights/baseline_2d_stride8_pooling_sqrt_video/
```

W&B:

```text
https://wandb.ai/nghiepphat4-ho-chi-minh-city-university-of-technology/real-time-exam-proctoring/runs/36oa6ck0
```

Kết quả:

```text
windows: 12000
best iteration: 55
val video macro-F1: 0.1837
test video macro-F1: 0.2381
test video accuracy: 0.3043
```

Kết luận: stride 8 không cải thiện đáng kể so với stride 4 và làm validation giảm. Giữ stride 4 làm cấu hình chuẩn.

## Vấn đề chính hiện tại

Vấn đề lớn nhất là domain shift/overfit theo video:

```text
validation video macro-F1 cao hơn test video macro-F1 rõ rệt
```

Đây không còn là vấn đề thiếu số lượng feature đơn thuần. Model học được pattern riêng của video/session nhưng tổng quát hóa yếu sang video test.

Chẩn đoán sâu hơn: đơn vị độc lập thật sự là 8 `split_group`, không phải 23,911 windows hay 93 video. Split cố định chỉ có 2 group validation và 2 group test, nên video macro-F1 có phương sai lớn. Cần leave-one-group-out CV và báo cáo mean +/- std trước khi kết luận ablation.

Variant `2d` chứa nhiều tọa độ mặt tuyệt đối, có thể mã hóa camera, identity và group thay vì hành vi. Cần normalize toàn thân, giảm face raw coordinates và giữ validity mask. `2d_quality` đã tồn tại nhưng chưa phải baseline chính.

Các lỗi trainer đã sửa: streaming scaler fallback khi scale <= 1e-12; metric dùng đúng `best_iteration`; hyperparameters XGBoost đã expose qua CLI; thêm `group`/`class_group` inverse-sqrt weighting; thêm per-class F1 và confusion matrix.

Các nguyên nhân cần xem xét:

1. Số video mỗi class thấp, đặc biệt `c2` và `c6`.
2. Nhiều windows từ cùng video rất giống nhau.
3. Background, camera, identity và chất lượng tracking khác nhau giữa video.
4. XGBoost pooling không tận dụng thứ tự thời gian đầy đủ.
5. Geometry hiện có validity semantics chưa hoàn hảo: missing landmark có thể bị chuyển thành `(0, 0)`.

## Việc nên làm tiếp theo

Ưu tiên 1: tạo leave-one-group-out CV trên 8 `split_group`, báo cáo video macro-F1 mean +/- std và per-class F1.

Ưu tiên 2: loại bỏ kênh identity từ tọa độ mặt tuyệt đối, normalize toàn thân và đưa validity mask vào model.

Ưu tiên 3: dùng `action_start_s`/`action_end_s` để loại phần video ngoài khoảng hành động, giảm nhãn nhiễu.

Ưu tiên 4: thử augmentation theo video:

```text
coordinate jitter
scale/translation normalization
random temporal crop
landmark dropout
```

Ưu tiên 5: thử GRU/TCN nhỏ với input `(batch, 16, 300)` và aggregate prediction theo video.

Ưu tiên 5: thêm action labels theo đoạn thời gian, không gán một action duy nhất cho toàn video nếu video chứa nhiều hành động.

## Các cảnh báo khi xem xét kết quả

- Không chọn model chỉ vì window-level macro-F1 cao.
- Metric chính là video-level macro-F1 trên video chưa thấy trong train.
- Không random split theo frame/window.
- Không fit scaler trên validation/test.
- Không train trên dataset stride 8 rồi so sánh trực tiếp với số window stride 4 mà bỏ qua thay đổi phân bố.
- Geometry và temporal pooling hiện là ablation, không được coi là cải thiện chắc chắn.
- Import package root có stale path `.tracking` trong `backend/ai_services/pose_gaze/__init__.py`; chạy script trực tiếp hoặc đặt `PYTHONPATH=../../../;.` như các lệnh trên.

## Lệnh kiểm tra nhanh

Kiểm tra NPZ:

```powershell
.\.venv\Scripts\python.exe -c "import numpy as np; d=np.load('data/processed/holistic_temporal/windows_2d.npz',allow_pickle=False); print(d['X'].shape, np.isfinite(d['X']).all(), np.unique(d['label']))"
```

Kiểm tra syntax:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/export_json_features.py `
  backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/build_temporal_dataset.py `
  backend/ai_services/pose_gaze/pose_gaze/holistic/feature_csv/train_baseline.py
```

## Tóm tắt cho Claude

Bắt đầu bằng việc đọc file này, sau đó đọc `train_baseline.py`, `build_temporal_dataset.py`, `temporal_windows.py` và `metadata.json` của các artifact. Không bắt đầu bằng việc train thêm feature mới. Trước tiên hãy kiểm tra video-level generalization, class confusion, per-video errors và split robustness. Dataset chuẩn là `data/processed/holistic_temporal/windows_2d.npz`, cấu hình chuẩn là stride 4, temporal pooling 3,300 features chỉ là một ablation.

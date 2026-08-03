# Realtime XGBoost main

## Ý tưởng

Entry point cuối cùng sau khi model đã train:

```text
webcam -> YOLO -> IoU tracking -> per-track Holistic
       -> fixed feature vector -> XGBoost -> class_code + confidence
```

Prediction được làm mượt EMA riêng theo `track_id` để giảm nhấp nháy giữa các
frame.

## Công năng và nhiệm vụ

- Nạp model XGBoost native `.json` hoặc `.ubj`.
- Kiểm tra feature names/feature order khớp CSV batch.
- Nạp thứ tự `class_codes` từ metadata, không suy diễn từ tên label.
- Classify từng người đang track và hiển thị class/confidence trên webcam.
- Không dùng pickle/joblib.

## Model input

Đặt file tại:

```text
model/xgboost_model.ubj
model/model_metadata.json
```

Metadata tối thiểu:

```json
{
  "format_version": 1,
  "class_codes": ["c1", "c2", "c3", "c4", "c5", "c6", "c7"],
  "class_labels": {
    "c5": "normal"
  }
}
```

Thứ tự `class_codes` phải đúng thứ tự encoder khi train. `feature_columns` có
thể bỏ trống để dùng schema hiện tại; nếu ghi vào metadata thì phải khớp chính
xác `MODEL_FEATURE_COLUMNS`.

Model phải được train từ feature một người/một frame của batch CSV. Model train
trên temporal window cần một runtime window riêng, chưa được entry point này giả
định thay.

## Output

Cửa sổ webcam có bbox, track ID, class code, label và confidence. Khi thoát vẫn
ghi tracking handoff JSON theo session.

## Chạy

```powershell
python -m backend.ai_services.pose_gaze `
  --xgboost-model backend/ai_services/pose_gaze/main/model/xgboost_model.ubj `
  --model-metadata backend/ai_services/pose_gaze/main/model/model_metadata.json `
  --device 0
```

Hoặc:

```powershell
python -m backend.ai_services.pose_gaze.main
```

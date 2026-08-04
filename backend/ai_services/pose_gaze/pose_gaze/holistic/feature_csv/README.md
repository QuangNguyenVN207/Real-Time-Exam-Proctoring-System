# Feature CSV

## Ý tưởng

Đây là hợp đồng feature duy nhất dùng chung cho batch dataset và realtime
XGBoost. Mỗi hàng biểu diễn một người trong một frame, với thứ tự cột cố định để
model train và model inference không bị lệch feature.

## Công năng và nhiệm vụ

- Flatten pose, pose world, hai tay, hand world và selected face.
- Chỉ có `x`, `y`, `visibility`, `presence`; không có `z`.
- Bổ sung tỷ lệ landmark hợp lệ cho từng nhóm.
- Giữ metadata annotation, tracking và lỗi tách khỏi input feature XGBoost.
- Tạo hàng rỗng có `status` cho ảnh lỗi hoặc không detect được người.

## Input

`TrackHolisticResult`, `TrackedPerson`, metadata ảnh và annotation tùy chọn.

## Output

- `CSV_FIELDNAMES`: toàn bộ cột của `train.csv`, `val.csv`, `test.csv`.
- `MODEL_FEATURE_COLUMNS`: đúng các cột phải đưa vào XGBoost.
- `model_features_from_result()`: mapping feature dùng ở cả train và realtime.

## Import

```python
from backend.ai_services.pose_gaze.holistic.feature_csv import (
    MODEL_FEATURE_COLUMNS,
    model_features_from_result,
)
```

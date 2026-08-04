# Trained model artifacts

Đặt model đã train tại đây:

- `xgboost_model.ubj` hoặc `xgboost_model.json`: model native được tạo bởi
  `Booster.save_model()`/`XGBClassifier.save_model()`.
- `model_metadata.json`: thứ tự `class_codes`, mapping `class_labels` và tùy
  chọn `feature_columns`.

Model thật và metadata thật bị Git ignore để tránh commit artifact lớn hoặc
taxonomy chưa chốt. Dùng `model_metadata.example.json` làm mẫu.

Không dùng pickle/joblib cho artifact lưu dài hạn.

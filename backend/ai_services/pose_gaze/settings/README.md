# Settings

## Ý tưởng

`settings.py` là nguồn cấu hình mặc định duy nhất cho cả tracking và Holistic,
tránh mỗi entry point tự hard-code một giá trị khác nhau.

## Công năng và nhiệm vụ

- Khai báo person confidence, IoU và grace period của tracker.
- Khai báo confidence chuẩn và ngưỡng mềm của Holistic.
- Cung cấp `PROJECT_ROOT` và cấu hình giảm log native MediaPipe/TFLite.

## Input

Module không nhận dữ liệu runtime. Ứng dụng có thể đặt trước biến môi trường log;
`configure_mediapipe_logging()` sẽ tôn trọng giá trị đã có.

## Output

Các constant và hàm cấu hình được export qua
`backend.ai_services.pose_gaze.settings`.

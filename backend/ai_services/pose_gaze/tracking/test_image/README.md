# Test image

## Ý tưởng

Kiểm tra nhanh riêng person detector trên một ảnh, không tạo tracking session và
không chạy MediaPipe.

## Công năng và nhiệm vụ

Chọn CPU/GPU, nạp YOLO, phát hiện người, vẽ bbox/confidence và hiển thị kết quả.
`image.png` được đặt cùng folder với code test.

## Input

Mặc định `image.png` trong folder này; có thể gọi `test_single_image()` với một
đường dẫn khác.

## Output

Cửa sổ OpenCV chứa ảnh annotated và số người phát hiện được trên console.

## Chạy

```powershell
python -m backend.ai_services.pose_gaze.tracking.test_image
```

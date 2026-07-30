import cv2
from pathlib import Path
import torch

# Import class Detector từ hệ thống của bạn[cite: 9]
from backend.ai_services.pose_gaze.tracking.detectors import UltralyticsPersonDetector

def test_single_image(image_path: str):
    print(f"Đang xử lý ảnh: {image_path}...")
    
    # 1. Khởi tạo mô hình YOLO[cite: 9]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_dir = Path("./weights")
    model_path = weights_dir / "yolov8n.pt"
    if not model_path.exists():
        model_path = "yolov8n.pt"

    # Đặt threshold = 0.5 để lọc kết quả nhiễu[cite: 9]
    detector = UltralyticsPersonDetector(model_path=model_path, confidence_threshold=0.5, device=device)
    
    # 2. Đọc ảnh bằng OpenCV
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Lỗi: Không thể tìm thấy hoặc đọc được ảnh tại '{image_path}'. Vui lòng kiểm tra đường dẫn!")
        return

    # 3. Tiến hành dò tìm người trong bức ảnh[cite: 9]
    detections = detector.detect(frame)
    print(f"✅ Đã phát hiện {len(detections)} người trong ảnh.")

    # 4. Vẽ khung bọc (Bounding Box) lên ảnh
    for index, det in enumerate(detections):
        # Lấy tọa độ x, y từ class BoundingBox[cite: 11]
        x1, y1, x2, y2 = int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)
        confidence = round(det.confidence, 2)
        
        # Vẽ hình chữ nhật màu xanh lá
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Ghi text độ tự tin (Confidence score) lên trên box[cite: 11]
        label = f"Person {index + 1} | Conf: {confidence}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # (Tùy chọn) Thu nhỏ ảnh nếu độ phân giải quá to (vượt màn hình)
    h, w = frame.shape[:2]
    if h > 800 or w > 1200:
        frame = cv2.resize(frame, (w // 2, h // 2))

    # 5. Hiển thị bức ảnh đã vẽ
    cv2.imshow("Test Detect Single Image", frame)
    print("Bấm phím bất kỳ trên cửa sổ ảnh để đóng lại...")
    
    # Lệnh waitKey(0) sẽ giữ cho cửa sổ ảnh không bị tắt cho đến khi bạn bấm một phím bất kỳ
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Đổi tên file ảnh dưới đây thành ảnh bạn muốn test
    # Lưu ý: Bức ảnh phải nằm chung thư mục với script này, hoặc bạn truyền đường dẫn tuyệt đối vào
    IMAGE_TO_TEST = "backend/ai_services/pose_gaze/tracking/image.png" 
    
    test_single_image(IMAGE_TO_TEST)
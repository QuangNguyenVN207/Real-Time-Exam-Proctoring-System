"""
Test object detection module bằng webcam thật — có vẽ bounding box
trực tiếp lên cửa sổ live để xem trực quan model detect gì.

Chạy từ trong thư mục YOLOv8/ (cùng cấp với config.py, object_detect.py):
    python test_webcam.py
"""

import cv2

from backend.ai_services.object_detect.object_detect import ObjectDetectModule


def draw_boxes(frame, raw_boxes: dict, raw_detections: dict, confirmed_classes: list):
    """Vẽ box lên frame — đỏ nếu class đã được confirm (đủ ngưỡng
    frame liên tiếp), vàng nếu chỉ mới detect thoáng qua (chưa đủ
    ngưỡng confirm)."""
    for class_name, bbox in raw_boxes.items():
        x1, y1, x2, y2 = bbox
        confidence = raw_detections.get(class_name, 0.0)
        is_confirmed = class_name in confirmed_classes
        color = (0, 0, 255) if is_confirmed else (0, 255, 255)  # đỏ / vàng

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.2f}" + (" [CONFIRMED]" if is_confirmed else "")
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main():
    module = ObjectDetectModule()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam")

    session_id = "manual_test_session"
    frame_id = 0

    # Vì model chỉ chạy inference mỗi N frame (xem
    # object_detect_every_n_frames trong config.py), giữ lại box của
    # lần inference gần nhất để vẽ liên tục — không thì box sẽ
    # nhấp nháy ẩn/hiện mỗi N frame, nhìn giật.
    last_boxes = {}
    last_detections = {}
    last_confirmed = []

    print("Nhấn 'q' để thoát. Box đỏ = đã confirm (đủ frame liên tiếp), vàng = mới detect thoáng qua.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        result = module.process(frame, session_id, frame_id)
        if result is not None:
            last_boxes = result.get("raw_boxes", last_boxes)
            last_detections = result.get("raw_detections", last_detections)
            last_confirmed = result.get("confirmed_classes", last_confirmed)
            if result["label"] != "clear":
                # Cứ mỗi 15 frame (~0.5 giây) mới in log ra một lần
                if frame_id % 15 == 0: 
                    print(f"[frame {frame_id}] {result}")

        draw_boxes(frame, last_boxes, last_detections, last_confirmed)
        cv2.imshow("Test object detect - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    module.cleanup_session(session_id)


if __name__ == "__main__":
    main()
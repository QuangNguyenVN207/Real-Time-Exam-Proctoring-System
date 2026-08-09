"""
Test face_verify bằng webcam thật — vẽ khung cảnh báo trực tiếp lên
cửa sổ live, giống style cảnh báo thật trong main.py.

Chạy từ thư mục gốc project:
    python -m backend.ai_services.face_verify.test_webcam

Phím tắt:
    s = chụp khung hình hiện tại, lưu thành ảnh gốc mới trong
        data/student_faces/ (tự đặt tên you_<n>.jpg) rồi nạp lại DB ngay
        — dùng để tự "đăng ký" khuôn mặt mình và xem có còn bị báo
        "người lạ" nữa không, không cần thoát chương trình.
    q = thoát
"""

import time

import cv2

from backend.ai_services.face_verify.face_verify import FaceVerifier


def draw_alert(frame, result: dict) -> None:
    bbox = result["details"]["unauthorized_bbox"]
    score = result["details"]["similarity_score"]
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
    label = f"NGUOI LA ({score:.2f})"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w + 10, y1), (0, 140, 255), -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    fv = FaceVerifier(db_path="data/student_faces/")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam")

    print("Nhấn 's' để tự đăng ký khuôn mặt hiện tại vào DB, 'q' để thoát.")
    frame_id = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        t0 = time.time()
        result = fv.verify_face(frame, time.time())
        dt_ms = (time.time() - t0) * 1000

        if result is not None:
            draw_alert(frame, result)
            print(f"[frame {frame_id}] {dt_ms:.0f}ms -> ALERT {result['details']}")
        elif frame_id % 15 == 0:
            print(f"[frame {frame_id}] {dt_ms:.0f}ms -> OK (không cảnh báo)")

        cv2.putText(frame, "s=dang ky mat  q=thoat", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Test face_verify - press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            saved_count += 1
            path = f"data/student_faces/you_{saved_count}.jpg"
            cv2.imwrite(path, frame)
            print(f"Đã lưu {path}, đang nạp lại DB...")
            fv._load_database()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

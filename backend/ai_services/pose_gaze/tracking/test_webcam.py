import cv2
import time
import threading
import queue
from pathlib import Path

from backend.ai_services.pose_gaze.tracking.detectors import UltralyticsPersonDetector
from backend.ai_services.pose_gaze.tracking.manager import TrackingManager, AssignmentError

def main():
    print("Khởi tạo hệ thống Tracking (Backend Re-Tracking)...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    weights_dir = Path("./weights")
    weights_dir.mkdir(exist_ok=True)
    model_path = weights_dir / "yolov8n.pt"
    if not model_path.exists():
        model_path = "yolov8n.pt"

    detector = UltralyticsPersonDetector(model_path=model_path, confidence_threshold=0.75, device=device)
    manager = TrackingManager(storage_root=Path("./test_data_tracking"), max_tracks=10)
    session_id = "exam_room_test"
    manager.create_session(session_id)
    
    ignored_tracks = set()
    input_queue = queue.Queue()
    is_prompting = False

    def ask_for_id_thread(track_id):
        user_val = input(f"\n🔔 Track {track_id} mới! Nhập ID (hoặc 'no' để bỏ qua, 'full' để chốt luồng): ").strip()
        input_queue.put((track_id, user_val))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    frame_id = 0
    end_module = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        timestamp_ms = int(time.time() * 1000)

        # Detect và Track liên tục không ngừng
        detections = detector.detect(frame)
        packet = manager.process_detections(
            session_id=session_id, frame_id=frame_id, timestamp_ms=timestamp_ms, detections=detections
        )

        # Xử lý kết quả gõ bàn phím từ giám thị
        if not input_queue.empty():
            prompted_track_id, user_input = input_queue.get()
            is_prompting = False
            
            if user_input.lower() == 'full':
                end_module = True
                break
            elif user_input.lower() == 'no':
                ignored_tracks.add(prompted_track_id)
            elif user_input:
                try:
                    # Gọi Backend: Nó sẽ tự xử lý gán mới hoặc Re-tracking (đổi ID)
                    manager.assign_student(session_id, track_id=prompted_track_id, student_id=user_input)
                    print(f"✅ Đã xử lý ID: {user_input}")
                except AssignmentError as e:
                    print(f"❌ Lỗi: {e}")

        # Tự động hỏi nếu có track trống mới
        if not is_prompting and not end_module:
            unassigned = [t for t in packet.tracks if not t.student_id and t.track_id not in ignored_tracks]
            if unassigned:
                is_prompting = True
                threading.Thread(target=ask_for_id_thread, args=(unassigned[0].track_id,), daemon=True).start()

        # Render giao diện
        for track in packet.tracks:
            x1, y1, x2, y2 = int(track.bbox.x1), int(track.bbox.y1), int(track.bbox.x2), int(track.bbox.y2)
            color = (0, 255, 0) if track.student_id else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"Track: {track.track_id}" + (f" | {track.student_id}" if track.student_id else "")
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("He thong Giam Sat Thi Sinh (Module 1)", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # GỌI HÀM CỦA BACKEND ĐỂ XUẤT FILE JSON ĐÚNG CHUẨN
    if end_module:
        json_path = manager.generate_final_output(session_id)
        print(f"\n🎉 MODULE TRACKER HOÀN TẤT!")
        print(f"📄 DỮ LIỆU ĐÃ ĐƯỢC LƯU TẠI: {json_path.absolute()}")
        print("\nNội dung file JSON:")
        print(json_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
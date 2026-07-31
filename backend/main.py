# from fastapi import FastAPI

# from backend.api.pose_gaze_routes import router as pose_gaze_router


# app = FastAPI(
#     title="Exam Proctoring System",
#     version="0.1.0",
#     description="Backend APIs for realtime exam supervision.",
# )
# app.include_router(pose_gaze_router)


# @app.get("/health", tags=["system"])
# def health_check() -> dict[str, str]:
#     return {"status": "ok"}

# ==========================================
# Các ưu điểm trong đoạn code vừa nâng cấp:
# 1. Không giật lag (Flicker-free): Nhờ cơ chế OVERLAY_TTL (Thời gian sống), các Bounding Box hiển thị ổn định, không bị chớp tắt khi AI chỉ chạy ở 5 FPS.

# 2. Xử lý màu sắc linh hoạt:

# Vật thể cấm (YOLOv8): Khung đỏ tươi (0, 0, 255) hiển thị trên vật thể.

# Người lạ (face_verify): Khung màu cam (0, 140, 255) quanh khuôn mặt lạ.

# Tư thế bất thường (pose_gaze): Băng đỏ cảnh báo nằm góc dưới màn hình.

# Âm thanh gian lận (audio_whisper): Dải thông báo màu vàng nổi bật ở góc trên màn hình.

# 3. An toàn đa luồng: Thao tác đọc/ghi vào ACTIVE_OVERLAYS được khóa cẩn thận bằng OVERLAY_LOCK tránh tình trạng đụng độ dữ liệu giữa luồng AI và luồng Camera.

import os
# --- ÉP TẮT CẢNH BÁO QUYỀN VÀ FONT CỦA OPENCV TRÊN LINUX ---
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false;qt.text.font.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL" # Thêm dòng này để tắt log lõi của OpenCV
import cv2
import threading
import time
import queue

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & HÀNG ĐỢI (QUEUES)
# ==========================================
# FIX LỖI 2: Giảm maxsize xuống 2. Nếu AI xử lý chậm, nó sẽ tự động bỏ qua các khung hình cũ
# giúp cảnh báo luôn bám sát thời gian thực (Real-time)
FRAME_QUEUE = queue.Queue(maxsize=2) 
AUDIO_QUEUE = queue.Queue(maxsize=50)
RESULT_QUEUE = queue.Queue(maxsize=100)
FPS_SKIP = 5
CAMERA_ID = 0

# ==========================================
# 2. CẤU HÌNH HIỂN THỊ (GUI OVERLAYS)
# ==========================================
ACTIVE_OVERLAYS = []
OVERLAY_LOCK = threading.Lock()
OVERLAY_TTL = 1  # Thời gian sống (giây): Khung cảnh báo sẽ hiển thị mượt mà trong 1.5s

# Giả lập import các module AI (bạn sẽ liên kết với code của cộng sự sau)
from backend.ai_services.object_detect.object_detect_test import ObjectDetector
from backend.ai_services.pose_gaze.pose_gaze_test import PoseGazeDetector
from backend.ai_services.face_verify.face_verify import FaceVerifier
from backend.ai_services.whisper.audio_whisper_test import AudioWhisper

# ==========================================
# 3. HÀM VẼ GIAO DIỆN LÊN OPENCV
# ==========================================
def draw_warning_overlays(frame):
    """Vẽ các Bounding Box, nhãn vi phạm và bảng cảnh báo lên khung hình."""
    current_time = time.time()
    
    with OVERLAY_LOCK:
        # FIX LỖI 1: Dùng 'display_timestamp' (lúc nhận được) thay vì 'timestamp' (lúc chụp ảnh)
        valid_overlays = [
            item for item in ACTIVE_OVERLAYS 
            if current_time - item.get('display_timestamp', current_time) <= OVERLAY_TTL
        ]
        ACTIVE_OVERLAYS.clear()
        ACTIVE_OVERLAYS.extend(valid_overlays)
        
        for alert in ACTIVE_OVERLAYS:
            module = alert.get("module")
            details = alert.get("details", {})
            
            # --- Vẽ Bounding Box vật thể cấm (YOLOv8) ---
            if module == "object_detect":
                detections = alert.get("detections", [])
                for det in detections:
                    bbox = det.get("bbox")
                    label = det.get("label", "Vat cam").upper()
                    conf = det.get("confidence", 0.0)
                    
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox) # Ép kiểu int để tránh lỗi float
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        text = f"{label} ({conf*100:.0f}%)"
                        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - 25), (x1 + w + 10, y1), (0, 0, 255), -1)
                        cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- Vẽ Bounding Box người lạ (FaceNet) ---
            elif module == "face_verify":
                bbox = details.get("unauthorized_bbox")
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                    text = "NGUOI LA"
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w + 10, y1), (0, 140, 255), -1)
                    cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- Hiển thị Banner tư thế/hướng nhìn (MediaPipe) ---
            elif module == "pose_gaze":
                action = details.get("action", "VI PHAM TU THE").upper()
                cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (0, 0, 200), -1)
                cv2.putText(frame, f"[CANH BAO TU THE]: {action}", (20, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # --- Hiển thị Banner âm thanh bất thường (Whisper) ---
            elif module == "audio_whisper":
                transcription = details.get("transcription", "")
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 165, 255), -1)
                cv2.putText(frame, f"[AM THANH BAT THUONG]: {transcription}", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


# ==========================================
# 4. ĐỊNH NGHĨA CÁC LUỒNG XỬ LÝ (THREADS)
# ==========================================
def io_camera_thread():
    """Luồng 1: Đọc Camera, hiển thị giao diện và đẩy khung hình vào hàng đợi."""
    cap = cv2.VideoCapture(CAMERA_ID)
    frame_count = 0
    
    print("[INFO] Đang khởi động luồng Camera...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[LỖI] Không thể đọc luồng video từ Camera.")
            break
            
        # 1. Vẽ cảnh báo đè lên khung hình
        rendered_frame = draw_warning_overlays(frame.copy())
        
        # 2. Hiển thị lên màn hình (GUI)
        cv2.imshow("Exam Proctoring System", rendered_frame)
            
        # 3. Áp dụng Frame Skipping: Lấy khung hình đẩy vào AI
        if frame_count % FPS_SKIP == 0:
            # FIX LỖI DELAY: Rút frame cũ ra vứt đi nếu Queue bị đầy, ép AI phải lấy frame mới nhất
            if FRAME_QUEUE.full():
                try:
                    FRAME_QUEUE.get_nowait()
                except queue.Empty:
                    pass
            
            FRAME_QUEUE.put((frame.copy(), time.time()))

        frame_count += 1
        
        # Nhấn 'q' trên cửa sổ video để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    FRAME_QUEUE.put(None) # Đánh dấu kết thúc luồng

def io_audio_thread():
    """Luồng giả lập thu âm từ Microphone."""
    print("[INFO] Đang khởi động luồng thu âm (Microphone)...")
    while True:
        time.sleep(3.0)  # Cứ 3 giây cắt 1 đoạn âm thanh giả lập
        if not AUDIO_QUEUE.full():
            # dummy_audio_chunk = b'\x00\x00\x00' 
            # Tạo 3 giây âm thanh im lặng chuẩn (16000 sample/s * 3s * 2 bytes/sample)
            dummy_audio_chunk = b'\x00' * (16000 * 3 * 2)
            AUDIO_QUEUE.put((dummy_audio_chunk, time.time()))

def vision_ai_thread():
    """Luồng 2: Lấy khung hình từ bộ đệm và chạy các mô hình AI thị giác."""
    print("[INFO] Đang khởi động luồng AI Thị giác...")
    
    # Khởi tạo các module (Bỏ comment khi ráp code thật của cộng sự)
    yolo_model = ObjectDetector(model_path="weights/yolov8_finetuned.pt")
    face_model = FaceVerifier(db_path="data/student_faces/")
    gaze_model = PoseGazeDetector()
    
    while True:
        data = FRAME_QUEUE.get()
        if data is None: 
            break
            
        frame, timestamp = data
        
        # --- CHẠY AI VÀ XỬ LÝ HỢP ĐỒNG DỮ LIỆU ---
        
        # 1. Quét vật thể cấm (YOLOv8)
        result_yolo = yolo_model.process_frame(frame, timestamp)
        if result_yolo is not None and not RESULT_QUEUE.full():
            RESULT_QUEUE.put(result_yolo)
            
        # 2. Quét người lạ (FaceNet) - Bỏ qua người quen, bắt người lạ
        result_face = face_model.verify_face(frame, timestamp)
        if result_face is not None and not RESULT_QUEUE.full():
            RESULT_QUEUE.put(result_face)
            
        # 3. Quét hành vi tư thế (MediaPipe)
        result_gaze = gaze_model.process_frame(frame, timestamp)
        if result_gaze is not None and not RESULT_QUEUE.full():
            RESULT_QUEUE.put(result_gaze)


def audio_ai_thread():
    """Luồng 3: Nhận chuỗi âm thanh và chạy VAD + Whisper."""
    print("[INFO] Đang khởi động luồng AI Âm thanh...")
    
    # Khởi tạo mô hình Whisper (Mock)
    audio_model = AudioWhisper()
    
    while True:
        data = AUDIO_QUEUE.get()
        if data is None: # Tín hiệu thoát
            break
            
        audio_chunk, timestamp = data
        
        # Chạy phân tích âm thanh
        result_audio = audio_model.process_audio(audio_chunk, timestamp)
        
        # Nếu phát hiện gian lận bằng giọng nói, đẩy vào hàng đợi cảnh báo chung
        if result_audio is not None and not RESULT_QUEUE.full():
            RESULT_QUEUE.put(result_audio)


# ==========================================
# 5. LUỒNG ĐIỀU PHỐI (MAIN)
# ==========================================
def main():
    """Nhận kết quả từ AI, cập nhật GUI và in Log."""
    print("=== HỆ THỐNG GIÁM SÁT PHÒNG THI AI ===")
    
    t_camera = threading.Thread(target=io_camera_thread, daemon=True)
    t_mic = threading.Thread(target=io_audio_thread, daemon=True)      # Luồng Micro mới
    t_vision = threading.Thread(target=vision_ai_thread, daemon=True)
    t_audio = threading.Thread(target=audio_ai_thread, daemon=True)
    
    t_camera.start()
    t_mic.start()
    t_vision.start()
    t_audio.start()
    
    print("[INFO] Toàn bộ hệ thống đang chạy. Nhấn 'q' trên cửa sổ Camera để thoát.")
    
    try:
        while t_camera.is_alive():
            if not RESULT_QUEUE.empty():
                alert = RESULT_QUEUE.get()
                
                module_name = alert.get("module")
                status = alert.get("status")
                timestamp = alert.get("timestamp", time.time())
                details = alert.get("details", {})
                
                # CẬP NHẬT FIX: Đóng dấu thời gian lúc luồng main NHẬN ĐƯỢC cảnh báo
                alert['display_timestamp'] = time.time()
                
                # --- A. ĐẨY VÀO ACTIVE_OVERLAYS ĐỂ VẼ LÊN GUI ---
                with OVERLAY_LOCK:
                    # 1. Lọc và xóa các Bounding Box cũ của CÙNG MỘT module
                    # (Chỉ giữ lại nếu nó là của module khác, hoặc có cùng timestamp với alert hiện tại)
                    filtered_overlays = [
                        a for a in ACTIVE_OVERLAYS 
                        if not (a.get("module") == module_name and a.get("timestamp", 0) < timestamp)
                    ]
                    
                    # 2. Cập nhật lại danh sách và thêm cảnh báo mới nhất
                    ACTIVE_OVERLAYS.clear()
                    ACTIVE_OVERLAYS.extend(filtered_overlays)
                    ACTIVE_OVERLAYS.append(alert)
                
                # --- B. IN LOG CHI TIẾT RA TERMINAL ---
                module_name = alert.get("module")
                details = alert.get("details", {})
                time_str = time.strftime('%H:%M:%S', time.localtime(alert.get("timestamp", time.time())))
                
                if module_name == "face_verify" and status == "alert":
                    score = details.get("similarity_score")
                    bbox = details.get("unauthorized_bbox")
                    print(f"[{time_str}] PHÁT HIỆN NGƯỜI LẠ! Độ tương đồng: {score}. Tọa độ: {details.get('unauthorized_bbox')}")
                    
                elif module_name == "object_detect" and status == "alert":
                    detections = alert.get("detections", [])
                    for item in detections:
                        print(f"[{time_str}] PHÁT HIỆN {item.get('label', 'Vật thể').upper()} trên bàn! Tọa độ: {item.get('bbox')}")
                        
                elif module_name == "pose_gaze" and status == "warning":
                    print(f"[{time_str}] VI PHẠM TƯ THẾ: {details.get('action')}")
                    
                elif module_name == "audio_whisper" and status == "alert":
                    print(f"[{time_str}] ÂM THANH BẤT THƯỜNG: {details.get('transcription')}")

            # Nhường CPU một nhịp để tránh luồng Main chiếm hết tài nguyên
            time.sleep(0.01) 
                
    except KeyboardInterrupt:
        print("\n[INFO] Đang tiến hành tắt hệ thống an toàn...")
        
    finally:
        # Không dùng join(timeout=2) cho camera để tránh kẹt thread
        print("[INFO] Đã tắt máy chủ thành công.")

if __name__ == "__main__":
    main()

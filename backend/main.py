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
import logging
import warnings

# ==========================================
# KHỐI LỆNH "BỊT MIỆNG" SPAM LOG TỪ THƯ VIỆN
# ==========================================
# 1. Tắt cảnh báo Python (Bao gồm cả UserWarning và FutureWarning của InsightFace)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 2. Tắt lời nhắc nhở HF_TOKEN của Hugging Face
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# 3. Tắt cảnh báo Wayland/Gnome của Linux & log OpenCV
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false;qt.text.font.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
# --- SAU ĐÓ MỚI IMPORT CÁC THƯ VIỆN CHÍNH ---
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
FPS_SKIP = 8
CAMERA_ID = 0

# --- ĐÈN TÍN HIỆU ĐỒNG BỘ LUỒNG ---
VISION_READY = threading.Event()
AUDIO_READY = threading.Event()

# --- CỜ VÀ BIẾN ĐỂ CHỤP ẢNH ĐĂNG KÝ ---
REGISTER_FACE_EVENT = threading.Event()
REGISTER_FRAME = None

# --- BIẾN TOÀN CỤC CHO CÁC MODEL AI ---
face_model = None
yolo_model = None
gaze_model = None

# ==========================================
# 2. CẤU HÌNH HIỂN THỊ (GUI OVERLAYS)
# ==========================================
ACTIVE_OVERLAYS = []
OVERLAY_LOCK = threading.Lock()
OVERLAY_TTL = 0.5  # Thời gian sống (giây) của khung cảnh báo

# Import các module AI
from backend.ai_services.object_detect.object_detect import ObjectDetector
from backend.ai_services.pose_gaze.pose_gaze_test import PoseGazeDetector
from backend.ai_services.face_verify.face_verify import FaceVerifier

# Import luồng Audio thực tế của cộng sự (RealtimeAudioWorker)
from backend.ai_services.whisper.realtime_audio_ubuntu import RealtimeAudioWorker

# ==========================================
# 3. HÀM VẼ GIAO DIỆN LÊN OPENCV
# ==========================================
def draw_warning_overlays(frame):
    """Vẽ các Bounding Box, nhãn vi phạm và bảng cảnh báo lên khung hình."""
    current_time = time.time()
    
    # TỐI ƯU: Nếu không có cảnh báo nào, trả về ảnh gốc ngay lập tức
    if not ACTIVE_OVERLAYS:
        return frame
    
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

            # --- Hiển thị Banner âm thanh bất thường (Whisper / Audio Pipeline) ---
            elif module in ["audio_whisper", "audio_phobert_pipeline"]:
                transcription = alert.get("transcription", details.get("transcription", ""))
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 165, 255), -1)
                cv2.putText(frame, f"[AM THANH BAT THUONG]: {transcription}", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


# ==========================================
# 4. ĐỊNH NGHĨA CÁC LUỒNG XỬ LÝ (THREADS)
# ==========================================
def vision_ai_thread():
    """Luồng 2: Lấy khung hình từ bộ đệm và chạy các mô hình AI thị giác."""
    print("[INFO] Đang khởi động luồng AI Thị giác...")
    global face_model, yolo_model, gaze_model, REGISTER_FRAME

    yolo_model = ObjectDetector(model_path="weights/yolov8_finetuned.pt")
    face_model = FaceVerifier(db_path="data/student_faces/")
    gaze_model = PoseGazeDetector()

    VISION_READY.set()
    print("[INFO] ✅ AI Thị giác đã nạp xong!")

    while True:
        if REGISTER_FACE_EVENT.is_set():
            if REGISTER_FRAME is not None:
                try:
                    faces = face_model._detect_faces(REGISTER_FRAME)
                    if faces:
                        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                        x1, y1, x2, y2 = map(int, face.bbox)
                        
                        save_frame = REGISTER_FRAME.copy()
                        cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(save_frame, "REGISTERED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        file_name = f"data/student_faces/you_{int(time.time())}.jpg"
                        cv2.imwrite(file_name, save_frame)
                        print(f"[INFO] Đã vẽ khung xanh và lưu ảnh thành công: {file_name}")
                        face_model._load_database()
                    else:
                        print("[WARNING] Không tìm thấy khuôn mặt nào!")
                except Exception as e:
                    print(f"[LỖI KHI CHỤP ẢNH] {e}")
            
            REGISTER_FACE_EVENT.clear()
            REGISTER_FRAME = None

        data = FRAME_QUEUE.get()
        if data is None: 
            break
            
        frame, timestamp = data
        
        result_yolo = yolo_model.process_frame(frame, timestamp)
        if result_yolo is not None and not RESULT_QUEUE.full():
            RESULT_QUEUE.put(result_yolo)
            
        result_face = face_model.verify_face(frame, timestamp)
        if result_face is not None and not RESULT_QUEUE.full():
            RESULT_QUEUE.put(result_face)
            
        result_gaze = gaze_model.process_frame(frame, timestamp)
        if result_gaze is not None and not RESULT_QUEUE.full():
            RESULT_QUEUE.put(result_gaze)


def audio_ai_thread():
    """Luồng 3: Khởi chạy RealtimeAudioWorker (PyAudio + VAD + Whisper + PhoBERT)."""
    print("[INFO] Đang khởi động luồng AI Âm thanh thực tế...")
    
    try:
        audio_worker = RealtimeAudioWorker()
        # Ghi đè phương thức xử lý kết quả của audio_worker để đẩy kết quả vào RESULT_QUEUE chung của hệ thống
        original_worker_loop = audio_worker._worker_loop
        
        # Bật đèn báo Âm thanh đã sẵn sàng
        AUDIO_READY.set()
        print("[INFO] ✅ AI Âm thanh đã nạp xong!")
        
        # Chạy worker chính của cộng sự
        audio_worker.start()
    except Exception as e:
        print(f"[LỖI LUỒNG ÂM THANH] {e}")
        AUDIO_READY.set()


# ==========================================
# 5. LUỒNG ĐIỀU PHỐI (MAIN)
# ==========================================
def main():
    """Nhận kết quả từ AI, cập nhật GUI và in Log."""
    print("=== HỆ THỐNG GIÁM SÁT PHÒNG THI AI ===")
    
    t_vision = threading.Thread(target=vision_ai_thread, daemon=True)
    t_audio = threading.Thread(target=audio_ai_thread, daemon=True)
    
    t_vision.start()
    t_audio.start()

    print("[INFO] Hệ thống đang nạp mô hình AI vào bộ nhớ. Vui lòng đợi 1-2 phút...")
    VISION_READY.wait()
    AUDIO_READY.wait()
    print("[INFO] 🚀 TOÀN BỘ MÔ HÌNH ĐÃ SẴN SÀNG! Đang bật Camera...")
    
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0

    cv2.namedWindow("Exam Proctoring System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Exam Proctoring System", 1024, 768)
    
    print("[INFO] Toàn bộ hệ thống đang chạy. Nhấn 'q' trên cửa sổ Camera để thoát. Nhấn 's' để đăng ký khuôn mặt.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[LỖI] Camera bị ngắt kết nối.")
                break

            while not RESULT_QUEUE.empty():
                alert = RESULT_QUEUE.get_nowait()
                
                module_name = alert.get("module")
                status = alert.get("status")
                timestamp = alert.get("timestamp", time.time())
                details = alert.get("details", {})
                
                alert['display_timestamp'] = time.time()
                
                with OVERLAY_LOCK:
                    filtered_overlays = [
                        a for a in ACTIVE_OVERLAYS 
                        if not (a.get("module") == module_name and a.get("timestamp", 0) < timestamp)
                    ]
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
                    
                elif module_name in ["audio_whisper", "audio_phobert_pipeline"] or status == "alert":
                    print(f"[{time_str}] 🚨 ÂM THANH GIAN LẬN: '{alert.get('transcription', '')}'")

            rendered_frame = draw_warning_overlays(frame.copy())
            cv2.imshow("Exam Proctoring System", rendered_frame)
            
            if frame_count % FPS_SKIP == 0:
                if FRAME_QUEUE.full():
                    try:
                        FRAME_QUEUE.get_nowait()
                    except queue.Empty:
                        pass
                FRAME_QUEUE.put((frame.copy(), time.time()))
                
            frame_count += 1

            # Lắng nghe phím tắt 'q' để thoát. 
            # cv2.waitKey(1) cũng đóng luôn vai trò nhường CPU (time.sleep) ở bản cũ.
            key = cv2.waitKey(1) & 0xFF
                        
            if key == ord('q'):
                break

            # Nhường CPU một nhịp để tránh luồng Main chiếm hết tài nguyên
            # (Đã được comment lại vì cv2.waitKey(1) ở trên đã làm nhiệm vụ này rồi)
            # time.sleep(0.01) 
            # Lắng nghe phím tắt điều khiển
                
            # --- TÍCH HỢP: NHẤN 'S' ĐỂ ĐĂNG KÝ KHUÔN MẶT CÓ VẼ KHUNG ---
            elif key == ord('s'):
                if face_model is None:
                    print("[WARNING] AI Thị giác chưa khởi tạo xong!")
                    continue
                print("\n[INFO] Đã gửi lệnh chụp ảnh...")
                global REGISTER_FRAME
                REGISTER_FRAME = frame.copy()
                REGISTER_FACE_EVENT.set()

    except KeyboardInterrupt:
        print("\n[INFO] Đang tiến hành tắt hệ thống an toàn...")
        
    finally:
        # Dọn dẹp tài nguyên Camera
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Đã tắt máy chủ thành công.")
        
        # FIX LỖI CRASH UBUNTU: Ép hệ điều hành đóng băng và hủy ngay lập tức 
        # toàn bộ tiến trình C++ đang chạy ngầm mà không kích hoạt quy trình dọn dẹp (destructor).
        import os
        os._exit(0)

if __name__ == "__main__":
    main()

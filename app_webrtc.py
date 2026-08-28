import os
import logging
import warnings

# ==========================================
# KHỐI LỆNH "BỊT MIỆNG" SPAM LOG TỪ THƯ VIỆN
# ==========================================
# Tối ưu hóa driver SYCL cho Intel Arc (Giảm độ trễ giao tiếp)
os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
# Ép FaceNet/ONNXRuntime sử dụng OpenVINO để tăng tốc bằng Intel GPU
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "OpenVINOExecutionProvider,CPUExecutionProvider"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR) # Thêm dòng này để tắt log W000

import transformers
transformers.logging.set_verbosity_error()

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false;qt.text.font.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

# ---> THÊM 3 DÒNG NÀY ĐỂ DIỆT GỌN LOG CỦA MEDIAPIPE, TENSORFLOW VÀ ONNX <---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["GLOG_minloglevel"] = "3"
os.environ["ORT_LOGGING_LEVEL"] = "4"

import cv2
import threading
import time
import queue
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

# ---> KIỂM TRA THIẾT BỊ OPENVINO CHO INTEL GPU <---
@st.cache_resource
def initialize_openvino():
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        print(f"[INFO] OpenVINO đã sẵn sàng trên các thiết bị: {devices}")
        return devices
    except Exception as e:
        print(f"[WARNING] Không khởi tạo được OpenVINO: {e}")
        return []
# Gọi hàm ngay phía dưới:
core, devices = initialize_openvino()

# Import các module AI của bạn
from backend.ai_services.object_detect.object_detect import ObjectDetector
from backend.ai_services.pose_gaze.pose_gaze_service import PoseGazeDetector
from backend.ai_services.face_verify.face_verify import FaceVerifier
from backend.ai_services.whisper.realtime_audio_ubuntu import RealtimeAudioWorker

# ==========================================
# 1. KHỞI TẠO TÀI NGUYÊN (CHẠY 1 LẦN DUY NHẤT)
# ==========================================
@st.cache_resource
def init_system_resources():
    print("[INFO] Khởi tạo trạm trung chuyển dữ liệu...")
    frame_q = queue.Queue(maxsize=2) 
    gaze_q = queue.Queue(maxsize=2)
    result_q = queue.Queue(maxsize=100)
    
    overlays = []
    overlay_lock = threading.Lock()
    
    vision_ready = threading.Event()
    audio_ready = threading.Event()
    gaze_ready = threading.Event()
    
    register_face_event = threading.Event()
    shared_state = {
        "register_frame": None,
        "gaze_model": None,
        "frame_count": 0,
        "logs": []
    }

    # --- LUỒNG AI THỊ GIÁC ---
    # LUỒNG 1: Xử lý nặng (YOLO + FaceNet) - Chạy chậm
    def heavy_vision_thread():
        print("[INFO] Đang khởi động luồng AI Thị giác (Nặng) bằng OpenVINO...")
        # Trỏ tới thư mục chứa file .xml và .bin của OpenVINO
        try:
            # Xóa tham số enable_smartphone_fallback
            # Thêm confidence_threshold để dễ dàng test (giảm xuống 0.3 để OpenVINO nhạy hơn)
            yolo_model = ObjectDetector(
                model_path="weights/best_openvino_model", 
                device="GPU",
                confidence_threshold=0.5
            )
            face_model = FaceVerifier(db_path="data/student_faces/")
        except Exception as e:
            print(f"[LỖI KHỞI TẠO AI THỊ GIÁC]: {e}")
            vision_ready.set()
            return
        
        # ---> BÍ QUYẾT ĐỒNG BỘ: Tạo hàm Masking chung cho cả Camera và Đăng ký <---
        def get_largest_stranger_with_masking(img, ts):
            faces = face_model._detect_faces(img)
            if not faces:
                return None, None
                
            # Sắp xếp khuôn mặt theo diện tích từ TO đến NHỎ
            faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            h, w = img.shape[:2]
            
            # Dùng kỹ thuật "Che phủ" (Masking) để test từng mặt từ to đến nhỏ
            for i, target_face in enumerate(faces):
                masked_frame = img.copy()
                
                # Vẽ ô đen che giấu TẤT CẢ các khuôn mặt khác
                for j, other_face in enumerate(faces):
                    if i != j:
                        ox1, oy1, ox2, oy2 = map(int, other_face.bbox)
                        # Nới rộng ô đen ra 10 pixel để xóa thật sạch
                        cv2.rectangle(masked_frame, 
                                      (max(0, ox1 - 10), max(0, oy1 - 10)), 
                                      (min(w, ox2 + 10), min(h, oy2 + 10)), 
                                      (0, 0, 0), -1)
                                      
                # Gửi ảnh đã che đen đi kiểm tra (AI giờ chỉ còn thấy 1 mặt duy nhất)
                verify_result = face_model.verify_face(masked_frame, ts)
                
                if verify_result and verify_result.get("status") == "alert":
                    # Ghi đè lại tọa độ gốc để khung vàng vẽ chính xác
                    verify_result["details"]["unauthorized_bbox"] = [
                        int(x) for x in target_face.bbox
                    ]
                    return verify_result, target_face.bbox
                    
            # Nếu toàn người quen (hoặc DB trống), trả về bbox của người to nhất làm dự phòng
            return None, faces[0].bbox
        # -------------------------------------------------------------------------
        
        vision_ready.set()
        print("[INFO] ✅ AI Thị giác (Nặng) đã nạp xong!")

        while True:
            # 1. Xử lý chụp ảnh đăng ký (Đồng bộ Masking)
            if register_face_event.is_set():
                frame_to_save = shared_state["register_frame"]
                if frame_to_save is not None:
                    try:
                        h, w = frame_to_save.shape[:2]
                        
                        # Gọi hàm Masking chung
                        _, target_bbox = get_largest_stranger_with_masking(frame_to_save, time.time())
                                
                        if target_bbox is not None:
                            # Bước 5: Nới rộng khung chữ nhật
                            x1, y1, x2, y2 = map(int, target_bbox)
                            margin = 40
                            x1_ext = max(0, x1 - margin)
                            y1_ext = max(0, y1 - margin)
                            x2_ext = min(w, x2 + margin)
                            y2_ext = min(h, y2 + margin)
                            
                            # 1. BẢN CHO AI: Chỉ cắt khuôn mặt lưu vào thư mục chuẩn
                            face_crop = frame_to_save[y1_ext:y2_ext, x1_ext:x2_ext]
                            if face_crop.size > 0:
                                ai_file = f"data/student_faces/student_{int(time.time())}.jpg"
                                cv2.imwrite(ai_file, face_crop)
                            
                            # 2. BẢN TOÀN CẢNH CHO BẠN: Vẽ khung xanh và lưu ra thư mục riêng
                            save_frame = frame_to_save.copy()
                            cv2.rectangle(save_frame, (x1_ext, y1_ext), (x2_ext, y2_ext), (0, 255, 0), 2)
                            cv2.putText(save_frame, "REGISTERED", (x1_ext, y1_ext - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            import os
                            os.makedirs("data/registered_logs", exist_ok=True)
                            human_file = f"data/registered_logs/full_student_{int(time.time())}.jpg"
                            cv2.imwrite(human_file, save_frame)
                            
                            # Nạp lại Face Vector (AI lúc này chỉ đọc bản cắt trong student_faces)
                            face_model._load_database()
                            
                            result_q.put({"module": "system", "status": "info", "message": f"📸 Đã đăng ký thành công!"})
                        else:
                            result_q.put({"module": "system", "status": "error", "message": "⚠️ Không tìm thấy khuôn mặt để lưu!"})
                            
                    except Exception as e:
                        print(f"[LỖI KHI CHỤP ẢNH] {e}")
                
                register_face_event.clear()
                shared_state["register_frame"] = None

            # 2. Xử lý khung hình
            data = frame_q.get()
            if data is None: break
            frame, timestamp = data
            # print(f"[DEBUG YOLO] Nhận frame từ queue: {timestamp}")
            result_yolo = yolo_model.process_frame(frame, timestamp)
            # [DEBUG 1] In ra toàn bộ kết quả thô trả về từ mô hình
            # print(f"[DEBUG YOLO] Kết quả thô từ ObjectDetector: {result_yolo}")
    
            if result_yolo is not None:
                detections = result_yolo.get("detections", [])
        
                # [DEBUG 2] Kiểm tra số lượng vật thể bắt được
                if len(detections) > 0:
                    # print(f"[DEBUG YOLO] Đã bắt được {len(detections)} vật thể. Cập nhật status thành 'alert'.")
                    result_yolo["status"] = "alert" # BẮT BUỘC: Đánh dấu là cảnh báo để hàm Main xử lý
            
                    # Đẩy vào queue nếu chưa đầy
                    if not result_q.full():
                        result_q.put(result_yolo)
                        # print("[DEBUG YOLO] Đã đẩy cảnh báo vật thể vào RESULT_QUEUE thành công.")
                    else:
                        print("[DEBUG YOLO] CẢNH BÁO: RESULT_QUEUE đã đầy, bị rớt frame cảnh báo!")

            if result_yolo:
                # Ép key "module" để WebRTC nhận diện luồng
                if "module" not in result_yolo:
                    result_yolo["module"] = "object_detect"
                
                # Logic xác định dị thường cho ObjectDetector: có vật thể trong list 'detections'
                has_anomaly = len(result_yolo.get("detections", [])) > 0
                
                if has_anomaly:
                    result_yolo["status"] = "alert"
                
                # if not result_q.full(): 
                #     result_q.put(result_yolo)

            # ---> ĐỒNG BỘ: Luồng Camera giám sát giờ cũng dùng Masking để quét người lạ <---
            result_face, _ = get_largest_stranger_with_masking(frame, timestamp)
            if result_face and not result_q.full(): 
                result_q.put(result_face)


    # LUỒNG 2: Xử lý nhẹ (MediaPipe PoseGaze) - Chạy siêu tốc
    def fast_gaze_thread():
        gaze_model = PoseGazeDetector()
        shared_state["gaze_model"] = gaze_model
        gaze_ready.set()
        print("[INFO] ✅ Fast Gaze Thread đã nạp xong!")
        while True:
            data = gaze_q.get()
            if data is None: break
            frame, timestamp = data
            result_gaze = gaze_model.process_frame(frame, timestamp)
            if result_gaze and not result_q.full(): result_q.put(result_gaze)


    # --- LUỒNG AI ÂM THANH (Dùng PyAudio độc lập) ---
    def audio_ai_thread():
        print("[INFO] Đang khởi động luồng AI Âm thanh thực tế...")
        try:
            audio_worker = RealtimeAudioWorker()
            
            original_pipeline = audio_worker.pipeline.process_audio
            def hooked_process_audio(*args, **kwargs):
                res = original_pipeline(*args, **kwargs)
                if res and not result_q.full():
                    res["module"] = "audio_phobert_pipeline"
                    result_q.put(res)
                return res
                
            audio_worker.pipeline.process_audio = hooked_process_audio
            audio_ready.set()
            print("[INFO] ✅ Audio Thread đã nạp xong!")
            audio_worker.start()
        except Exception as e:
            print(f"[LỖI LUỒNG ÂM THANH] {e}")
            audio_ready.set()

    # Khởi chạy các luồng
    t_vision = threading.Thread(target=heavy_vision_thread, daemon=True)
    t_audio = threading.Thread(target=audio_ai_thread, daemon=True)
    t_gaze = threading.Thread(target=fast_gaze_thread, daemon=True)
    
    t_vision.start()
    t_audio.start()
    t_gaze.start()
    
    vision_ready.wait()
    audio_ready.wait()
    gaze_ready.wait()

    return frame_q, gaze_q, result_q, overlays, overlay_lock, register_face_event, shared_state

# Kích hoạt Cache
FRAME_QUEUE, GAZE_QUEUE, RESULT_QUEUE, ACTIVE_OVERLAYS, OVERLAY_LOCK, REG_EVENT, SHARED_STATE = init_system_resources()
OVERLAY_TTL = 1.5
FPS_SKIP = 2

# ==========================================
# 2. HÀM VẼ GIAO DIỆN (ĐƯỢC GỌI TRONG WEBRTC)
# ==========================================
def draw_warning_overlays(frame):
    current_time = time.time()
    
    # 1. Vẽ Skeleton 3D trước (để nó nằm dưới)
    gaze_ai = SHARED_STATE.get("gaze_model")
    if gaze_ai and hasattr(gaze_ai, 'draw_skeleton'):
        gaze_ai.draw_skeleton(frame)
        
    if not ACTIVE_OVERLAYS: 
        return frame
    
    # 2. Vẽ Bounding Box Cảnh báo (nằm trên)
    with OVERLAY_LOCK:
        valid_overlays = [item for item in ACTIVE_OVERLAYS if current_time - item.get('display_timestamp', current_time) <= OVERLAY_TTL]
        ACTIVE_OVERLAYS.clear()
        ACTIVE_OVERLAYS.extend(valid_overlays)
        
        for alert in ACTIVE_OVERLAYS:
            module = alert.get("module", "")
            details = alert.get("details", {})
            
            if module == "object_detect":
                # Lấy danh sách vật thể chuẩn từ ObjectDetector
                detections = alert.get("detections", [])

                # [DEBUG 4] Báo cáo số lượng bounding box đang được OpenCV vẽ
                # if len(detections) > 0:
                #     print(f"[DEBUG DRAW] Đang tiến hành vẽ {len(detections)} khung (Bounding Box) lên màn hình...")

                for det in detections:
                    bbox = det.get("bbox")
                    label = det.get("label", "VAT CAM")
                    conf = det.get("confidence", 0.0)
                    # [DEBUG 5] Kiểm tra xem tọa độ bbox có bị rỗng hay sai định dạng không
                    # print(f"[DEBUG DRAW] Tọa độ vẽ: {bbox}, Nhãn: {label}, Confidence: {conf}")
                    
                    if isinstance(label, str):
                        label = label.upper()

                    if bbox is not None and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif module == "face_verify":
                bbox = details.get("unauthorized_bbox")
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                    cv2.putText(frame, "NGUOI LA", (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif module == "pose_gaze":
                action = details.get("action", "VI PHAM TU THE").upper()
                cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (0, 0, 200), -1)
                cv2.putText(frame, f"[CANH BAO TU THE]: {action}", (20, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

# ==========================================
# 3. CALLBACK VIDEO XỬ LÝ FRAME WEBRTC
# ==========================================
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # 1. Chuyển đổi định dạng frame từ WebRTC sang Numpy (OpenCV)
    img = frame.to_ndarray(format="bgr24")
    
    SHARED_STATE["frame_count"] += 1
    current_time = time.time()

    # 2. Xử lý Hàng đợi Kết quả & Đóng gói Cảnh báo
    while not RESULT_QUEUE.empty():
        try:
            alert = RESULT_QUEUE.get_nowait()

            module_name = alert.get("module", "")
            status = alert.get("status")
            timestamp = alert.get("timestamp", current_time)
            details = alert.get("details", {})
            # [DEBUG 3] In ra xem Main Thread có bắt được alert từ YOLO không
            # if module_name == "object_detect":
            #     print(f"[DEBUG MAIN] Nhận được từ Queue - Module: {module_name}, Status: {status}, Detections: {alert.get('detections')}")
            # CẬP NHẬT FIX: Đóng dấu thời gian lúc luồng main NHẬN ĐƯỢC cảnh báo
            alert['display_timestamp'] = time.time()

            if module_name == "system":
                SHARED_STATE["logs"].insert(0, f"**[{time.strftime('%H:%M:%S')}]** {alert.get('message')}")
                continue

            alert['display_timestamp'] = current_time
            time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
            
            with OVERLAY_LOCK:
                filtered_overlays = [a for a in ACTIVE_OVERLAYS if not (a.get("module") == module_name and a.get("timestamp", 0) < timestamp)]
                ACTIVE_OVERLAYS.clear()
                ACTIVE_OVERLAYS.extend(filtered_overlays)
                ACTIVE_OVERLAYS.append(alert)
                
            # Tạo log hiển thị bên cột phải
            log_msg = ""
            status = alert.get("status", "")
            if module_name == "face_verify" and status == "alert":
                log_msg = f"**[{time_str}]** 🕵️‍♂️ PHÁT HIỆN NGƯỜI LẠ!"
            elif module_name == "object_detect" and status == "alert":
                log_msg = f"**[{time_str}]** 📱 PHÁT HIỆN VẬT CẤM!"
            elif module_name == "pose_gaze" and status == "warning":
                log_msg = f"**[{time_str}]** ⚠️ TƯ THẾ: {details.get('action', 'Bất thường')}"
            elif "audio" in module_name or "transcription" in alert or "transcription" in details:
                transcription = alert.get("transcription", details.get("transcription", "")).strip()
                if transcription:  
                    risk_level = str(alert.get('risk', '')).lower()
                    audio_status = str(alert.get('status', '')).lower()
                    if risk_level in ['cheating', 'high', 'medium'] or audio_status == 'alert':
                        log_msg = f"**[{time_str}]** <span style='color:red'>🚨 GIAN LẬN ÂM THANH: '{transcription}'</span>"
                    else:
                        log_msg = f"**[{time_str}]** 🎙️ Hội thoại: '{transcription}'"
            
            if log_msg:
                SHARED_STATE["logs"].insert(0, log_msg)
                if len(SHARED_STATE["logs"]) > 20:
                    SHARED_STATE["logs"].pop()

        except queue.Empty:
            break

    # 3. Đẩy ảnh mới vào Queue cho AI xử lý (Frame Skipping)

    # Bơm ảnh cho luồng Skeleton (Tốc độ cao: Lấy 1 ảnh mỗi 2 frame ~ 10 FPS)
    if SHARED_STATE["frame_count"] % 2 == 0:
        if GAZE_QUEUE.full():
            try: GAZE_QUEUE.get_nowait()
            except queue.Empty: pass
        GAZE_QUEUE.put((img.copy(), current_time))

    if SHARED_STATE["frame_count"] % FPS_SKIP == 0:
        if FRAME_QUEUE.full():
            try: FRAME_QUEUE.get_nowait()
            except queue.Empty: pass
        FRAME_QUEUE.put((img.copy(), current_time))

    # 4. Chụp ảnh đăng ký khuôn mặt
    if SHARED_STATE.get("save_face_flag"):
        SHARED_STATE["register_frame"] = img.copy()
        REG_EVENT.set()
        SHARED_STATE["logs"].insert(0, "⏳ Đang gửi lệnh trích xuất khuôn mặt...")
        SHARED_STATE["save_face_flag"] = False

    # 5. Vẽ giao diện và trả về cho WebRTC
    rendered_img = draw_warning_overlays(img)
    return av.VideoFrame.from_ndarray(rendered_img, format="bgr24")

# ==========================================
# 4. GIAO DIỆN TRANG WEB STREAMLIT
# ==========================================
st.set_page_config(layout="wide", page_title="AI Exam Proctoring (WebRTC)")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>HỆ THỐNG GIÁM SÁT PHÒNG THI AI (WEBRTC)</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([7, 3])

with col1:
    st.markdown("### 📷 Camera Giám Sát Real-time (Zero Latency)")
    
    # Nút bấm đăng ký khuôn mặt
    if st.button("📸 Đăng ký khuôn mặt", use_container_width=True):
        SHARED_STATE["save_face_flag"] = True

    # Cấu hình WebRTC (Tắt Audio của WebRTC vì Audio đã chạy độc lập qua PyAudio)
    webrtc_streamer(
        key="exam-proctor",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": True,
            "audio": False # Chặn WebRTC thu âm để tránh đụng độ với luồng PyAudio
        },
        async_processing=True,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    )

with col2:
    st.markdown("### 📜 Lịch Sử Cảnh Báo")

    # ---> LỆNH TỰ ĐỘNG LÀM MỚI GIAO DIỆN MỖI 3 GIÂY (3000 mili-giây) <---
    st_autorefresh(interval=3000, key="auto_refresh_log")

    # Lấy Log mới nhất từ Shared State
    log_html = "<br>".join(SHARED_STATE["logs"]) if SHARED_STATE["logs"] else "Hệ thống đang chạy..."
    st.markdown(f"<div style='height: 500px; overflow-y: auto; background-color: #f0f2f6; padding: 10px; border-radius: 10px;'>{log_html}</div>", unsafe_allow_html=True)
    
    # Nút Refresh Log bằng tay (vì WebRTC chạy ngầm, giao diện cần trigger để hiện Log mới)
    # st.button("🔄 Làm mới Log", use_container_width=True)
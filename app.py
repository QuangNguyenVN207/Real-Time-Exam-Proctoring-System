import os
import logging
import warnings

# ==========================================
# KHỐI LỆNH "BỊT MIỆNG" SPAM LOG TỪ THƯ VIỆN
# ==========================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # Chặn lỗi chia cho 0 của Noisereduce
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false;qt.text.font.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import cv2
import threading
import time
import queue
import streamlit as st

# Import các module AI của bạn
from backend.ai_services.object_detect.object_detect import ObjectDetector
from backend.ai_services.pose_gaze.pose_gaze_test import PoseGazeDetector
from backend.ai_services.face_verify.face_verify import FaceVerifier
from backend.ai_services.whisper.realtime_audio_ubuntu import RealtimeAudioWorker

# ==========================================
# 1. KHỞI TẠO TÀI NGUYÊN (CACHE CHO STREAMLIT)
# ==========================================
@st.cache_resource
def init_system_resources():
    frame_q = queue.Queue(maxsize=2) 
    result_q = queue.Queue(maxsize=100)
    overlays = []
    overlay_lock = threading.Lock()
    
    vision_ready = threading.Event()
    audio_ready = threading.Event()
    
    # THÊM CỜ VÀ BIẾN ĐỂ CHỤP ẢNH ĐĂNG KÝ XUYÊN LUỒNG
    register_face_event = threading.Event()
    shared_state = {"register_frame": None}

    def vision_ai_thread():
        print("[INFO] Đang khởi động luồng AI Thị giác...")
        yolo_model = ObjectDetector(model_path="weights/yolov8_finetuned.pt")
        face_model = FaceVerifier(db_path="data/student_faces/")
        gaze_model = PoseGazeDetector()
        
        vision_ready.set()
        print("[INFO] ✅ AI Thị giác đã nạp xong!")

        while True:
            # --- KIỂM TRA LỆNH CHỤP ẢNH TỪ STREAMLIT ---
            if register_face_event.is_set():
                frame_to_save = shared_state["register_frame"]
                if frame_to_save is not None:
                    try:
                        faces = face_model._detect_faces(frame_to_save)
                        if faces:
                            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                            x1, y1, x2, y2 = map(int, face.bbox)
                            
                            save_frame = frame_to_save.copy()
                            cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(save_frame, "REGISTERED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            file_name = f"data/student_faces/you_{int(time.time())}.jpg"
                            cv2.imwrite(file_name, save_frame)
                            print(f"[INFO] Đã vẽ khung xanh và lưu ảnh thành công: {file_name}")
                            
                            face_model._load_database()
                            result_q.put({"module": "system", "status": "info", "message": f"📸 Đã lưu thành công: {file_name}"})
                        else:
                            result_q.put({"module": "system", "status": "error", "message": "⚠️ Không tìm thấy khuôn mặt nào để lưu!"})
                    except Exception as e:
                        print(f"[LỖI KHI CHỤP ẢNH] {e}")
                
                register_face_event.clear()
                shared_state["register_frame"] = None

            # --- XỬ LÝ KHUNG HÌNH BÌNH THƯỜNG ---
            data = frame_q.get()
            if data is None: break
            frame, timestamp = data
            
            result_yolo = yolo_model.process_frame(frame, timestamp)
            if result_yolo and not result_q.full(): result_q.put(result_yolo)
                
            result_face = face_model.verify_face(frame, timestamp)
            if result_face and not result_q.full(): result_q.put(result_face)
                
            result_gaze = gaze_model.process_frame(frame, timestamp)
            if result_gaze and not result_q.full(): result_q.put(result_gaze)

    def audio_ai_thread():
        print("[INFO] Đang khởi động luồng AI Âm thanh thực tế...")
        try:
            audio_worker = RealtimeAudioWorker()
            
            original_pipeline = audio_worker.pipeline.process_audio
            def hooked_process_audio(clean_audio, timestamp, source):
                res = original_pipeline(clean_audio, timestamp, source)
                if res:
                    # BẮT BUỘC: Đóng dấu thẻ tên module để Streamlit phân biệt được
                    res["module"] = "audio_phobert_pipeline"
                    
                    if not result_q.full():
                        result_q.put(res)
                return res
            audio_worker.pipeline.process_audio = hooked_process_audio

            audio_ready.set()
            print("[INFO] ✅ AI Âm thanh đã nạp xong!")
            audio_worker.start()
        except Exception as e:
            print(f"[LỖI LUỒNG ÂM THANH] {e}")
            audio_ready.set()

    t_vision = threading.Thread(target=vision_ai_thread, daemon=True)
    t_audio = threading.Thread(target=audio_ai_thread, daemon=True)
    
    t_vision.start()
    t_audio.start()
    
    vision_ready.wait()
    audio_ready.wait()

    return frame_q, result_q, overlays, overlay_lock, register_face_event, shared_state

# Gọi hàm khởi tạo
FRAME_QUEUE, RESULT_QUEUE, ACTIVE_OVERLAYS, OVERLAY_LOCK, REG_EVENT, SHARED_STATE = init_system_resources()
OVERLAY_TTL = 1.5

# ==========================================
# 2. HÀM VẼ GIAO DIỆN (OVERLAYS)
# ==========================================
def draw_warning_overlays(frame):
    current_time = time.time()
    if not ACTIVE_OVERLAYS: return frame
    
    with OVERLAY_LOCK:
        valid_overlays = [item for item in ACTIVE_OVERLAYS if current_time - item.get('display_timestamp', current_time) <= OVERLAY_TTL]
        ACTIVE_OVERLAYS.clear()
        ACTIVE_OVERLAYS.extend(valid_overlays)
        
        for alert in ACTIVE_OVERLAYS:
            module = alert.get("module")
            details = alert.get("details", {})
            
            if module == "object_detect":
                for det in alert.get("detections", []):
                    bbox = det.get("bbox")
                    label = det.get("label", "Vat cam").upper()
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif module == "face_verify":
                bbox = details.get("unauthorized_bbox")
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                    cv2.putText(frame, "NGUOI LA", (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif module == "pose_gaze":
                action = details.get("action", "VI PHAM TU THE").upper()
                cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (0, 0, 200), -1)
                cv2.putText(frame, f"[CANH BAO TU THE]: {action}", (20, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            elif module in ["audio_whisper", "audio_phobert_pipeline"]:
                transcription = alert.get("transcription", details.get("transcription", ""))
                if transcription.strip():
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 165, 255), -1)
                    cv2.putText(frame, f"[AM THANH]: {transcription}", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame

# ==========================================
# 3. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(layout="wide", page_title="AI Exam Proctoring")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>HỆ THỐNG GIÁM SÁT PHÒNG THI AI</h1>", unsafe_allow_html=True)

# BIẾN STATE ĐỂ QUẢN LÝ NÚT BẤM
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "save_face_flag" not in st.session_state:
    st.session_state.save_face_flag = False
if "logs" not in st.session_state:
    st.session_state.logs = []

col1, col2 = st.columns([7, 3])

with col1:
    st.markdown("### 📷 Camera Giám Sát Real-time")
    
    # 2 NÚT BẤM NẰM CẠNH NHAU
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if not st.session_state.run_camera:
            if st.button("▶️ Bắt đầu giám sát", use_container_width=True):
                st.session_state.run_camera = True
                st.session_state.logs.insert(0, "✅ Hệ thống đã bắt đầu giám sát...")
                st.rerun()
        else:
            if st.button("⏹️ Dừng giám sát", type="primary", use_container_width=True):
                st.session_state.run_camera = False
                st.session_state.logs.insert(0, "🛑 Đã dừng giám sát.")
                st.rerun()
                
    with btn_col2:
        if st.session_state.run_camera:
            if st.button("📸 Đăng ký khuôn mặt", use_container_width=True):
                st.session_state.save_face_flag = True
                
    video_placeholder = st.empty()

with col2:
    st.markdown("### 📜 Lịch Sử Cảnh Báo")
    log_placeholder = st.empty()


# ==========================================
# 4. VÒNG LẶP ĐIỀU PHỐI (MAIN LOOP)
# ==========================================
if st.session_state.run_camera:
    # 1. LƯU CAMERA VÀO SESSION STATE ĐỂ TRÁNH BỊ KHÓA KHI TẢI LẠI TRANG
    if "camera_obj" not in st.session_state or st.session_state.camera_obj is None:
        st.session_state.camera_obj = cv2.VideoCapture(0)
        # HẠ ĐỘ PHÂN GIẢI XUỐNG ĐỂ TĂNG GẤP ĐÔI TỐC ĐỘ STREAMLIT VÀ AI
        st.session_state.camera_obj.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        st.session_state.camera_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        # ÉP CAMERA CHẠY Ở 30FPS CỨNG TỪ PHẦN CỨNG
        st.session_state.camera_obj.set(cv2.CAP_PROP_FPS, 30)
        # ---> THÊM DÒNG NÀY ĐỂ CHỐNG DELAY VIDEO <---
        st.session_state.camera_obj.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    # TĂNG TẦN SUẤT BỎ QUA KHUNG HÌNH (GIAO DIỆN MƯỢT, AI VẪN ĐỦ CHẠY)
    FPS_SKIP = 10

    while st.session_state.run_camera:
        # Sử dụng camera từ session_state thay vì cap cục bộ
        ret, frame = st.session_state.camera_obj.read()
        
        if not ret:
            st.error("Lỗi: Không thể kết nối với Camera! (Thiết bị có thể đang bị chiếm dụng)")
            st.session_state.run_camera = False
            # Dọn dẹp nếu lỗi
            if st.session_state.camera_obj is not None:
                st.session_state.camera_obj.release()
                st.session_state.camera_obj = None
            break
            
        # NẾU BẤM NÚT CHỤP ẢNH -> TRUYỀN FRAME XUYÊN LUỒNG CHO AI
        if st.session_state.save_face_flag:
            SHARED_STATE["register_frame"] = frame.copy()
            REG_EVENT.set()
            st.session_state.logs.insert(0, "⏳ Đang gửi lệnh trích xuất khuôn mặt...")
            st.session_state.save_face_flag = False

        # Đọc dữ liệu từ hàng đợi AI
        while not RESULT_QUEUE.empty():
            alert = RESULT_QUEUE.get_nowait()
            module_name = alert.get("module")
            status = alert.get("status")
            timestamp = alert.get("timestamp", time.time())
            details = alert.get("details", {})
            
            # Xử lý thông báo từ hệ thống (khi chụp ảnh thành công/thất bại)
            if module_name == "system":
                st.session_state.logs.insert(0, f"**[{time.strftime('%H:%M:%S')}]** {alert.get('message')}")
                continue
            
            alert['display_timestamp'] = time.time()
            time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
            
            # Đẩy vào Overlay để vẽ lên Camera
            with OVERLAY_LOCK:
                filtered_overlays = [a for a in ACTIVE_OVERLAYS if not (a.get("module") == module_name and a.get("timestamp", 0) < timestamp)]
                ACTIVE_OVERLAYS.clear()
                ACTIVE_OVERLAYS.extend(filtered_overlays)
                ACTIVE_OVERLAYS.append(alert)
            
            # XỬ LÝ LOG HIỂN THỊ CỘT BÊN PHẢI
            # ---> THÊM CỜ ĐÁNH DẤU <---
        logs_changed = False 

        # Đọc dữ liệu từ hàng đợi AI
        while not RESULT_QUEUE.empty():
            alert = RESULT_QUEUE.get_nowait()
            module_name = alert.get("module")
            status = alert.get("status")
            timestamp = alert.get("timestamp", time.time())
            details = alert.get("details", {})
            
            if module_name == "system":
                st.session_state.logs.insert(0, f"**[{time.strftime('%H:%M:%S')}]** {alert.get('message')}")
                logs_changed = True # Đánh dấu có thay đổi
                continue
            
            alert['display_timestamp'] = time.time()
            time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
            
            with OVERLAY_LOCK:
                filtered_overlays = [a for a in ACTIVE_OVERLAYS if not (a.get("module") == module_name and a.get("timestamp", 0) < timestamp)]
                ACTIVE_OVERLAYS.clear()
                ACTIVE_OVERLAYS.extend(filtered_overlays)
                ACTIVE_OVERLAYS.append(alert)
            
            log_msg = ""
            if module_name == "face_verify" and status == "alert":
                log_msg = f"**[{time_str}]** 🕵️‍♂️ PHÁT HIỆN NGƯỜI LẠ!"
            elif module_name == "object_detect" and status == "alert":
                log_msg = f"**[{time_str}]** 📱 PHÁT HIỆN VẬT CẤM!"
            elif module_name == "pose_gaze" and status == "warning":
                log_msg = f"**[{time_str}]** ⚠️ TƯ THẾ: {details.get('action')}"
            
            elif module_name == "audio_phobert_pipeline":
                transcription = alert.get("transcription", details.get("transcription", "")).strip()
                if transcription:  
                    risk_level = str(alert.get('risk', '')).lower()
                    audio_status = str(alert.get('status', '')).lower()
                    if risk_level in ['cheating', 'high', 'medium'] or audio_status == 'alert':
                        log_msg = f"**[{time_str}]** <span style='color:red'>🚨 GIAN LẬN ÂM THANH: '{transcription}'</span>"
                    else:
                        log_msg = f"**[{time_str}]** 🎙️ Hội thoại: '{transcription}'"

            if log_msg:
                st.session_state.logs.insert(0, log_msg)
                if len(st.session_state.logs) > 20:
                    st.session_state.logs.pop()
                logs_changed = True # Đánh dấu có thay đổi

        # ---> CHỈ CẬP NHẬT GIAO DIỆN KHI CÓ LOG MỚI <---
        if logs_changed:
            log_html = "<br>".join(st.session_state.logs)
            log_placeholder.markdown(f"<div style='height: 500px; overflow-y: auto; background-color: #f0f2f6; padding: 10px; border-radius: 10px;'>{log_html}</div>", unsafe_allow_html=True)

        # Vẽ đè Bounding Box và đưa lên giao diện Streamlit
        rendered_frame = draw_warning_overlays(frame.copy())

        # Hiển thị cột Logs
        log_html = "<br>".join(st.session_state.logs)
        log_placeholder.markdown(f"<div style='height: 500px; overflow-y: auto; background-color: #f0f2f6; padding: 10px; border-radius: 10px;'>{log_html}</div>", unsafe_allow_html=True)

        # Vẽ đè Bounding Box và đưa lên giao diện Streamlit
        rendered_frame = draw_warning_overlays(frame.copy())
        
        rendered_frame_rgb = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
        
        video_placeholder.image(rendered_frame_rgb, width='stretch', output_format="JPEG")
        
        if frame_count % FPS_SKIP == 0:
            if FRAME_QUEUE.full():
                try: FRAME_QUEUE.get_nowait()
                except queue.Empty: pass
            FRAME_QUEUE.put((frame.copy(), time.time()))
            
        frame_count += 1
        
else:
    # 2. DỌN DẸP CAMERA KHI BẤM DỪNG HOẶC CHƯA BẬT
    if "camera_obj" in st.session_state and st.session_state.camera_obj is not None:
        st.session_state.camera_obj.release()
        st.session_state.camera_obj = None
        
    video_placeholder.info("Nhấn 'Bắt đầu giám sát' để mở camera.")
    log_html = "<br>".join(st.session_state.logs)
    log_placeholder.markdown(f"<div style='height: 500px; overflow-y: auto; background-color: #f0f2f6; padding: 10px; border-radius: 10px;'>{log_html}</div>", unsafe_allow_html=True)
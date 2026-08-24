import os
import logging
import warnings

# Ẩn các cảnh báo liên quan đến ScriptRunContext của Streamlit (Kế thừa từ app_win.py)
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.state.session_state').setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# ==========================================
# KHỐI LỆNH "BỊT MIỆNG" SPAM LOG TỪ THƯ VIỆN
# ==========================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

# ĐÃ BỎ CÁC CẤU HÌNH ĐẶC TRƯNG CỦA UBUNTU (X11, XCB) ĐỂ CHẠY ĐƯỢC TRÊN WINDOWS
# os.environ["QT_QPA_PLATFORM"] = "xcb"
# os.environ["XDG_SESSION_TYPE"] = "x11"
# os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false;qt.text.font.*=false"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import cv2
import threading
import time
import queue
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

# Import các module AI của bạn
from backend.ai_services.object_detect.object_detect import ObjectDetector
from backend.ai_services.pose_gaze.pose_gaze_service import PoseGazeDetector
from backend.ai_services.face_verify.face_verify import FaceVerifier

# ĐỔI IMPORT SANG BẢN WORKER CHO WINDOWS THEO YÊU CẦU:
from backend.ai_services.whisper.realtime_audio_worker import RealtimeAudioWorker

# ==========================================
# 1. KHỞI TẠO TÀI NGUYÊN (CHẠY 1 LẦN DUY NHẤT)
# ==========================================
@st.cache_resource
def init_system_resources():
    print("[INFO] Khởi tạo trạm trung chuyển dữ liệu...")
    frame_q = queue.Queue(maxsize=2) 
    result_q = queue.Queue(maxsize=100)
    
    overlays = []
    overlay_lock = threading.Lock()
    
    vision_ready = threading.Event()
    audio_ready = threading.Event()
    
    register_face_event = threading.Event()
    shared_state = {
        "register_frame": None,
        "gaze_model": None,
        "frame_count": 0,
        "logs": []
    }

    # --- LUỒNG AI THỊ GIÁC ---
    def vision_ai_thread():
        print("[INFO] Đang khởi động luồng AI Thị giác...")
        yolo_model = ObjectDetector(model_path="weights/yolov8_finetuned.pt")
        face_model = FaceVerifier(db_path="data/student_faces/")
        gaze_model = PoseGazeDetector()
        
        shared_state["gaze_model"] = gaze_model
        vision_ready.set()
        print("[INFO] ✅ AI Thị giác đã nạp xong!")

        while True:
            # 1. Xử lý chụp ảnh đăng ký
            if register_face_event.is_set():
                frame_to_save = shared_state["register_frame"]
                if frame_to_save is not None:
                    try:
                        target_bbox = None
                        h, w = frame_to_save.shape[:2]
                        
                        # Bước 1: Trích xuất TẤT CẢ khuôn mặt có trong ảnh
                        faces = face_model._detect_faces(frame_to_save)
                        
                        if faces:
                            # Bước 2: Sắp xếp khuôn mặt theo diện tích từ TO đến NHỎ
                            faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                            
                            # Bước 3: Dùng kỹ thuật "Che phủ" (Masking) để test từng mặt
                            for i, target_face in enumerate(faces):
                                masked_frame = frame_to_save.copy()
                                
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
                                verify_result = face_model.verify_face(masked_frame, time.time())
                                
                                if verify_result and verify_result.get("status") == "alert":
                                    target_bbox = target_face.bbox
                                    break  # Chốt người lạ to nhất và thoát vòng lặp
                                    
                            # Bước 4: Phương án dự phòng (Nếu DB trống hoặc toàn người quen)
                            if target_bbox is None:
                                target_bbox = faces[0].bbox
                                
                            # Bước 5: Nới rộng khung chữ nhật, vẽ và lưu ảnh gốc
                            x1, y1, x2, y2 = map(int, target_bbox)
                            margin = 40
                            x1_ext = max(0, x1 - margin)
                            y1_ext = max(0, y1 - margin)
                            x2_ext = min(w, x2 + margin)
                            y2_ext = min(h, y2 + margin)
                            
                            save_frame = frame_to_save.copy()
                            cv2.rectangle(save_frame, (x1_ext, y1_ext), (x2_ext, y2_ext), (0, 255, 0), 2)
                            cv2.putText(save_frame, "REGISTERED", (x1_ext, y1_ext - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            file_name = f"data/student_faces/student_{int(time.time())}.jpg"
                            cv2.imwrite(file_name, save_frame)
                            
                            # Nạp lại Face Vector
                            face_model._load_database()
                            
                            result_q.put({"module": "system", "status": "info", "message": f"📸 Đã đăng ký khuôn mặt: {file_name}"})
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
            
            result_yolo = yolo_model.process_frame(frame, timestamp)
            if result_yolo and not result_q.full(): result_q.put(result_yolo)
                
            result_face = face_model.verify_face(frame, timestamp)
            if result_face and not result_q.full(): result_q.put(result_face)
                
            result_gaze = gaze_model.process_frame(frame, timestamp)
            if result_gaze and not result_q.full(): result_q.put(result_gaze)

    # --- LUỒNG AI ÂM THANH (Dùng RealtimeAudioWorker Windows) ---
    def audio_ai_thread():
        print("[INFO] Đang khởi động luồng AI Âm thanh thực tế...")
        try:
            audio_worker = RealtimeAudioWorker()
            
            # Khôi phục lại mưu kế "Bọc lót" hoàn hảo, KHÔNG cần đụng vào file của cộng sự
            original_pipeline = audio_worker.pipeline.process_audio
            def hooked_process_audio(*args, **kwargs):
                res = original_pipeline(*args, **kwargs)
                if res and not result_q.full():
                    res["module"] = "audio_phobert_pipeline"
                    result_q.put(res)
                return res
                
            audio_worker.pipeline.process_audio = hooked_process_audio

            audio_ready.set()
            print("[INFO] ✅ AI Âm thanh đã nạp xong!")
            audio_worker.start()
        except Exception as e:
            print(f"[LỖI LUỒNG ÂM THANH] {e}")
            audio_ready.set()

    # Khởi chạy các luồng
    t_vision = threading.Thread(target=vision_ai_thread, daemon=True)
    t_audio = threading.Thread(target=audio_ai_thread, daemon=True)
    
    t_vision.start()
    t_audio.start()
    
    vision_ready.wait()
    audio_ready.wait()

    return frame_q, result_q, overlays, overlay_lock, register_face_event, shared_state

# Kích hoạt Cache
FRAME_QUEUE, RESULT_QUEUE, ACTIVE_OVERLAYS, OVERLAY_LOCK, REG_EVENT, SHARED_STATE = init_system_resources()
OVERLAY_TTL = 1.5
FPS_SKIP = 8

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
            timestamp = alert.get("timestamp", current_time)
            details = alert.get("details", {})
            
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

    # Cấu hình WebRTC (Tắt Audio của WebRTC vì Audio đã chạy độc lập qua Worker bên dưới)
    webrtc_streamer(
        key="exam-proctor",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640, "max": 640},
                "height": {"ideal": 480, "max": 480},
                "frameRate": {"ideal": 24, "max": 30},
            },
            "audio": False 
        },
        async_processing=True,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    )

with col2:
    st.markdown("### 📜 Lịch Sử Cảnh Báo")
    
    # LỆNH TỰ ĐỘNG LÀM MỚI GIAO DIỆN MỖI 3 GIÂY (3000 mili-giây)
    st_autorefresh(interval=3000, key="auto_refresh_log")
    
    # Lấy Log mới nhất từ Shared State
    log_html = "<br>".join(SHARED_STATE["logs"]) if SHARED_STATE["logs"] else "Hệ thống đang chạy..."
    st.markdown(f"<div style='height: 500px; overflow-y: auto; background-color: #f0f2f6; padding: 10px; border-radius: 10px;'>{log_html}</div>", unsafe_allow_html=True)
    
    # Nút Refresh Log bằng tay (vì WebRTC chạy ngầm, giao diện cần trigger để hiện Log mới)
    # st.button("🔄 Làm mới Log", use_container_width=True)
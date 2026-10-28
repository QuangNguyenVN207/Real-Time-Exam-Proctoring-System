import os
import sys
import logging
import warnings

# ==========================================
# KHỐI LỆNH TẮT TRIỆT ĐỂ LOG RÁC (C++ & TENSORFLOW)
# ==========================================
try:
    # Mở file devnull (hoặc NUL trên Windows)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    # Chuyển hướng File Descriptor 2 (stderr của C/C++) vào devnull
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
except Exception:
    pass
# Tắt cảnh báo NNPACK của PyTorch
os.environ["TORCH_SHOW_CPP_WARNINGS"] = "0"
logging.getLogger("torch.nnpack").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"         # Tắt toàn bộ log C++ của TensorFlow / TFLite
os.environ["GLOG_minloglevel"] = "3"             # Tắt toàn bộ log C++ của Google Logging / MediaPipe
os.environ["ORT_LOGGING_LEVEL"] = "4"            # Tắt log của ONNX Runtime (4 = FATAL)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
# 2. Điều hướng toàn bộ STDERR vào hư vô (Nul) trong lúc khởi tạo MediaPipe

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

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

import cv2
import threading
import time
import queue
from typing import Optional
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

# ---> ENTRY POINT DUY NHẤT: thiết bị/model chọn theo GPU đã resolve, chỉ worker micro là theo OS <---
# Hai entry point cũ tách theo OS, nên chạy WSL + NVIDIA vẫn rơi vào nhánh Ubuntu (paper + tracking
# bị ép chạy CPU). Nay chọn theo `select_device().kind`; `PROCTOR_DEVICE=cuda|intel|cpu` vẫn ép được.
IS_WINDOWS = sys.platform.startswith("win")


@st.cache_resource
def _openvino_devices():
    """Liệt kê thiết bị OpenVINO (chẩn đoán iGPU). Trả [] nếu máy không cài OpenVINO."""
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        print(f"[INFO] OpenVINO đã sẵn sàng trên các thiết bị: {devices}")
        return devices
    except Exception as e:
        print(f"[WARNING] Không khởi tạo được OpenVINO: {e}")
        return []


# (đường dẫn model vật cấm, device cho ObjectDetector, device cho paper + person tracking)
_DEVICE_PROFILE = {
    "cuda": ("weights/best (1).pt", None, "auto"),
    "intel": ("weights/best_openvino_model", "GPU", "cpu"),
    "cpu": ("weights/best (1).pt", None, "cpu"),
}

from backend.core.device import select_device
DEVICE = select_device()
VISION_MODEL_PATH, VISION_DEVICE, PERSON_DEVICE = _DEVICE_PROFILE.get(DEVICE.kind, _DEVICE_PROFILE["cpu"])
OPENVINO_DEVICES = _openvino_devices() if DEVICE.kind == "intel" else []
# print(f"[INFO] Thiết bị suy luận: {DEVICE.kind} ({DEVICE.name}) -> vật cấm={VISION_MODEL_PATH}"
    #   f"/{VISION_DEVICE}, paper+tracking={PERSON_DEVICE}")

# Import các module AI của bạn
from pathlib import Path
from backend.ai_services.pose_gaze import paper_pipeline
from backend.ai_services.object_detect.paper_count_pipeline import PaperCountPipeline
from backend.ai_services.pose_gaze.tracking.detectors import UltralyticsPersonDetector
from backend.ai_services.object_detect.object_detect import ObjectDetector, ObjectDetectModule
from backend.ai_services.pose_gaze.pose_gaze_service import PoseGazeDetector
from backend.ai_services.face_verify.face_verify import FaceVerifier
# Micro: PyAudio trên Linux, sounddevice trên Windows (hai worker riêng biệt)
if IS_WINDOWS:
    from backend.ai_services.whisper.realtime_audio_worker import RealtimeAudioWorker
else:
    from backend.ai_services.whisper.realtime_audio_ubuntu import RealtimeAudioWorker
# Logic per-module đã tách ra package dùng chung (realtime + offline chạy ra cảnh báo y hệt nhau)
from backend.ai_services.pipeline import audio_scan, events, face_scan, gaze_scan, object_scan, paper_scan
from backend.ai_services.pipeline import overlays as overlay_draw
from offline_job import OfflineJob

# ==========================================
# 1. KHỞI TẠO TÀI NGUYÊN (CHẠY 1 LẦN DUY NHẤT)
# ==========================================
@st.cache_resource
def init_system_resources():
    print("[INFO] Khởi tạo trạm trung chuyển dữ liệu...")
    frame_q = queue.Queue(maxsize=2) 
    paper_q = queue.Queue(maxsize=2)
    gaze_q = queue.Queue(maxsize=2)
    result_q = queue.Queue(maxsize=100)
    
    overlays = []
    overlay_lock = threading.Lock()
    
    vision_ready = threading.Event()
    paper_ready = threading.Event()
    audio_ready = threading.Event()
    gaze_ready = threading.Event()
    
    register_face_event = threading.Event()
    shared_state = {
        "register_frame": None,
        "gaze_model": None,
        "holistic_model": None,       # <--- THÊM MỚI
        "live_classifier": None,     # <--- THÊM MỚI
        "tracking_manager": None,    # <--- THÊM MỚI
        # Bổ sung vào dictionary SHARED_STATE / shared_state hiện có
        "action_arm_paper": False,       # Thay cho phím A
        "action_disarm_paper": False,    # Thay cho phím D
        "paper_status_msg": "SETUP - Đang chờ cố định baseline",
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
                model_path=VISION_MODEL_PATH,
                device=VISION_DEVICE,
                confidence_threshold=0.7
            )
            face_model = FaceVerifier(db_path="data/student_faces/")
        except Exception as e:
            print(f"[LỖI KHỞI TẠO AI THỊ GIÁC]: {e}")
            vision_ready.set()
            return
        
        # ---> BÍ QUYẾT ĐỒNG BỘ: Tạo hàm Masking chung cho cả Camera và Đăng ký <---
        def get_largest_stranger_with_masking(img, ts):
            return face_scan.scan_stranger_with_masking(face_model, img, ts)
        # -------------------------------------------------------------------------
        
        vision_ready.set()
        print("[INFO] ✅ AI Thị giác (Nặng) đã nạp xong!")

        while True:
            # 1. Xử lý chụp ảnh đăng ký (Đồng bộ Masking)
            if register_face_event.is_set():
                frame_to_save = shared_state["register_frame"]
                if frame_to_save is not None:
                    try:
                        # Gọi hàm Masking chung
                        _, target_bbox = get_largest_stranger_with_masking(frame_to_save, time.time())
                                
                        if target_bbox is not None:
                            crop = face_scan.crop_for_registration(frame_to_save, target_bbox)

                            # 1. BẢN CHO AI: Chỉ cắt khuôn mặt lưu vào thư mục chuẩn
                            if crop["face_crop"].size > 0:
                                ai_file = f"data/student_faces/student_{int(time.time())}.jpg"
                                cv2.imwrite(ai_file, crop["face_crop"])
                            
                            # 2. BẢN TOÀN CẢNH CHO BẠN: Vẽ khung xanh và lưu ra thư mục riêng
                            import os
                            os.makedirs("data/registered_logs", exist_ok=True)
                            human_file = f"data/registered_logs/full_student_{int(time.time())}.jpg"
                            cv2.imwrite(human_file, crop["full_view"])
                            
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

            # Vật cấm (đã lọc nhãn giấy trong object_scan) + người lạ (masking) đi qua queue chung
            alert_object = object_scan.scan_objects(yolo_model, frame, timestamp)
            if alert_object is not None:
                if not result_q.full():
                    result_q.put(alert_object)
                else:
                    print("[DEBUG YOLO] CẢNH BÁO: RESULT_QUEUE đã đầy, bị rớt frame cảnh báo!")

            result_face, _ = get_largest_stranger_with_masking(frame, timestamp)
            if result_face and not result_q.full(): 
                result_q.put(result_face)

    # ==========================================
    # LUỒNG AI ĐẾM GIẤY (ĐẶC TRỊ RIÊNG CHO PAPER PIPELINE)
    # ==========================================
    def paper_vision_thread():
        print("[INFO] Đang khởi động luồng AI Đếm Giấy (Paper Pipeline) độc lập...")
        try:
            person_detector = UltralyticsPersonDetector(
                model_path=Path("weights/yolov8n.pt"),
                confidence_threshold=0.55, 
                device=PERSON_DEVICE
            )
            paper_object_detector = ObjectDetectModule(device=PERSON_DEVICE)
            
            session_id = f"paper_session_{int(time.time())}"
            paper_pipeline = PaperCountPipeline(
                person_detector=person_detector,
                object_detector=paper_object_detector,
                storage_root=Path("test_data_tracking"),
                max_people=2
            )
            
            # Vô hiệu hóa smartphone fallback bên trong pipeline giấy
            if hasattr(paper_pipeline, "object_detector") and paper_pipeline.object_detector is not None:
                if hasattr(paper_pipeline.object_detector, "_smartphone_fallback_enabled"):
                    paper_pipeline.object_detector._smartphone_fallback_enabled = False
                if hasattr(paper_pipeline.object_detector, "_auxiliary_class_ids"):
                    paper_pipeline.object_detector._auxiliary_class_ids = []
            if hasattr(paper_pipeline, "enable_smartphone_detection"):
                paper_pipeline.enable_smartphone_detection = False

            paper_pipeline.create_session(session_id, restore_existing=True)
        except Exception as e:
            print(f"[LỖI KHỞI TẠO PAPER PIPELINE]: {e}")
            paper_ready.set()
            return

        paper_ready.set()
        print("[INFO] ✅ AI Đếm Giấy đã sẵn sàng trên luồng riêng!")

        while True:
            data = paper_q.get()
            if data is None: break
            frame, timestamp = data

            # 1. Xử lý nút bấm ARM / DISARM
            try:
                if shared_state.get("action_arm_paper"):
                    shared_state["paper_status_msg"] = paper_scan.arm_baseline(
                        paper_pipeline, session_id
                    )["status_msg"]
                    shared_state["action_arm_paper"] = False
                    
                if shared_state.get("action_disarm_paper"):
                    session_id = f"paper_session_{int(time.time())}"
                    paper_pipeline.create_session(session_id, restore_existing=False)
                    shared_state["paper_status_msg"] = "DISARMED! Đã quay lại chế độ SETUP (Baseline = 0)."
                    shared_state["action_disarm_paper"] = False
            except Exception as ex:
                print(f"[LỖI ĐIỀU KHIỂN PIPELINE GIẤY]: {ex}")
                shared_state["action_arm_paper"] = False
                shared_state["action_disarm_paper"] = False

            # 2. Xử lý frame đếm giấy
            try:
                result_paper = paper_pipeline.process_frame(
                    frame, 
                    session_id=session_id, 
                    frame_id=shared_state["frame_count"], 
                    timestamp_ms=int(timestamp * 1000)
                )
            except Exception as ex:
                print(f"[LỖI XỬ LÝ FRAME GIẤY]: {ex}")
                result_paper = None

            # 3. Đẩy kết quả giấy & cảnh báo vào RESULT_QUEUE chung
            alert_paper = paper_scan.build_alert(result_paper, timestamp)
            if alert_paper is not None and not result_q.full():
                if alert_paper["alerts"]:
                    print(f"[DEBUG PAPER] 🚨 Kích hoạt báo động! Danh sách lỗi: {alert_paper['alerts']} | Số lượng tờ giấy trên bàn: {len(alert_paper['papers'])}")
                result_q.put(alert_paper)

    # LUỒNG 2: Xử lý nhẹ (MediaPipe PoseGaze) - Chạy siêu tốc
    def fast_gaze_thread():
        print("[INFO] Đang khởi động Fast Gaze Thread (tích hợp Holistic & Action Classifier)...")
        
        # 1. Khởi tạo PoseGaze gốc
        gaze_model = PoseGazeDetector()
        shared_state["gaze_model"] = gaze_model

        # 2+3. Tracking + Holistic + Causal Classifier (c2 Exchange, c3 Looking, suspicious)
        stack = gaze_scan.build_stack(device=PERSON_DEVICE, live_pair=["student_01:student_02"])
        shared_state["tracking_manager"] = stack.manager
        shared_state["holistic_model"] = stack.holistic
        shared_state["live_classifier"] = stack.live_classifier

        gaze_ready.set()
        print("[INFO] ✅ Fast Gaze Thread (Nâng cao) đã nạp xong!")
        # --- BỘ NHỚ THEO DÕI TRẠNG THÁI HÀNH VI CỦA TỪNG HỌC SINH ---
        active_violations = {} # {actor_id: pred_class}

        while True:
            data = gaze_q.get()
            if data is None: break
            frame, timestamp = data
            
            # --- Step A: Chạy PoseGaze cơ bản ---
            result_gaze = gaze_model.process_frame(frame, timestamp)
            if result_gaze and not result_q.full():
                result_q.put(result_gaze)

            # --- Step B: Chạy Tracking & Holistic & Causal Classifier (Exchange, Looking) ---
            for c2_c3_alert in gaze_scan.scan_frame(stack, frame, timestamp):
                if not result_q.full():
                    result_q.put(c2_c3_alert)

    # --- LUỒNG AI ÂM THANH (Dùng PyAudio độc lập) ---
    def audio_ai_thread():
        print("[INFO] Đang khởi động luồng AI Âm thanh thực tế...")
        try:
            audio_worker = RealtimeAudioWorker()
            print("[DEBUG AUDIO] ✅ Đã khởi tạo RealtimeAudioWorker thành công!")

            original_pipeline = audio_worker.pipeline.process_audio
            def hooked_process_audio(*args, **kwargs):
                # Kiểm tra các tham số đầu vào (thường là mảng numpy hoặc bytes chứa audio chunk)
                if args:
                    raw_audio = args[0]
                    if isinstance(raw_audio, np.ndarray) and raw_audio.size > 0:
                        # Tính độ lớn sóng âm (RMS Volume)
                        rms_volume = np.sqrt(np.mean(raw_audio**2))
                        # print(f"[DEBUG AUDIO] 🔊 Biên độ âm thanh (RMS từ Mic): {rms_volume:.5f}")
                        if rms_volume < 0.001:
                            print("[DEBUG AUDIO] ⚠️ CẢNH BÁO: Mic quá nhỏ hoặc đang bị mute/chọn nhầm thiết bị!")

                # print("[DEBUG AUDIO] 🎙️ VAD đã cắt được câu! Đang đẩy vào AudioPipeline...")

                res = original_pipeline(*args, **kwargs)

                alert_audio = audio_scan.to_alert(res)
                if alert_audio is not None:
                    if not result_q.full():
                        result_q.put(alert_audio)
                    else:
                        print("[DEBUG AUDIO] ⚠️ RESULT_QUEUE đã đầy, rớt gói tin âm thanh!")
                    
                return res
                
            audio_worker.pipeline.process_audio = hooked_process_audio
            audio_ready.set()
            print("[INFO] ✅ Audio Thread đã nạp xong và chuẩn bị lắng nghe micro!")
            audio_worker.start()
        except Exception as e:
            print(f"[LỖI LUỒNG ÂM THANH] Khởi tạo thất bại: {e}")
            import traceback
            traceback.print_exc()
            audio_ready.set()

    # Khởi chạy các luồng
    t_vision = threading.Thread(target=heavy_vision_thread, daemon=True)
    t_audio = threading.Thread(target=audio_ai_thread, daemon=True)
    t_gaze = threading.Thread(target=fast_gaze_thread, daemon=True)
    t_paper = threading.Thread(target=paper_vision_thread, daemon=True)

    t_vision.start()
    t_audio.start()
    t_gaze.start()
    t_paper.start()
    
    vision_ready.wait()
    audio_ready.wait()
    gaze_ready.wait()
    paper_ready.wait()

    return frame_q, paper_q, gaze_q, result_q, overlays, overlay_lock, register_face_event, shared_state

# Kích hoạt Cache
FRAME_QUEUE, PAPER_QUEUE, GAZE_QUEUE, RESULT_QUEUE, ACTIVE_OVERLAYS, OVERLAY_LOCK, REG_EVENT, SHARED_STATE = init_system_resources()
OVERLAY_TTL = 1.5
FPS_SKIP = 1

# ==========================================
# 2. HÀM VẼ GIAO DIỆN (ĐƯỢC GỌI TRONG WEBRTC)
# ==========================================
def draw_warning_overlays(frame):
    gaze_ai = SHARED_STATE.get("gaze_model")
    with OVERLAY_LOCK:
        current_time = time.time()
        events.expire_overlays(ACTIVE_OVERLAYS, current_time, OVERLAY_TTL)
        snapshot = list(ACTIVE_OVERLAYS)
    return overlay_draw.draw(frame, snapshot, gaze_model=gaze_ai)

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
            alert['display_timestamp'] = current_time

            log_msg = events.to_log_line(alert, now=current_time)

            if module_name == "system":
                if log_msg:
                    SHARED_STATE["logs"].insert(0, log_msg)
                continue

            with OVERLAY_LOCK:
                events.prune_overlays(ACTIVE_OVERLAYS, alert)

            if log_msg:
                SHARED_STATE["logs"].insert(0, log_msg)
                if len(SHARED_STATE["logs"]) > 20:
                    SHARED_STATE["logs"].pop()

        except queue.Empty:
            break

    # 3. Đẩy ảnh mới vào Queue cho AI xử lý (Frame Skipping)

    # Bơm ảnh cho luồng Skeleton (Tốc độ cao: Lấy 1 ảnh mỗi 2 frame ~ 10 FPS)
    if SHARED_STATE["frame_count"] % 1 == 0:
        if GAZE_QUEUE.full():
            try: GAZE_QUEUE.get_nowait()
            except queue.Empty: pass
        GAZE_QUEUE.put((img.copy(), current_time))
    # Bơm ảnh cho luồng AI Đếm Giấy (Chạy ngầm độc lập - Lấy 1 ảnh mỗi 2 frame)
    if SHARED_STATE["frame_count"] % 5 == 0:
        if PAPER_QUEUE.full():
            try: PAPER_QUEUE.get_nowait()
            except queue.Empty: pass
        PAPER_QUEUE.put((img.copy(), current_time))
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
    st.markdown("### 📷 Camera Giám Sát Real-time")
    
    # # Nút bấm đăng ký khuôn mặt
    # if st.button("📸 Đăng ký khuôn mặt", use_container_width=True):
    #     SHARED_STATE["save_face_flag"] = True
    # st.markdown("#### 📄 Cấu hình Giấy thi & Cheatsheet")
    # btn_col1, btn_col2 = st.columns(2)
    # with btn_col1:
    #     if st.button("🔒 Cố định Baseline (A)", use_container_width=True):
    #         SHARED_STATE["action_arm_paper"] = True
    # with btn_col2:
    #     if st.button("🔄 Reset Setup (D)", use_container_width=True):
    #         SHARED_STATE["action_disarm_paper"] = True
            
    # # Hiển thị trạng thái đếm giấy hiện tại
    # st.info(SHARED_STATE.get("paper_status_msg", "Trạng thái: SETUP - Đang chờ cố định baseline"))
    # Cấu hình WebRTC (Tắt Audio của WebRTC vì Audio đã chạy độc lập qua PyAudio)
    webrtc_ctx = webrtc_streamer(
        key="exam-proctor",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": True,
            # "video": {
            #     "width": {"min": 1280, "ideal": 1920, "max": 1920},
            #     "height": {"min": 720, "ideal": 1080, "max": 1080},
            #     "frameRate": {"ideal": 30, "max": 30}
            # },
            "audio": False # Chặn WebRTC thu âm để tránh đụng độ với luồng PyAudio
        },
        async_processing=True,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    )

with col2:
    # Nút bấm đăng ký khuôn mặt
    if st.button("📸 Đăng ký khuôn mặt", use_container_width=True):
        SHARED_STATE["save_face_flag"] = True
    st.markdown("#### 📄 Cấu hình Giấy thi & Cheatsheet")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🔒 Cố định Baseline (A)", use_container_width=True):
            SHARED_STATE["action_arm_paper"] = True
    with btn_col2:
        if st.button("🔄 Reset Setup (D)", use_container_width=True):
            SHARED_STATE["action_disarm_paper"] = True
            
    # Hiển thị trạng thái đếm giấy hiện tại
    st.info(SHARED_STATE.get("paper_status_msg", "Trạng thái: SETUP - Đang chờ cố định baseline"))

    st.markdown("### 📜 Lịch Sử Cảnh Báo")

    # ---> LỆNH TỰ ĐỘNG LÀM MỚI GIAO DIỆN MỖI 3 GIÂY (3000 mili-giây) <---
    st_autorefresh(interval=3000, key="auto_refresh_log")

    # Lấy Log mới nhất từ Shared State
    log_html = "<br>".join(SHARED_STATE["logs"]) if SHARED_STATE["logs"] else "Hệ thống đang chạy..."
    st.markdown(f"<div style='height: 500px; overflow-y: auto; background-color: #f0f2f6; padding: 10px; border-radius: 10px;'>{log_html}</div>", unsafe_allow_html=True)
    
    # Nút Refresh Log bằng tay (vì WebRTC chạy ngầm, giao diện cần trigger để hiện Log mới)
    # st.button("🔄 Làm mới Log", use_container_width=True)

# ==========================================
# 5. CHẠY OFFLINE TRÊN MỘT FILE VIDEO (không cần camera/micro)
# ==========================================
OFFLINE_MODULES = ["object", "face", "paper", "pose", "action", "audio"]
OFFLINE_UPLOAD_DIR = Path("offline_out/_uploads")
# Mỗi lần Streamlit rerun là giành GIL với luồng suy luận: đo trên iGPU Arc, refresh 1.2s làm mỗi frame
# lâu gấp ~3 lần (face 0.35s -> 1.20s); 4s thì trở lại bằng CLI chạy một mình.
OFFLINE_REFRESH_MS = 4000


def _offline_input_path(uploaded, typed_path: str) -> Optional[Path]:
    """`run_offline` cần đường dẫn thật: file upload được ghi ra đĩa, đường dẫn thì dùng thẳng."""
    if uploaded is not None:
        OFFLINE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = OFFLINE_UPLOAD_DIR / f"{uploaded.size}_{uploaded.name}"
        if not target.exists():
            with open(target, "wb") as fh:
                fh.write(uploaded.getbuffer())
        return target
    typed_path = typed_path.strip()
    return Path(typed_path).expanduser() if typed_path else None


def _render_offline_result(snap: dict) -> None:
    """Video phát trực tiếp trong trình duyệt + tóm tắt theo từng actor + nút tải kết quả."""
    if snap["playback"]:
        st.markdown("#### 📺 Video kết quả (H.264, phát ngay tại đây)")
        st.video(snap["playback"])
    else:
        st.warning("Không tạo được bản H.264 (thiếu ffmpeg?) — tải `annotated.mp4` bên dưới và mở bằng VLC.")

    report = snap["report"] or {}
    by_module = report.get("alerts_by_module", {})
    actors = report.get("actors", {})
    m1, m2, m3 = st.columns(3)
    m1.metric("Frame đã xử lý", report.get("frames_processed", snap["frame_id"]))
    m2.metric("Sự kiện ghi nhận", sum(by_module.values()))
    m3.metric("Thời gian xử lý", f"{report.get('wall_time_s', snap['elapsed'])}s")

    if by_module:
        st.markdown("**Theo module:** " + " · ".join(f"`{k}` {v}" for k, v in sorted(by_module.items())))
    if actors:
        st.markdown("**Theo actor** (điểm max kể cả khi kết luận `c5`):")
        for actor_id, info in sorted(actors.items()):
            scores = ", ".join(f"{k.replace('_score', '')}={v}" for k, v in (info.get("max_score") or {}).items())
            st.text(f"{actor_id}: class={info.get('final_class') or info.get('predicted_class')} "
                    f"frames={info.get('frames')} alerts={info.get('alerts')} max({scores})")

    st.markdown("**Tải kết quả**")
    d1, d2, d3, d4 = st.columns(4)
    out_dir = Path(snap["out_dir"])
    for col, name in ((d1, "playback_h264.mp4"), (d2, "log.txt"), (d3, "report.json"), (d4, "events.jsonl")):
        path = out_dir / name
        if path.exists():
            with open(path, "rb") as fh:
                col.download_button(f"⬇ {name}", fh.read(), file_name=name, use_container_width=True)


with st.expander("🎬 Chạy offline trên file video — xem kết quả ngay trong trình duyệt", expanded=False):
    st.caption("Dùng đúng bộ phát hiện của camera (`backend/ai_services/pipeline/`): vừa chạy vừa xem khung hình "
               "đã burn cảnh báo, xong thì nhận video phát trực tiếp + `events.jsonl` / `report.json`. "
               "Quá trình chạy nằm ở luồng nền nên giao diện không bị đơ.")

    job = st.session_state.get("offline_job")
    busy = bool(job and job.running)
    camera_on = bool(webrtc_ctx is not None and webrtc_ctx.state and webrtc_ctx.state.playing)

    if camera_on:
        st.error("⛔ Camera realtime đang phát — bấm STOP trước khi chạy offline. "
                 "Luồng offline nạp một bộ model RIÊNG nên chạy cả hai sẽ nhân đôi RAM/iGPU.")

    src_mode = st.radio("Nguồn file", ["Đường dẫn trên máy", "Tải file lên"],
                        horizontal=True, key="off_src", disabled=busy)
    if src_mode == "Tải file lên":
        uploaded = st.file_uploader("Video (.mp4 / .MOV / .mkv / .avi) — Streamlit chặn ở 200MB/lần",
                                    type=["mp4", "mov", "mkv", "avi", "m4v"], key="off_file")
        typed_path = ""
    else:
        uploaded = None
        typed_path = st.text_input("Đường dẫn file trên máy chạy Streamlit",
                                   value="data/IMG_7817.MOV", key="off_path")
    video_path = _offline_input_path(uploaded, typed_path)

    c1, c2, c3 = st.columns(3)
    with c1:
        max_seconds = st.number_input("Chỉ chạy N giây đầu (0 = cả file)", 0.0, 3600.0, 20.0, 5.0,
                                 key="off_maxs")
        setup_seconds = st.number_input("ARM baseline giấy sau (giây)", 0.0, 300.0, 5.0, 1.0, key="off_setup")
    with c2:
        stride = st.slider("Stride nặng (YOLO/mặt/pose/action) — lớn = nhanh", 1, 8, 3, key="off_stride")
        stride_paper = st.slider("Stride giấy", 1, 10, 5, key="off_stride_paper")
    with c3:
        modules = st.multiselect("Module bật", OFFLINE_MODULES, default=OFFLINE_MODULES, key="off_modules")
        ttl = st.number_input("TTL overlay (giây video)", 0.5, 10.0, 1.5, 0.5, key="off_ttl")

    if video_path is None:
        start_enabled = False
        st.info("Chọn file hoặc nhập đường dẫn để bắt đầu.")
    elif not video_path.exists():
        start_enabled = False
        st.error(f"Không thấy file: {video_path}")
    elif not modules:
        start_enabled = False
        st.warning("Chọn ít nhất một module phát hiện.")
    else:
        start_enabled = True

    if st.button("▶ Bắt đầu chạy offline", type="primary", key="off_start",
                 disabled=(busy or camera_on or not start_enabled)):
        out_dir = Path("offline_out") / f"ui_{video_path.stem}_{int(time.time())}"
        argv = [str(video_path), "--output", str(out_dir),
                "--max-seconds", str(max_seconds), "--setup-seconds", str(setup_seconds), "--ttl", str(ttl),
                "--stride-heavy", str(stride), "--stride-paper", str(stride_paper),
                "--stride-pose", str(stride), "--stride-action", str(stride)]
        argv += ["--modules", ",".join(modules)]
        st.session_state.offline_job = OfflineJob(argv, out_dir, video_path).start()
        st.rerun()

    job = st.session_state.get("offline_job")
    if job is not None:
        snap = job.snapshot()
        if snap["status"] == "running":
            text = f"Frame {snap['frame_id']}/{snap['total_frames'] or '?'} · {snap['alerts']} cảnh báo"
            if snap["detail"]:
                text += f" · {snap['detail']}"
            st.progress(snap["fraction"], text=text)
            if snap["preview"]:
                st.image(snap["preview"], caption="Khung hình offline mới nhất (đã burn cảnh báo)",
                         use_container_width=True)
            if snap["logs"]:
                st.markdown("**Cảnh báo mới nhất**")
                for line in snap["logs"]:
                    st.text(overlay_draw.plain_text(line))
            st_autorefresh(interval=OFFLINE_REFRESH_MS, key="off_refresh")
        elif snap["status"] == "error":
            st.error(f"Lỗi khi chạy offline: {snap['error']}")
            st.caption(f"Log chi tiết trong {snap['out_dir']}")
        elif snap["status"] == "done":
            _render_offline_result(snap)
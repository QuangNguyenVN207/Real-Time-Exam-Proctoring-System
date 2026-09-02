import os
import sys
import logging
import warnings

# ==========================================
# KHỐI LỆNH TẮT TRIỆT ĐỂ LOG RÁC (C++ & TENSORFLOW)
# ==========================================
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
from pathlib import Path
from backend.ai_services.pose_gaze import paper_pipeline
from backend.ai_services.object_detect.paper_count_pipeline import PaperCountPipeline
from backend.ai_services.pose_gaze.tracking.detectors import UltralyticsPersonDetector
from backend.ai_services.object_detect.object_detect import ObjectDetector, ObjectDetectModule
from backend.ai_services.pose_gaze.pose_gaze_service import PoseGazeDetector
from backend.ai_services.face_verify.face_verify import FaceVerifier
from backend.ai_services.whisper.realtime_audio_ubuntu import RealtimeAudioWorker
from backend.ai_services.pose_gaze.pose_gaze.holistic.landmark import HolisticLandmarkExtractor
from backend.ai_services.pose_gaze.pose_gaze.tracking.manager import TrackingManager
from backend.ai_services.pose_gaze.pose_gaze.tracking.webcam import PersonTrackingModule, PersonTrackingConfig
from backend.ai_services.pose_gaze.pose_gaze.holistic.test_media.test_media import create_live_classifier
from backend.ai_services.pose_gaze.pose_gaze.settings import (
    DEFAULT_HOLISTIC_CONFIDENCE,
    DEFAULT_HOLISTIC_SOFT_CONFIDENCE,
    DEFAULT_MAX_MISSED_FRAMES,
    DEFAULT_MIN_IOU,
    DEFAULT_PERSON_CONFIDENCE,
    PROJECT_ROOT,
)

DEFAULT_ACTION_ARTIFACTS = {
    "extended": PROJECT_ROOT / "stage6_bundle_exact" / "causal_8fps_stage6_mixed_084699_final_20260827",
}

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
                model_path="weights/best_openvino_model", 
                device="GPU",
                confidence_threshold=0.7
            )
            face_model = FaceVerifier(db_path="data/student_faces/")
        except Exception as e:
            print(f"[LỖI KHỞI TẠO AI THỊ GIÁC]: {e}")
            vision_ready.set()
            return
        
        # ---> BÍ QUYẾT ĐỒNG BỘ: Tạo hàm Masking chung cho cả Camera và Đăng ký <---
        def get_largest_stranger_with_masking(img, ts):
            faces = face_model._detect_faces(img)
            # print(f"[DEBUG FACE] Số lượng khuôn mặt phát hiện trong frame: {len(faces) if faces else 0}")
            if not faces:
                # print("[DEBUG FACE] Không tìm thấy khuôn mặt nào trong frame.")
                return None, None
                
            # Sắp xếp khuôn mặt theo diện tích từ TO đến NHỎ
            faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            h, w = img.shape[:2]
            
            # Kiểm tra từng khuôn mặt bằng cách cắt trực tiếp (Crop)
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
                    # print(f"[DEBUG FACE] 🚨 CẢNH BÁO: Phát hiện người lạ (alert) thực sự tại bbox: {target_face.bbox}")
                    # Ghi đè lại tọa độ gốc để khung vàng vẽ đúng vị trí trên video gốc
                    verify_result["details"]["unauthorized_bbox"] = [int(x) for x in target_face.bbox]
                    return verify_result, target_face.bbox
                else:
                    # print(f"[DEBUG FACE] ✅ Xác thực thành công: Là người quen (Không phải người lạ).")
                    pass
                    
            # Nếu tất cả các mặt đều là người quen
            # print("[DEBUG FACE] Tất cả các mặt đều được nhận diện là người quen.")
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
                # 🛑 LỌC BỎ CÁC NHÃN LIÊN QUAN ĐẾN GIẤY / TÀI LIỆU CỦA MODEL CŨ
                filtered_detections = []
                for det in detections:
                    label = str(det.get("label", "")).lower()
                    # 1. Chặn các nhãn giấy/tài liệu nếu lọt vào đây
                    if label in ["paper", "document", "cheatsheet", "sheet", "tai_lieu", "giay", "book"]:
                        continue
                  
                    filtered_detections.append(det)
                
                result_yolo["detections"] = filtered_detections
        
                # [DEBUG 2] Kiểm tra số lượng vật thể bắt được
                if len(filtered_detections) > 0:
                    # print(f"[DEBUG YOLO] Đã bắt được {len(detections)} vật thể. Cập nhật status thành 'alert'.")
                    result_yolo["status"] = "alert" # BẮT BUỘC: Đánh dấu là cảnh báo để hàm Main xử lý
                    result_yolo["module"] = "object_detect"
                    # Đẩy vào queue nếu chưa đầy
                    if not result_q.full():
                        result_q.put(result_yolo)
                        # print("[DEBUG YOLO] Đã đẩy cảnh báo vật thể vào RESULT_QUEUE thành công.")
                    else:
                        print("[DEBUG YOLO] CẢNH BÁO: RESULT_QUEUE đã đầy, bị rớt frame cảnh báo!")
                else:
                    # Nếu sau khi lọc mà không còn gì, bỏ qua trạng thái alert để không gửi rác lên queue
                    result_yolo = None

            if result_yolo:
                # Ép key "module" để WebRTC nhận diện luồng
                if "module" not in result_yolo:
                    result_yolo["module"] = "object_detect"
                
                # Logic xác định dị thường cho ObjectDetector: có vật thể trong list 'detections'
                has_anomaly = len(result_yolo.get("detections", [])) > 0
                
                if has_anomaly:
                    result_yolo["status"] = "alert"
            
            # ---> ĐỒNG BỘ: Luồng Camera giám sát giờ cũng dùng Masking để quét người lạ <---
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
                device="cpu"
            )
            paper_object_detector = ObjectDetectModule(device="cpu")
            
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
                    state_info = paper_pipeline.arm_paper_monitoring(session_id)
                    baseline_count = state_info.get('baseline_count', 0) if isinstance(state_info, dict) else 0
                    shared_state["paper_status_msg"] = f"ARMED! Baseline: {baseline_count} giấy."
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
            if result_paper and not result_q.full():
                alerts = result_paper.get("alerts", [])
                papers = result_paper.get("papers", [])
                # [DEBUG PAPER] Bắt bệnh lý do cảnh báo lạ
                if alerts:
                    print(f"[DEBUG PAPER] 🚨 Kích hoạt báo động! Danh sách lỗi: {alerts} | Số lượng tờ giấy trên bàn: {len(papers)}")
    
                    # Sửa lại 'label' thành string key và dùng .get() để an toàn tuyệt đối
                    first_alert = alerts[0]
                    if isinstance(first_alert, dict) and first_alert.get('label') == 'smartphone_detected':
                        alerts = []
                        print("[DEBUG PAPER] ⚠️ Cảnh báo: Phát hiện smartphone trên bàn (bị nhầm là giấy).")
        
                    for p in papers:
                        print(f"   -> Giấy #{p.get('observation_index')}: Trạng thái = {p.get('status')}")
                paper_data = {
                    "module": "paper_count",
                    "status": "alert" if result_paper.get("alerts") else "info",
                    "timestamp": timestamp,
                    "alerts": alerts,
                    "papers": papers
                }
                result_q.put(paper_data)

    # LUỒNG 2: Xử lý nhẹ (MediaPipe PoseGaze) - Chạy siêu tốc
    def fast_gaze_thread():
        print("[INFO] Đang khởi động Fast Gaze Thread (tích hợp Holistic & Action Classifier)...")
        
        # 1. Khởi tạo PoseGaze gốc
        gaze_model = PoseGazeDetector()
        shared_state["gaze_model"] = gaze_model
        # 2. Khởi tạo Tracking & Holistic từ test_webcam.py (Dùng session động để tránh trùng lặp)
        session_id = f"gaze_session_{int(time.time())}"
        # 2. Khởi tạo Tracking & Holistic từ test_webcam.py
        tracking_module = PersonTrackingModule(
            PersonTrackingConfig(
                session_id=session_id,
                confidence_threshold=0.5,
                device="cpu",
                max_tracks=2,
            )
        )
        # Thêm dòng này để xử lý an toàn nếu manager giữ lại session cũ
        try:
            tracking_module.manager.create_session(session_id, restore_existing=True)
        except Exception:
            pass
        shared_state["tracking_manager"] = tracking_module.manager
        
        holistic = HolisticLandmarkExtractor(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            soft_landmark_confidence=0.2,
            crop_padding=0.15,
            face_hold_frames=3,
        )
        holistic.__enter__()
        shared_state["holistic_model"] = holistic

        # 3. Khởi tạo Causal Classifier (c2: Exchange, c3: Looking, suspicious_activity)
        live_classifier = None
        try:
            class ArgsMock:
                causal_model_dir = DEFAULT_ACTION_ARTIFACTS["extended"]
                xgboost_model_dir = DEFAULT_ACTION_ARTIFACTS["extended"]
                xgboost_device = "cpu"
                actions = "c2,c3,suspicious_activity"
                live_pair = ["student_01:student_02"]
                student_prefix = "student_"

            load_res = create_live_classifier(ArgsMock(), clip_id=f"webrtc_{session_id}", structured=True)
            live_classifier = load_res.classifier
            shared_state["live_classifier"] = live_classifier
            print("[INFO] ✅ Nạp thành công Causal Action Classifier (Exchange, Looking,...)")
        except Exception as e:
            print(f"[WARNING] Không thể nạp Causal Action Classifier: {e}")

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
            try:
                packet = tracking_module.process_frame(frame)
                
                # Tự động gán student_id nếu chưa có
                for track in packet.tracks:
                    if track.is_present and not track.student_id:
                        packet = tracking_module.manager.assign_student(
                            session_id,
                            track_id=track.track_id,
                            student_id=f"student_{track.track_id:02d}",
                        )
                
                # Trích xuất Holistic Landmarks
                holistic_results = holistic.process_packet(frame, packet)
                labels = {
                    "c2": "EXCHANGE (Trao đổi)", 
                    "c3": "LOOKING (Quay ngó)", 
                    "suspicious_activity": "HOẠT ĐỘNG BẤT THƯỜNG"
                }
                # Cập nhật phân loại hành vi
                if live_classifier is not None:
                    classifications = live_classifier.update(
                        frame_index=packet.frame_id,
                        timestamp_ms=packet.timestamp_ms,
                        results=holistic_results,
                    )
                    current_frame_actors = set(classifications.keys())
                    # Bắn cảnh báo nếu phát hiện hành vi gian lận c2 (Exchange), c3 (Looking), v.v.
                    for actor_id, clf in classifications.items():
                        pred_class = clf.get("predicted_class", "c5")
                        if pred_class in ["c2", "c3", "suspicious_activity"]:
                            
                            action_label = labels.get(pred_class, pred_class.upper())

                            c2_c3_alert = {
                                "module": "pose_gaze",
                                "status": "warning",
                                "timestamp": timestamp,
                                "details": {
                                    "action": f"{actor_id}: {action_label}"
                                }
                            }
                            if not result_q.full():
                                result_q.put(c2_c3_alert)

            except Exception as ex:
                pass  # Bỏ qua lỗi suy luận nhỏ để giữ luồng realtime

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
                # In toàn bộ kết quả thô trả về từ PhoWhisper & PhoBERT
                # print(f"[DEBUG AUDIO] 🧠 Kết quả trả về từ Backend: {res}")

                if res:
                    transcription = res.get("transcription", "").strip()
                    if transcription:
                        if not result_q.full():
                            res["module"] = "audio_phobert_pipeline"
                            result_q.put(res)
                            # print(f"[DEBUG AUDIO] 📤 Đã đẩy kết quả transcription ('{transcription}') vào hàng đợi UI thành công.")
                        else:
                            print("[DEBUG AUDIO] ⚠️ RESULT_QUEUE đã đầy, rớt gói tin âm thanh!")
                    else:
                        # print("[DEBUG AUDIO] ⚠️ VAD cắt nhầm tạp âm (Transcription bị rỗng ''), bỏ qua không đẩy lên UI.")
                        pass
                else:
                    # print("[DEBUG AUDIO] ⚠️ AudioPipeline trả về None.")
                    pass
                    
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
    current_time = time.time()
    
    # 1. Vẽ Skeleton 3D trước (để nó nằm dưới)
    gaze_ai = SHARED_STATE.get("gaze_model")
    if gaze_ai and hasattr(gaze_ai, 'draw_skeleton'):
        gaze_ai.draw_skeleton(frame)
    # 1b. THÊM MỚI: Vẽ Holistic Landmarks nếu có
    holistic_ai = SHARED_STATE.get("holistic_model")
    tracking_mgr = SHARED_STATE.get("tracking_manager")
    if holistic_ai and hasattr(holistic_ai, "draw_results"):
        # Lấy kết quả holistic gần nhất để vẽ lên khung hình
        pass # holistic_ai tự quản lý vẽ khi được gọi
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
                # print(f"[DEBUG DRAW] Đang vẽ khung nhận diện người lạ với bbox: {bbox}")
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                    cv2.putText(frame, "NGUOI LA", (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif module == "pose_gaze":
                action = details.get("action", "VI PHAM TU THE").upper()
                cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (0, 0, 200), -1)
                cv2.putText(frame, f"[CANH BAO TU THE]: {action}", (20, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # =========================================================
            # ---> BẮT ĐẦU CHÈN LOGIC 5: VẼ KHUNG GIẤY Ở ĐÂY <---
            # =========================================================
            elif module == "paper_count" and alert.get("papers"):
                for paper in alert["papers"]:
                    x1, y1, x2, y2 = paper["bbox_xyxy"]
                    status_paper = paper["status"]
                    
                    # Phân biệt Baseline (màu Vàng/Cyan) và Cheatsheet (Màu Đỏ)
                    is_suspicious = (status_paper == "suspicious_new_paper")
                    color = (0, 0, 255) if is_suspicious else (0, 255, 255)
                    
                    # Nhãn text
                    label = "CHEAT SHEET (NEW)" if is_suspicious else f"Paper #{paper.get('observation_index', '?')}"
                    
                    # Vẽ Box và Text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # =========================================================
            # ---> KẾT THÚC LOGIC 5 <---
            # =========================================================
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
            # print(f"[DEBUG MAIN] Nhận alert từ queue - Module: {module_name}, Status: {status}")
            if module_name == "face_verify" and status == "alert":
                log_msg = f"**[{time_str}]** 🕵️‍♂️ PHÁT HIỆN NGƯỜI LẠ!"
                # print(f"[DEBUG MAIN] 🚨 Kích hoạt log 'PHÁT HIỆN NGƯỜI LẠ' lên giao diện lịch sử!")
            elif module_name == "object_detect" and status == "alert":
                # 🔥 TÁCH RIÊNG LOG VẬT CẤM (SMARTPHONE, EARPHONE,...)
                detected_labels = [det.get("label", "vật thể") for det in alert.get("detections", [])]
                label_str = ", ".join(detected_labels).upper()
                log_msg = f"**[{time_str}]** <span style='color:red'>📱 PHÁT HIỆN VẬT CẤM: {label_str}</span>"
            elif module_name == "paper_count" and status == "alert":
                alerts = alert.get("alerts", [])
                papers = alert.get("papers", [])
                
                has_true_paper_violation = False
                detected_device_label = ""
                
                # 1. Kiểm tra xem trong danh sách papers có tờ nào là phao mới (suspicious_new_paper) không
                for p in papers:
                    if p.get("status") == "suspicious_new_paper":
                        has_true_paper_violation = True
                        
                # 2. Kiểm tra chi tiết từng alert bên trong
                for a in alerts:
                    label = str(a.get("label", "")).lower()
                    source = str(a.get("source", "")).lower()
                    
                    if "smartphone" in label or "phone" in label or "device" in label or "object_detect" in source:
                        detected_device_label = label.upper() if label else "SMARTPHONE"
                    if "paper" in label or "cheat" in label or "sheet" in label or "monitor" in source:
                        has_true_paper_violation = True

                # 3. Phân tách rõ ràng log hiển thị dựa trên bản chất sự thật
                if detected_device_label and not has_true_paper_violation:
                    # Nếu thực chất là do object detector bắt nhầm điện thoại
                    pass
                    # log_msg = f"**[{time_str}]** <span style='color:red'>📱 PHÁT HIỆN VẬT CẤM 2: {detected_device_label}</span>"
                elif has_true_paper_violation:
                    # Chỉ khi thực sự có vi phạm liên quan đến giấy/phao thi mới báo cheatsheet
                    log_msg = f"**[{time_str}]** <span style='color:red'>📄 PHÁT HIỆN TÀI LIỆU LẠ (CHEATSHEET)!</span>"
                # else:
                #     # Dự phòng nếu có alert khác chung chung
                #     log_msg = f"**[{time_str}]** <span style='color:red'>⚠️ CẢNH BÁO BẤT THƯỜNG TRÊN BÀN THI</span>"
            elif module_name == "pose_gaze" and status == "warning":
                action_text = details.get('action', 'Bất thường')
                if "EXCHANGE" in action_text or "LOOKING" in action_text or "HOẠT ĐỘNG BẤT THƯỜNG" in action_text:
                    log_msg = f"**[{time_str}]** <span style='color:red'>🚨 GIAN LẬN HÀNH VI: {action_text}</span>"
                else:
                    log_msg = f"**[{time_str}]** ⚠️ TƯ THẾ: {action_text}"
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
    st.markdown("### 📷 Camera Giám Sát Real-time (Zero Latency)")
    
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
    # Cấu hình WebRTC (Tắt Audio của WebRTC vì Audio đã chạy độc lập qua PyAudio)
    webrtc_streamer(
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
    st.markdown("### 📜 Lịch Sử Cảnh Báo")

    # ---> LỆNH TỰ ĐỘNG LÀM MỚI GIAO DIỆN MỖI 3 GIÂY (3000 mili-giây) <---
    st_autorefresh(interval=3000, key="auto_refresh_log")

    # Lấy Log mới nhất từ Shared State
    log_html = "<br>".join(SHARED_STATE["logs"]) if SHARED_STATE["logs"] else "Hệ thống đang chạy..."
    st.markdown(f"<div style='height: 500px; overflow-y: auto; background-color: #f0f2f6; padding: 10px; border-radius: 10px;'>{log_html}</div>", unsafe_allow_html=True)
    
    # Nút Refresh Log bằng tay (vì WebRTC chạy ngầm, giao diện cần trigger để hiện Log mới)
    # st.button("🔄 Làm mới Log", use_container_width=True)
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

from backend.core.device import select_device, configure_cpu_threads
DEVICE = select_device()
# Bốn luồng model chạy song song; mỗi thư viện mặc định lấy hết core (torch 16 thread) nên
# chúng tranh CPU lẫn nhau. Đặt ngân sách thread TRƯỚC khi nạp model (xem device.CPU_THREADS).
CPU_THREADS = configure_cpu_threads()
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
from backend.ai_services.face_verify.face_verify import FaceVerifier
# Micro: PyAudio trên Linux, sounddevice trên Windows (hai worker riêng biệt)
if IS_WINDOWS:
    from backend.ai_services.whisper.realtime_audio_worker import RealtimeAudioWorker
else:
    from backend.ai_services.whisper.realtime_audio_ubuntu import RealtimeAudioWorker
# Logic per-module đã tách ra package dùng chung (realtime + offline chạy ra cảnh báo y hệt nhau)
from backend.ai_services.pipeline import audio_scan, events, face_scan, gaze_scan, object_scan, paper_scan
from backend.ai_services.pipeline import overlays as overlay_draw
from backend.ai_services.pipeline.metrics import METRICS
from offline_job import OfflineJob

# Tên module realtime — giống luồng offline (`run_offline.ALL_MODULES`) để người dùng và log
# dùng cùng một từ vựng. "pose" ở offline chỉ là bí danh cũ của "action" nên realtime bỏ.
REALTIME_MODULES = ("object", "face", "paper", "action", "audio")
FACES_DIR = Path("data/student_faces")

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
        "last_gaze_push": 0.0,
        "last_vision_push": 0.0,
        "logs": [],
        "log_cooldown": {},
        # Bật/tắt module realtime lúc đang chạy. Tên trùng `run_offline.ALL_MODULES` để hai
        # luồng dùng cùng từ vựng. Thread đọc cờ này MỖI vòng lặp nên đổi trên UI có hiệu lực
        # ngay, không phải restart (model vẫn nằm trong RAM, chỉ bỏ qua phần suy luận).
        "modules": set(REALTIME_MODULES),
        "face_model": None,
    }

    # --- LUỒNG AI THỊ GIÁC ---
    # `os.nice` chỉ hạ ưu tiên của ĐÚNG thread đang gọi (Linux/WSL). Khi CPU bị AI chiếm hết,
    # event loop của aiortc không kịp nhận/ACK gói -> bộ ước lượng băng thông tưởng nghẽn ->
    # gửi REMB thấp -> Chrome hạ độ phân giải (mờ) và trả frame cũ (lag). Nhường CPU cho
    # callback WebRTC là cách rẻ nhất để cắt vòng lặp đó; Windows không hỗ trợ -> bỏ qua.
    def _be_nice() -> None:
        try:
            os.nice(5)
        except (AttributeError, OSError):
            pass

    def _module_on(name: str) -> bool:
        """Module có đang bật không (UI ghi vào `shared_state["modules"]`, mặc định bật hết)."""
        return name in shared_state.get("modules", REALTIME_MODULES)

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
            # UI cần model này để nạp lại DB sau khi xoá/thêm ảnh học sinh.
            shared_state["face_model"] = face_model
        except Exception as e:
            print(f"[LỖI KHỞI TẠO AI THỊ GIÁC]: {e}")
            vision_ready.set()
            return
        
        # ---> BÍ QUYẾT ĐỒNG BỘ: Tạo hàm Masking chung cho cả Camera và Đăng ký <---
        def get_largest_stranger_with_masking(img, ts):
            with METRICS.time("face"):
                return face_scan.scan_stranger_with_masking(face_model, img, ts)
        # -------------------------------------------------------------------------
        
        vision_ready.set()
        print("[INFO] ✅ AI Thị giác (Nặng) đã nạp xong!")
        _be_nice()

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

            # Nạp lại DB khuôn mặt theo yêu cầu từ UI (xoá/thêm ảnh). Làm Ở THREAD NÀY vì
            # `_load_database()` ghi đè `_known_names`/`_known_vectors`; gọi từ luồng Streamlit
            # có thể để lọt trạng thái nửa cũ nửa mới khi đang có frame được xử lý.
            if shared_state.pop("reload_face_db", False):
                try:
                    face_model._load_database()
                except Exception as exc:
                    print(f"[LỖI NẠP LẠI FACE DB] {exc}")

            # 2. Xử lý khung hình. Cờ module đọc lại mỗi frame -> tắt trên UI là dừng ngay
            # (không chạy model, không đẩy cảnh báo) mà không phải restart app.
            # `timeout` để vòng lặp vẫn thức dậy xử lý cờ `reload_face_db` khi camera chưa chạy.
            try:
                data = frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if data is None: break
            frame, timestamp = data

            # Vật cấm (đã lọc nhãn giấy trong object_scan) + người lạ (masking) đi qua queue chung
            if _module_on("object"):
                with METRICS.time("object"):
                    alert_object = object_scan.scan_objects(yolo_model, frame, timestamp)
                if alert_object is not None:
                    if not result_q.full():
                        result_q.put(alert_object)
                    else:
                        METRICS.drop("queue.result")
                        print("[DEBUG YOLO] CẢNH BÁO: RESULT_QUEUE đã đầy, bị rớt frame cảnh báo!")

            if _module_on("face"):
                result_face, _ = get_largest_stranger_with_masking(frame, timestamp)
                if result_face:
                    if not result_q.full():
                        result_q.put(result_face)
                    else:
                        METRICS.drop("queue.result")

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
            # `device=None` -> auto: trên máy Intel là OV/iGPU, trên NVIDIA là CUDA. Trước đây ép
            # `PERSON_DEVICE` (= "cpu") nên model vật cấm của luồng giấy chạy .pt trên CPU, cộng
            # thêm person detector cũng CPU -> hai model xếp hàng trên cùng CPU. Tách sang iGPU
            # (person vẫn CPU) thì hai model chạy song song: đo 2026-10-29, paper p50 1089 -> 319 ms.
            paper_object_detector = ObjectDetectModule(device=None)
            
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
        _be_nice()

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
            if not _module_on("paper"):
                continue  # frame đã được rút khỏi queue ở trên -> không đọng lại
            try:
                with METRICS.time("paper"):
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
            if alert_paper is not None:
                if not result_q.full():
                    if alert_paper["alerts"]:
                        print(f"[DEBUG PAPER] 🚨 Kích hoạt báo động! Danh sách lỗi: {alert_paper['alerts']} | Số lượng tờ giấy trên bàn: {len(alert_paper['papers'])}")
                    result_q.put(alert_paper)
                else:
                    METRICS.drop("queue.result")

    # LUỒNG 2: Hành vi (Tracking + Holistic + Causal Classifier)
    def fast_gaze_thread():
        print("[INFO] Đang khởi động Fast Gaze Thread (Tracking & Holistic & Action Classifier)...")

        # Stack DUY NHẤT cho hành vi: tracking + holistic + classifier (c2 Exchange, c3 Looking, suspicious).
        # Khung xương trên video hiển thị được vẽ lại từ latest_packet/latest_results, không chạy model lần 2.
        # `isolate_holistic=True`: MediaPipe giữ GIL ~100 ms/tick nên chạy ở tiến trình riêng —
        # nếu để trong app, callback WebRTC bị chặn và hình giật/trễ dần.
        stack = gaze_scan.build_stack(
            device=PERSON_DEVICE,
            live_pair=["student_01:student_02"],
            isolate_holistic=True,
        )
        shared_state["gaze_model"] = stack
        shared_state["gaze_worker"] = stack.holistic
        shared_state["tracking_manager"] = stack.manager
        shared_state["holistic_model"] = stack.holistic
        shared_state["live_classifier"] = stack.live_classifier

        gaze_ready.set()
        print("[INFO] ✅ Fast Gaze Thread (Nâng cao) đã nạp xong!")
        # KHÔNG `_be_nice()` cho luồng này: phần nặng (MediaPipe) đã ở tiến trình con, còn lại
        # là chờ socket + tracking CPU. Hạ ưu tiên chỉ làm khung xương cập nhật chậm hơn (đứng
        # hình ở chỗ cũ) mà không giúp gì cho callback video.

        while True:
            data = gaze_q.get()
            if data is None: break
            frame, timestamp = data

            if not _module_on("action"):
                # Xoá kết quả cũ, nếu không overlay sẽ vẽ lại khung xương đã "đóng băng".
                stack.latest_packet = None
                stack.latest_results = ()
                continue

            with METRICS.time("action"):
                c2_c3_alerts = gaze_scan.scan_frame(stack, frame, timestamp)
            for c2_c3_alert in c2_c3_alerts:
                if not result_q.full():
                    result_q.put(c2_c3_alert)
                else:
                    METRICS.drop("queue.result")

    # --- LUỒNG AI ÂM THANH (Dùng PyAudio độc lập) ---
    def audio_ai_thread():
        print("[INFO] Đang khởi động luồng AI Âm thanh thực tế...")
        try:
            audio_worker = RealtimeAudioWorker()
            # Phơi ra cho panel "Sức khoẻ module": nếu mic không nghe thấy gì (RMS thấp hơn
            # ngưỡng VAD) thì không câu nào được cắt và giao diện im lặng dù người dùng đang nói.
            shared_state["audio_worker"] = audio_worker
            print("[DEBUG AUDIO] ✅ Đã khởi tạo RealtimeAudioWorker thành công!")

            original_pipeline = audio_worker.pipeline.process_audio
            def hooked_process_audio(*args, **kwargs):
                if not _module_on("audio"):
                    return None  # worker coi None là "không có gì để báo"
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

                with METRICS.time("audio"):
                    res = original_pipeline(*args, **kwargs)

                alert_audio = audio_scan.to_alert(res)
                if alert_audio is not None:
                    if not result_q.full():
                        result_q.put(alert_audio)
                    else:
                        METRICS.drop("queue.result")
                        print("[DEBUG AUDIO] ⚠️ RESULT_QUEUE đã đầy, rớt gói tin âm thanh!")
                    
                return res
                
            audio_worker.pipeline.process_audio = hooked_process_audio
            audio_ready.set()
            print("[INFO] ✅ Audio Thread đã nạp xong và chuẩn bị lắng nghe micro!")
            audio_worker.start()
        except Exception as e:
            shared_state["audio_error"] = f"{type(e).__name__}: {e}"
            print(f"[LỖI LUỒNG ÂM THANH] Khởi tạo thất bại: {e}")
            import traceback
            traceback.print_exc()
            audio_ready.set()

    # Nạp trước các thư viện nặng, TUẦN TỰ ở luồng chính. Python 3.12 coi việc hai thread cùng
    # import một module là deadlock (`_ModuleLock`) và đã ném thật:
    #   "[LỖI KHỞI TẠO PAPER PIPELINE]: deadlock detected by _ModuleLock('ultralytics.utils')"
    # khi luồng thị giác và luồng đếm giấy cùng import ultralytics. Import ở đây chỉ chạy một lần
    # trong `st.cache_resource`, các luồng sau đó chỉ tra `sys.modules`.
    for _module in ("ultralytics", "ultralytics.utils", "torch", "mediapipe",
                    "onnxruntime", "insightface", "faiss"):
        try:
            __import__(_module)
        except Exception as _exc:  # thiếu gói là chuyện của từng module, không chặn khởi động
            print(f"[INFO] Bỏ qua preload {_module}: {_exc}")

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
# Nhịp đẩy frame cho luồng "nặng" (YOLO vật cấm + FaceNet). Đo trên RTX 4050: 22 + 32 ms nên
# trần ~18 fps; đẩy mỗi frame chỉ sinh ra drop liên tục và tranh GIL với callback. 10 fps là dư
# khi mỗi model đã có ngân sách thread riêng (device.CPU_THREADS) thay vì mỗi cái chiếm 16 thread.
VISION_TARGET_FPS = 10.0
VISION_MIN_INTERVAL = 1.0 / VISION_TARGET_FPS
# Nhịp đẩy frame vào luồng HÀNH VI. Model causal được train ở 8 fps
# (stage6_bundle_exact/.../temporal_policy.json: target_fps=8, max_derivative_gap_ms=450) nên đẩy
# 8 fps là đúng phân bố lúc train. Trước đây đẩy MỌI frame: queue maxsize=2 vứt bớt 2/3 số tick,
# chỉ tốn CPU/GPU và tranh GIL với callback WebRTC.
GAZE_TARGET_FPS = 8.0
GAZE_MIN_INTERVAL = 1.0 / GAZE_TARGET_FPS
# Cùng một nội dung cảnh báo không lặp lại trong sidebar trong N giây. Chỉ gộp DÒNG LOG:
# overlay vẫn được làm mới mỗi alert, và offline/`events.jsonl` vẫn ghi đủ từng lần phát hiện.
LOG_COOLDOWN = 2.0
# Không có frame trong khoảng này = phiên stream mới (STOP rồi START, hoặc lần đầu bấm START).
# Dùng để bỏ cảnh báo đã sinh ra trong lúc camera chưa chạy (model âm thanh vẫn chạy nền).
# 10 s (không phải 3): lúc mở stream Chrome có thể ngừng gửi vài giây để thương lượng, hạ
# ngưỡng quá thấp sẽ xoá oan cảnh báo vừa sinh ra trong lúc đang chạy.
STREAM_GAP_RESET_S = 10.0
# Sàn REMB (bps). aiortc ước lượng băng thông Chrome -> server từ jitter gói RTP
# (aiortc/rate.py: RemoteBitrateEstimator) rồi gửi NGUYÊN giá trị đó về Chrome qua REMB.
# Khi tiến trình Python bận, gói đến dồn cục -> estimator tưởng nghẽn -> REMB thấp -> Chrome
# bật scaleResolutionDownBy (960x540 -> 640x360 -> 320x180) nên hình "lúc rõ lúc mờ". Loopback
# thừa băng thông nên đây là nghẽn giả. Sàn này khác MIN_BITRATE của VP8: MIN_BITRATE chỉ chặn
# encoder server -> Chrome, không chặn REMB Chrome -> server. PROCTOR_REMB_FLOOR=0 để tắt.
REMB_FLOOR = int(os.environ.get("PROCTOR_REMB_FLOOR", "1200000") or 0)


def _apply_remb_floor() -> None:
    if REMB_FLOOR <= 0 or SHARED_STATE.get("remb_floor_applied"):
        return
    try:
        from aiortc.codecs import vpx
        from aiortc.rate import RemoteBitrateEstimator

        original_add = RemoteBitrateEstimator.add

        def add_with_floor(self, *args, **kwargs):
            estimate = original_add(self, *args, **kwargs)
            if estimate is None:
                return None
            bitrate, ssrcs = estimate
            return max(bitrate, REMB_FLOOR), ssrcs

        RemoteBitrateEstimator.add = add_with_floor
        vpx.MIN_BITRATE = min(REMB_FLOOR, 2_000_000)
        vpx.DEFAULT_BITRATE = max(vpx.DEFAULT_BITRATE, REMB_FLOOR)
        SHARED_STATE["remb_floor_applied"] = True
        print(
            f"[WEBRTC] Sàn REMB {REMB_FLOOR // 1000} kbps "
            f"(VP8 min {vpx.MIN_BITRATE // 1000} kbps) — Chrome không tự hạ 960x540."
        )
    except Exception as exc:
        print(f"[WEBRTC] Không áp được sàn REMB: {type(exc).__name__}: {exc}")


# ==========================================
# 2. HÀM VẼ GIAO DIỆN (ĐƯỢC GỌI TRONG WEBRTC)
# ==========================================
def draw_warning_overlays(frame):
    gaze_ai = SHARED_STATE.get("gaze_model")
    with OVERLAY_LOCK:
        current_time = time.time()
        events.expire_overlays(ACTIVE_OVERLAYS, current_time, OVERLAY_TTL)
        snapshot = list(ACTIVE_OVERLAYS)
    with METRICS.time("draw"):
        return overlay_draw.draw(frame, snapshot, gaze_model=gaze_ai)

def _log_allowed(module: str, log_msg: str, now: float) -> bool:
    """True nếu chưa có dòng log cùng nội dung trong LOG_COOLDOWN giây (bỏ tiền tố giờ)."""
    key = (module, log_msg.partition("]** ")[2] or log_msg)
    seen = SHARED_STATE["log_cooldown"]
    if now - seen.get(key, 0.0) < LOG_COOLDOWN:
        METRICS.drop("log.suppressed")
        return False
    seen[key] = now
    if len(seen) > 64:  # dọn key cũ để dict không phình mãi
        cutoff = now - 60
        for k, ts in list(seen.items()):
            if ts < cutoff:
                del seen[k]
    return True


# ==========================================
# 3. CALLBACK VIDEO XỬ LÝ FRAME WEBRTC
# ==========================================
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    with METRICS.time("callback"):
        return _process_incoming_frame(frame)


def _process_incoming_frame(frame: av.VideoFrame) -> av.VideoFrame:
    # 1. Chuyển đổi định dạng frame từ WebRTC sang Numpy (OpenCV)
    img = frame.to_ndarray(format="bgr24")

    # Kích thước + nhịp frame THẬT trình duyệt đang gửi. Chrome tự hạ 960x540 -> 480x270 khi
    # ước lượng băng thông (aiortc gửi REMB ngược lại) tụt vì jitter -> hình "lúc rõ lúc mờ".
    # Đây là số đo duy nhất phân biệt mờ-do-trình-duyệt với mờ-do-encoder của server.
    SHARED_STATE["in_size"] = (frame.width, frame.height)
    in_now = time.monotonic()
    in_prev = SHARED_STATE.get("in_last_ts") or 0.0
    prev_size = SHARED_STATE.get("in_size_prev")
    if prev_size and prev_size != (frame.width, frame.height):
        print(f"[WEBRTC] đổi kích thước vào: {prev_size[0]}x{prev_size[1]} -> {frame.width}x{frame.height}")
    SHARED_STATE["in_size_prev"] = (frame.width, frame.height)
    if in_prev and 0.005 < in_now - in_prev < 1.0:
        inst_fps = 1.0 / (in_now - in_prev)
        prev_fps = SHARED_STATE.get("in_fps") or 0.0
        SHARED_STATE["in_fps"] = inst_fps if prev_fps <= 0 else prev_fps * 0.9 + inst_fps * 0.1
    SHARED_STATE["in_last_ts"] = in_now

    # Phiên stream mới (frame đầu tiên sau khi bấm START). Model âm thanh chạy suốt từ lúc mở
    # app nên `RESULT_QUEUE` đã đầy cảnh báo cũ; nếu để nguyên thì vừa START là panel tuôn ra
    # một loạt log của lúc chưa có camera. Bỏ chúng và ghi mốc để lọc tiếp phần còn lại.
    # Console vẫn in đủ — cái này chỉ ảnh hưởng panel/lịch sử cảnh báo.
    if not in_prev or in_now - in_prev > STREAM_GAP_RESET_S:
        SHARED_STATE["stream_epoch"] = time.time()
        dropped = 0
        while not RESULT_QUEUE.empty():
            try:
                RESULT_QUEUE.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        if dropped:
            print(f"[INFO] Bỏ {dropped} cảnh báo tồn đọng trước khi START (vẫn xem được ở terminal).")

    SHARED_STATE["frame_count"] += 1
    current_time = time.time()

    # 2. Xử lý Hàng đợi Kết quả & Đóng gói Cảnh báo
    stream_epoch = SHARED_STATE.get("stream_epoch", 0.0)
    while not RESULT_QUEUE.empty():
        try:
            alert = RESULT_QUEUE.get_nowait()

            # Cảnh báo sinh ra TRƯỚC khi phiên stream này bắt đầu thì không đưa lên panel.
            event_ts = alert.get("timestamp")
            if event_ts is not None and event_ts < stream_epoch:
                continue

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

            if log_msg and _log_allowed(module_name, log_msg, current_time):
                SHARED_STATE["logs"].insert(0, log_msg)
                if len(SHARED_STATE["logs"]) > 20:
                    SHARED_STATE["logs"].pop()

        except queue.Empty:
            break

    # 3. Đẩy ảnh mới vào Queue cho AI xử lý (frame skipping + chỉ copy MỘT lần mỗi frame).
    #
    # Bước 5 vẽ overlay đè lên `img`, nên queue phải nhận bản copy; nhưng bản cũ copy 3 lần ở
    # full-res cho 3 queue. Nay copy một lần rồi chia sẻ — các luồng AI chỉ đọc frame này.
    analysis_img = None
    frame_no = SHARED_STATE["frame_count"]

    def _push(target, metric_key: str) -> None:
        nonlocal analysis_img
        if target.full():
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            METRICS.drop(f"queue.{metric_key}")
        if analysis_img is None:
            analysis_img = img.copy()
        target.put((analysis_img, current_time))

    # Hành vi: 8 fps theo đồng hồ thật (xem GAZE_TARGET_FPS), không phải mỗi frame.
    now_mono = time.monotonic()
    if now_mono - SHARED_STATE.get("last_gaze_push", 0.0) >= GAZE_MIN_INTERVAL:
        SHARED_STATE["last_gaze_push"] = now_mono
        _push(GAZE_QUEUE, "gaze")
    # Đếm giấy: giấy nằm yên trên bàn, 4 fps là quá đủ.
    if frame_no % 5 == 0:
        _push(PAPER_QUEUE, "paper")
    if now_mono - SHARED_STATE.get("last_vision_push", 0.0) >= VISION_MIN_INTERVAL:
        SHARED_STATE["last_vision_push"] = now_mono
        _push(FRAME_QUEUE, "vision")

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
_METRIC_LABELS = {
    "callback": "Callback WebRTC (cả frame)",
    "object": "YOLO vật cấm",
    "face": "FaceNet người lạ",
    "paper": "Đếm giấy",
    "action": "Tracking + action",
    "audio": "Âm thanh (VAD→PhoBERT)",
    "draw": "Vẽ overlay",
    "queue.gaze": "Rớt frame → luồng gaze",
    "queue.paper": "Rớt frame → luồng giấy",
    "queue.vision": "Rớt frame → luồng thị giác",
    "queue.result": "Rớt cảnh báo → RESULT_QUEUE",
    "log.suppressed": "Log trùng đã gộp (sidebar)",
}


def _render_metrics_table():
    """Bảng sức khoẻ module: đọc snapshot của METRICS (thread-safe) tại thời điểm rerun."""
    snap = METRICS.snapshot()
    rows = []
    for name, stat in snap.items():
        label = _METRIC_LABELS.get(name, name)
        if name.startswith("queue.") or stat["runs"] == 0:  # chỉ đếm sự kiện, không đo thời gian
            rows.append({"Module": label, "fps": None, "p50": None, "p95": None, "max": None,
                         "rớt": stat["drops"]})
            continue
        rows.append({"Module": label, "fps": stat["fps"], "p50": stat["p50_ms"], "p95": stat["p95_ms"],
                     "max": stat["max_ms"], "rớt": stat["drops"]})
    if not rows:
        st.caption("Chưa có số đo — bấm START camera rồi chờ vài giây.")
        return
    st.dataframe(rows, hide_index=True, use_container_width=True)
    bad = [f"`{name}` {stat['errors']} lần" for name, stat in snap.items() if stat["errors"]]
    if bad:
        st.caption("Lỗi: " + " · ".join(bad))
    errors = [f"`{name}`: {stat['last_error']}" for name, stat in snap.items() if stat["last_error"]]
    if errors:
        st.caption("Lỗi gần nhất — " + " · ".join(errors))


def _render_stream_status() -> None:
    """Độ phân giải/nhịp frame trình duyệt đang gửi — thứ quyết định hình rõ hay mờ."""
    size = SHARED_STATE.get("in_size")
    if not size:
        st.caption("📥 WebRTC: chưa nhận frame nào (bấm START).")
        return
    width, height = size
    fps = float(SHARED_STATE.get("in_fps") or 0.0)
    st.caption(f"📥 WebRTC vào: **{width}×{height}** @ {fps:.1f} fps · server trả về cùng kích thước")
    if width < 900 or height < 500:
        st.caption(
            f"⚠️ Trình duyệt đang gửi {width}×{height} (thấp hơn 960×540): Chrome tự hạ độ phân giải "
            "khi jitter/CPU tăng, nên hình mờ dù pipeline không đổi. Tắt START rồi bật lại chỉ tạm thời; "
            "giảm tải máy (đóng app khác, hạ `stride` giấy) mới bền."
        )


def _render_student_manager() -> None:
    """Danh sách học sinh đã đăng ký (`data/student_faces`) — xem, xoá, nạp lại.

    Không xoá trực tiếp DB trong RAM: đặt cờ `reload_face_db` để luồng thị giác tự nạp lại
    (xem `_load_database` được gọi trong `heavy_vision_thread`).
    """
    files = []
    if FACES_DIR.exists():
        files = sorted(p for p in FACES_DIR.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    model = SHARED_STATE.get("face_model")
    loaded = len(getattr(model, "_known_names", []) or []) if model is not None else None

    with st.expander(f"👥 Học sinh đã đăng ký ({len(files)})", expanded=False):
        if loaded is not None and loaded != len(files):
            st.caption(f"⚠️ AI đang giữ {loaded} khuôn mặt (khác số file) — bấm **Nạp lại** bên dưới.")
        if not files:
            st.caption("Chưa có ai. Bấm **📸 Đăng ký khuôn mặt** khi khuôn mặt đang trong khung hình.")
            return

        cols = st.columns(2)
        for i, path in enumerate(files):
            with cols[i % 2]:
                st.image(str(path), caption=path.name, use_container_width=True)
                if st.button("🗑 Xoá", key=f"faces_del_{path.name}", use_container_width=True):
                    st.session_state["faces_delete_target"] = path.name

        target = st.session_state.get("faces_delete_target")
        if target:
            st.warning(f"Xoá vĩnh viễn `{target}` khỏi danh sách học sinh?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Xác nhận xoá", type="primary", key="faces_del_confirm"):
                    try:
                        (FACES_DIR / target).unlink()
                        SHARED_STATE["reload_face_db"] = True
                        st.session_state["faces_delete_target"] = None
                        st.rerun()
                    except OSError as exc:
                        st.error(f"Không xoá được: {exc}")
            with c2:
                if st.button("Huỷ", key="faces_del_cancel"):
                    st.session_state["faces_delete_target"] = None
                    st.rerun()

        if st.button("↻ Nạp lại danh sách vào AI", key="faces_reload"):
            SHARED_STATE["reload_face_db"] = True
            st.rerun()


def _render_mic_status() -> None:
    """Hiện mức mic thật để phân biệt "mic câm/chọn nhầm thiết bị" với "VAD chưa cắt câu"."""
    error = SHARED_STATE.get("audio_error")
    if error:
        st.caption(f"🎙️ Micro: ❌ khởi tạo lỗi — {error}")
        return
    worker = SHARED_STATE.get("audio_worker")
    if worker is None:
        st.caption("🎙️ Micro: đang khởi động…")
        return

    threshold = float(getattr(getattr(worker, "chunker", None), "threshold", 0.0) or 0.0)
    rms = float(getattr(worker, "last_rms", 0.0) or 0.0)
    max_rms = float(getattr(worker, "max_rms", 0.0) or 0.0)
    chunks = int(getattr(worker, "chunks_captured", 0) or 0)
    segments = int(getattr(worker, "segments_cut", 0) or 0)
    device = getattr(worker, "device_name", "") or "(mặc định)"
    sr = int(getattr(worker, "input_sr", 0) or 0)

    st.caption(
        f"🎙️ Micro: **{device}** @ {sr} Hz · RMS {rms:.4f} / ngưỡng VAD {threshold:.4f} · "
        f"cao nhất {max_rms:.4f} · {chunks} chunk · {segments} câu đã cắt"
    )
    worker_error = getattr(worker, "last_error", None)
    if worker_error:
        st.caption(f"❌ Micro không mở được: {worker_error} — xem console để chọn thiết bị khác.")
    elif chunks == 0:
        st.caption("🎙️ Chưa nhận được mẫu âm thanh nào — kiểm tra quyền micro / thiết bị đầu vào.")
    elif max_rms < threshold:
        st.caption(
            f"⚠️ Mức thu cao nhất {max_rms:.4f} vẫn dưới ngưỡng {threshold:.4f}: nếu bạn **đã nói to mà dòng "
            "này không đổi**, không câu nào được cắt. Hạ ngưỡng bằng `PROCTOR_VAD_THRESHOLD=0.012`, hoặc chọn "
            "mic khác bằng `PROCTOR_MIC_DEVICE=<số>` (index in ra ở console), rồi chạy lại app."
        )
    elif segments == 0:
        st.caption("🎙️ Đã nghe thấy tiếng nhưng chưa cắt được câu — cần ≥2 giây nói liên tục trong một câu.")


@st.fragment(run_every="1s")
def _render_live_panel() -> None:
    """Cột phải (lịch sử cảnh báo + sức khoẻ module) tự làm mới mỗi 1 s.

    Trước đây dùng `st_autorefresh(interval=3000)`: mỗi 3 s Streamlit **rerun cả trang** — render
    lại dataframe/log, tạo lại component WebRTC và tranh GIL với luồng suy luận. Hình vì thế
    khựng đúng nhịp 3 s. `st.fragment(run_every=...)` chỉ chạy lại đúng khối này.
    """
    st.markdown("### 📜 Lịch Sử Cảnh Báo")
    logs = list(SHARED_STATE["logs"])
    log_html = "<br>".join(logs) if logs else "Hệ thống đang chạy..."
    st.markdown(
        f"<div style='height: 500px; overflow-y: auto; background-color: #f0f2f6; padding: 10px; "
        f"border-radius: 10px;'>{log_html}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("🩺 Sức khoẻ module (fps / độ trễ / rớt frame)", expanded=True):
        _render_metrics_table()
        worker = SHARED_STATE.get("gaze_worker")
        if worker is not None:
            stats = getattr(worker, "stats", None) or {}
            st.caption(
                f"MediaPipe Holistic: **{worker.mode}** · "
                f"remote={stats.get('remote', 0)} · tại chỗ={stats.get('fallback', 0)} · "
                f"restart={stats.get('restarts', 0)}"
                + (f" · lỗi gần nhất: {worker.last_error}" if worker.last_error else "")
            )
        _render_stream_status()
        _render_mic_status()
        if st.button("♻️ Reset số đo", key="reset_metrics"):
            METRICS.reset()
            st.rerun(scope="fragment")


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
    # Cấu hình WebRTC. Audio tắt ở đây vì micro đã chạy độc lập qua PyAudio.
    #
    # 960x540 @ 20 fps, không để Chrome tự chọn: VP8 encode ở 720p mất ~25 ms khi máy bận
    # (52% hạn 50 ms của 20 fps) và p95 vượt hạn -> aiortc trả lại frame cũ -> hình giật.
    # Ở 540p encode còn ~11 ms (p95 ~22 ms). Mọi model chỉ dùng tối đa 640 px nên không mất
    # độ chính xác; ảnh đăng ký mặt vẫn nét hơn hẳn mức 320x240 mà Chrome tự hạ trước đây.
    # KHÔNG dùng STUN: trình duyệt và Streamlit chạy cùng máy (Windows -> WSL2), host candidate
    # là đủ; STUN chỉ thêm một vòng ICE ra Internet và treo nếu mạng công ty chặn UDP.
    _apply_remb_floor()
    webrtc_ctx = webrtc_streamer(
        key="exam-proctor",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            # `max` cũng phải bằng 960x540: trước đây max 1280x720 nên Chrome có thể chọn 720p,
            # decode + encode 720p tốn gấp đôi -> callback vượt hạn 50 ms của 20 fps -> aiortc
            # báo REMB thấp -> Chrome hạ tiếp độ phân giải (mờ) và trả frame cũ (lag).
            "video": {
                "width": {"ideal": 960, "max": 960},
                "height": {"ideal": 540, "max": 540},
                "frameRate": {"ideal": 20, "max": 20},
            },
            "audio": False,  # Chặn WebRTC thu âm để tránh đụng độ với luồng PyAudio
        },
        async_processing=True,
        rtc_configuration=RTCConfiguration({"iceServers": []}),
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

    # --- Bật/tắt module ngay khi đang chạy (giống ô "Module bật" của luồng offline) ---
    st.markdown("#### ⚙️ Module realtime")
    if "rt_modules" not in st.session_state:
        st.session_state.rt_modules = list(REALTIME_MODULES)
    rt_modules = st.multiselect("Module bật", REALTIME_MODULES, key="rt_modules",
                                help="Tắt module để giảm tải máy; có hiệu lực ngay, không cần restart.")
    # Thread đọc lại tập này mỗi vòng lặp (xem `_module_on` trong init_system_resources).
    SHARED_STATE["modules"] = set(rt_modules)
    if not rt_modules:
        st.caption("Không module nào bật — chỉ còn video, không có cảnh báo nào.")
    else:
        off = [m for m in REALTIME_MODULES if m not in rt_modules]
        if off:
            st.caption("Đang tắt: " + ", ".join(f"`{m}`" for m in off))

    _render_student_manager()

    # Khối log + bảng sức khoẻ tự làm mới mỗi 1 s (chỉ khối này, không rerun cả trang).
    _render_live_panel()

# ==========================================
# 5. CHẠY OFFLINE TRÊN MỘT FILE VIDEO (không cần camera/micro)
# ==========================================
OFFLINE_MODULES = ["object", "face", "paper", "action", "audio"]
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


def _available_ram_gb() -> Optional[float]:
    """RAM còn dùng được (Linux: MemAvailable). None nếu nền tảng không đọc được."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return None


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
               "Job chạy ở **tiến trình riêng** nên giao diện vẫn phản hồi và huỷ được giữa chừng.")

    job = st.session_state.get("offline_job")
    busy = bool(job and job.running)
    camera_on = bool(webrtc_ctx is not None and webrtc_ctx.state and webrtc_ctx.state.playing)

    if camera_on:
        st.error("⛔ Camera realtime đang phát — bấm STOP trước khi chạy offline. "
                 "Luồng offline nạp một bộ model RIÊNG nên chạy cả hai sẽ nhân đôi RAM/iGPU.")
    ram_gb = _available_ram_gb()
    ram_blocked = ram_gb is not None and ram_gb < 2.0
    if ram_gb is not None and not busy:
        if ram_blocked:
            st.error(f"⛔ RAM trống chỉ còn **{ram_gb:.1f} GB** — job offline cần ~3 GB nên chưa cho chạy "
                     "(chạy lúc này sẽ tràn RAM và bị OOM-kill). Đóng bớt ứng dụng rồi tải lại trang.")
        elif ram_gb < 3.5:
            st.warning(f"⚠️ RAM trống còn **{ram_gb:.1f} GB**, job offline cần ~3 GB. Nên đóng bớt ứng dụng, "
                       "hoặc bỏ module `audio`/`paper` cho nhẹ.")

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
        stride = st.slider("Stride nặng (YOLO/mặt/action) — lớn = nhanh", 1, 8, 3, key="off_stride")
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
                 disabled=(busy or camera_on or ram_blocked or not start_enabled)):
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
            if st.button("⏹ Huỷ job", key="off_cancel"):
                job.cancel()
                st.rerun()
            if snap["preview"]:
                st.image(snap["preview"], caption="Khung hình offline mới nhất (đã burn cảnh báo)",
                         use_container_width=True)
            if snap["logs"]:
                st.markdown("**Cảnh báo mới nhất**")
                for line in snap["logs"]:
                    st.text(overlay_draw.plain_text(line))
            st_autorefresh(interval=OFFLINE_REFRESH_MS, key="off_refresh")
        elif snap["status"] == "cancelled":
            st.info(f"Đã huỷ job offline. Phần đã chạy nằm trong {snap['out_dir']}")
        elif snap["status"] == "error":
            st.error(f"Lỗi khi chạy offline: {snap['error']}")
            st.caption(f"Log chi tiết trong {snap['out_dir']}")
        elif snap["status"] == "done":
            _render_offline_result(snap)
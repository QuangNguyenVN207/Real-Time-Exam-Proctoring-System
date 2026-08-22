import time
from pathlib import Path
from time import monotonic

import os
import sys
import mediapipe as mp

class SuppressOutput:
    """Bịt miệng log C++ ở tầng OS, tuyệt đối an toàn cho đa luồng."""
    def __enter__(self):
        self.devnull = open(os.devnull, 'w')
        # Lưu lại File Descriptor gốc của hệ điều hành
        self.old_stdout_fd = os.dup(sys.stdout.fileno())
        self.old_stderr_fd = os.dup(sys.stderr.fileno())
        # Chuyển hướng stdout và stderr vào hố đen
        os.dup2(self.devnull.fileno(), sys.stdout.fileno())
        os.dup2(self.devnull.fileno(), sys.stderr.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Trả lại luồng in chữ bình thường cho Python
        os.dup2(self.old_stdout_fd, sys.stdout.fileno())
        os.dup2(self.old_stderr_fd, sys.stderr.fileno())
        os.close(self.old_stdout_fd)
        os.close(self.old_stderr_fd)
        self.devnull.close()

# Yêu cầu cộng sự import đúng đường dẫn tương đối từ code của họ
from backend.ai_services.pose_gaze.pose_gaze.holistic.landmark import HolisticLandmarkExtractor
from backend.ai_services.pose_gaze.tracking.manager import TrackingManager
from backend.ai_services.pose_gaze.tracking.webcam import PersonTrackingConfig, PersonTrackingModule
from backend.ai_services.pose_gaze.pose_gaze.holistic.test_media.test_media import create_live_classifier

class PoseGazeDetector:
    def __init__(self, causal_model_dir="tmp/behavior_actor_causal_pose_only_20260812"):
        print("[INFO] Đang khởi tạo PoseGazeDetector (Thực tế)...")
        self.session_id = TrackingManager.generate_session_id("webcam_holistic")
        
        # 1. Khởi tạo Tracking
        self.tracking = PersonTrackingModule(
            PersonTrackingConfig(
                session_id=self.session_id,
                restore_session=False,
                max_tracks=2,
            )
        )
        
        # 2. Khởi tạo Classifier
        class DummyArgs:
            xgboost_model_dir = Path(causal_model_dir)
            student_prefix = "student_"
            live_pair = ["student_01:student_02"]

            c1_model_dir = None
            c4_model_dir = None
            c7_model_dir = None
        self.live_classifier = create_live_classifier(DummyArgs(), clip_id=f"webcam_{self.session_id}")

        with SuppressOutput():
        # 3. Khởi tạo Holistic
            self.holistic = HolisticLandmarkExtractor(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.holistic.__enter__() # Khởi động context manager
            # ---> BỔ SUNG BƯỚC KHỞI ĐỘNG NÓNG (WARM-UP) <---
            # Tạo 1 ảnh đen để ép lõi C++ chạy và xả hết log rác vào lỗ đen
            import numpy as np
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_packet = self.tracking.process_frame(dummy_frame)
            self.holistic.process_packet(dummy_frame, dummy_packet)

    def process_frame(self, frame, timestamp):
        """Hàm này sẽ được app.py gọi liên tục ở luồng Vision"""
        try:
            # Chạy Tracking & Holistic
            latest_packet = self.tracking.process_frame(frame)
            latest_holistic_results = self.holistic.process_packet(frame, latest_packet)

            # ---> LƯU LẠI 2 BIẾN NÀY ĐỂ XUẤT RA NGOÀI <---
            self.latest_results = latest_holistic_results
            self.latest_packet = latest_packet

            self.holistic.draw_results(frame, latest_holistic_results)
            
            # ---> THÊM 1 DÒNG NÀY ĐỂ XUẤT TỌA ĐỘ XƯƠNG RA NGOÀI <---
            self.latest_results = latest_holistic_results

            # Chạy phân loại hành vi
            classifications = self.live_classifier.update(
                frame_index=latest_packet.frame_id,
                timestamp_ms=latest_packet.timestamp_ms,
                results=latest_holistic_results,
            ) if self.live_classifier else {}

            # Kiểm tra xem có hành vi gian lận (c1, c2, c3, c4) không
            for result in latest_holistic_results:
                actor_id = result.student_id or f"student_{result.track_id:02d}"
                classification = classifications.get(actor_id)
                if not classification:
                    continue
                    
                predicted = str(classification["predicted_class"])
                
                # Bỏ qua c5 (Normal) và warmup, chỉ bắt gian lận
                if predicted != "c5" and "warming" not in str(classification):
                    labels = {
                        "c1": "Sử dụng điện thoại", 
                        "c2": "Trao đổi đồ vật", 
                        "c3": "Nhìn bài/Quay ngang",
                        "c4": "Sử dụng phao thi",
                        "suspicious_activity": "Hành vi khả nghi"
                    }
                    action_name = labels.get(predicted, predicted)
                    
                    # Trả về đúng định dạng hợp đồng dữ liệu cho app.py
                    return {
                        "module": "pose_gaze",
                        "status": "warning",
                        "timestamp": timestamp,
                        "details": {
                            "action": action_name,
                            "actor_id": actor_id,
                            "score": classification.get("evidence_score", 0.0)
                        }
                    }
            return None
            
        except Exception as e:
            print(f"[LỖI POSE_GAZE] {e}")
            return None

    def draw_skeleton(self, frame):
        """Hàm công khai để giao diện gọi và vẽ khung xương + tracking 30 FPS"""
        try:
            # Vẽ khung xương 3D và lưới khuôn mặt
            if hasattr(self, 'latest_results') and self.latest_results:
                self.holistic.draw_results(frame, self.latest_results)
            # Vẽ Bounding Box của thuật toán Tracking
            if hasattr(self, 'latest_packet') and self.latest_packet:
                self.tracking.draw_tracks(frame, self.latest_packet)
        except Exception:
            pass
        
    def __del__(self):
        if hasattr(self, 'holistic') and self.holistic is not None: 
            self.holistic.__exit__(None, None, None)
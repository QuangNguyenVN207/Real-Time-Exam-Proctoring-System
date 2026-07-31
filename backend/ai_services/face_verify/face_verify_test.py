import random
import onnxruntime as ort
import logging
ort.set_default_logger_severity(3)
logging.getLogger('insightface').setLevel(logging.ERROR)

class FaceVerifier:
    def __init__(self, db_path=None):
        print("[MOCK] Đã khởi tạo FaceVerifier (FaceNet giả lập)")

    def verify_face(self, frame, timestamp):
        # Tỉ lệ 3% phát hiện người lạ
        if random.random() < 0.03:
            bbox = [240, 180, 400, 380] # Tọa độ khuôn mặt người lạ
            
            return {
                "module": "face_verify",
                "status": "alert",
                "timestamp": timestamp,
                "details": {
                    "similarity_score": 0.25, # Điểm thấp -> người lạ
                    "unauthorized_bbox": bbox
                }
            }
        return None # Đúng thí sinh
import random

class PoseGazeDetector:
    def __init__(self):
        print("[MOCK] Đã khởi tạo PoseGazeDetector (MediaPipe giả lập)")
        self.actions = ["CUI MAT XUONG GAM BAN", "NHIN SANG TRAI LIEN TUC"]

    def process_frame(self, frame, timestamp):
        # Tỉ lệ 4% phát hiện tư thế bất thường
        if random.random() < 0.04:
            action = random.choice(self.actions)
            
            return {
                "module": "pose_gaze",
                "status": "warning",
                "timestamp": timestamp,
                "details": {
                    "action": action,
                    "angles": {"pitch": -25.5, "yaw": 15.0, "roll": 0.0}
                }
            }
        return None # Tư thế bình thường
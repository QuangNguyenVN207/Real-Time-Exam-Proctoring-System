import random

class ObjectDetector:
    def __init__(self, model_path=None):
        print("[MOCK] Đã khởi tạo ObjectDetector (YOLOv8 giả lập)")
        self.banned_items = ["smartphone", "cheat_sheet"]

    def process_frame(self, frame, timestamp):
        # Tỉ lệ 5% phát hiện ra đồ vật cấm
        if random.random() < 0.05:
            # Sinh tọa độ khung chữ nhật giả định (x1, y1, x2, y2)
            bbox = [100, 150, 250, 350] 
            item = random.choice(self.banned_items)
            
            return {
                "module": "object_detect",
                "status": "alert",
                "timestamp": timestamp,
                "detections": [
                    {"label": item, "confidence": 0.95, "bbox": bbox}
                ]
            }
        return None # Trạng thái an toàn
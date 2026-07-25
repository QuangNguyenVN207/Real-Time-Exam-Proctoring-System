import queue
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent # thư mục YOLOv8/ hiện tại

# --- CẤU HÌNH HỆ THỐNG ---
CAMERA_ID = 0
FPS_SKIP = 5 # Kỹ thuật Frame Skipping: Lấy 5 khung hình/giây để xử lý AI
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- KHỞI TẠO HÀNG ĐỢI (QUEUES) ---
# Dùng maxsize để tránh tràn RAM (Memory Leak) nếu AI xử lý không kịp
frame_queue = queue.Queue(maxsize=30)
audio_queue = queue.Queue(maxsize=50)
result_queue = queue.Queue(maxsize=100)

# Danh sách nhãn cấm của YOLOv8
BANNED_ITEMS = ["cheat_sheet","earphone","smartwatch","smartphone"]

class Settings:
    yolo_model_path: str = str(BASE_DIR / "weights" / "yolov8_finetuned.pt")
    yolo_confidence_threshold: float = 0.5
    frame_resize_width: int = 640
    flagged_classes: list[str] = [
        "cheat_sheet",
        "earphone",
        "smartwatch",
        "smartphone",
    ]
    object_confirm_frames: int = 3         # số frame liên tiếp để xác nhận (giảm false positive)
    object_detect_every_n_frames: int = 3  # chạy YOLO mỗi N frame, đỡ tốn CPU

    object_class_display_names: dict[str, str] = {}

    # --- Storage ---
    session_log_dir: Path = BASE_DIR / "data" / "sessions"


settings = Settings()
settings.session_log_dir.mkdir(parents=True, exist_ok=True)
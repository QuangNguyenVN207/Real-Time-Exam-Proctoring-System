from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent  # thư mục YOLOv8/ hiện tại
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
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
    yolo_model_path: str = str(BASE_DIR / "weights" / "best (1).pt")
    yolo_confidence_threshold: float = 0.5
    object_class_confidence_thresholds: dict[str, float] = {
        "earphone": 0.55,
        "smartphone": 0.55,
    }
    paper_detection_confidence_threshold: float = 0.20
    object_inference_size: int = 640
    frame_resize_width: int = 640
    # Paper classes are tracked by physical paper_id. They must not go through
    # the legacy direct-alert counter because the same paper may flip labels.
    paper_class_names: list[str] = [
        "cheat_sheet",
        "test_paper",
        "paper_unknown",
    ]
    flagged_classes: list[str] = [
        "earphone",
        "smartwatch",
        "smartphone",
    ]
    object_confirm_frames: int = 3         # số lần phát hiện cần có trong cửa sổ xác nhận
    object_confirm_window: int = 5         # chịu được tối đa hai inference hụt detection
    object_detect_every_n_frames: int = 2  # chạy YOLO mỗi N frame, đỡ tốn CPU
    smartphone_fallback_enabled: bool = False
    smartphone_fallback_model_path: str = str(
        BASE_DIR / "weights" / "yolov8n.pt"
    )
    smartphone_fallback_confidence_threshold: float = 0.35
    smartphone_fallback_nms_iou: float = 0.45
    # COCO has no calculator class. Ask the pretrained model for nearby
    # confuser classes as negative evidence, then use a button-grid check on
    # phone candidates before raising a smartphone alert.
    smartphone_confuser_class_names: list[str] = ["remote"]
    smartphone_confuser_confidence_threshold: float = 0.20
    smartphone_confuser_overlap_threshold: float = 0.20
    smartphone_calculator_grid_filter_enabled: bool = True
    # COCO occasionally labels a bright marker/pen as ``cell phone`` when a
    # person ROI magnifies it. Reject only the narrow bright-component pattern
    # seen in those false positives; broad light-coloured phones remain valid.
    smartphone_pen_shape_filter_enabled: bool = True
    # Class presence alone is not enough for a temporal alert: detections in
    # the rolling window must also refer to the same physical image region.
    object_spatial_match_overlap_threshold: float = 0.20
    object_spatial_match_center_distance_ratio: float = 1.25
    # The COCO fallback calls notebooks/books ``book``. In this exam domain a
    # visible book is treated exactly like a cheat sheet and routed through the
    # paper identity tracker instead of the legacy object alert counter.
    book_as_cheatsheet_enabled: bool = True
    book_fallback_confidence_threshold: float = 0.15
    book_fallback_detail_inference_size: int = 960
    book_fallback_min_short_side_ratio: float = 0.015
    book_fallback_max_aspect_ratio: float = 3.5
    person_roi_object_enabled: bool = True
    person_roi_object_inference_size: int = 960
    # A phone held edge-on or partly covered by a hand is often only 0.20-0.30
    # confident even after the person ROI is enlarged. Keep the full-frame
    # threshold conservative, but recover these hand-held phones inside the
    # tracked interaction ROI; temporal/spatial confirmation still applies.
    person_roi_phone_confidence_threshold: float = 0.15
    person_roi_book_confidence_threshold: float = 0.10
    person_roi_phone_min_short_side_ratio: float = 0.0125
    person_roi_phone_min_aspect_ratio: float = 1.55
    person_roi_phone_max_aspect_ratio: float = 4.5
    # The project checkpoint knows the domain-specific cheat_sheet class but
    # misses folded paper at full-frame 640 inference. Run only its paper
    # classes again inside each tracked person's interaction ROI.
    person_roi_custom_paper_enabled: bool = True
    person_roi_custom_paper_inference_size: int = 768
    person_roi_custom_paper_confidence_threshold: float = 0.15
    person_roi_horizontal_expansion: float = 0.15
    person_roi_top_expansion: float = 0.08
    person_roi_full_body_bottom_ratio: float = 0.82
    person_roi_torso_bottom_expansion: float = 0.10
    person_roi_full_body_aspect_ratio: float = 1.60
    person_roi_context_width_ratio: float = 0.40
    person_roi_context_height_ratio: float = 0.75
    person_roi_context_vertical_center_ratio: float = 0.35
    person_roi_max_people: int = 2

    # Live webcam defaults. Keeping the longest side at 960 avoids the
    # four-tile fallback used by Full-HD frames, which is the main source of
    # latency on CPU.
    webcam_max_width: int = 960
    webcam_max_height: int = 540
    webcam_person_detect_every_n_frames: int = 2
    webcam_object_detect_every_n_frames: int = 4
    # Do not override the ROI threshold with a stricter live-only floor. The
    # calculator/remote/marker guards and 3-in-5 confirmation reject confusers.
    webcam_phone_confidence_floor: float = 0.15

    person_appearance_match_threshold: float = 0.78
    paper_registration_frames: int = 5
    paper_alert_confirm_frames: int = 3
    paper_max_missed_frames: int = 12
    paper_auto_register_first: bool = True
    paper_appearance_match_threshold: float = 0.86

    # Count-only paper experiment. Duplicate/partial boxes are clustered into
    # one physical observation; a changed count must persist across two
    # actual object-inference frames before it becomes an alert.
    paper_count_confirm_inferences: int = 2
    paper_count_duplicate_overlap_threshold: float = 0.35
    paper_count_duplicate_center_distance_ratio: float = 0.70

    object_class_aliases: dict[str, str] = {
        "cheatsheet": "cheat_sheet",
        "cheat_paper": "cheat_sheet",
        "testpaper": "test_paper",
        "exam_paper": "test_paper",
        "question_paper": "test_paper",
        "cell_phone": "smartphone",
        "book": "cheat_sheet",
    }
    object_class_display_names: dict[str, str] = {
        "cheat_sheet": "cheat_sheet",
        "test_paper": "test_paper",
        "paper_unknown": "paper_unknown",
    }

    # --- Face Verify ---
    face_db_path: str = str(BASE_DIR / "data" / "student_faces")
    face_similarity_threshold: float = 0.4  # Cosine similarity (ArcFace); < ngưỡng này -> người lạ
    face_model_name: str = "buffalo_l"       # Model pack của insightface (RetinaFace + ArcFace)
    face_det_size: tuple[int, int] = (640, 640)

    # --- Storage ---
    session_log_dir: Path = BASE_DIR / "data" / "sessions"


settings = Settings()
settings.session_log_dir.mkdir(parents=True, exist_ok=True)

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent # thư mục YOLOv8/ hiện tại
class Settings:
    yolo_model_path: str = str(BASE_DIR / "weights" / "best (1).pt")
    yolo_confidence_threshold: float = 0.5
    object_class_confidence_thresholds: dict[str, float] = {
        "earphone": 0.55,
        "smartphone": 0.55,
    }
    paper_detection_confidence_threshold: float = 0.30
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
    object_detect_every_n_frames: int = 3  # chạy YOLO mỗi N frame, đỡ tốn CPU
    smartphone_fallback_enabled: bool = True
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
    person_roi_phone_confidence_threshold: float = 0.35
    person_roi_book_confidence_threshold: float = 0.10
    person_roi_phone_min_short_side_ratio: float = 0.0125
    person_roi_phone_min_aspect_ratio: float = 1.55
    person_roi_phone_max_aspect_ratio: float = 4.5
    # The project checkpoint knows the domain-specific cheat_sheet class but
    # misses folded paper at full-frame 640 inference. Run only its paper
    # classes again inside each tracked person's interaction ROI.
    person_roi_custom_paper_enabled: bool = True
    person_roi_custom_paper_inference_size: int = 768
    person_roi_custom_paper_confidence_threshold: float = 0.20
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
    webcam_phone_confidence_floor: float = 0.50

    person_appearance_match_threshold: float = 0.78
    paper_registration_frames: int = 5
    paper_alert_confirm_frames: int = 3
    paper_max_missed_frames: int = 12
    paper_auto_register_first: bool = True
    paper_appearance_match_threshold: float = 0.86

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

    # --- Storage ---
    session_log_dir: Path = BASE_DIR / "data" / "sessions"


settings = Settings()
settings.session_log_dir.mkdir(parents=True, exist_ok=True)

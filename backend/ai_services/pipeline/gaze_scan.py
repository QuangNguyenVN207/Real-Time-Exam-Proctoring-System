"""Luồng hành vi: tracking người -> holistic landmark -> causal classifier.

Sao y phần khởi tạo (`fast_gaze_thread`, app_webrtc.py:404-459) và phần xử lý frame
(app_webrtc.py:474-518) của bản realtime. `scan_frame()` trả về *danh sách* alert
(mỗi actor một alert) thay vì đẩy thẳng vào queue, nên offline runner và app cùng dùng.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from backend.core.device import resolve_device
from backend.ai_services.pose_gaze.pose_gaze.holistic.landmark import HolisticLandmarkExtractor
from backend.ai_services.pose_gaze.pose_gaze.holistic.test_media.test_media import (
    create_live_classifier,
)
from backend.ai_services.pose_gaze.pose_gaze.settings import PROJECT_ROOT
from backend.ai_services.pose_gaze.pose_gaze.tracking.webcam import (
    PersonTrackingConfig,
    PersonTrackingModule,
)

__all__ = [
    "DEFAULT_ACTION_ARTIFACT_DIR",
    "ALERT_CLASSES",
    "ACTION_LABELS",
    "GazeStack",
    "build_stack",
    "scan_frame",
]

DEFAULT_ACTION_ARTIFACT_DIR = (
    PROJECT_ROOT / "stage6_bundle_exact" / "causal_8fps_stage6_mixed_084699_final_20260827"
)

ALERT_CLASSES = ("c2", "c3", "suspicious_activity")
ACTION_LABELS = {
    "c2": "EXCHANGE (Trao đổi)",
    "c3": "LOOKING (Quay ngó)",
    "suspicious_activity": "HOẠT ĐỘNG BẤT THƯỜNG",
}


class _ClassifierArgs:
    """Tham số dạng CLI mà `create_live_classifier()` mong nhận — bản app dùng mock giống hệt."""

    def __init__(self, model_dir: Path, live_pair: list[str]) -> None:
        self.causal_model_dir = model_dir
        self.xgboost_model_dir = model_dir
        self.xgboost_device = "cpu"
        self.actions = "c2,c3,suspicious_activity"
        self.live_pair = live_pair
        self.student_prefix = "student_"


@dataclass
class GazeStack:
    session_id: str
    tracking_module: PersonTrackingModule
    holistic: HolisticLandmarkExtractor
    live_classifier: Optional[Any] = None
    classifier_error: Optional[str] = None
    # Phân loại mới nhất của *mọi* actor (kể cả c5) — offline dùng để báo cáo theo từng người.
    last_classifications: dict[str, Any] = field(default_factory=dict)

    @property
    def manager(self) -> Any:
        return self.tracking_module.manager

    def close(self) -> None:
        try:
            self.holistic.__exit__(None, None, None)
        except Exception:
            pass


def build_stack(
    *,
    session_id: Optional[str] = None,
    device: str = "cpu",
    confidence_threshold: float = 0.5,
    max_tracks: int = 2,
    model_dir: Path = DEFAULT_ACTION_ARTIFACT_DIR,
    live_pair: Optional[list[str]] = None,
    init_holistic: bool = True,
) -> GazeStack:
    # `tracking.webcam` đưa thẳng device cho ultralytics, nên "auto"/"intel" phải đổi về
    # giá trị torch hợp lệ ("0" trên CUDA, "cpu" trên iGPU/CPU) — bản Ubuntu vẫn hay truyền "cpu".
    tracking_device = resolve_device(device).torch_device

    session_id = session_id or f"gaze_session_{int(time.time())}"
    tracking_module = PersonTrackingModule(
        PersonTrackingConfig(
            session_id=session_id,
            confidence_threshold=confidence_threshold,
            device=tracking_device,
            max_tracks=max_tracks,
        )
    )
    try:
        tracking_module.manager.create_session(session_id, restore_existing=True)
    except Exception:
        pass

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
    if init_holistic:
        holistic.__enter__()

    stack = GazeStack(session_id=session_id, tracking_module=tracking_module, holistic=holistic)

    try:
        pair = live_pair or ["student_01:student_02"]
        loaded = create_live_classifier(_ClassifierArgs(model_dir, pair), clip_id=f"live_{session_id}", structured=True)
        stack.live_classifier = loaded.classifier
        print("[INFO] ✅ Nạp thành công Causal Action Classifier (Exchange, Looking,...)")
    except Exception as e:
        stack.classifier_error = str(e)
        print(f"[WARNING] Không thể nạp Causal Action Classifier: {e}")

    return stack


def scan_frame(stack: GazeStack, frame: Any, timestamp: float) -> list[dict[str, Any]]:
    """Chạy tracking + holistic + classifier trên 1 frame; trả lời alert cho từng actor vi phạm."""
    alerts: list[dict[str, Any]] = []
    try:
        packet = stack.tracking_module.process_frame(frame)

        # Tự động gán student_id cho track chưa có tên (actor-level, theo đúng contract dự án)
        for track in packet.tracks:
            if track.is_present and not track.student_id:
                packet = stack.manager.assign_student(
                    stack.session_id,
                    track_id=track.track_id,
                    student_id=f"student_{track.track_id:02d}",
                )

        holistic_results = stack.holistic.process_packet(frame, packet)

        if stack.live_classifier is not None:
            classifications = stack.live_classifier.update(
                frame_index=packet.frame_id,
                timestamp_ms=packet.timestamp_ms,
                results=holistic_results,
            )
            stack.last_classifications = dict(classifications or {})
            for actor_id, clf in (classifications or {}).items():
                pred_class = clf.get("predicted_class", "c5")
                if pred_class in ALERT_CLASSES:
                    alerts.append(
                        {
                            "module": "pose_gaze",
                            "status": "warning",
                            "timestamp": timestamp,
                            "actor_id": actor_id,
                            "details": {
                                "action": f"{actor_id}: {ACTION_LABELS.get(pred_class, pred_class.upper())}",
                                "predicted_class": pred_class,
                                "score": clf.get("score"),
                            },
                        }
                    )
    except Exception as ex:  # giữ luồng sống sót qua frame lỗi, giống hành vi realtime
        print(f"[DEBUG GAZE] Lỗi xử lý frame: {ex}")
    return alerts

"""Luồng hành vi: tracking người -> holistic landmark -> causal classifier.

Dùng chung cho realtime (`fast_gaze_thread` trong app_webrtc.py) và offline (`run_offline.py`).
`scan_frame()` trả về *danh sách* alert (mỗi actor một alert) thay vì đẩy thẳng vào queue.
Đây là stack hành vi DUY NHẤT trong hệ thống: khung xương hiển thị được vẽ lại từ
`latest_packet`/`latest_results` bằng `GazeStack.draw_skeleton()`, không chạy model lần hai.

Realtime truyền `isolate_holistic=True`: MediaPipe giữ GIL khi suy luận nên nó được đẩy sang
tiến trình riêng (`holistic_worker.RemoteHolistic`), nhờ vậy luồng video không bị chặn.
Offline (chạy theo lô, không ai xem trực tiếp) giữ nguyên chạy tại chỗ.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from backend.core.device import resolve_device
from backend.ai_services.pose_gaze.pose_gaze.holistic.landmark import HolisticLandmarkExtractor
from backend.ai_services.pipeline.holistic_worker import RemoteHolistic
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
    # HolisticLandmarkExtractor, hoặc RemoteHolistic khi chạy MediaPipe ở tiến trình riêng.
    holistic: Any
    live_classifier: Optional[Any] = None
    classifier_error: Optional[str] = None
    # Phân loại mới nhất của *mọi* actor (kể cả c5) — offline dùng để báo cáo theo từng người.
    last_classifications: dict[str, Any] = field(default_factory=dict)
    # Frame gần nhất đã suy luận: UI vẽ lại khung xương từ đây (không suy luận thêm lần nữa).
    latest_packet: Optional[Any] = None
    latest_results: Optional[list[Any]] = None
    # Toạ độ landmark/bbox là PIXEL của frame đã suy luận. Chrome có thể đổi độ phân giải gửi
    # lên giữa chừng (960x540 <-> 640x360), nên phải so kích thước trước khi vẽ — vẽ kết quả
    # của frame cũ lên frame mới cùng toạ độ pixel = khung xương lệch xa khỏi người.
    latest_at: float = 0.0
    latest_frame_size: Optional[tuple[int, int]] = None      # của `latest_results`
    latest_packet_size: Optional[tuple[int, int]] = None     # của `latest_packet`
    #: Mốc bắt đầu "không còn track nào hiện diện" (0 = đang có người).
    latest_no_person_since: float = 0.0

    @property
    def manager(self) -> Any:
        return self.tracking_module.manager

    #: Chỉ bỏ vẽ khi kết quả đã quá cũ (luồng hành vi chết hẳn). Để rộng tay: tick 200-500 ms
    #: là bình thường, ngưỡng 0.75 s trước đây làm khung xương nhấp nháy tắt/bật.
    SKELETON_MAX_AGE_S = 3.0
    #: Tracker ở 8 fps đặt `is_present=False` chỉ vì MỘT tick trượt detection (track chỉ bị
    #: xoá sau 30 tick, ~3.75 s). Nếu xoá khung xương ngay tick đó thì ở 20 fps khung hình
    #: hiển thị sẽ tắt ~125 ms rồi bật lại -> nhìn như nhấp nháy liên tục. Chỉ xoá khi người
    #: vắng thật sự quá ngưỡng này; đủ ngắn để không "đóng băng" khung xương sau khi họ đi.
    NO_PERSON_GRACE_S = 0.75
    #: Cache (khoá, results, packet) đã scale — chỉ tính lại khi có lần suy luận mới hoặc
    #: kích thước khung hình hiển thị đổi, không phải mỗi frame.
    _scale_cache: Optional[tuple[Any, Any, Any]] = field(
        default=None, init=False, repr=False, compare=False
    )

    def draw_skeleton(self, frame: Any) -> None:
        """overlays.draw() gọi hàm này trên frame hiển thị — chỉ vẽ, không chạy model."""
        results, packet = self._drawables_for(frame)
        if not results and not packet:
            return
        try:
            if results:
                self.holistic.draw_results(frame, results)
            if packet:
                self.tracking_module.draw_tracks(frame, packet)
        except Exception:  # frame hiển thị khác kích thước/định dạng thì bỏ qua, không làm chết callback
            pass

    def _drawables_for(self, frame: Any) -> tuple[Any, Any]:
        """(results, packet) đã quy về kích thước frame hiển thị hiện tại.

        Toạ độ landmark/bbox là pixel của frame đã suy luận. Chrome có thể đổi độ phân giải
        gửi lên giữa chừng (960x540 <-> 640x360), nên phải scale theo tỉ lệ thay vì vẽ nguyên
        toạ độ (khung xương lệch xa khỏi người) hoặc bỏ vẽ (khung xương biến mất).
        """
        if not self.latest_results and not self.latest_packet:
            return None, None
        if self.latest_at and time.monotonic() - self.latest_at > self.SKELETON_MAX_AGE_S:
            return None, None
        if not hasattr(frame, "shape"):
            return self.latest_results, self.latest_packet

        height, width = frame.shape[:2]
        key = (self.latest_at, self.latest_frame_size, self.latest_packet_size, width, height)
        cached = self._scale_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]

        results = self._scaled_results(width, height)
        packet = self._scaled_packet(width, height)
        self._scale_cache = (key, results, packet)
        return results, packet

    def _scaled_results(self, width: int, height: int) -> Any:
        if not self.latest_results:
            return self.latest_results
        sx, sy = _factors(self.latest_frame_size, width, height)
        if sx == 1.0 and sy == 1.0:
            return self.latest_results
        scaled = [_scale_result(r, sx, sy) for r in self.latest_results]
        return tuple(r for r in scaled if r is not None) or self.latest_results

    def _scaled_packet(self, width: int, height: int) -> Any:
        if self.latest_packet is None:
            return None
        sx, sy = _factors(self.latest_packet_size, width, height)
        if sx == 1.0 and sy == 1.0:
            return self.latest_packet
        return _scale_packet(self.latest_packet, sx, sy) or self.latest_packet

    def close(self) -> None:
        try:
            self.holistic.__exit__(None, None, None)
        except Exception:
            pass


def _holistic_kwargs() -> dict[str, Any]:
    """Tham số MediaPipe Holistic dùng chung cho cả realtime (tiến trình riêng) và offline."""
    return dict(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        soft_landmark_confidence=0.2,
        crop_padding=0.15,
        face_hold_frames=3,
    )


def build_stack(
    *,
    session_id: Optional[str] = None,
    device: str = "cpu",
    confidence_threshold: float = 0.5,
    max_tracks: int = 2,
    model_dir: Path = DEFAULT_ACTION_ARTIFACT_DIR,
    live_pair: Optional[list[str]] = None,
    init_holistic: bool = True,
    isolate_holistic: bool = False,
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

    # `isolate_holistic=True` -> MediaPipe chạy ở tiến trình riêng (không giành GIL với video).
    holistic: Any = (
        RemoteHolistic(extractor_kwargs=_holistic_kwargs())
        if isolate_holistic
        else HolisticLandmarkExtractor(**_holistic_kwargs())
    )
    if init_holistic:
        if isolate_holistic:
            holistic.start()
        else:
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


def _factors(size: Optional[tuple[int, int]], width: int, height: int) -> tuple[float, float]:
    """Tỉ lệ scale từ kích thước frame đã suy luận sang frame đang hiển thị."""
    if not size or size[0] <= 0 or size[1] <= 0:
        return 1.0, 1.0
    return width / float(size[0]), height / float(size[1])


def _scale_bbox(bbox: Any, sx: float, sy: float) -> Any:
    """BoundingBox pixel -> pixel theo tỉ lệ khung hình; lỗi kiểu dữ liệu thì giữ nguyên."""
    if bbox is None:
        return bbox
    try:
        return replace(bbox, x1=bbox.x1 * sx, y1=bbox.y1 * sy, x2=bbox.x2 * sx, y2=bbox.y2 * sy)
    except Exception:
        return bbox


def _scale_frame_points(points: Any, sx: float, sy: float) -> Any:
    """Scale `frame_x/frame_y` (pixel gốc). KHÔNG đụng `x/y` (crop-normalized) hay world."""
    if not points:
        return points
    scaled = []
    for point in points:
        if point.frame_x is None or point.frame_y is None:
            scaled.append(point)
            continue
        try:
            scaled.append(replace(point, frame_x=point.frame_x * sx, frame_y=point.frame_y * sy))
        except Exception:
            scaled.append(point)
    return tuple(scaled)


def _scale_result(result: Any, sx: float, sy: float) -> Any:
    if result is None:
        return None
    try:
        return replace(
            result,
            bbox=_scale_bbox(result.bbox, sx, sy),
            crop_bbox=_scale_bbox(getattr(result, "crop_bbox", None), sx, sy),
            pose_landmarks=_scale_frame_points(result.pose_landmarks, sx, sy),
            left_hand_landmarks=_scale_frame_points(result.left_hand_landmarks, sx, sy),
            right_hand_landmarks=_scale_frame_points(result.right_hand_landmarks, sx, sy),
            selected_face_landmarks=_scale_frame_points(result.selected_face_landmarks, sx, sy),
        )
    except Exception:
        return None


def _scale_packet(packet: Any, sx: float, sy: float) -> Any:
    if packet is None:
        return None
    try:
        tracks = tuple(replace(t, bbox=_scale_bbox(t.bbox, sx, sy)) for t in packet.tracks)
        return replace(packet, tracks=tracks)
    except Exception:
        return None


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
        # Giữ khung xương LIÊN TỤC: một tick có track nhưng Holistic không trả landmark (tiến
        # trình con đang dựng lại, timeout, frame mờ) thì giữ kết quả gần nhất thay vì xoá —
        # nếu không, khung xương nhấp nháy tắt/bật theo từng tick lỗi. Chỉ xoá khi người đã
        # vắng quá NO_PERSON_GRACE_S để khung xương không "đóng băng" sau khi họ đi.
        frame_size = (frame.shape[1], frame.shape[0]) if hasattr(frame, "shape") else None
        has_present_track = any(track.is_present for track in packet.tracks)
        now = time.monotonic()
        stack.latest_packet = packet
        stack.latest_packet_size = frame_size
        if holistic_results:
            stack.latest_results = holistic_results
            stack.latest_at = now
            stack.latest_frame_size = frame_size
            stack.latest_no_person_since = 0.0
        elif has_present_track:
            pass  # Holistic rỗng nhưng người vẫn còn: giữ landmark của tick trước
        elif not stack.latest_no_person_since:
            stack.latest_no_person_since = now
        elif now - stack.latest_no_person_since >= stack.NO_PERSON_GRACE_S:
            stack.latest_results = ()
            stack.latest_at = now

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

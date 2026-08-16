"""Reusable webcam person-detection and tracking module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, time
from typing import Any

from .detectors import PersonDetector, UltralyticsPersonDetector
from .manager import TrackingManager
from .schemas import TrackPacket


PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True, slots=True)
class PersonTrackingConfig:
    """Configuration for person detection and per-session tracking."""

    model_path: str | Path | None = None
    storage_root: str | Path | None = None
    session_id: str = field(
        default_factory=lambda: TrackingManager.generate_session_id("tracking")
    )
    restore_session: bool = False
    confidence_threshold: float = 0.50
    device: str | None = None
    max_tracks: int = 2
    min_iou: float = 0.20
    max_missed_frames: int = 15


class ProcessingRateController:
    """Set an integer full-loop FPS limit and report measured FPS."""

    def __init__(self, target_fps: float = 10.0) -> None:
        self._target_fps = 1
        self.target_fps = target_fps
        self._last_started_at: float | None = None
        self._measured_fps = 0.0

    @property
    def target_fps(self) -> int:
        return self._target_fps

    @target_fps.setter
    def target_fps(self, value: float) -> None:
        self._target_fps = max(1, min(60, round(float(value))))

    @property
    def measured_fps(self) -> float:
        return self._measured_fps

    @property
    def frame_interval(self) -> float:
        return 1.0 / self._target_fps

    def should_process(self, now: float | None = None) -> bool:
        current = monotonic() if now is None else now
        if self._last_started_at is None:
            return True
        return current - self._last_started_at >= self.frame_interval

    def remaining_seconds(
        self,
        frame_started_at: float,
        now: float | None = None,
    ) -> float:
        current = monotonic() if now is None else now
        return max(0.0, self.frame_interval - (current - frame_started_at))

    def mark_processed(self, started_at: float, finished_at: float | None = None) -> None:
        finished = monotonic() if finished_at is None else finished_at
        if self._last_started_at is not None:
            elapsed = max(1e-6, started_at - self._last_started_at)
            instantaneous_fps = 1.0 / elapsed
            if self._measured_fps == 0.0:
                self._measured_fps = instantaneous_fps
            else:
                self._measured_fps = 0.85 * self._measured_fps + 0.15 * instantaneous_fps
        self._last_started_at = started_at

    def increase(self, step: float = 1.0) -> None:
        self.target_fps += step

    def decrease(self, step: float = 1.0) -> None:
        self.target_fps -= step


class PersonTrackingModule:
    """Detect and track people while preserving a stable session contract."""

    def __init__(
        self,
        config: PersonTrackingConfig | None = None,
        *,
        detector: PersonDetector | None = None,
        manager: TrackingManager | None = None,
    ) -> None:
        self.config = config or PersonTrackingConfig()

        if detector is None:
            model_path = (
                Path(self.config.model_path)
                if self.config.model_path is not None
                else PROJECT_ROOT / "weights" / "yolov8n.pt"
            )
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Person detector weights were not found: {model_path}"
                )
            detector = UltralyticsPersonDetector(
                model_path=model_path,
                confidence_threshold=self.config.confidence_threshold,
                device=self.config.device,
            )
        self.detector = detector

        if manager is None:
            storage_root = (
                Path(self.config.storage_root)
                if self.config.storage_root is not None
                else PROJECT_ROOT / "test_data_tracking"
            )
            manager = TrackingManager(
                storage_root=storage_root,
                max_tracks=self.config.max_tracks,
                min_iou=self.config.min_iou,
                max_missed_frames=self.config.max_missed_frames,
            )
        self.manager = manager
        if self.config.restore_session:
            self.manager.restore_session(self.config.session_id)
        else:
            self.manager.create_session(self.config.session_id)
        restored_frame_id = self.manager.get_packet(self.config.session_id).frame_id
        self._frame_id = max(0, restored_frame_id)

    def process_frame(
        self,
        frame: Any,
        *,
        timestamp_ms: int | None = None,
    ) -> TrackPacket:
        """Run person detection and tracking on one OpenCV BGR frame."""

        self._frame_id += 1
        detections = self.detector.detect(frame)
        return self.manager.process_detections(
            self.config.session_id,
            frame_id=self._frame_id,
            timestamp_ms=timestamp_ms if timestamp_ms is not None else int(time() * 1000),
            detections=detections,
        )

    def draw_tracks(
        self,
        frame: Any,
        packet: TrackPacket,
        *,
        include_missing: bool = False,
    ) -> Any:
        """Draw bounding boxes and tracking/student IDs in-place."""

        try:
            import cv2
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install opencv-python to draw tracks") from error

        for track in packet.tracks:
            if not track.is_present and not include_missing:
                continue
            x1, y1, x2, y2 = track.bbox.to_list()
            if not track.is_present:
                color = (128, 128, 128)
            elif track.student_id:
                color = (0, 200, 0)
            else:
                color = (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Track {track.track_id}"
            if track.student_id:
                label += f" | {track.student_id}"
            label += f" | {track.confidence:.2f}"
            text_y = max(20, y1 - 8)
            cv2.putText(
                frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame

    def open_webcam(
        self,
        camera_index: int = 0,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> Any:
        """Open and configure an OpenCV webcam capture."""

        try:
            import cv2
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install opencv-python to use a webcam") from error

        capture = cv2.VideoCapture(camera_index)
        if width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open webcam index {camera_index}")
        return capture

    def run_webcam(
        self,
        *,
        camera_index: int = 0,
        target_fps: float = 10.0,
        width: int | None = None,
        height: int | None = None,
        window_name: str = "Person Tracking",
    ) -> None:
        """Run the standalone realtime tracking preview.

        The FPS limit applies to capture, detection, tracking, and display.
        """

        try:
            import cv2
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install opencv-python to use a webcam") from error

        capture = self.open_webcam(camera_index, width=width, height=height)
        rate = ProcessingRateController(target_fps)
        latest_packet = self.manager.get_packet(self.config.session_id)
        from .interactive import (
            WebcamInteractionController,
            pump_keyboard_until_frame_deadline,
        )

        interaction = WebcamInteractionController(
            manager=self.manager,
            session_id=self.config.session_id,
            rate=rate,
        )

        try:
            while True:
                started_at = monotonic()
                ok, frame = capture.read()
                if not ok:
                    break

                latest_packet = self.process_frame(frame)
                rate.mark_processed(started_at)
                interaction.update_tracks(latest_packet)
                packet_update = interaction.consume_packet_update()
                if packet_update is not None:
                    latest_packet = packet_update

                self.draw_tracks(frame, latest_packet)
                cv2.putText(
                    frame,
                    (
                        f"Actual FPS {rate.measured_fps:.0f} | "
                        f"Limit {rate.target_fps} FPS"
                    ),
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    interaction.overlay_lines()[0],
                    (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    interaction.overlay_lines()[1],
                    (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, frame)

                pump_keyboard_until_frame_deadline(
                    cv2,
                    frame_started_at=started_at,
                    rate=rate,
                    interaction=interaction,
                )
                packet_update = interaction.consume_packet_update()
                if packet_update is not None:
                    latest_packet = packet_update
                if interaction.quit_requested:
                    break
        finally:
            output_path = self.manager.generate_final_output(
                self.config.session_id
            )
            capture.release()
            cv2.destroyAllWindows()
            print(f"\nTracking JSON saved to: {output_path.resolve()}")

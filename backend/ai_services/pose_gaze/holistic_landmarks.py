"""MediaPipe Holistic extraction for each tracked student ROI.

Pose and both hands are retained in full. Face output is intentionally limited
to lips, face oval, forehead, and a small head-direction axis.

MediaPipe 0.10.30 removed the legacy ``mp.solutions`` package. This module
supports both the legacy API and the current MediaPipe Tasks API without
depending on MediaPipe's internal protobuf modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from time import monotonic
from typing import Any, Iterable
from urllib.request import Request, urlopen

from backend.ai_services.pose_gaze.tracking.schemas import (
    BoundingBox,
    TrackPacket,
    TrackedPerson,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASK_MODEL_PATH = (
    PROJECT_ROOT / "weights" / "mediapipe" / "holistic_landmarker.task"
)
DEFAULT_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)

# Connections are defined locally so drawing does not depend on the removed
# mediapipe.framework protobuf package or the legacy drawing utilities.
POSE_CONNECTIONS = frozenset(
    {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        (17, 19),
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        (18, 20),
        (11, 23),
        (12, 24),
        (23, 24)

        # nửa thân dưới
        # (23, 25),
        # (24, 26),
        # (25, 27),
        # (26, 28),
        # (27, 29),
        # (28, 30),
        # (29, 31),
        # (30, 32),
        # (27, 31),
        # (28, 32),
    }
)
HAND_CONNECTIONS = frozenset(
    {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    }
)
LIP_CONNECTIONS = frozenset(
    {
        (61, 146),
        (146, 91),
        (91, 181),
        (181, 84),
        (84, 17),
        (17, 314),
        (314, 405),
        (405, 321),
        (321, 375),
        (375, 291),
        (61, 185),
        (185, 40),
        (40, 39),
        (39, 37),
        (37, 0),
        (0, 267),
        (267, 269),
        (269, 270),
        (270, 409),
        (409, 291),
        (78, 95),
        (95, 88),
        (88, 178),
        (178, 87),
        (87, 14),
        (14, 317),
        (317, 402),
        (402, 318),
        (318, 324),
        (324, 308),
        (78, 191),
        (191, 80),
        (80, 81),
        (81, 82),
        (82, 13),
        (13, 312),
        (312, 311),
        (311, 310),
        (310, 415),
        (415, 308),
    }
)
FACE_OVAL_CONNECTIONS = frozenset(
    {
        (10, 338),
        (338, 297),
        (297, 332),
        (332, 284),
        (284, 251),
        (251, 389),
        (389, 356),
        (356, 454),
        (454, 323),
        (323, 361),
        (361, 288),
        (288, 397),
        (397, 365),
        (365, 379),
        (379, 378),
        (378, 400),
        (400, 377),
        (377, 152),
        (152, 148),
        (148, 176),
        (176, 149),
        (149, 150),
        (150, 136),
        (136, 172),
        (172, 58),
        (58, 132),
        (132, 93),
        (93, 234),
        (234, 127),
        (127, 162),
        (162, 21),
        (21, 54),
        (54, 103),
        (103, 67),
        (67, 109),
        (109, 10),
    }
)
FOREHEAD_CONNECTIONS = frozenset(
    {
        (10, 109),
        (109, 67),
        (67, 103),
        (103, 54),
        (10, 338),
        (338, 297),
        (297, 332),
        (332, 284),
    }
)
HEAD_DIRECTION_CONNECTIONS = frozenset(
    {
        (10, 1),  # forehead to nose tip
        (1, 152),  # nose tip to chin
        (234, 1),  # left cheek to nose
        (1, 454),  # nose to right cheek
    }
)
SELECTED_FACE_CONNECTIONS = frozenset(
    set(LIP_CONNECTIONS)
    | set(FACE_OVAL_CONNECTIONS)
    | set(FOREHEAD_CONNECTIONS)
    | set(HEAD_DIRECTION_CONNECTIONS)
)


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    """One landmark in crop-normalized and original-frame coordinates."""

    index: int
    x: float
    y: float
    z: float
    frame_x: float | None = None
    frame_y: float | None = None
    visibility: float | None = None
    presence: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "frame_x": self.frame_x,
            "frame_y": self.frame_y,
            "visibility": self.visibility,
            "presence": self.presence,
        }


@dataclass(frozen=True, slots=True)
class _LetterboxTransform:
    """Map normalized landmarks from a fixed input canvas to the source crop."""

    input_width: int
    input_height: int
    content_width: int
    content_height: int
    pad_left: int
    pad_top: int

    def to_crop_normalized(
        self,
        x: float,
        y: float,
        z: float,
    ) -> tuple[float, float, float]:
        crop_x = (x * self.input_width - self.pad_left) / self.content_width
        crop_y = (y * self.input_height - self.pad_top) / self.content_height
        # MediaPipe normalized z uses approximately the same scale as x.
        crop_z = z * self.input_width / self.content_width
        return crop_x, crop_y, crop_z


@dataclass(slots=True)
class TrackHolisticResult:
    """Serializable landmark output for one tracked person."""

    track_id: int
    student_id: str | None
    bbox: BoundingBox
    crop_bbox: BoundingBox
    pose_landmarks: tuple[LandmarkPoint, ...] = ()
    pose_world_landmarks: tuple[LandmarkPoint, ...] = ()
    left_hand_landmarks: tuple[LandmarkPoint, ...] = ()
    left_hand_world_landmarks: tuple[LandmarkPoint, ...] = ()
    right_hand_landmarks: tuple[LandmarkPoint, ...] = ()
    right_hand_world_landmarks: tuple[LandmarkPoint, ...] = ()
    selected_face_landmarks: tuple[LandmarkPoint, ...] = ()

    @property
    def has_pose(self) -> bool:
        return bool(self.pose_landmarks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "student_id": self.student_id,
            "bbox_xyxy": self.bbox.to_list(),
            "crop_bbox_xyxy": self.crop_bbox.to_list(),
            "pose_landmarks": [point.to_dict() for point in self.pose_landmarks],
            "pose_world_landmarks": [
                point.to_dict() for point in self.pose_world_landmarks
            ],
            "left_hand_landmarks": [
                point.to_dict() for point in self.left_hand_landmarks
            ],
            "left_hand_world_landmarks": [
                point.to_dict() for point in self.left_hand_world_landmarks
            ],
            "right_hand_landmarks": [
                point.to_dict() for point in self.right_hand_landmarks
            ],
            "right_hand_world_landmarks": [
                point.to_dict() for point in self.right_hand_world_landmarks
            ],
            "selected_face_landmarks": [
                point.to_dict() for point in self.selected_face_landmarks
            ],
        }


class HolisticLandmarkExtractor:
    """Run one MediaPipe Holistic temporal context per tracking ID."""

    def __init__(
        self,
        *,
        static_image_mode: bool = False,
        model_complexity: int = 2,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        crop_padding: float = 0.15,
        task_model_path: str | Path | None = None,
        task_model_url: str = DEFAULT_TASK_MODEL_URL,
        task_input_size: int = 512,
    ) -> None:
        if not 0.0 <= crop_padding <= 1.0:
            raise ValueError("crop_padding must be in [0, 1]")
        if task_input_size < 128:
            raise ValueError("task_input_size must be at least 128 pixels")

        try:
            import cv2
            import mediapipe as mp
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Install opencv-python and mediapipe to use "
                "HolisticLandmarkExtractor"
            ) from error

        self._cv2 = cv2
        self._mp = mp
        self._static_image_mode = static_image_mode
        self._model_complexity = model_complexity
        self._smooth_landmarks = smooth_landmarks
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._crop_padding = crop_padding
        self._task_input_size = int(task_input_size)
        self._processors: dict[int, Any] = {}
        self._last_timestamps: dict[int, int] = {}
        self._face_connections = SELECTED_FACE_CONNECTIONS
        self.selected_face_indices = frozenset(
            index for connection in self._face_connections for index in connection
        )

        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "holistic"):
            self._backend = "legacy-solutions"
            self._holistic_api = solutions.holistic
            self._task_model_path: Path | None = None
            return

        self._configure_tasks_backend(
            Path(task_model_path) if task_model_path is not None else None,
            task_model_url,
        )

    @property
    def backend(self) -> str:
        """Name of the MediaPipe API selected at runtime."""

        return self._backend

    @property
    def task_model_path(self) -> Path | None:
        """Resolved Tasks model path, or ``None`` for legacy MediaPipe."""

        return self._task_model_path

    @property
    def task_input_size(self) -> int:
        """Fixed square input size used by the MediaPipe Tasks video graph."""

        return self._task_input_size

    def _configure_tasks_backend(
        self,
        task_model_path: Path | None,
        task_model_url: str,
    ) -> None:
        try:
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                VisionTaskRunningMode,
            )
            from mediapipe.tasks.python.vision.holistic_landmarker import (
                HolisticLandmarker,
                HolisticLandmarkerOptions,
            )
        except ImportError as error:  # pragma: no cover - version dependent
            version = getattr(self._mp, "__version__", "unknown")
            raise RuntimeError(
                "MediaPipe "
                f"{version} does not provide either legacy Holistic or the "
                "current HolisticLandmarker Tasks API. Upgrade with: "
                'python -m pip install --upgrade "mediapipe>=0.10.33,<0.11"'
            ) from error

        model_path = (task_model_path or DEFAULT_TASK_MODEL_PATH).resolve()
        self._ensure_task_model(model_path, task_model_url)

        self._backend = "tasks"
        self._task_model_path = model_path
        self._task_base_options = BaseOptions
        self._task_image = Image
        self._task_image_format = ImageFormat
        self._task_running_mode = VisionTaskRunningMode
        self._task_holistic = HolisticLandmarker
        self._task_options = HolisticLandmarkerOptions

        if self._model_complexity != 2:
            LOGGER.warning(
                "--model-complexity only applies to legacy MediaPipe; "
                "the Tasks Holistic model has a fixed complexity"
            )

    @staticmethod
    def _ensure_task_model(model_path: Path, model_url: str) -> None:
        if model_path.is_file() and model_path.stat().st_size > 0:
            return

        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = model_path.with_suffix(model_path.suffix + ".download")
        LOGGER.warning(
            "MediaPipe Holistic model is missing; downloading it once to %s",
            model_path,
        )
        request = Request(model_url, headers={"User-Agent": "exam-proctoring/1.0"})
        try:
            with urlopen(request, timeout=180) as response:
                with temporary_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if temporary_path.stat().st_size == 0:
                raise RuntimeError("downloaded model is empty")
            temporary_path.replace(model_path)
        except Exception as error:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the MediaPipe Holistic model. Download "
                f"{model_url} manually to {model_path}, or pass "
                "task_model_path to HolisticLandmarkExtractor."
            ) from error

    def _new_processor(self) -> Any:
        if self._backend == "legacy-solutions":
            return self._holistic_api.Holistic(
                static_image_mode=self._static_image_mode,
                model_complexity=self._model_complexity,
                smooth_landmarks=self._smooth_landmarks,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )

        running_mode = (
            self._task_running_mode.IMAGE
            if self._static_image_mode
            else self._task_running_mode.VIDEO
        )
        options = self._task_options(
            base_options=self._task_base_options(
                model_asset_path=str(self._task_model_path)
            ),
            running_mode=running_mode,
            min_face_detection_confidence=self._min_detection_confidence,
            min_face_landmarks_confidence=self._min_tracking_confidence,
            min_pose_detection_confidence=self._min_detection_confidence,
            min_pose_landmarks_confidence=self._min_tracking_confidence,
            min_hand_landmarks_confidence=self._min_tracking_confidence,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        return self._task_holistic.create_from_options(options)

    def process_packet(
        self,
        frame: Any,
        packet: TrackPacket,
    ) -> tuple[TrackHolisticResult, ...]:
        """Extract landmarks for all visible tracks in a tracking packet."""

        packet_track_ids = {track.track_id for track in packet.tracks}
        self._close_processors_not_in(packet_track_ids)

        output: list[TrackHolisticResult] = []
        for track in packet.tracks:
            if not track.is_present:
                continue
            try:
                result = self.process_track(
                    frame,
                    track,
                    timestamp_ms=packet.timestamp_ms,
                )
            except RuntimeError:
                LOGGER.exception(
                    "MediaPipe Holistic failed for track %s; resetting its "
                    "processor and continuing the webcam loop",
                    track.track_id,
                )
                self._discard_processor(track.track_id)
                continue
            if result is not None:
                output.append(result)
        return tuple(output)

    def process_track(
        self,
        frame: Any,
        track: TrackedPerson,
        *,
        timestamp_ms: int | None = None,
    ) -> TrackHolisticResult | None:
        """Extract pose, hands, and selected face points from one track crop."""

        frame_height, frame_width = frame.shape[:2]
        crop_bbox = self._expand_and_clip_bbox(
            track.bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if crop_bbox is None:
            return None

        x1, y1, x2, y2 = crop_bbox.to_list()
        if x2 <= x1 or y2 <= y1:
            return None
        crop_bbox = BoundingBox(float(x1), float(y1), float(x2), float(y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        processor = self._processors.get(track.track_id)
        if processor is None:
            processor = self._new_processor()
            self._processors[track.track_id] = processor

        rgb_input = self._cv2.cvtColor(crop, self._cv2.COLOR_BGR2RGB)
        transform: _LetterboxTransform | None = None
        if self._backend == "tasks":
            rgb_input, transform = self._letterbox_task_input(rgb_input)
        rgb_input.flags.writeable = False
        holistic_result = self._run_processor(
            processor,
            rgb_input,
            track_id=track.track_id,
            timestamp_ms=timestamp_ms,
        )

        pose = self._normalized_points(
            holistic_result.pose_landmarks,
            crop_bbox,
            transform,
        )
        pose_world = self._world_points(holistic_result.pose_world_landmarks)
        left_hand = self._normalized_points(
            holistic_result.left_hand_landmarks,
            crop_bbox,
            transform,
        )
        left_hand_world = self._world_points(
            getattr(holistic_result, "left_hand_world_landmarks", None)
        )
        right_hand = self._normalized_points(
            holistic_result.right_hand_landmarks,
            crop_bbox,
            transform,
        )
        right_hand_world = self._world_points(
            getattr(holistic_result, "right_hand_world_landmarks", None)
        )
        full_face = self._normalized_points(
            holistic_result.face_landmarks,
            crop_bbox,
            transform,
        )
        selected_face = tuple(
            point for point in full_face if point.index in self.selected_face_indices
        )

        return TrackHolisticResult(
            track_id=track.track_id,
            student_id=track.student_id,
            bbox=track.bbox,
            crop_bbox=crop_bbox,
            pose_landmarks=pose,
            pose_world_landmarks=pose_world,
            left_hand_landmarks=left_hand,
            left_hand_world_landmarks=left_hand_world,
            right_hand_landmarks=right_hand,
            right_hand_world_landmarks=right_hand_world,
            selected_face_landmarks=selected_face,
        )

    def _letterbox_task_input(
        self,
        rgb_crop: Any,
    ) -> tuple[Any, _LetterboxTransform]:
        """Letterbox a changing person ROI to one stable Tasks input size."""

        source_height, source_width = rgb_crop.shape[:2]
        target_size = self._task_input_size
        scale = min(target_size / source_width, target_size / source_height)
        content_width = max(1, min(target_size, round(source_width * scale)))
        content_height = max(1, min(target_size, round(source_height * scale)))
        interpolation = (
            self._cv2.INTER_AREA if scale < 1.0 else self._cv2.INTER_LINEAR
        )
        resized = self._cv2.resize(
            rgb_crop,
            (content_width, content_height),
            interpolation=interpolation,
        )

        pad_left = (target_size - content_width) // 2
        pad_right = target_size - content_width - pad_left
        pad_top = (target_size - content_height) // 2
        pad_bottom = target_size - content_height - pad_top
        letterboxed = self._cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            self._cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        transform = _LetterboxTransform(
            input_width=target_size,
            input_height=target_size,
            content_width=content_width,
            content_height=content_height,
            pad_left=pad_left,
            pad_top=pad_top,
        )
        return letterboxed, transform

    def _run_processor(
        self,
        processor: Any,
        rgb_crop: Any,
        *,
        track_id: int,
        timestamp_ms: int | None,
    ) -> Any:
        if self._backend == "legacy-solutions":
            return processor.process(rgb_crop)

        mp_image = self._task_image(
            image_format=self._task_image_format.SRGB,
            data=rgb_crop,
        )
        if self._static_image_mode:
            return processor.detect(mp_image)

        timestamp = self._monotonic_timestamp(track_id, timestamp_ms)
        return processor.detect_for_video(mp_image, timestamp)

    def _monotonic_timestamp(
        self,
        track_id: int,
        timestamp_ms: int | None,
    ) -> int:
        candidate = (
            int(timestamp_ms)
            if timestamp_ms is not None
            else int(monotonic() * 1000)
        )
        previous = self._last_timestamps.get(track_id)
        if previous is not None:
            candidate = max(candidate, previous + 1)
        self._last_timestamps[track_id] = candidate
        return candidate

    def draw_results(
        self,
        frame: Any,
        results: tuple[TrackHolisticResult, ...] | list[TrackHolisticResult],
    ) -> Any:
        """Draw full pose/hands and only the selected face regions in-place."""

        for result in results:
            self._draw_landmark_set(
                frame,
                result.left_hand_landmarks,
                HAND_CONNECTIONS,
                line_color=(255, 80, 80),
                point_color=(255, 220, 80),
                thickness=2,
            )
            self._draw_landmark_set(
                frame,
                result.right_hand_landmarks,
                HAND_CONNECTIONS,
                line_color=(80, 80, 255),
                point_color=(80, 220, 255),
                thickness=2,
            )
            self._draw_landmark_set(
                frame,
                result.pose_landmarks,
                POSE_CONNECTIONS,
                line_color=(80, 220, 80),
                point_color=(0, 80, 255),
                thickness=2,
            )
            self._draw_landmark_set(
                frame,
                result.selected_face_landmarks,
                self._face_connections,
                line_color=(255, 180, 0),
                point_color=(0, 255, 255),
                thickness=1,
                radius=1,
            )
        return frame

    def _draw_landmark_set(
        self,
        frame: Any,
        points: Iterable[LandmarkPoint],
        connections: Iterable[tuple[int, int]],
        *,
        line_color: tuple[int, int, int],
        point_color: tuple[int, int, int],
        thickness: int,
        radius: int = 2,
    ) -> None:
        indexed = {
            point.index: point
            for point in points
            if point.frame_x is not None and point.frame_y is not None
        }
        if not indexed:
            return

        for start_index, end_index in connections:
            start = indexed.get(start_index)
            end = indexed.get(end_index)
            if start is None or end is None:
                continue
            self._cv2.line(
                frame,
                (round(start.frame_x), round(start.frame_y)),
                (round(end.frame_x), round(end.frame_y)),
                line_color,
                thickness,
                self._cv2.LINE_AA,
            )

        for point in indexed.values():
            self._cv2.circle(
                frame,
                (round(point.frame_x), round(point.frame_y)),
                radius,
                point_color,
                -1,
                self._cv2.LINE_AA,
            )

    def _close_processors_not_in(self, track_ids: set[int]) -> None:
        for track_id in tuple(self._processors):
            if track_id in track_ids:
                continue
            self._discard_processor(track_id)

    def _discard_processor(self, track_id: int) -> None:
        processor = self._processors.pop(track_id, None)
        self._last_timestamps.pop(track_id, None)
        if processor is None:
            return
        try:
            processor.close()
        except Exception:
            # A failed native graph can also throw during close. Do not mask
            # the original inference error or terminate the realtime preview.
            LOGGER.warning(
                "MediaPipe processor for track %s also failed while closing",
                track_id,
                exc_info=True,
            )

    def close(self) -> None:
        for track_id in tuple(self._processors):
            self._discard_processor(track_id)

    def __enter__(self) -> "HolisticLandmarkExtractor":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _expand_and_clip_bbox(
        self,
        bbox: BoundingBox,
        *,
        frame_width: int,
        frame_height: int,
    ) -> BoundingBox | None:
        horizontal_padding = bbox.width * self._crop_padding
        vertical_padding = bbox.height * self._crop_padding
        x1 = max(0.0, bbox.x1 - horizontal_padding)
        y1 = max(0.0, bbox.y1 - vertical_padding)
        x2 = min(float(frame_width), bbox.x2 + horizontal_padding)
        y2 = min(float(frame_height), bbox.y2 + vertical_padding)
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1, y1, x2, y2)

    @staticmethod
    def _optional_landmark_value(landmark: Any, name: str) -> float | None:
        value = getattr(landmark, name, None)
        return float(value) if value is not None else None

    @staticmethod
    def _landmarks_sequence(landmark_list: Any) -> tuple[Any, ...]:
        if landmark_list is None:
            return ()
        landmarks = getattr(landmark_list, "landmark", landmark_list)
        return tuple(landmarks)

    def _normalized_points(
        self,
        landmark_list: Any,
        crop_bbox: BoundingBox,
        transform: _LetterboxTransform | None = None,
    ) -> tuple[LandmarkPoint, ...]:
        output: list[LandmarkPoint] = []
        for index, landmark in enumerate(
            self._landmarks_sequence(landmark_list)
        ):
            x = float(landmark.x)
            y = float(landmark.y)
            z = float(landmark.z)
            if transform is not None:
                x, y, z = transform.to_crop_normalized(x, y, z)
            output.append(
                LandmarkPoint(
                    index=index,
                    x=x,
                    y=y,
                    z=z,
                    frame_x=crop_bbox.x1 + x * crop_bbox.width,
                    frame_y=crop_bbox.y1 + y * crop_bbox.height,
                    visibility=self._optional_landmark_value(
                        landmark, "visibility"
                    ),
                    presence=self._optional_landmark_value(landmark, "presence"),
                )
            )
        return tuple(output)

    def _world_points(self, landmark_list: Any) -> tuple[LandmarkPoint, ...]:
        return tuple(
            LandmarkPoint(
                index=index,
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                visibility=self._optional_landmark_value(landmark, "visibility"),
                presence=self._optional_landmark_value(landmark, "presence"),
            )
            for index, landmark in enumerate(
                self._landmarks_sequence(landmark_list)
            )
        )

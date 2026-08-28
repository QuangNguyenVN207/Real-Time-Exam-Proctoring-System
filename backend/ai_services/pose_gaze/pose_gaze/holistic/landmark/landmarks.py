"""MediaPipe Holistic landmark extraction for each tracked student ROI.

Only upper-body pose, both hands, lips, forehead, and a small head-direction
axis are retained. Unused model landmarks are never materialized or serialized.

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
from types import SimpleNamespace
from typing import Any, Callable, Collection, Iterable
from urllib.request import Request, urlopen

from ...settings import (
    DEFAULT_HOLISTIC_CONFIDENCE,
    DEFAULT_HOLISTIC_SOFT_CONFIDENCE,
    PROJECT_ROOT,
    configure_mediapipe_logging,
)
from ...tracking.schemas import (
    BoundingBox,
    TrackPacket,
    TrackedPerson,
)
from .face_mapping import OneEuroFilter2D, fit_pose_anchor_transform


LOGGER = logging.getLogger(__name__)
configure_mediapipe_logging()
DEFAULT_TASK_MODEL_PATH = (
    PROJECT_ROOT / "weights" / "mediapipe" / "holistic_landmarker.task"
)
DEFAULT_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)
DEFAULT_FACE_TASK_MODEL_PATH = PROJECT_ROOT / "weights" / "mediapipe" / "face_landmarker.task"
DEFAULT_FACE_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
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
        (23, 24),
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
LEFT_EYE_CONNECTIONS = frozenset(
    {(33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133)}
)
RIGHT_EYE_CONNECTIONS = frozenset(
    {(362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390), (390, 249), (249, 263)}
)
NOSE_CONNECTIONS = frozenset({(1, 2), (2, 98), (98, 327), (327, 326), (326, 1)})
LEFT_EAR_CONNECTIONS = frozenset({(234, 93), (93, 132), (132, 58), (58, 172)})
RIGHT_EAR_CONNECTIONS = frozenset({(454, 323), (323, 361), (361, 288), (288, 397)})
FEATURE_FACE_CONNECTIONS = frozenset(
    set(LIP_CONNECTIONS)
    | set(LEFT_EYE_CONNECTIONS)
    | set(RIGHT_EYE_CONNECTIONS)
    | set(NOSE_CONNECTIONS)
    | set(LEFT_EAR_CONNECTIONS)
    | set(RIGHT_EAR_CONNECTIONS)
)
SELECTED_FACE_CONNECTIONS = frozenset(
    set(FEATURE_FACE_CONNECTIONS)
    | set(FOREHEAD_CONNECTIONS)
    | set(HEAD_DIRECTION_CONNECTIONS)
)
POSE_LANDMARK_INDICES = frozenset(
    index for connection in POSE_CONNECTIONS for index in connection
)
HAND_LANDMARK_INDICES = frozenset(
    index for connection in HAND_CONNECTIONS for index in connection
)
SELECTED_FACE_LANDMARK_INDICES = frozenset(
    index for connection in SELECTED_FACE_CONNECTIONS for index in connection
)
MOUTH_LANDMARK_INDICES = frozenset(
    index for connection in LIP_CONNECTIONS for index in connection
)


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    """One landmark in crop-normalized and original-frame coordinates."""

    index: int
    x: float | None = None
    y: float | None = None
    frame_x: float | None = None
    frame_y: float | None = None
    world_x: float | None = None
    world_y: float | None = None
    visibility: float | None = None
    presence: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        output = {
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "frame_x": self.frame_x,
            "frame_y": self.frame_y,
            "visibility": self.visibility,
            "presence": self.presence,
        }
        if self.world_x is not None and self.world_y is not None:
            output["world_x"] = self.world_x
            output["world_y"] = self.world_y
        return output


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
    ) -> tuple[float, float]:
        crop_x = (x * self.input_width - self.pad_left) / self.content_width
        crop_y = (y * self.input_height - self.pad_top) / self.content_height
        return crop_x, crop_y


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
    face_valid: bool = False
    mouth_valid: bool = False
    face_predicted: bool = False
    face_world_valid: bool = False
    face_anchor_source: str | None = None

    @property
    def has_pose(self) -> bool:
        return any(
            point.x is not None and point.y is not None
            for point in self.pose_landmarks
        )

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
            "face_valid": self.face_valid,
            "mouth_valid": self.mouth_valid,
            "face_predicted": self.face_predicted,
            "face_world_valid": self.face_world_valid,
            "face_anchor_source": self.face_anchor_source,
        }


class HolisticLandmarkExtractor:
    """Run one MediaPipe Holistic temporal context per tracking ID."""

    def __init__(
        self,
        *,
        static_image_mode: bool = False,
        model_complexity: int = 2,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = DEFAULT_HOLISTIC_CONFIDENCE,
        face_detection_confidence: float = 0.25,
        min_tracking_confidence: float = DEFAULT_HOLISTIC_CONFIDENCE,
        soft_landmark_confidence: float = DEFAULT_HOLISTIC_SOFT_CONFIDENCE,
        crop_padding: float = 0.15,
        crop_stabilization_alpha: float = 1.0,
        # Face holding is disabled by default. A held face is a prediction,
        # not an observation, and must never silently enter the canonical
        # landmark artifact.
        face_hold_frames: int = 0,
        face_fallback_model_path: str | Path | None = None,
        face_fallback_after_frames: int = 2,
        face_fallback_interval: int = 10,
        task_model_path: str | Path | None = None,
        task_model_url: str = DEFAULT_TASK_MODEL_URL,
        task_input_size: int = 512,
        processor_factory: Callable[[], Any] | None = None,
    ) -> None:
        if not 0.0 <= crop_padding <= 1.0:
            raise ValueError("crop_padding must be in [0, 1]")
        if not 0.0 < crop_stabilization_alpha <= 1.0:
            raise ValueError("crop_stabilization_alpha must be in (0, 1]")
        if face_hold_frames < 0:
            raise ValueError("face_hold_frames must be non-negative")
        if face_fallback_after_frames < 1 or face_fallback_interval < 1:
            raise ValueError("face fallback frame settings must be positive")
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be in [0, 1]")
        if not 0.0 <= face_detection_confidence <= 1.0:
            raise ValueError("face_detection_confidence must be in [0, 1]")
        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be in [0, 1]")
        if not 0.0 <= soft_landmark_confidence <= min_tracking_confidence:
            raise ValueError(
                "soft_landmark_confidence must be in [0, min_tracking_confidence]"
            )
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
        self._face_detection_confidence = face_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._soft_landmark_confidence = soft_landmark_confidence
        self._crop_padding = crop_padding
        self._crop_stabilization_alpha = crop_stabilization_alpha
        self._face_hold_frames = face_hold_frames
        self._face_fallback_model_path = (
            Path(face_fallback_model_path).resolve()
            if face_fallback_model_path is not None
            else None
        )
        self._face_fallback_after_frames = face_fallback_after_frames
        self._face_fallback_interval = face_fallback_interval
        self._face_fallback_processor: Any | None = None
        self._face_fallback_timestamp = -1
        self._task_input_size = int(task_input_size)
        self._processor_factory = processor_factory
        self._processors: dict[int, Any] = {}
        self._crop_bboxes: dict[int, BoundingBox] = {}
        self._last_face_landmarks: dict[int, tuple[LandmarkPoint, ...]] = {}
        self._smoothed_face_landmarks: dict[int, tuple[LandmarkPoint, ...]] = {}
        self._face_filters: dict[int, dict[int, OneEuroFilter2D]] = {}
        self._hand_filters: dict[
            int, dict[str, dict[int, OneEuroFilter2D]]
        ] = {}
        self._pose_filters: dict[int, dict[str, dict[int, OneEuroFilter2D]]] = {}
        self._last_face_bboxes: dict[int, BoundingBox] = {}
        self._face_missing_frames: dict[int, int] = {}
        self._last_timestamps: dict[int, int] = {}
        self._face_connections = SELECTED_FACE_CONNECTIONS
        self.selected_face_indices = SELECTED_FACE_LANDMARK_INDICES

        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "holistic"):
            self._backend = "legacy-solutions"
            self._holistic_api = solutions.holistic
            self._task_model_path: Path | None = None
            self._face_fallback_model_path = None
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
                'python -m pip install --upgrade "mediapipe==0.10.33"'
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

        if self._face_fallback_model_path is not None:
            self._ensure_task_model(
                self._face_fallback_model_path,
                DEFAULT_FACE_TASK_MODEL_URL,
            )

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
        if self._processor_factory is not None:
            return self._processor_factory()
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
            min_face_detection_confidence=self._face_detection_confidence,
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

        packet_track_ids = {
            track.track_id for track in packet.tracks if track.is_present
        }
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
        crop_bbox = self._stabilize_crop_bbox(track.track_id, crop_bbox)

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
        original_rgb_crop = rgb_input
        transform: _LetterboxTransform | None = None
        face_transform = transform
        face_crop_bbox = crop_bbox
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
            included_indices=POSE_LANDMARK_INDICES,
        )
        pose_world = self._world_points(
            holistic_result.pose_world_landmarks,
            included_indices=POSE_LANDMARK_INDICES,
        )
        pose, pose_world = self._smooth_pose_points(
            track.track_id,
            pose,
            pose_world,
            crop_bbox=crop_bbox,
            timestamp_ms=timestamp_ms,
        )
        face_fallback = self._run_face_fallback(
            original_rgb_crop,
            track_id=track.track_id,
            timestamp_ms=timestamp_ms,
            face_landmarks=holistic_result.face_landmarks,
            pose_landmarks=holistic_result.pose_landmarks,
            base_result=holistic_result,
            crop_bbox=crop_bbox,
            pose_transform=transform,
        )
        if face_fallback is not None:
            face_result, face_crop_bbox, face_transform = face_fallback
            holistic_result = face_result
            self._last_face_bboxes[track.track_id] = face_crop_bbox
        left_hand = self._normalized_points(
            holistic_result.left_hand_landmarks,
            crop_bbox,
            transform,
            included_indices=HAND_LANDMARK_INDICES,
        )
        left_hand_world = self._world_points(
            getattr(holistic_result, "left_hand_world_landmarks", None),
            included_indices=HAND_LANDMARK_INDICES,
        )
        right_hand = self._normalized_points(
            holistic_result.right_hand_landmarks,
            crop_bbox,
            transform,
            included_indices=HAND_LANDMARK_INDICES,
        )
        right_hand_world = self._world_points(
            getattr(holistic_result, "right_hand_world_landmarks", None),
            included_indices=HAND_LANDMARK_INDICES,
        )
        left_hand, left_hand_world = self._smooth_hand_points(
            track.track_id,
            "left",
            left_hand,
            left_hand_world,
            crop_bbox=crop_bbox,
            timestamp_ms=timestamp_ms,
        )
        right_hand, right_hand_world = self._smooth_hand_points(
            track.track_id,
            "right",
            right_hand,
            right_hand_world,
            crop_bbox=crop_bbox,
            timestamp_ms=timestamp_ms,
        )
        # Holistic can emit a complete-looking hand from weak/no hand
        # evidence.  Do not render or serialize it unless the hand wrist is
        # geometrically supported by the corresponding pose wrist.
        left_hand, left_hand_world = self._validate_hand_support(
            pose, left_hand, left_hand_world, pose_wrist_index=15
        )
        right_hand, right_hand_world = self._validate_hand_support(
            pose, right_hand, right_hand_world, pose_wrist_index=16
        )
        left_hand, left_hand_world, right_hand, right_hand_world = (
            self._reject_duplicate_hands(
                pose,
                left_hand,
                left_hand_world,
                right_hand,
                right_hand_world,
            )
        )
        observed_face = self._normalized_points(
            holistic_result.face_landmarks,
            face_crop_bbox,
            face_transform,
            included_indices=self.selected_face_indices,
            apply_confidence_gate=False,
        )
        face_valid = any(point.x is not None for point in observed_face)
        mouth_valid = any(
            point.x is not None and point.index in MOUTH_LANDMARK_INDICES
            for point in observed_face
        )
        face_predicted = False
        face_world_valid = False
        face_anchor_source: str | None = None
        if face_valid:
            smoothed_face = self._smooth_face_points(
                track.track_id,
                observed_face,
                timestamp_ms=timestamp_ms,
            )
            selected_face = self._remap_frame_points_to_crop(
                smoothed_face,
                crop_bbox,
            )
            selected_face, face_world_valid, face_anchor_source = (
                self._map_face_to_pseudo_world(
                    selected_face,
                    pose,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )
            self._last_face_landmarks[track.track_id] = selected_face
            self._last_face_bboxes[track.track_id] = face_crop_bbox
            self._face_missing_frames[track.track_id] = 0
        else:
            missing_frames = self._face_missing_frames.get(track.track_id, 0) + 1
            self._face_missing_frames[track.track_id] = missing_frames
            held_face = self._last_face_landmarks.get(track.track_id, ())
            if held_face and missing_frames <= self._face_hold_frames:
                selected_face = self._remap_frame_points_to_crop(
                    held_face,
                    crop_bbox,
                )
                face_predicted = True
                mouth_valid = any(
                    point.x is not None and point.index in MOUTH_LANDMARK_INDICES
                    for point in selected_face
                )
            else:
                selected_face = observed_face
                if held_face and missing_frames > self._face_hold_frames:
                    self._last_face_landmarks.pop(track.track_id, None)
                    self._last_face_bboxes.pop(track.track_id, None)
                    self._smoothed_face_landmarks.pop(track.track_id, None)
                    self._face_filters.pop(track.track_id, None)

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
            face_valid=face_valid,
            mouth_valid=mouth_valid,
            face_predicted=face_predicted,
            face_world_valid=face_world_valid,
            face_anchor_source=face_anchor_source,
        )

    @staticmethod
    def _map_face_to_pseudo_world(
        points: Collection[LandmarkPoint],
        pose: Collection[LandmarkPoint],
        *,
        frame_width: int,
        frame_height: int,
    ) -> tuple[tuple[LandmarkPoint, ...], bool, str | None]:
        pose_points = {
            point.index: (point.frame_x / frame_width, point.frame_y / frame_height)
            for point in pose
            if point.frame_x is not None and point.frame_y is not None
        }
        fitted = fit_pose_anchor_transform(pose_points)
        if fitted is None:
            return tuple(points), False, None
        transform, anchor_source = fitted
        mapped: list[LandmarkPoint] = []
        valid = False
        for point in points:
            if point.frame_x is None or point.frame_y is None:
                mapped.append(point)
                continue
            world = transform.apply(
                (point.frame_x / frame_width, point.frame_y / frame_height)
            )
            if world is None:
                mapped.append(point)
                continue
            valid = True
            mapped.append(
                LandmarkPoint(
                    index=point.index,
                    x=point.x,
                    y=point.y,
                    frame_x=point.frame_x,
                    frame_y=point.frame_y,
                    world_x=world[0],
                    world_y=world[1],
                    visibility=point.visibility,
                    presence=point.presence,
                )
            )
        return tuple(mapped), valid, anchor_source

    @staticmethod
    def _remap_frame_points_to_crop(
        points: Collection[LandmarkPoint],
        crop_bbox: BoundingBox,
    ) -> tuple[LandmarkPoint, ...]:
        if crop_bbox.width <= 0.0 or crop_bbox.height <= 0.0:
            return tuple(points)
        remapped: list[LandmarkPoint] = []
        for point in points:
            if point.frame_x is None or point.frame_y is None:
                remapped.append(point)
                continue
            x = (point.frame_x - crop_bbox.x1) / crop_bbox.width
            y = (point.frame_y - crop_bbox.y1) / crop_bbox.height
            remapped.append(
                LandmarkPoint(
                    index=point.index,
                    x=x,
                    y=y,
                    frame_x=point.frame_x,
                    frame_y=point.frame_y,
                    visibility=point.visibility,
                    presence=point.presence,
                )
            )
        return tuple(remapped)

    def _smooth_face_points(
        self,
        track_id: int,
        points: Collection[LandmarkPoint],
        *,
        timestamp_ms: int | None,
    ) -> tuple[LandmarkPoint, ...]:
        filters = self._face_filters.setdefault(track_id, {})
        timestamp = float(timestamp_ms if timestamp_ms is not None else monotonic() * 1000.0)
        smoothed: list[LandmarkPoint] = []
        for point in points:
            if point.frame_x is None or point.frame_y is None:
                smoothed.append(point)
                continue
            filter_ = filters.setdefault(point.index, OneEuroFilter2D())
            frame_x, frame_y = filter_.update(
                (point.frame_x, point.frame_y),
                timestamp,
            )
            smoothed.append(
                LandmarkPoint(
                    index=point.index,
                    frame_x=frame_x,
                    frame_y=frame_y,
                    visibility=point.visibility,
                    presence=point.presence,
                )
            )
        result = tuple(smoothed)
        self._smoothed_face_landmarks[track_id] = result
        return result

    def _smooth_hand_points(
        self,
        track_id: int,
        side: str,
        points: Collection[LandmarkPoint],
        world_points: Collection[LandmarkPoint],
        *,
        crop_bbox: BoundingBox,
        timestamp_ms: int | None,
    ) -> tuple[tuple[LandmarkPoint, ...], tuple[LandmarkPoint, ...]]:
        """Smooth observed hand points without filling missing observations."""

        side_filters = self._hand_filters.setdefault(track_id, {}).setdefault(side, {})
        filters = side_filters.setdefault("frame", {})
        world_filters = side_filters.setdefault("world", {})
        timestamp = float(timestamp_ms if timestamp_ms is not None else monotonic() * 1000.0)
        observed_indices = {point.index for point in points if point.frame_x is not None}
        for index in tuple(filters):
            if index not in observed_indices:
                filters[index].reset()

        smoothed: list[LandmarkPoint] = []
        for point in points:
            if point.frame_x is None or point.frame_y is None:
                smoothed.append(point)
                continue
            filter_ = filters.setdefault(point.index, OneEuroFilter2D())
            frame_x, frame_y = filter_.update(
                (point.frame_x, point.frame_y),
                timestamp,
            )
            x = (frame_x - crop_bbox.x1) / crop_bbox.width
            y = (frame_y - crop_bbox.y1) / crop_bbox.height
            smoothed.append(
                LandmarkPoint(
                    index=point.index,
                    x=x,
                    y=y,
                    frame_x=frame_x,
                    frame_y=frame_y,
                    visibility=point.visibility,
                    presence=point.presence,
                )
            )

        observed_world_indices = {
            point.index for point in world_points if point.x is not None
        }
        for index in tuple(world_filters):
            if index not in observed_world_indices:
                world_filters[index].reset()
        smoothed_world: list[LandmarkPoint] = []
        for point in world_points:
            if point.x is None or point.y is None:
                smoothed_world.append(point)
                continue
            filter_ = world_filters.setdefault(point.index, OneEuroFilter2D())
            world_x, world_y = filter_.update((point.x, point.y), timestamp)
            smoothed_world.append(
                LandmarkPoint(
                    index=point.index,
                    x=world_x,
                    y=world_y,
                    visibility=point.visibility,
                    presence=point.presence,
                )
            )
        return tuple(smoothed), tuple(smoothed_world)

    @staticmethod
    def _validate_hand_support(
        pose: Collection[LandmarkPoint],
        hand: tuple[LandmarkPoint, ...],
        world_hand: tuple[LandmarkPoint, ...],
        *,
        pose_wrist_index: int,
    ) -> tuple[tuple[LandmarkPoint, ...], tuple[LandmarkPoint, ...]]:
        """Reject unsupported hand predictions before render/feature export."""
        visible = [
            point for point in hand
            if point.frame_x is not None and point.frame_y is not None
        ]
        pose_by_index = {point.index: point for point in pose}
        pose_wrist = pose_by_index.get(pose_wrist_index)
        pose_left = pose_by_index.get(11)
        pose_right = pose_by_index.get(12)
        hand_wrist = next((point for point in visible if point.index == 0), None)
        if (
            not hand_wrist
            or not pose_wrist
            or not pose_left
            or not pose_right
            or pose_wrist.frame_x is None
            or pose_wrist.frame_y is None
            or pose_left.frame_x is None
            or pose_left.frame_y is None
            or pose_right.frame_x is None
            or pose_right.frame_y is None
        ):
            return (), ()
        shoulder_width = ((pose_left.frame_x - pose_right.frame_x) ** 2 +
                          (pose_left.frame_y - pose_right.frame_y) ** 2) ** 0.5
        wrist_distance = ((hand_wrist.frame_x - pose_wrist.frame_x) ** 2 +
                          (hand_wrist.frame_y - pose_wrist.frame_y) ** 2) ** 0.5
        if shoulder_width <= 1.0 or wrist_distance > shoulder_width * 0.35:
            return (), ()
        if len(visible) < 4:
            return (), ()
        return hand, world_hand

    @staticmethod
    def _reject_duplicate_hands(
        pose: Collection[LandmarkPoint],
        left_hand: tuple[LandmarkPoint, ...],
        left_world: tuple[LandmarkPoint, ...],
        right_hand: tuple[LandmarkPoint, ...],
        right_world: tuple[LandmarkPoint, ...],
    ) -> tuple[
        tuple[LandmarkPoint, ...],
        tuple[LandmarkPoint, ...],
        tuple[LandmarkPoint, ...],
        tuple[LandmarkPoint, ...],
    ]:
        """Suppress a second hand when Holistic emits a duplicate overlay.

        This keeps observed coordinates untouched.  Only an ambiguous hand set
        is invalidated, using bbox overlap plus proximity and pose-wrist support.
        """
        left = [p for p in left_hand if p.frame_x is not None and p.frame_y is not None]
        right = [p for p in right_hand if p.frame_x is not None and p.frame_y is not None]
        if not left or not right:
            return left_hand, left_world, right_hand, right_world

        def bounds(points: list[LandmarkPoint]) -> tuple[float, float, float, float]:
            return (
                min(p.frame_x for p in points), max(p.frame_x for p in points),
                min(p.frame_y for p in points), max(p.frame_y for p in points),
            )

        lb = bounds(left)
        rb = bounds(right)
        intersection = max(0.0, min(lb[1], rb[1]) - max(lb[0], rb[0])) * max(
            0.0, min(lb[3], rb[3]) - max(lb[2], rb[2])
        )
        left_area = max(1.0, (lb[1] - lb[0]) * (lb[3] - lb[2]))
        right_area = max(1.0, (rb[1] - rb[0]) * (rb[3] - rb[2]))
        union = left_area + right_area - intersection
        iou = intersection / union
        lc = ((lb[0] + lb[1]) / 2.0, (lb[2] + lb[3]) / 2.0)
        rc = ((rb[0] + rb[1]) / 2.0, (rb[2] + rb[3]) / 2.0)
        center_distance = ((lc[0] - rc[0]) ** 2 + (lc[1] - rc[1]) ** 2) ** 0.5
        scale = max(lb[1] - lb[0], lb[3] - lb[2], rb[1] - rb[0], rb[3] - rb[2])
        if iou < 0.45 or center_distance > 0.35 * max(scale, 1.0):
            return left_hand, left_world, right_hand, right_world

        pose_by_index = {point.index: point for point in pose}

        def wrist_distance(points: list[LandmarkPoint], pose_index: int) -> float:
            hand_wrist = next((p for p in points if p.index == 0), None)
            pose_wrist = pose_by_index.get(pose_index)
            if not hand_wrist or not pose_wrist or pose_wrist.frame_x is None:
                return float("inf")
            return ((hand_wrist.frame_x - pose_wrist.frame_x) ** 2 +
                    (hand_wrist.frame_y - pose_wrist.frame_y) ** 2) ** 0.5

        left_score = wrist_distance(left, 15)
        right_score = wrist_distance(right, 16)
        if left_score <= right_score:
            return left_hand, left_world, tuple(LandmarkPoint(index=p.index) for p in right_hand), tuple(LandmarkPoint(index=p.index) for p in right_world)
        return tuple(LandmarkPoint(index=p.index) for p in left_hand), tuple(LandmarkPoint(index=p.index) for p in left_world), right_hand, right_world

    def _smooth_pose_points(
        self,
        track_id: int,
        points: Collection[LandmarkPoint],
        world_points: Collection[LandmarkPoint],
        *,
        crop_bbox: BoundingBox,
        timestamp_ms: int | None,
    ) -> tuple[tuple[LandmarkPoint, ...], tuple[LandmarkPoint, ...]]:
        """Smooth observed pose points without filling missing observations."""

        filters = self._pose_filters.setdefault(track_id, {})
        frame_filters = filters.setdefault("frame", {})
        world_filters = filters.setdefault("world", {})
        timestamp = float(timestamp_ms if timestamp_ms is not None else monotonic() * 1000.0)

        observed_indices = {point.index for point in points if point.frame_x is not None}
        for index in tuple(frame_filters):
            if index not in observed_indices:
                frame_filters[index].reset()

        smoothed: list[LandmarkPoint] = []
        for point in points:
            if point.frame_x is None or point.frame_y is None:
                smoothed.append(point)
                continue
            filter_ = frame_filters.setdefault(
                point.index,
                OneEuroFilter2D(min_cutoff=2.0, beta=0.08),
            )
            frame_x, frame_y = filter_.update(
                (point.frame_x, point.frame_y),
                timestamp,
            )
            smoothed.append(
                LandmarkPoint(
                    index=point.index,
                    x=(frame_x - crop_bbox.x1) / crop_bbox.width,
                    y=(frame_y - crop_bbox.y1) / crop_bbox.height,
                    frame_x=frame_x,
                    frame_y=frame_y,
                    visibility=point.visibility,
                    presence=point.presence,
                )
            )

        observed_world_indices = {point.index for point in world_points if point.x is not None}
        for index in tuple(world_filters):
            if index not in observed_world_indices:
                world_filters[index].reset()
        smoothed_world: list[LandmarkPoint] = []
        for point in world_points:
            if point.x is None or point.y is None:
                smoothed_world.append(point)
                continue
            filter_ = world_filters.setdefault(
                point.index,
                OneEuroFilter2D(min_cutoff=2.0, beta=0.08),
            )
            world_x, world_y = filter_.update((point.x, point.y), timestamp)
            smoothed_world.append(
                LandmarkPoint(
                    index=point.index,
                    x=world_x,
                    y=world_y,
                    visibility=point.visibility,
                    presence=point.presence,
                )
            )
        return tuple(smoothed), tuple(smoothed_world)

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

    def _run_face_fallback(
        self,
        rgb_input: Any,
        *,
        track_id: int,
        timestamp_ms: int | None,
        face_landmarks: Any,
        pose_landmarks: Any,
        base_result: Any,
        crop_bbox: BoundingBox,
        pose_transform: _LetterboxTransform | None,
    ) -> tuple[Any, BoundingBox, _LetterboxTransform | None] | None:
        """Run standalone Face Landmarker only after repeated face loss."""

        if self._face_fallback_model_path is None:
            return None
        if self._landmarks_sequence(face_landmarks):
            return None
        missing_frames = self._face_missing_frames.get(track_id, 0) + 1
        if missing_frames < self._face_fallback_after_frames:
            return None
        if (missing_frames - self._face_fallback_after_frames) % self._face_fallback_interval:
            return None
        if self._face_fallback_processor is None:
            from mediapipe.tasks.python.vision.face_landmarker import (
                FaceLandmarker,
                FaceLandmarkerOptions,
            )

            options = FaceLandmarkerOptions(
                base_options=self._task_base_options(
                    model_asset_path=str(self._face_fallback_model_path)
                ),
                running_mode=self._task_running_mode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=self._face_detection_confidence,
                min_face_presence_confidence=self._min_tracking_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._face_fallback_processor = FaceLandmarker.create_from_options(options)
        source_height, source_width = rgb_input.shape[:2]
        pose_face_bbox = self._pose_face_roi(
            pose_landmarks,
            crop_bbox=crop_bbox,
            transform=pose_transform,
            source_width=source_width,
            source_height=source_height,
        )
        previous_face_bbox = self._last_face_bboxes.get(track_id)
        face_bbox = self._union_face_rois(
            pose_face_bbox,
            previous_face_bbox,
            crop_bbox=crop_bbox,
            source_width=source_width,
            source_height=source_height,
        )
        if face_bbox is None:
            face_height = max(1, round(source_height * 0.70))
            face_input = rgb_input[:face_height, :]
            face_bbox = BoundingBox(
                crop_bbox.x1,
                crop_bbox.y1,
                crop_bbox.x2,
                crop_bbox.y1 + crop_bbox.height * 0.70,
            )
        else:
            local_x1 = max(0, round(face_bbox.x1 - crop_bbox.x1))
            local_y1 = max(0, round(face_bbox.y1 - crop_bbox.y1))
            local_x2 = min(source_width, round(face_bbox.x2 - crop_bbox.x1))
            local_y2 = min(source_height, round(face_bbox.y2 - crop_bbox.y1))
            face_input = rgb_input[local_y1:local_y2, local_x1:local_x2]
            if face_input.size == 0:
                return None
        face_input, face_transform = self._letterbox_task_input(face_input)
        image = self._task_image(self._task_image_format.SRGB, face_input)
        result = self._face_fallback_processor.detect(image)
        if not getattr(result, "face_landmarks", None):
            return None
        return SimpleNamespace(
            pose_landmarks=base_result.pose_landmarks,
            pose_world_landmarks=base_result.pose_world_landmarks,
            left_hand_landmarks=base_result.left_hand_landmarks,
            left_hand_world_landmarks=getattr(base_result, "left_hand_world_landmarks", None),
            right_hand_landmarks=base_result.right_hand_landmarks,
            right_hand_world_landmarks=getattr(base_result, "right_hand_world_landmarks", None),
            face_landmarks=result.face_landmarks[0],
        ), face_bbox, face_transform

    def _pose_face_roi(
        self,
        pose_landmarks: Any,
        *,
        crop_bbox: BoundingBox,
        transform: _LetterboxTransform | None,
        source_width: int,
        source_height: int,
    ) -> BoundingBox | None:
        points = tuple(self._landmarks_sequence(pose_landmarks))
        if len(points) <= 12:
            return None

        nose = points[0]
        left_shoulder = points[11]
        right_shoulder = points[12]

        def to_local(point: Any) -> tuple[float, float]:
            x = float(point.x)
            y = float(point.y)
            if transform is not None:
                x, y = transform.to_crop_normalized(x, y)
            return x * crop_bbox.width, y * crop_bbox.height

        nose_x, nose_y = to_local(nose)
        left_x, left_y = to_local(left_shoulder)
        right_x, right_y = to_local(right_shoulder)
        shoulder_width = abs(right_x - left_x)
        if shoulder_width < 8.0:
            return None

        center_x = nose_x
        center_y = nose_y + shoulder_width * 0.12
        roi_width = shoulder_width * 1.00
        roi_height = roi_width * 1.70
        x1 = max(0.0, center_x - roi_width * 0.5)
        y1 = max(0.0, center_y - roi_height * 0.42)
        x2 = min(float(source_width), center_x + roi_width * 0.5)
        y2 = min(float(source_height), center_y + roi_height * 0.58)
        if x2 - x1 < 8.0 or y2 - y1 < 8.0:
            return None
        return BoundingBox(
            crop_bbox.x1 + x1,
            crop_bbox.y1 + y1,
            crop_bbox.x1 + x2,
            crop_bbox.y1 + y2,
        )

    @staticmethod
    def _union_face_rois(
        current: BoundingBox | None,
        previous: BoundingBox | None,
        *,
        crop_bbox: BoundingBox,
        source_width: int,
        source_height: int,
    ) -> BoundingBox | None:
        boxes = [box for box in (current, previous) if box is not None]
        if not boxes:
            return None
        x1 = max(crop_bbox.x1, min(box.x1 for box in boxes))
        y1 = max(crop_bbox.y1, min(box.y1 for box in boxes))
        x2 = min(crop_bbox.x2, max(box.x2 for box in boxes))
        y2 = min(crop_bbox.y2, max(box.y2 for box in boxes))
        x1 = max(crop_bbox.x1, min(x1, crop_bbox.x1 + source_width))
        y1 = max(crop_bbox.y1, min(y1, crop_bbox.y1 + source_height))
        x2 = min(crop_bbox.x2, max(x2, crop_bbox.x1))
        y2 = min(crop_bbox.y2, max(y2, crop_bbox.y1))
        if x2 - x1 < 8.0 or y2 - y1 < 8.0:
            return None
        return BoundingBox(x1, y1, x2, y2)

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

    def _stabilize_crop_bbox(self, track_id: int, measurement: BoundingBox) -> BoundingBox:
        previous = self._crop_bboxes.get(track_id)
        if previous is None:
            self._crop_bboxes[track_id] = measurement
            return measurement
        alpha = self._crop_stabilization_alpha
        stabilized = BoundingBox(
            x1=previous.x1 + alpha * (measurement.x1 - previous.x1),
            y1=previous.y1 + alpha * (measurement.y1 - previous.y1),
            x2=previous.x2 + alpha * (measurement.x2 - previous.x2),
            y2=previous.y2 + alpha * (measurement.y2 - previous.y2),
        )
        self._crop_bboxes[track_id] = stabilized
        return stabilized

    def _discard_processor(self, track_id: int) -> None:
        processor = self._processors.pop(track_id, None)
        self._last_timestamps.pop(track_id, None)
        self._crop_bboxes.pop(track_id, None)
        self._last_face_landmarks.pop(track_id, None)
        self._smoothed_face_landmarks.pop(track_id, None)
        self._face_filters.pop(track_id, None)
        self._hand_filters.pop(track_id, None)
        self._pose_filters.pop(track_id, None)
        self._last_face_bboxes.pop(track_id, None)
        self._face_missing_frames.pop(track_id, None)
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

    def face_trace_state(self, track_id: int) -> dict[str, int | bool]:
        """Return non-coordinate face-hold telemetry for one current track."""

        missing_frames = int(self._face_missing_frames.get(int(track_id), 0))
        return {
            "missing_frames": missing_frames,
            "hold_frames": int(self._face_hold_frames),
            "hold_expired": missing_frames > self._face_hold_frames,
        }

    def reset(self) -> None:
        """Clear every per-track landmark, smoothing, and held-face state."""

        track_ids = set(self._processors) | set(self._last_face_landmarks)
        for track_id in tuple(track_ids):
            self._discard_processor(track_id)

    def close(self) -> None:
        self.reset()
        if self._face_fallback_processor is not None:
            self._face_fallback_processor.close()
            self._face_fallback_processor = None

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
        has_field = getattr(landmark, "HasField", None)
        if callable(has_field):
            try:
                if not has_field(name):
                    return None
            except (TypeError, ValueError):
                # Some protobuf variants expose HasField but reject scalar
                # fields. Fall back to the attribute representation below.
                pass
        value = getattr(landmark, name, None)
        return float(value) if value is not None else None

    @staticmethod
    def _landmarks_sequence(landmark_list: Any) -> Iterable[Any]:
        if landmark_list is None:
            return ()
        return getattr(landmark_list, "landmark", landmark_list)

    def _normalized_points(
        self,
        landmark_list: Any,
        crop_bbox: BoundingBox,
        transform: _LetterboxTransform | None = None,
        *,
        included_indices: Collection[int] | None = None,
        apply_confidence_gate: bool = True,
    ) -> tuple[LandmarkPoint, ...]:
        output: list[LandmarkPoint] = []
        for index, landmark in enumerate(
            self._landmarks_sequence(landmark_list)
        ):
            if included_indices is not None and index not in included_indices:
                continue

            visibility = self._optional_landmark_value(landmark, "visibility")
            presence = self._optional_landmark_value(landmark, "presence")
            confidence = self._landmark_confidence(visibility, presence)
            if apply_confidence_gate and (
                confidence is not None
                and confidence < self._soft_landmark_confidence
            ):
                output.append(LandmarkPoint(index=index))
                continue

            x = float(landmark.x)
            y = float(landmark.y)
            if transform is not None:
                x, y = transform.to_crop_normalized(x, y)
            if apply_confidence_gate and (
                confidence is not None
                and confidence < self._min_tracking_confidence
            ):
                visibility = None
            output.append(
                LandmarkPoint(
                    index=index,
                    x=x,
                    y=y,
                    frame_x=crop_bbox.x1 + x * crop_bbox.width,
                    frame_y=crop_bbox.y1 + y * crop_bbox.height,
                    visibility=visibility,
                    presence=presence,
                )
            )
        return tuple(output)

    def _world_points(
        self,
        landmark_list: Any,
        *,
        included_indices: Collection[int] | None = None,
    ) -> tuple[LandmarkPoint, ...]:
        output: list[LandmarkPoint] = []
        for index, landmark in enumerate(self._landmarks_sequence(landmark_list)):
            if included_indices is not None and index not in included_indices:
                continue

            visibility = self._optional_landmark_value(landmark, "visibility")
            presence = self._optional_landmark_value(landmark, "presence")
            confidence = self._landmark_confidence(visibility, presence)
            if (
                confidence is not None
                and confidence < self._soft_landmark_confidence
            ):
                output.append(LandmarkPoint(index=index))
                continue
            if (
                confidence is not None
                and confidence < self._min_tracking_confidence
            ):
                visibility = None
            output.append(
                LandmarkPoint(
                    index=index,
                    x=float(landmark.x),
                    y=float(landmark.y),
                    visibility=visibility,
                    presence=presence,
                )
            )
        return tuple(output)

    @staticmethod
    def _landmark_confidence(
        visibility: float | None,
        presence: float | None,
    ) -> float | None:
        """Return the conservative per-point score exposed by MediaPipe."""

        available = tuple(
            value for value in (visibility, presence) if value is not None
        )
        return min(available) if available else None

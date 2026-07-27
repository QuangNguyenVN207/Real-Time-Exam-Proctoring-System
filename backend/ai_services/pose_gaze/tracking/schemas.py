"""Dependency-light data contracts shared by tracking and pose/gaze stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Image-space bounding box using ``[x1, y1, x2, y2]`` pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive width and height")

    @classmethod
    def from_xyxy(cls, values: Sequence[float]) -> "BoundingBox":
        if len(values) != 4:
            raise ValueError("bbox_xyxy must contain exactly four values")
        return cls(*(float(value) for value in values))

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        left, top = max(self.x1, other.x1), max(self.y1, other.y1)
        right, bottom = min(self.x2, other.x2), min(self.y2, other.y2)
        intersection_width = max(0.0, right - left)
        intersection_height = max(0.0, bottom - top)
        intersection = intersection_width * intersection_height
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def to_list(self) -> list[int]:
        return [round(value) for value in (self.x1, self.y1, self.x2, self.y2)]


@dataclass(frozen=True, slots=True)
class PersonDetection:
    """A single person detection emitted by any detector implementation."""

    bbox: BoundingBox
    confidence: float
    class_name: str = "person"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TrackedPerson:
    """Tracker output consumed by the later pose/gaze feature extractor."""

    track_id: int
    bbox: BoundingBox
    confidence: float
    age_frames: int
    missed_frames: int
    is_present: bool
    student_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "student_id": self.student_id,
            "bbox_xyxy": self.bbox.to_list(),
            "track_confidence": round(self.confidence, 4),
            "age_frames": self.age_frames,
            "missed_frames": self.missed_frames,
            "is_present": self.is_present,
        }


@dataclass(frozen=True, slots=True)
class TrackPacket:
    """Stable contract from tracking to Module 1 pose/gaze processing."""

    session_id: str
    frame_id: int
    timestamp_ms: int
    tracks: tuple[TrackedPerson, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "tracks": [track.to_dict() for track in self.tracks],
        }


def detection_from_dict(payload: dict[str, Any]) -> PersonDetection:
    """Convert an API/detector payload to a validated domain object."""

    return PersonDetection(
        bbox=BoundingBox.from_xyxy(payload["bbox_xyxy"]),
        confidence=float(payload["confidence"]),
        class_name=str(payload.get("class_name", "person")),
    )

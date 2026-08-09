"""Dependency-light data contracts shared by tracking and pose/gaze stages."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
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

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def diagonal(self) -> float:
        return hypot(self.width, self.height)

    def contains_point(self, point: tuple[float, float]) -> bool:
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def expanded(
        self,
        *,
        left: float = 0.0,
        right: float = 0.0,
        top: float = 0.0,
        bottom: float = 0.0,
    ) -> "BoundingBox":
        return BoundingBox(
            self.x1 - left,
            self.y1 - top,
            self.x2 + right,
            self.y2 + bottom,
        )

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
    appearance_fingerprint: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be in [0, 1]")
        if self.appearance_fingerprint is not None:
            if len(self.appearance_fingerprint) < 8:
                raise ValueError(
                    "appearance_fingerprint must contain at least 8 values"
                )
            if not all(isfinite(value) for value in self.appearance_fingerprint):
                raise ValueError(
                    "appearance_fingerprint must contain finite values"
                )


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
    appearance_identity_registered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            # ``track_id`` is the tracker's numeric handle. ``person_id`` is
            # assigned by the proctor and is the stable identity consumers
            # should display/store. Keep student_id for existing clients.
            "person_id": self.student_id,
            "student_id": self.student_id,
            "appearance_identity_registered": (
                self.appearance_identity_registered
            ),
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

    raw_fingerprint = payload.get("appearance_fingerprint")
    return PersonDetection(
        bbox=BoundingBox.from_xyxy(payload["bbox_xyxy"]),
        confidence=float(payload["confidence"]),
        class_name=str(payload.get("class_name", "person")),
        appearance_fingerprint=(
            tuple(float(value) for value in raw_fingerprint)
            if raw_fingerprint is not None
            else None
        ),
    )

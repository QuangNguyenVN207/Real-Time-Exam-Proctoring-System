"""Small, deterministic multi-person tracker for the two-student MVP.

The tracker intentionally owns only association.  A detector may be replaced with
YOLO/ByteTrack later without changing the packet passed to pose/gaze.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schemas import BoundingBox, PersonDetection, TrackedPerson


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox: BoundingBox
    confidence: float
    age_frames: int = 1
    missed_frames: int = 0
    is_present: bool = True


class IoUPersonTracker:
    """Associates up to ``max_tracks`` person detections using greedy IoU matching.

    It is sufficient for two seated students in a fixed scene and is deliberately
    isolated behind the same output contract a stronger tracker will use later.
    """

    def __init__(
        self,
        *,
        max_tracks: int = 2,
        min_iou: float = 0.20,
        max_missed_frames: int = 15,
    ) -> None:
        if max_tracks < 1:
            raise ValueError("max_tracks must be at least 1")
        if not 0.0 <= min_iou <= 1.0:
            raise ValueError("min_iou must be in [0, 1]")
        self.max_tracks = max_tracks
        self.min_iou = min_iou
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    def update(self, detections: Iterable[PersonDetection]) -> list[TrackedPerson]:
        """Update state and return all non-expired tracks for the current frame."""

        candidates = sorted(
            (detection for detection in detections if detection.class_name == "person"),
            key=lambda detection: detection.confidence,
            reverse=True,
        )
        # Extra people should not evict assigned students in this two-person MVP.
        candidates = candidates[: self.max_tracks]

        track_ids = list(self._tracks)
        matched_track_ids: set[int] = set()
        matched_detection_indexes: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for track_id in track_ids:
            for index, detection in enumerate(candidates):
                pairs.append((self._tracks[track_id].bbox.iou(detection.bbox), track_id, index))

        for overlap, track_id, index in sorted(pairs, reverse=True):
            if overlap < self.min_iou or track_id in matched_track_ids or index in matched_detection_indexes:
                continue
            track = self._tracks[track_id]
            detection = candidates[index]
            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.age_frames += 1
            track.missed_frames = 0
            track.is_present = True
            matched_track_ids.add(track_id)
            matched_detection_indexes.add(index)

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_track_ids:
                continue
            track.age_frames += 1
            track.missed_frames += 1
            track.is_present = False
            if track.missed_frames > self.max_missed_frames:
                del self._tracks[track_id]

        available_slots = self.max_tracks - len(self._tracks)
        for index, detection in enumerate(candidates):
            if index in matched_detection_indexes or available_slots <= 0:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )
            available_slots -= 1

        return self.snapshot()

    def snapshot(self) -> list[TrackedPerson]:
        return [
            TrackedPerson(
                track_id=track.track_id,
                bbox=track.bbox,
                confidence=track.confidence,
                age_frames=track.age_frames,
                missed_frames=track.missed_frames,
                is_present=track.is_present,
            )
            for track in sorted(self._tracks.values(), key=lambda value: value.track_id)
        ]

    def has_track(self, track_id: int) -> bool:
        return track_id in self._tracks
    
    def remap_track(self, current_id: int, target_id: int) -> None:
        """Đổi track_id mới về lại track_id cũ để phục vụ Re-tracking."""
        if current_id in self._tracks:
            track = self._tracks.pop(current_id)
            track.track_id = target_id
            self._tracks[target_id] = track

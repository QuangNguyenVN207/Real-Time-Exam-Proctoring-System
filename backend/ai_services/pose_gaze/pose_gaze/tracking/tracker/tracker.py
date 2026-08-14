"""Small, deterministic multi-person tracker for the two-student MVP.

The tracker intentionally owns only association.  A detector may be replaced with
YOLO/ByteTrack later without changing the packet passed to pose/gaze.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ...settings import DEFAULT_MAX_MISSED_FRAMES, DEFAULT_MIN_IOU
from ..schemas import BoundingBox, PersonDetection, TrackedPerson
from ..detectors.detectors import suppress_duplicate_detections


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
        min_iou: float = DEFAULT_MIN_IOU,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        max_center_distance_ratio: float = 1.5,
        max_area_ratio: float = 3.0,
        bbox_smoothing_alpha: float = 1.0,
    ) -> None:
        if max_tracks < 1:
            raise ValueError("max_tracks must be at least 1")
        if not 0.0 <= min_iou <= 1.0:
            raise ValueError("min_iou must be in [0, 1]")
        if max_center_distance_ratio <= 0.0:
            raise ValueError("max_center_distance_ratio must be positive")
        if max_area_ratio < 1.0:
            raise ValueError("max_area_ratio must be at least 1")
        if not 0.0 < bbox_smoothing_alpha <= 1.0:
            raise ValueError("bbox_smoothing_alpha must be in (0, 1]")
        self.max_tracks = max_tracks
        self.min_iou = min_iou
        self.max_missed_frames = max_missed_frames
        self.max_center_distance_ratio = max_center_distance_ratio
        self.max_area_ratio = max_area_ratio
        self.bbox_smoothing_alpha = bbox_smoothing_alpha
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1
        self._telemetry = {
            "frames": 0,
            "detections_seen": 0,
            "duplicates_suppressed": 0,
            "matches": 0,
            "new_tracks": 0,
            "misses": 0,
        }

    def update(self, detections: Iterable[PersonDetection]) -> list[TrackedPerson]:
        """Update state and return all non-expired tracks for the current frame."""

        raw_candidates = [detection for detection in detections if detection.class_name == "person"]
        candidates = suppress_duplicate_detections(raw_candidates)
        self._telemetry["frames"] += 1
        self._telemetry["detections_seen"] += len(raw_candidates)
        self._telemetry["duplicates_suppressed"] += len(raw_candidates) - len(candidates)

        track_ids = list(self._tracks)
        matched_track_ids: set[int] = set()
        matched_detection_indexes: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for track_id in track_ids:
            for index, detection in enumerate(candidates):
                track_box = self._tracks[track_id].bbox
                detection_box = detection.bbox
                track_center = ((track_box.x1 + track_box.x2) / 2, (track_box.y1 + track_box.y2) / 2)
                detection_center = ((detection_box.x1 + detection_box.x2) / 2, (detection_box.y1 + detection_box.y2) / 2)
                center_distance = ((track_center[0] - detection_center[0]) ** 2 + (track_center[1] - detection_center[1]) ** 2) ** 0.5
                scale = max(track_box.width, track_box.height, detection_box.width, detection_box.height)
                area_ratio = max(track_box.area, detection_box.area) / min(track_box.area, detection_box.area)
                if center_distance > self.max_center_distance_ratio * scale or area_ratio > self.max_area_ratio:
                    continue
                pairs.append((track_box.iou(detection_box), track_id, index))

        for overlap, track_id, index in sorted(pairs, reverse=True):
            if overlap < self.min_iou or track_id in matched_track_ids or index in matched_detection_indexes:
                continue
            track = self._tracks[track_id]
            detection = candidates[index]
            track.bbox = self._smooth_bbox(track.bbox, detection.bbox)
            track.confidence = detection.confidence
            track.age_frames += 1
            track.missed_frames = 0
            track.is_present = True
            matched_track_ids.add(track_id)
            matched_detection_indexes.add(index)
            self._telemetry["matches"] += 1

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_track_ids:
                continue
            track.age_frames += 1
            track.missed_frames += 1
            track.is_present = False
            self._telemetry["misses"] += 1
            if track.missed_frames > self.max_missed_frames:
                del self._tracks[track_id]

        available_slots = self.max_tracks - len(self._tracks)
        available_candidates = sorted(
            ((index, detection) for index, detection in enumerate(candidates) if index not in matched_detection_indexes),
            key=lambda item: self._foreground_score(item[1]),
            reverse=True,
        )
        for index, detection in available_candidates:
            if index in matched_detection_indexes or available_slots <= 0:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )
            self._telemetry["new_tracks"] += 1
            available_slots -= 1

        return self.snapshot()

    @staticmethod
    def _foreground_score(detection: PersonDetection) -> float:
        """Prefer confident, substantial detections when creating new tracks."""

        return detection.confidence * detection.bbox.area ** 0.5

    def _smooth_bbox(self, current: BoundingBox, measurement: BoundingBox) -> BoundingBox:
        alpha = self.bbox_smoothing_alpha
        return BoundingBox(
            x1=current.x1 + alpha * (measurement.x1 - current.x1),
            y1=current.y1 + alpha * (measurement.y1 - current.y1),
            x2=current.x2 + alpha * (measurement.x2 - current.x2),
            y2=current.y2 + alpha * (measurement.y2 - current.y2),
        )

    def telemetry(self) -> dict[str, int]:
        """Return counters useful for tracking-quality verification."""

        return dict(self._telemetry)

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

    def is_track_present(self, track_id: int) -> bool:
        """Return whether a track is visible in the current frame."""

        track = self._tracks.get(track_id)
        return track is not None and track.is_present

    def remap_track(self, current_id: int, target_id: int) -> None:
        """Move a visible re-detected track onto a previous, non-visible ID."""

        if current_id == target_id:
            return
        current = self._tracks.get(current_id)
        if current is None:
            raise KeyError(f"Unknown current track_id: {current_id}")
        if not current.is_present:
            raise ValueError(f"Current track_id {current_id} is not present")

        target = self._tracks.get(target_id)
        if target is not None and target.is_present:
            raise ValueError(f"Target track_id {target_id} is already present")

        self._tracks.pop(current_id)
        self._tracks.pop(target_id, None)
        current.track_id = target_id
        self._tracks[target_id] = current
        self._next_track_id = max(self._next_track_id, target_id + 1)

    def export_state(self) -> dict[str, Any]:
        """Return JSON-serializable state used for backend restart recovery."""

        return {
            "next_track_id": self._next_track_id,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "bbox_xyxy": [
                        track.bbox.x1,
                        track.bbox.y1,
                        track.bbox.x2,
                        track.bbox.y2,
                    ],
                    "confidence": track.confidence,
                    "age_frames": track.age_frames,
                    "missed_frames": track.missed_frames,
                    "is_present": track.is_present,
                }
                for track in sorted(self._tracks.values(), key=lambda value: value.track_id)
            ],
        }

    def restore_state(self, payload: dict[str, Any]) -> None:
        """Restore state produced by :meth:`export_state`."""

        raw_tracks = payload.get("tracks", [])
        if not isinstance(raw_tracks, list):
            raise ValueError("tracker.tracks must be a list")
        if len(raw_tracks) > self.max_tracks:
            raise ValueError("Persisted tracker state exceeds max_tracks")

        restored: dict[int, _TrackState] = {}
        for item in raw_tracks:
            if not isinstance(item, dict):
                raise ValueError("Each persisted track must be an object")
            track_id = int(item["track_id"])
            if track_id < 1 or track_id in restored:
                raise ValueError(f"Invalid persisted track_id: {track_id}")
            restored[track_id] = _TrackState(
                track_id=track_id,
                bbox=BoundingBox.from_xyxy(item["bbox_xyxy"]),
                confidence=float(item["confidence"]),
                age_frames=max(1, int(item.get("age_frames", 1))),
                missed_frames=max(0, int(item.get("missed_frames", 0))),
                # A persisted bbox is useful for re-association, but nobody is
                # present in a newly restored runtime until a fresh frame says
                # so. This prevents stale people from appearing at startup.
                is_present=False,
            )

        minimum_next_id = max(restored, default=0) + 1
        self._tracks = restored
        self._next_track_id = max(minimum_next_id, int(payload.get("next_track_id", minimum_next_id)))

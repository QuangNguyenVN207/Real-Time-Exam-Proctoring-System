"""Small, deterministic multi-person tracker for the two-student MVP.

The tracker intentionally owns only association.  A detector may be replaced with
YOLO/ByteTrack later without changing the packet passed to pose/gaze.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, hypot
from typing import Any, Iterable

from .schemas import BoundingBox, PersonDetection, TrackedPerson


def _fingerprint_similarity(
    first: tuple[float, ...] | None,
    second: tuple[float, ...] | None,
) -> float | None:
    if first is None or second is None or len(first) != len(second):
        return None
    first_norm = sum(value * value for value in first) ** 0.5
    second_norm = sum(value * value for value in second) ** 0.5
    if first_norm <= 1e-8 or second_norm <= 1e-8:
        return None
    similarity = sum(
        left * right for left, right in zip(first, second, strict=True)
    ) / (first_norm * second_norm)
    return max(-1.0, min(1.0, similarity))


def _blend_fingerprints(
    previous: tuple[float, ...],
    current: tuple[float, ...],
    current_weight: float,
) -> tuple[float, ...]:
    if len(previous) != len(current):
        return current
    blended = tuple(
        (1.0 - current_weight) * old + current_weight * new
        for old, new in zip(previous, current, strict=True)
    )
    norm = sum(value * value for value in blended) ** 0.5
    if norm <= 1e-8:
        return current
    return tuple(value / norm for value in blended)


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox: BoundingBox
    confidence: float
    age_frames: int = 1
    missed_frames: int = 0
    is_present: bool = True
    appearance_fingerprint: tuple[float, ...] | None = None
    latest_appearance_fingerprint: tuple[float, ...] | None = None


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
        max_center_distance_ratio: float = 0.75,
        max_missed_frames: int = 15,
        appearance_match_threshold: float = 0.78,
    ) -> None:
        if max_tracks < 1:
            raise ValueError("max_tracks must be at least 1")
        if not 0.0 <= min_iou <= 1.0:
            raise ValueError("min_iou must be in [0, 1]")
        if max_center_distance_ratio <= 0:
            raise ValueError("max_center_distance_ratio must be positive")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")
        if not 0.0 <= appearance_match_threshold <= 1.0:
            raise ValueError("appearance_match_threshold must be in [0, 1]")
        self.max_tracks = max_tracks
        self.min_iou = min_iou
        self.max_center_distance_ratio = max_center_distance_ratio
        self.max_missed_frames = max_missed_frames
        self.appearance_match_threshold = appearance_match_threshold
        self._tracks: dict[int, _TrackState] = {}
        # Only identities explicitly confirmed by the proctor are archived.
        # The archive survives normal track expiry for the lifetime of a session.
        self._registered_fingerprints: dict[
            int,
            list[tuple[float, ...]],
        ] = {}
        self._next_track_id = 1

    def update(self, detections: Iterable[PersonDetection]) -> list[TrackedPerson]:
        """Update state and return all non-expired tracks for the current frame."""

        candidates = sorted(
            (detection for detection in detections if detection.class_name == "person"),
            # In a classroom a small, high-confidence person in the back row
            # must not displace one of the two foreground examinees. Combining
            # confidence with box scale prioritizes the people nearest camera.
            key=lambda detection: (
                detection.confidence * detection.bbox.area**0.5,
                detection.confidence,
            ),
            reverse=True,
        )
        # Extra people should not evict assigned students in this two-person MVP.
        candidates = candidates[: self.max_tracks]

        track_ids = list(self._tracks)
        matched_track_ids: set[int] = set()
        matched_detection_indexes: set[int] = set()
        pairs: list[tuple[float, int, int]] = []

        # A strong, unambiguous registered appearance match takes precedence
        # over geometry. Otherwise a nearby temporary box could consume the
        # returning person's detection before the identity archive is checked.
        for index, detection in enumerate(candidates):
            registered_track_id = self._best_registered_identity(
                detection.appearance_fingerprint,
                excluded_ids=set(),
            )
            if registered_track_id is None:
                continue
            similarity = _fingerprint_similarity(
                self._best_gallery_fingerprint(
                    registered_track_id,
                    detection.appearance_fingerprint,
                ),
                detection.appearance_fingerprint,
            )
            if similarity is not None:
                pairs.append((10.0 + similarity, registered_track_id, index))

        for track_id in track_ids:
            for index, detection in enumerate(candidates):
                match_score = self._association_score(
                    track_id,
                    self._tracks[track_id],
                    detection.bbox,
                    detection.appearance_fingerprint,
                )
                if match_score is not None:
                    pairs.append((match_score, track_id, index))

        for _, track_id, index in sorted(pairs, reverse=True):
            if track_id in matched_track_ids or index in matched_detection_indexes:
                continue
            existing = self._tracks.get(track_id)
            if existing is None:
                self._tracks[track_id] = self._new_track(
                    track_id,
                    candidates[index],
                )
            else:
                self._update_track(existing, candidates[index])
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

        # A confirmed person can return anywhere in the frame after the old
        # geometric track has expired (or while it is merely marked missing).
        claimed_registered_ids = set(matched_track_ids)
        for index, detection in enumerate(candidates):
            if index in matched_detection_indexes:
                continue
            registered_track_id = self._best_registered_identity(
                detection.appearance_fingerprint,
                excluded_ids=claimed_registered_ids,
            )
            if registered_track_id is None:
                continue

            existing = self._tracks.get(registered_track_id)
            if existing is None:
                self._tracks[registered_track_id] = self._new_track(
                    registered_track_id,
                    detection,
                )
            else:
                self._update_track(existing, detection)
            matched_detection_indexes.add(index)
            claimed_registered_ids.add(registered_track_id)

        # Missing tracks are history, not occupied seats. Count only people
        # visible in this frame when deciding whether a temporary ID may start.
        available_slots = self.max_tracks - sum(
            track.is_present for track in self._tracks.values()
        )
        for index, detection in enumerate(candidates):
            if index in matched_detection_indexes or available_slots <= 0:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = self._new_track(track_id, detection)
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
                appearance_identity_registered=self.has_registered_identity(
                    track.track_id
                ),
            )
            for track in sorted(self._tracks.values(), key=lambda value: value.track_id)
        ]

    def has_track(self, track_id: int) -> bool:
        return track_id in self._tracks

    def is_present(self, track_id: int) -> bool:
        track = self._tracks.get(track_id)
        return track is not None and track.is_present

    def is_track_present(self, track_id: int) -> bool:
        """Backward-compatible alias used by the develop branch APIs."""

        return self.is_present(track_id)

    def has_registered_identity(self, track_id: int) -> bool:
        return track_id in self._registered_fingerprints

    def register_identity(self, track_id: int) -> bool:
        """Archive a confirmed person's appearance for automatic re-ID."""

        track = self._tracks.get(track_id)
        if track is None:
            return False
        fingerprint = (
            track.latest_appearance_fingerprint
            or track.appearance_fingerprint
        )
        if fingerprint is None:
            return False

        gallery = self._registered_fingerprints.setdefault(track_id, [])
        if not gallery:
            gallery.append(fingerprint)
            return True

        similarities = [
            _fingerprint_similarity(template, fingerprint) or -1.0
            for template in gallery
        ]
        closest_index = max(range(len(gallery)), key=similarities.__getitem__)
        closest_similarity = similarities[closest_index]
        if closest_similarity >= 0.96:
            gallery[closest_index] = _blend_fingerprints(
                gallery[closest_index],
                fingerprint,
                0.08,
            )
        elif closest_similarity < 0.92:
            # Preserve several distinct seated poses/crop scales. Keep the
            # first enrollment template and rotate later samples when full.
            if len(gallery) >= 12:
                gallery.pop(1 if len(gallery) > 1 else 0)
            gallery.append(fingerprint)
        return True

    def unregister_identity(self, track_id: int) -> None:
        self._registered_fingerprints.pop(track_id, None)

    def remap_track(self, current_id: int, target_id: int) -> None:
        """Restore a newly detected person to a previously assigned track ID."""

        if current_id == target_id:
            return
        try:
            track = self._tracks[current_id]
        except KeyError as error:
            raise ValueError(f"Unknown current track_id: {current_id}") from error

        previous = self._tracks.get(target_id)
        if previous is not None and previous.is_present:
            raise ValueError(f"target track_id {target_id} is still present")
        if previous is not None:
            del self._tracks[target_id]

        del self._tracks[current_id]
        track.track_id = target_id
        self._tracks[target_id] = track
        current_gallery = self._registered_fingerprints.pop(current_id, [])
        target_gallery = self._registered_fingerprints.get(target_id, [])
        if current_gallery:
            merged_gallery = list(target_gallery)
            for fingerprint in current_gallery:
                if not merged_gallery or max(
                    (
                        _fingerprint_similarity(template, fingerprint) or -1.0
                        for template in merged_gallery
                    ),
                    default=-1.0,
                ) < 0.92:
                    merged_gallery.append(fingerprint)
            self._registered_fingerprints[target_id] = merged_gallery[-12:]
        self._next_track_id = max(self._next_track_id, target_id + 1)

    def export_state(self) -> dict[str, Any]:
        """Return restart-safe tracker and enrolled appearance state."""

        return {
            "next_track_id": self._next_track_id,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "bbox_xyxy": track.bbox.to_list(),
                    "confidence": track.confidence,
                    "age_frames": track.age_frames,
                    "missed_frames": track.missed_frames,
                    "is_present": track.is_present,
                    "appearance_fingerprint": (
                        list(track.appearance_fingerprint)
                        if track.appearance_fingerprint is not None
                        else None
                    ),
                    "latest_appearance_fingerprint": (
                        list(track.latest_appearance_fingerprint)
                        if track.latest_appearance_fingerprint is not None
                        else None
                    ),
                }
                for track in sorted(
                    self._tracks.values(),
                    key=lambda value: value.track_id,
                )
            ],
            "registered_fingerprints": {
                str(track_id): [list(template) for template in gallery]
                for track_id, gallery in sorted(
                    self._registered_fingerprints.items()
                )
            },
        }

    def restore_state(self, payload: dict[str, Any]) -> None:
        """Restore state produced by :meth:`export_state`.

        Restored boxes are intentionally marked absent until a fresh frame
        confirms them. Enrolled appearance galleries remain available so a
        returning student can recover the same stable track ID after restart.
        """

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
            appearance = item.get("appearance_fingerprint")
            latest_appearance = item.get("latest_appearance_fingerprint")
            restored[track_id] = _TrackState(
                track_id=track_id,
                bbox=BoundingBox.from_xyxy(item["bbox_xyxy"]),
                confidence=float(item["confidence"]),
                age_frames=max(1, int(item.get("age_frames", 1))),
                missed_frames=max(0, int(item.get("missed_frames", 0))),
                is_present=False,
                appearance_fingerprint=(
                    tuple(float(value) for value in appearance)
                    if appearance is not None
                    else None
                ),
                latest_appearance_fingerprint=(
                    tuple(float(value) for value in latest_appearance)
                    if latest_appearance is not None
                    else None
                ),
            )

        raw_registered = payload.get("registered_fingerprints", {})
        if not isinstance(raw_registered, dict):
            raise ValueError("tracker.registered_fingerprints must be an object")
        registered: dict[int, list[tuple[float, ...]]] = {}
        for raw_track_id, raw_gallery in raw_registered.items():
            track_id = int(raw_track_id)
            if track_id < 1 or not isinstance(raw_gallery, list):
                raise ValueError("Invalid registered fingerprint gallery")
            gallery = [
                tuple(float(value) for value in template)
                for template in raw_gallery
                if isinstance(template, list) and template
            ]
            if gallery:
                registered[track_id] = gallery[-12:]

        minimum_next_id = max(
            [*restored, *registered],
            default=0,
        ) + 1
        self._tracks = restored
        self._registered_fingerprints = registered
        self._next_track_id = max(
            minimum_next_id,
            int(payload.get("next_track_id", minimum_next_id)),
        )

    @staticmethod
    def _update_track(
        track: _TrackState,
        detection: PersonDetection,
    ) -> None:
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.age_frames += 1
        track.missed_frames = 0
        track.is_present = True
        if detection.appearance_fingerprint is not None:
            track.latest_appearance_fingerprint = (
                detection.appearance_fingerprint
            )
            track.appearance_fingerprint = (
                detection.appearance_fingerprint
                if track.appearance_fingerprint is None
                else _blend_fingerprints(
                    track.appearance_fingerprint,
                    detection.appearance_fingerprint,
                    0.20,
                )
            )

    @staticmethod
    def _new_track(
        track_id: int,
        detection: PersonDetection,
    ) -> _TrackState:
        return _TrackState(
            track_id=track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            appearance_fingerprint=detection.appearance_fingerprint,
            latest_appearance_fingerprint=detection.appearance_fingerprint,
        )

    def _association_score(
        self,
        track_id: int,
        track: _TrackState,
        current_bbox: BoundingBox,
        current_fingerprint: tuple[float, ...] | None,
    ) -> float | None:
        geometry_score = self._match_score(track.bbox, current_bbox)
        if geometry_score is None:
            return None

        if track.is_present:
            # While a person remains continuously visible, pose changes can
            # alter the face/torso crop far more than identity does. Geometry
            # keeps that active track stable and the current appearance only
            # acts as a soft preference. Strict identity gating is reserved for
            # re-entry after at least one missed frame.
            similarity = _fingerprint_similarity(
                track.appearance_fingerprint,
                current_fingerprint,
            )
            return geometry_score + 0.25 * max(similarity or 0.0, 0.0)

        registered = self._best_gallery_fingerprint(
            track_id,
            current_fingerprint,
        )
        similarity = _fingerprint_similarity(
            registered,
            current_fingerprint,
        )
        if similarity is None:
            return geometry_score
        if similarity < self.appearance_match_threshold:
            # A different person entering the same chair must not inherit a
            # confirmed ID merely because the bounding boxes overlap.
            return None
        return geometry_score + similarity

    def _best_registered_identity(
        self,
        fingerprint: tuple[float, ...] | None,
        *,
        excluded_ids: set[int],
    ) -> int | None:
        if fingerprint is None:
            return None

        candidates: list[tuple[float, int]] = []
        for track_id in self._registered_fingerprints:
            if track_id in excluded_ids:
                continue
            similarity = _fingerprint_similarity(
                self._best_gallery_fingerprint(track_id, fingerprint),
                fingerprint,
            )
            if (
                similarity is not None
                and similarity >= self.appearance_match_threshold
            ):
                candidates.append((similarity, track_id))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.03:
            # If two registered appearances are nearly indistinguishable,
            # require manual confirmation instead of guessing an identity.
            return None
        return candidates[0][1]

    def _best_gallery_fingerprint(
        self,
        track_id: int,
        fingerprint: tuple[float, ...] | None,
    ) -> tuple[float, ...] | None:
        gallery = self._registered_fingerprints.get(track_id, [])
        if not gallery or fingerprint is None:
            return None
        return max(
            gallery,
            key=lambda template: (
                _fingerprint_similarity(template, fingerprint) or -1.0
            ),
        )

    def _match_score(
        self,
        previous: BoundingBox,
        current: BoundingBox,
    ) -> float | None:
        """Score plausible matches using overlap, motion, and box scale.

        IoU alone assigns a new ID whenever a person moves far enough that two
        consecutive boxes no longer overlap. Center distance keeps the existing
        ID through that common case while the area-ratio guard rejects boxes
        whose scale changed too drastically.
        """

        overlap = previous.iou(current)
        center_distance = hypot(
            previous.center[0] - current.center[0],
            previous.center[1] - current.center[1],
        )
        scale = max(previous.diagonal, current.diagonal, 1.0)
        normalized_distance = center_distance / scale
        area_ratio = min(previous.area, current.area) / max(
            previous.area,
            current.area,
        )

        if overlap < self.min_iou and normalized_distance > self.max_center_distance_ratio:
            return None
        if area_ratio < 0.25 and overlap < 0.35:
            return None

        center_similarity = max(
            0.0,
            1.0 - normalized_distance / self.max_center_distance_ratio,
        )
        return overlap + 0.35 * center_similarity + 0.10 * area_ratio


def person_fingerprint_from_frame(
    frame: Any,
    bbox: BoundingBox,
) -> tuple[float, ...] | None:
    """Build a same-session appearance fingerprint from face and clothing.

    This intentionally uses local image descriptors rather than assigning an
    identity by geometry. Face texture contributes most of the descriptor,
    while clothing colour/edges make re-entry more robust when the face is
    small. Low-detail crops are rejected so an unsafe identity is not guessed.
    """

    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover - runtime dependencies
        return None

    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return None
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(frame_width, floor(bbox.x1)))
    y1 = max(0, min(frame_height, floor(bbox.y1)))
    x2 = max(0, min(frame_width, ceil(bbox.x2)))
    y2 = max(0, min(frame_height, ceil(bbox.y2)))
    if x2 - x1 < 32 or y2 - y1 < 48:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    normalized_crop = cv2.resize(
        crop,
        (96, 160),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(normalized_crop, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray))
    edge_energy = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))))
    if contrast < 5.0 and edge_energy < 2.0:
        return None

    # A normal person detector box places the head in the upper half and the
    # torso below it. Central crops reduce background leakage around the body.
    face_gray = gray[0:88, 14:82]
    torso_bgr = normalized_crop[72:156, 10:86]
    torso_gray = gray[72:156, 10:86]

    face_equalized = cv2.equalizeHist(face_gray)
    face_float = face_equalized.astype(np.float32)
    face_float -= float(np.mean(face_float))
    face_float /= max(float(np.std(face_float)), 1.0)
    face_dct = cv2.dct(
        cv2.resize(face_float, (48, 48), interpolation=cv2.INTER_AREA)
    )
    face_values = face_dct[:9, :9].reshape(-1)[1:]
    face_norm = float(np.linalg.norm(face_values))
    if face_norm > 1e-8:
        face_values /= face_norm

    face_edges = cv2.Canny(face_equalized, 60, 160)
    face_edge_values = cv2.resize(
        face_edges,
        (12, 12),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32).reshape(-1)
    face_edge_values /= 255.0
    face_edge_norm = float(np.linalg.norm(face_edge_values))
    if face_edge_norm > 1e-8:
        face_edge_values /= face_edge_norm

    torso_hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
    torso_hist = cv2.calcHist(
        [torso_hsv],
        [0, 1],
        None,
        [16, 8],
        [0, 180, 0, 256],
    ).astype(np.float32).reshape(-1)
    torso_hist_norm = float(np.linalg.norm(torso_hist))
    if torso_hist_norm > 1e-8:
        torso_hist /= torso_hist_norm

    torso_edges = cv2.Canny(cv2.equalizeHist(torso_gray), 60, 160)
    torso_edge_values = cv2.resize(
        torso_edges,
        (10, 10),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32).reshape(-1)
    torso_edge_values /= 255.0
    torso_edge_norm = float(np.linalg.norm(torso_edge_values))
    if torso_edge_norm > 1e-8:
        torso_edge_values /= torso_edge_norm

    fingerprint = np.concatenate(
        (
            0.50 * face_values,
            0.20 * face_edge_values,
            0.22 * torso_hist,
            0.08 * torso_edge_values,
        )
    )
    fingerprint_norm = float(np.linalg.norm(fingerprint))
    if fingerprint_norm <= 1e-8:
        return None
    fingerprint /= fingerprint_norm
    return tuple(float(value) for value in fingerprint)

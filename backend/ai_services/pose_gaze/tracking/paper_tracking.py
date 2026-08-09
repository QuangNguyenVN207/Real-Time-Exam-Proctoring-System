"""Identity-first paper tracking for exam-paper/cheat-sheet disambiguation.

The object model is allowed to change its semantic label between frames.  This
module first associates detections by geometry, then smooths label evidence on
the resulting stable ``paper_id``.  The policy layer treats the registered exam
paper as allowed and evaluates *new physical paper tracks* as suspicious.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor, hypot, isfinite
from typing import Any, Iterable

from .schemas import BoundingBox, TrackedPerson


PAPER_CLASS_NAMES = frozenset({"cheat_sheet", "test_paper", "paper_unknown"})


def fingerprint_similarity(
    first: tuple[float, ...] | None,
    second: tuple[float, ...] | None,
) -> float | None:
    """Return cosine similarity for equal-length normalized fingerprints."""

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


@dataclass(frozen=True, slots=True)
class PaperDetection:
    bbox: BoundingBox
    confidence: float
    class_name: str
    appearance_fingerprint: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Paper detection confidence must be in [0, 1]")
        if self.class_name not in PAPER_CLASS_NAMES:
            raise ValueError(f"Unsupported paper class: {self.class_name}")
        if self.appearance_fingerprint is not None:
            if len(self.appearance_fingerprint) < 8:
                raise ValueError(
                    "appearance_fingerprint must contain at least 8 values"
                )
            if not all(isfinite(value) for value in self.appearance_fingerprint):
                raise ValueError("appearance_fingerprint must contain finite values")


@dataclass(frozen=True, slots=True)
class TrackedPaper:
    paper_id: int
    bbox: BoundingBox
    confidence: float
    raw_class_name: str
    stable_label: str
    label_confidence: float
    owner_track_id: int | None
    age_frames: int
    visible_frames: int
    missed_frames: int
    is_present: bool


@dataclass(frozen=True, slots=True)
class PaperAssessment:
    paper: TrackedPaper
    authorized: bool
    status: str
    risk_score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper.paper_id,
            "owner_track_id": self.paper.owner_track_id,
            "bbox_xyxy": self.paper.bbox.to_list(),
            "confidence": round(self.paper.confidence, 4),
            "raw_class_name": self.paper.raw_class_name,
            "stable_label": self.paper.stable_label,
            "label_confidence": round(self.paper.label_confidence, 4),
            "age_frames": self.paper.age_frames,
            "visible_frames": self.paper.visible_frames,
            "missed_frames": self.paper.missed_frames,
            "is_present": self.paper.is_present,
            "authorized": self.authorized,
            "status": self.status,
            "risk_score": self.risk_score,
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class _PaperTrackState:
    paper_id: int
    bbox: BoundingBox
    confidence: float
    raw_class_name: str
    age_frames: int = 1
    visible_frames: int = 1
    missed_frames: int = 0
    is_present: bool = True
    label_scores: dict[str, float] = field(default_factory=dict)
    owner_scores: dict[int, float] = field(default_factory=dict)
    owner_track_id: int | None = None
    owner_locked: bool = False
    appearance_fingerprint: tuple[float, ...] | None = None


class IoUPaperTracker:
    """Track physical papers even when YOLO flips their semantic class label."""

    def __init__(
        self,
        *,
        min_iou: float = 0.10,
        max_center_distance_ratio: float = 1.35,
        max_missed_frames: int = 12,
        label_score_decay: float = 0.85,
        appearance_match_threshold: float = 0.86,
        supports_test_paper: bool = False,
    ) -> None:
        if not 0.0 <= min_iou <= 1.0:
            raise ValueError("min_iou must be in [0, 1]")
        if max_center_distance_ratio <= 0:
            raise ValueError("max_center_distance_ratio must be positive")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")
        if not 0.0 <= label_score_decay <= 1.0:
            raise ValueError("label_score_decay must be in [0, 1]")
        if not 0.0 <= appearance_match_threshold <= 1.0:
            raise ValueError("appearance_match_threshold must be in [0, 1]")

        self.min_iou = min_iou
        self.max_center_distance_ratio = max_center_distance_ratio
        self.max_missed_frames = max_missed_frames
        self.label_score_decay = label_score_decay
        self.appearance_match_threshold = appearance_match_threshold
        self.supports_test_paper = supports_test_paper
        self._tracks: dict[int, _PaperTrackState] = {}
        self._registered_fingerprints: dict[int, tuple[float, ...]] = {}
        self._registered_owners: dict[int, int] = {}
        self._next_paper_id = 1

    def set_model_capabilities(self, *, supports_test_paper: bool) -> None:
        self.supports_test_paper = bool(supports_test_paper)

    def update(
        self,
        detections: Iterable[PaperDetection],
        *,
        people: Iterable[TrackedPerson] = (),
    ) -> list[TrackedPaper]:
        candidates = sorted(detections, key=lambda item: item.confidence, reverse=True)
        present_people = [person for person in people if person.is_present]
        detected_owner_ids = [
            (
                owner[0]
                if (owner := self._best_owner(detection.bbox, present_people))
                is not None
                else None
            )
            for detection in candidates
        ]
        matched_track_ids: set[int] = set()
        matched_detection_indexes: set[int] = set()
        pairs: list[tuple[float, int, int]] = []

        for paper_id, track in self._tracks.items():
            for index, detection in enumerate(candidates):
                match_score = self._association_score(
                    paper_id,
                    track,
                    detection,
                    detected_owner_id=detected_owner_ids[index],
                )
                if match_score is not None:
                    pairs.append((match_score, paper_id, index))

        for _, paper_id, index in sorted(pairs, reverse=True):
            if paper_id in matched_track_ids or index in matched_detection_indexes:
                continue
            self._update_track(self._tracks[paper_id], candidates[index])
            matched_track_ids.add(paper_id)
            matched_detection_indexes.add(index)

        for paper_id, track in list(self._tracks.items()):
            if paper_id in matched_track_ids:
                continue
            track.age_frames += 1
            track.missed_frames += 1
            track.is_present = False
            if track.missed_frames > self.max_missed_frames:
                del self._tracks[paper_id]

        claimed_registered_ids = set(matched_track_ids)
        for index, detection in enumerate(candidates):
            if index in matched_detection_indexes:
                continue
            registered_paper_id = self._best_registered_identity(
                detection,
                people=present_people,
                excluded_ids=claimed_registered_ids,
            )
            if registered_paper_id is None:
                continue

            existing = self._tracks.get(registered_paper_id)
            if existing is None:
                self._tracks[registered_paper_id] = self._new_track(
                    registered_paper_id,
                    detection,
                )
            else:
                self._update_track(existing, detection)
            matched_detection_indexes.add(index)
            claimed_registered_ids.add(registered_paper_id)

        for index, detection in enumerate(candidates):
            if index in matched_detection_indexes:
                continue
            paper_id = self._next_paper_id
            self._next_paper_id += 1
            self._tracks[paper_id] = self._new_track(paper_id, detection)

        return self.snapshot()

    def associate_owners(self, people: Iterable[TrackedPerson]) -> list[TrackedPaper]:
        present_people = [person for person in people if person.is_present]
        for track in self._tracks.values():
            if not track.is_present:
                continue
            if track.owner_locked:
                continue

            candidate = self._best_owner(track.bbox, present_people)
            if candidate is None:
                continue

            owner_track_id, score = candidate
            for known_owner in tuple(track.owner_scores):
                track.owner_scores[known_owner] *= 0.80
            track.owner_scores[owner_track_id] = track.owner_scores.get(owner_track_id, 0.0) + score
            track.owner_track_id = max(track.owner_scores, key=track.owner_scores.get)

        return self.snapshot()

    def snapshot(self) -> list[TrackedPaper]:
        return [
            self._to_tracked_paper(track)
            for track in sorted(self._tracks.values(), key=lambda item: item.paper_id)
        ]

    def has_track(self, paper_id: int) -> bool:
        return paper_id in self._tracks

    def is_present(self, paper_id: int) -> bool:
        track = self._tracks.get(paper_id)
        return track is not None and track.is_present

    def has_registered_identity(self, paper_id: int) -> bool:
        return paper_id in self._registered_fingerprints

    def remap_track(self, *, current_paper_id: int, target_paper_id: int) -> None:
        """Replace an automatic paper ID with a proctor-assigned stable ID."""

        if target_paper_id < 1:
            raise ValueError("paper_id must be at least 1")
        if current_paper_id == target_paper_id:
            return
        try:
            track = self._tracks[current_paper_id]
        except KeyError as error:
            raise ValueError(
                f"Unknown current paper_id: {current_paper_id}"
            ) from error

        previous = self._tracks.get(target_paper_id)
        if previous is not None and previous.is_present:
            raise ValueError(f"paper_id {target_paper_id} is already visible")
        if previous is not None:
            del self._tracks[target_paper_id]

        del self._tracks[current_paper_id]
        track.paper_id = target_paper_id
        self._tracks[target_paper_id] = track
        registered_fingerprint = self._registered_fingerprints.pop(
            current_paper_id,
            None,
        )
        if registered_fingerprint is not None:
            self._registered_fingerprints[target_paper_id] = (
                registered_fingerprint
            )
        registered_owner = self._registered_owners.pop(current_paper_id, None)
        if registered_owner is not None:
            self._registered_owners[target_paper_id] = registered_owner
        self._next_paper_id = max(self._next_paper_id, target_paper_id + 1)

    def register_identity(
        self,
        *,
        paper_id: int,
        owner_track_id: int | None = None,
    ) -> bool:
        """Remember an authorized paper's appearance for automatic re-ID."""

        track = self._tracks.get(paper_id)
        if track is None or track.appearance_fingerprint is None:
            return False
        current = self._registered_fingerprints.get(paper_id)
        self._registered_fingerprints[paper_id] = (
            track.appearance_fingerprint
            if current is None
            else _blend_fingerprints(current, track.appearance_fingerprint, 0.10)
        )
        resolved_owner = (
            owner_track_id
            if owner_track_id is not None
            else track.owner_track_id
        )
        if resolved_owner is not None:
            self._registered_owners[paper_id] = resolved_owner
        return True

    def assign_owner(self, *, paper_id: int, owner_track_id: int) -> None:
        """Force an owner after a proctor/dashboard manual authorization."""

        try:
            track = self._tracks[paper_id]
        except KeyError as error:
            raise ValueError(f"Unknown paper_id: {paper_id}") from error
        track.owner_track_id = owner_track_id
        track.owner_scores = {owner_track_id: 1000.0}
        track.owner_locked = True

    def remap_owner(self, *, current_track_id: int, target_track_id: int) -> None:
        """Move paper ownership when a known person is manually re-identified."""

        if current_track_id == target_track_id:
            return
        for track in self._tracks.values():
            current_score = track.owner_scores.pop(current_track_id, None)
            if current_score is not None:
                track.owner_scores[target_track_id] = max(
                    current_score,
                    track.owner_scores.get(target_track_id, 0.0),
                )
            if track.owner_track_id == current_track_id:
                track.owner_track_id = target_track_id
        for paper_id, owner_track_id in tuple(self._registered_owners.items()):
            if owner_track_id == current_track_id:
                self._registered_owners[paper_id] = target_track_id

    def export_state(self) -> dict[str, Any]:
        """Return restart-safe physical-paper identity state."""

        return {
            "next_paper_id": self._next_paper_id,
            "supports_test_paper": self.supports_test_paper,
            "tracks": [
                {
                    "paper_id": track.paper_id,
                    "bbox_xyxy": track.bbox.to_list(),
                    "confidence": track.confidence,
                    "raw_class_name": track.raw_class_name,
                    "age_frames": track.age_frames,
                    "visible_frames": track.visible_frames,
                    "missed_frames": track.missed_frames,
                    "label_scores": track.label_scores,
                    "owner_scores": {
                        str(owner): score
                        for owner, score in track.owner_scores.items()
                    },
                    "owner_track_id": track.owner_track_id,
                    "owner_locked": track.owner_locked,
                    "appearance_fingerprint": (
                        list(track.appearance_fingerprint)
                        if track.appearance_fingerprint is not None
                        else None
                    ),
                }
                for track in sorted(
                    self._tracks.values(),
                    key=lambda value: value.paper_id,
                )
            ],
            "registered_fingerprints": {
                str(paper_id): list(fingerprint)
                for paper_id, fingerprint in sorted(
                    self._registered_fingerprints.items()
                )
            },
            "registered_owners": {
                str(paper_id): owner_track_id
                for paper_id, owner_track_id in sorted(
                    self._registered_owners.items()
                )
            },
        }

    def restore_state(self, payload: dict[str, Any]) -> None:
        """Restore paper identities while requiring a new frame for presence."""

        raw_tracks = payload.get("tracks", [])
        if not isinstance(raw_tracks, list):
            raise ValueError("paper_tracker.tracks must be a list")

        tracks: dict[int, _PaperTrackState] = {}
        for item in raw_tracks:
            if not isinstance(item, dict):
                raise ValueError("Each persisted paper track must be an object")
            paper_id = int(item["paper_id"])
            if paper_id < 1 or paper_id in tracks:
                raise ValueError(f"Invalid persisted paper_id: {paper_id}")
            raw_fingerprint = item.get("appearance_fingerprint")
            raw_owner = item.get("owner_track_id")
            raw_label_scores = item.get("label_scores", {})
            raw_owner_scores = item.get("owner_scores", {})
            if not isinstance(raw_label_scores, dict) or not isinstance(
                raw_owner_scores,
                dict,
            ):
                raise ValueError("Invalid persisted paper score state")
            tracks[paper_id] = _PaperTrackState(
                paper_id=paper_id,
                bbox=BoundingBox.from_xyxy(item["bbox_xyxy"]),
                confidence=float(item["confidence"]),
                raw_class_name=str(item["raw_class_name"]),
                age_frames=max(1, int(item.get("age_frames", 1))),
                visible_frames=max(1, int(item.get("visible_frames", 1))),
                missed_frames=max(0, int(item.get("missed_frames", 0))),
                is_present=False,
                label_scores={
                    str(label): float(score)
                    for label, score in raw_label_scores.items()
                },
                owner_scores={
                    int(owner): float(score)
                    for owner, score in raw_owner_scores.items()
                },
                owner_track_id=(int(raw_owner) if raw_owner is not None else None),
                owner_locked=bool(item.get("owner_locked", False)),
                appearance_fingerprint=(
                    tuple(float(value) for value in raw_fingerprint)
                    if raw_fingerprint is not None
                    else None
                ),
            )

        raw_registered = payload.get("registered_fingerprints", {})
        raw_registered_owners = payload.get("registered_owners", {})
        if not isinstance(raw_registered, dict) or not isinstance(
            raw_registered_owners,
            dict,
        ):
            raise ValueError("Invalid persisted paper identity state")
        registered = {
            int(paper_id): tuple(float(value) for value in fingerprint)
            for paper_id, fingerprint in raw_registered.items()
        }
        registered_owners = {
            int(paper_id): int(owner_track_id)
            for paper_id, owner_track_id in raw_registered_owners.items()
        }
        minimum_next_id = max([*tracks, *registered], default=0) + 1
        self._tracks = tracks
        self._registered_fingerprints = registered
        self._registered_owners = registered_owners
        self._next_paper_id = max(
            minimum_next_id,
            int(payload.get("next_paper_id", minimum_next_id)),
        )
        self.supports_test_paper = bool(
            payload.get("supports_test_paper", self.supports_test_paper)
        )

    def _update_track(self, track: _PaperTrackState, detection: PaperDetection) -> None:
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.raw_class_name = detection.class_name
        track.age_frames += 1
        track.visible_frames += 1
        track.missed_frames = 0
        track.is_present = True
        if detection.appearance_fingerprint is not None:
            track.appearance_fingerprint = (
                detection.appearance_fingerprint
                if track.appearance_fingerprint is None
                else _blend_fingerprints(
                    track.appearance_fingerprint,
                    detection.appearance_fingerprint,
                    0.20,
                )
            )
        for class_name in tuple(track.label_scores):
            track.label_scores[class_name] *= self.label_score_decay
        track.label_scores[detection.class_name] = (
            track.label_scores.get(detection.class_name, 0.0) + detection.confidence
        )

    @staticmethod
    def _new_track(
        paper_id: int,
        detection: PaperDetection,
    ) -> _PaperTrackState:
        return _PaperTrackState(
            paper_id=paper_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            raw_class_name=detection.class_name,
            label_scores={detection.class_name: detection.confidence},
            appearance_fingerprint=detection.appearance_fingerprint,
        )

    def _match_score(self, previous: BoundingBox, current: BoundingBox) -> float | None:
        overlap = previous.iou(current)
        previous_center = previous.center
        current_center = current.center
        center_distance = hypot(
            previous_center[0] - current_center[0],
            previous_center[1] - current_center[1],
        )
        scale = max(previous.diagonal, current.diagonal, 1.0)
        normalized_distance = center_distance / scale
        area_ratio = min(previous.area, current.area) / max(previous.area, current.area)

        if overlap < self.min_iou and normalized_distance > self.max_center_distance_ratio:
            return None
        if area_ratio < 0.18 and overlap < 0.35:
            return None

        center_similarity = max(0.0, 1.0 - normalized_distance / self.max_center_distance_ratio)
        return overlap + 0.25 * center_similarity + 0.10 * area_ratio

    def _association_score(
        self,
        paper_id: int,
        track: _PaperTrackState,
        detection: PaperDetection,
        *,
        detected_owner_id: int | None,
    ) -> float | None:
        geometry_score = self._match_score(track.bbox, detection.bbox)
        if geometry_score is None:
            return None

        registered = self._registered_fingerprints.get(paper_id)
        registered_owner = self._registered_owners.get(paper_id)
        if (
            detected_owner_id is not None
            and registered_owner is not None
            and detected_owner_id != registered_owner
        ):
            return None
        similarity = fingerprint_similarity(
            registered,
            detection.appearance_fingerprint,
        )
        if similarity is None:
            return geometry_score
        if similarity < self.appearance_match_threshold:
            # A different physical sheet in the same desk position must not
            # inherit the authorized paper ID merely because its box overlaps.
            return None
        return geometry_score + similarity

    def _best_registered_identity(
        self,
        detection: PaperDetection,
        *,
        people: list[TrackedPerson],
        excluded_ids: set[int],
    ) -> int | None:
        if detection.appearance_fingerprint is None:
            return None

        owner_candidate = self._best_owner(detection.bbox, people)
        detected_owner = owner_candidate[0] if owner_candidate is not None else None
        candidates: list[tuple[float, int]] = []
        for paper_id, fingerprint in self._registered_fingerprints.items():
            if paper_id in excluded_ids:
                continue
            registered_owner = self._registered_owners.get(paper_id)
            if (
                detected_owner is not None
                and registered_owner is not None
                and detected_owner != registered_owner
            ):
                continue
            similarity = fingerprint_similarity(
                fingerprint,
                detection.appearance_fingerprint,
            )
            if (
                similarity is not None
                and similarity >= self.appearance_match_threshold
            ):
                candidates.append((similarity, paper_id))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        if (
            detected_owner is None
            and len(candidates) > 1
            and candidates[0][0] - candidates[1][0] < 0.02
        ):
            # Identical exam sheets from different students are ambiguous
            # without an owner/location hint; generating a temporary ID is safer.
            return None
        return candidates[0][1]

    @staticmethod
    def _best_owner(
        paper_bbox: BoundingBox,
        people: list[TrackedPerson],
    ) -> tuple[int, float] | None:
        paper_center = paper_bbox.center
        candidates: list[tuple[float, int]] = []
        for person in people:
            body = person.bbox
            expanded = body.expanded(
                left=0.35 * body.width,
                right=0.35 * body.width,
                top=0.10 * body.height,
                bottom=0.45 * body.height,
            )
            anchor = (body.center[0], body.y1 + 0.82 * body.height)
            distance = hypot(paper_center[0] - anchor[0], paper_center[1] - anchor[1])
            normalized_distance = distance / max(body.diagonal, 1.0)
            inside_bonus = 1.0 if expanded.contains_point(paper_center) else 0.0
            score = inside_bonus + max(0.0, 1.0 - normalized_distance)
            if inside_bonus > 0.0 or normalized_distance <= 0.80:
                candidates.append((score, person.track_id))

        if not candidates:
            return None
        score, owner_track_id = max(candidates)
        return owner_track_id, score

    def _to_tracked_paper(self, track: _PaperTrackState) -> TrackedPaper:
        stable_label, label_confidence = self._stable_label(track)
        return TrackedPaper(
            paper_id=track.paper_id,
            bbox=track.bbox,
            confidence=track.confidence,
            raw_class_name=track.raw_class_name,
            stable_label=stable_label,
            label_confidence=label_confidence,
            owner_track_id=track.owner_track_id,
            age_frames=track.age_frames,
            visible_frames=track.visible_frames,
            missed_frames=track.missed_frames,
            is_present=track.is_present,
        )

    def _stable_label(self, track: _PaperTrackState) -> tuple[str, float]:
        if not self.supports_test_paper:
            # A four-class checkpoint labels every kind of paper as cheat_sheet.
            # Expose uncertainty so policy relies on physical identity/count.
            return "paper_unknown", 0.0

        relevant_scores = {
            name: score
            for name, score in track.label_scores.items()
            if name in {"cheat_sheet", "test_paper"}
        }
        if not relevant_scores:
            return "paper_unknown", 0.0
        best_label = max(relevant_scores, key=relevant_scores.get)
        total = sum(relevant_scores.values())
        return best_label, relevant_scores[best_label] / total if total > 0 else 0.0


class PaperAuthorizationPolicy:
    """Register one allowed exam paper per owner and evaluate later paper IDs."""

    def __init__(
        self,
        *,
        registration_frames: int = 5,
        alert_confirm_frames: int = 3,
        auto_register_first_paper: bool = True,
        semantic_alert_threshold: float = 0.65,
        supports_test_paper: bool = False,
    ) -> None:
        if registration_frames < 1:
            raise ValueError("registration_frames must be at least 1")
        if alert_confirm_frames < 1:
            raise ValueError("alert_confirm_frames must be at least 1")
        if not 0.0 <= semantic_alert_threshold <= 1.0:
            raise ValueError("semantic_alert_threshold must be in [0, 1]")

        self.registration_frames = registration_frames
        self.alert_confirm_frames = alert_confirm_frames
        self.auto_register_first_paper = auto_register_first_paper
        self.semantic_alert_threshold = semantic_alert_threshold
        self.supports_test_paper = supports_test_paper
        self.armed = False
        self._authorized_by_owner: dict[int, int] = {}
        self._suspicious_counts: dict[int, int] = {}
        self._last_assessments: list[PaperAssessment] = []

    def set_model_capabilities(self, *, supports_test_paper: bool) -> None:
        self.supports_test_paper = bool(supports_test_paper)

    def arm(self) -> None:
        """Lock registration. Any new stable paper is evaluated, never auto-allowed."""

        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def register(self, *, owner_track_id: int, paper_id: int, replace: bool = False) -> None:
        current = self._authorized_by_owner.get(owner_track_id)
        if current is not None and current != paper_id and not replace:
            raise ValueError(
                f"owner track {owner_track_id} already has authorized paper_id {current}"
            )
        for owner, authorized_paper_id in tuple(self._authorized_by_owner.items()):
            if authorized_paper_id == paper_id and owner != owner_track_id:
                del self._authorized_by_owner[owner]
        self._authorized_by_owner[owner_track_id] = paper_id
        self._suspicious_counts.pop(paper_id, None)

    def unregister(self, *, paper_id: int) -> None:
        for owner, authorized_paper_id in tuple(self._authorized_by_owner.items()):
            if authorized_paper_id == paper_id:
                del self._authorized_by_owner[owner]

    def evaluate(self, papers: Iterable[TrackedPaper]) -> list[PaperAssessment]:
        paper_list = list(papers)
        self._auto_register(paper_list)
        assessments: list[PaperAssessment] = []
        present_ids = {paper.paper_id for paper in paper_list if paper.is_present}

        for paper in paper_list:
            authorized = self._is_authorized(paper)
            reasons = self._suspicion_reasons(paper, authorized, present_ids)

            if not paper.is_present:
                self._suspicious_counts[paper.paper_id] = 0
                status, risk = "missing", 0.0
            elif authorized:
                self._suspicious_counts[paper.paper_id] = 0
                status, risk = "authorized_exam_paper", 0.0
            elif reasons:
                count = self._suspicious_counts.get(paper.paper_id, 0) + 1
                self._suspicious_counts[paper.paper_id] = count
                if count >= self.alert_confirm_frames:
                    status, risk = "suspicious", 0.95
                else:
                    status, risk = "watching", 0.55
            elif paper.visible_frames < self.registration_frames:
                self._suspicious_counts[paper.paper_id] = 0
                status, risk = "registration_pending", 0.05
            else:
                self._suspicious_counts[paper.paper_id] = 0
                status, risk = "observed", 0.10

            assessments.append(
                PaperAssessment(
                    paper=paper,
                    authorized=authorized,
                    status=status,
                    risk_score=risk,
                    reasons=tuple(reasons),
                )
            )

        active_ids = {paper.paper_id for paper in paper_list}
        for paper_id in tuple(self._suspicious_counts):
            if paper_id not in active_ids:
                del self._suspicious_counts[paper_id]
        self._last_assessments = assessments
        return assessments

    def snapshot(self) -> list[PaperAssessment]:
        return list(self._last_assessments)

    def authorized_mapping(self) -> dict[int, int]:
        return dict(self._authorized_by_owner)

    def export_state(self) -> dict[str, Any]:
        return {
            "armed": self.armed,
            "authorized_by_owner": {
                str(owner_track_id): paper_id
                for owner_track_id, paper_id in sorted(
                    self._authorized_by_owner.items()
                )
            },
            "suspicious_counts": {
                str(paper_id): count
                for paper_id, count in sorted(self._suspicious_counts.items())
            },
            "supports_test_paper": self.supports_test_paper,
        }

    def restore_state(self, payload: dict[str, Any]) -> None:
        raw_authorized = payload.get("authorized_by_owner", {})
        raw_counts = payload.get("suspicious_counts", {})
        if not isinstance(raw_authorized, dict) or not isinstance(
            raw_counts,
            dict,
        ):
            raise ValueError("Invalid persisted paper policy state")
        self._authorized_by_owner = {
            int(owner): int(paper_id)
            for owner, paper_id in raw_authorized.items()
        }
        self._suspicious_counts = {
            int(paper_id): max(0, int(count))
            for paper_id, count in raw_counts.items()
        }
        self.armed = bool(payload.get("armed", False))
        self.supports_test_paper = bool(
            payload.get("supports_test_paper", self.supports_test_paper)
        )
        self._last_assessments = []

    def remap_owner(self, *, current_track_id: int, target_track_id: int) -> None:
        """Keep an authorization attached to a manually re-identified person."""

        if current_track_id == target_track_id:
            return
        current_paper_id = self._authorized_by_owner.pop(current_track_id, None)
        if (
            current_paper_id is not None
            and target_track_id not in self._authorized_by_owner
        ):
            self._authorized_by_owner[target_track_id] = current_paper_id

    def remap_paper_id(
        self,
        *,
        current_paper_id: int,
        target_paper_id: int,
    ) -> None:
        """Move authorization and alert counters to a stable manual paper ID."""

        if current_paper_id == target_paper_id:
            return

        target_already_authorized = (
            target_paper_id in self._authorized_by_owner.values()
        )
        for owner_track_id, authorized_paper_id in tuple(
            self._authorized_by_owner.items()
        ):
            if authorized_paper_id != current_paper_id:
                continue
            if target_already_authorized:
                del self._authorized_by_owner[owner_track_id]
            else:
                self._authorized_by_owner[owner_track_id] = target_paper_id
                target_already_authorized = True

        current_count = self._suspicious_counts.pop(current_paper_id, None)
        if current_count is not None:
            self._suspicious_counts[target_paper_id] = max(
                current_count,
                self._suspicious_counts.get(target_paper_id, 0),
            )

    def _auto_register(self, papers: list[TrackedPaper]) -> None:
        if self.armed or not self.auto_register_first_paper:
            return

        owner_ids = {
            paper.owner_track_id
            for paper in papers
            if paper.is_present and paper.owner_track_id is not None
        }
        for owner_track_id in owner_ids:
            if owner_track_id in self._authorized_by_owner:
                continue
            candidates = [
                paper
                for paper in papers
                if paper.is_present
                and paper.owner_track_id == owner_track_id
                and paper.visible_frames >= self.registration_frames
                and not (
                    self.supports_test_paper
                    and paper.stable_label == "cheat_sheet"
                    and paper.label_confidence >= self.semantic_alert_threshold
                )
            ]
            if not candidates:
                continue
            selected = max(candidates, key=self._registration_rank)
            self.register(owner_track_id=owner_track_id, paper_id=selected.paper_id)

    @staticmethod
    def _registration_rank(paper: TrackedPaper) -> tuple[int, int, int, float, int]:
        return (
            int(paper.stable_label == "test_paper"),
            int(paper.stable_label != "cheat_sheet"),
            paper.visible_frames,
            paper.bbox.area,
            -paper.paper_id,
        )

    def _is_authorized(self, paper: TrackedPaper) -> bool:
        return (
            paper.owner_track_id is not None
            and self._authorized_by_owner.get(paper.owner_track_id) == paper.paper_id
        )

    def _suspicion_reasons(
        self,
        paper: TrackedPaper,
        authorized: bool,
        present_ids: set[int],
    ) -> list[str]:
        if authorized or not paper.is_present:
            return []
        if paper.visible_frames < self.registration_frames:
            return []

        reasons: list[str] = []
        owner_track_id = paper.owner_track_id
        if owner_track_id is None:
            reasons.append("unassigned_paper")
        elif owner_track_id in self._authorized_by_owner:
            authorized_id = self._authorized_by_owner[owner_track_id]
            reasons.append(
                "additional_paper"
                if authorized_id in present_ids
                else "paper_replacement"
            )
        elif self.armed:
            reasons.append("unregistered_paper")

        if (
            self.supports_test_paper
            and paper.stable_label == "cheat_sheet"
            and paper.label_confidence >= self.semantic_alert_threshold
        ):
            reasons.append("classifier_cheat_sheet")
        return reasons


def paper_detection_from_dict(payload: dict[str, Any]) -> PaperDetection:
    raw_fingerprint = payload.get("appearance_fingerprint")
    return PaperDetection(
        bbox=BoundingBox.from_xyxy(payload["bbox_xyxy"]),
        confidence=float(payload["confidence"]),
        class_name=str(payload["class_name"]),
        appearance_fingerprint=(
            tuple(float(value) for value in raw_fingerprint)
            if raw_fingerprint is not None
            else None
        ),
    )


def paper_fingerprint_from_frame(
    frame: Any,
    bbox: BoundingBox,
) -> tuple[float, ...] | None:
    """Build a lighting-normalized visual fingerprint from a paper crop.

    The low-frequency DCT block captures page layout while the edge thumbnail
    captures text/diagram structure. A nearly textureless crop is intentionally
    rejected because two blank sheets cannot be identified safely from pixels.
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
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    contrast = float(np.std(gray))
    edge_energy = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))))
    if contrast < 4.0 and edge_energy < 2.0:
        return None

    equalized = cv2.equalizeHist(gray)
    normalized = equalized.astype(np.float32)
    normalized -= float(np.mean(normalized))
    normalized /= max(float(np.std(normalized)), 1.0)

    dct = cv2.dct(normalized)
    dct_values = dct[:12, :12].reshape(-1)[1:]
    dct_norm = float(np.linalg.norm(dct_values))
    if dct_norm > 1e-8:
        dct_values = dct_values / dct_norm

    edges = cv2.Canny(equalized, 60, 160)
    edge_values = cv2.resize(
        edges,
        (16, 16),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32).reshape(-1)
    edge_values /= 255.0
    edge_norm = float(np.linalg.norm(edge_values))
    if edge_norm > 1e-8:
        edge_values = edge_values / edge_norm

    fingerprint = np.concatenate(
        (0.72 * dct_values, 0.28 * edge_values),
    )
    fingerprint_norm = float(np.linalg.norm(fingerprint))
    if fingerprint_norm <= 1e-8:
        return None
    fingerprint /= fingerprint_norm
    return tuple(float(value) for value in fingerprint)

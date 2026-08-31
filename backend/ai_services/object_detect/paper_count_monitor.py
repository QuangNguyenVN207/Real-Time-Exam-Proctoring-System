"""Count-based paper monitoring without persistent paper identities.

The monitor deliberately does not create or track ``paper_id`` values.  It
deduplicates the paper boxes produced by the detector, learns the normal paper
count during setup, and reports a new cheat sheet only when a larger count is
seen for several consecutive inference frames.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import hypot
from typing import Any, Iterable


BBox = list[int]


@dataclass(slots=True)
class _CountSession:
    armed: bool = False
    setup_counts: list[int] = field(default_factory=list)
    setup_snapshots: list[list[dict[str, Any]]] = field(default_factory=list)
    baseline_count: int | None = None
    baseline_clusters: list[dict[str, Any]] = field(default_factory=list)
    stable_count: int | None = None
    candidate_count: int | None = None
    candidate_streak: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)
    active_alerts: list[dict[str, Any]] = field(default_factory=list)
    new_events: list[dict[str, Any]] = field(default_factory=list)
    event_sequence: int = 0
    last_frame_id: int = -1
    last_timestamp_ms: int = 0


class CountBasedPaperMonitor:
    """Detect additions by paper count rather than long-term paper tracking.

    ``confirmation_frames`` counts object-inference frames, not camera frames.
    Short detector flicker is therefore ignored while a real added paper is
    still confirmed quickly.
    """

    def __init__(
        self,
        *,
        confirmation_frames: int = 3,
        duplicate_overlap_threshold: float = 0.25,
        duplicate_center_distance_ratio: float = 0.90,
        match_overlap_threshold: float = 0.10,
        match_center_distance_ratio: float = 1.35,
        max_setup_samples: int = 60,
    ) -> None:
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least 1")
        self.confirmation_frames = confirmation_frames
        self.duplicate_overlap_threshold = duplicate_overlap_threshold
        self.duplicate_center_distance_ratio = (
            duplicate_center_distance_ratio
        )
        self.match_overlap_threshold = match_overlap_threshold
        self.match_center_distance_ratio = match_center_distance_ratio
        self.max_setup_samples = max_setup_samples
        self._sessions: dict[str, _CountSession] = {}

    def create_session(self, session_id: str) -> None:
        if not session_id:
            raise ValueError("session_id cannot be empty")
        self._sessions[session_id] = _CountSession()

    def ensure_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            self.create_session(session_id)

    def cleanup_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def arm(self, session_id: str) -> dict[str, Any]:
        """Freeze the setup count and begin looking for count increases."""

        session = self._get_session(session_id)
        if session.baseline_count is None:
            session.baseline_count = self._mode_count(session.setup_counts)
        session.baseline_clusters = self._best_setup_snapshot(
            session,
            session.baseline_count,
        )
        session.stable_count = session.baseline_count
        session.armed = True
        session.candidate_count = None
        session.candidate_streak = 0
        session.active_alerts = []
        session.new_events = []
        return self._state_payload(session)

    def update(
        self,
        session_id: str,
        *,
        paper_detections: Iterable[dict[str, Any]],
        people: Iterable[dict[str, Any]],
        frame_id: int,
        timestamp_ms: int,
    ) -> dict[str, Any]:
        """Consume one real object-inference result.

        The caller must not invoke this with cached detections from skipped
        frames because that would incorrectly advance the debounce counter.
        """

        session = self._get_session(session_id)
        clusters = self.cluster_papers(paper_detections)
        people_list = [dict(person) for person in people]
        observed_count = len(clusters)
        session.last_frame_id = int(frame_id)
        session.last_timestamp_ms = int(timestamp_ms)
        session.new_events = []

        if not session.armed:
            session.setup_counts.append(observed_count)
            session.setup_snapshots.append(clusters)
            if len(session.setup_counts) > self.max_setup_samples:
                session.setup_counts.pop(0)
                session.setup_snapshots.pop(0)
            session.baseline_count = self._mode_count(session.setup_counts)
            session.stable_count = session.baseline_count
            session.baseline_clusters = self._best_setup_snapshot(
                session,
                session.baseline_count,
            )
            session.observations = self._decorate_observations(
                clusters,
                [],
                people_list,
            )
            session.active_alerts = []
            session.candidate_count = None
            session.candidate_streak = 0
            return self._state_payload(session)

        baseline = session.baseline_count or 0
        stable = session.stable_count if session.stable_count is not None else baseline

        # Find current clusters unmatched with baseline positions
        unmatched_clusters = self._unmatched_current(
            session.baseline_clusters,
            clusters,
        )

        # Count baseline papers per owner (person_id or track_id)
        baseline_owner_counts: Counter[Any] = Counter()
        for b_cluster in session.baseline_clusters:
            b_owner = self._nearest_person(b_cluster["bbox_xyxy"], people_list)
            b_key = b_owner.get("person_id") or b_owner.get("track_id")
            if b_key is not None:
                baseline_owner_counts[b_key] += 1

        # Count current papers per owner
        current_owner_counts: Counter[Any] = Counter()
        for c_cluster in clusters:
            c_owner = self._nearest_person(c_cluster["bbox_xyxy"], people_list)
            c_key = c_owner.get("person_id") or c_owner.get("track_id")
            if c_key is not None:
                current_owner_counts[c_key] += 1

        # Determine if per-person count or global count exceeds baseline
        per_person_increase = any(
            c_key is not None and c_count > baseline_owner_counts.get(c_key, 1)
            for c_key, c_count in current_owner_counts.items()
        )
        has_increase = per_person_increase or observed_count > baseline

        # Debounce/streak handling for increase/decrease
        if not has_increase and observed_count <= baseline:
            if stable > baseline:
                if session.candidate_count == baseline:
                    session.candidate_streak += 1
                else:
                    session.candidate_count = baseline
                    session.candidate_streak = 1

                if session.candidate_streak >= self.confirmation_frames:
                    previous_count = stable
                    session.stable_count = baseline
                    session.candidate_count = None
                    session.candidate_streak = 0
                    session.event_sequence += 1
                    session.new_events.append(
                        {
                            "event_id": session.event_sequence,
                            "type": "paper_count_decreased",
                            "reason": "paper_count_decreased",
                            "previous_count": previous_count,
                            "current_count": baseline,
                            "removed_count": previous_count - baseline,
                            "frame_id": int(frame_id),
                            "timestamp_ms": int(timestamp_ms),
                        }
                    )
            else:
                session.candidate_count = None
                session.candidate_streak = 0
        elif session.candidate_count == observed_count or (
            has_increase and session.candidate_count is not None
        ):
            session.candidate_streak += 1
        else:
            session.candidate_count = observed_count
            session.candidate_streak = 1

        if (
            has_increase
            and session.candidate_streak >= self.confirmation_frames
        ):
            previous_count = stable
            new_stable = max(observed_count, baseline + len(unmatched_clusters), baseline + 1)
            if new_stable != previous_count:
                session.stable_count = new_stable
                if new_stable > previous_count:
                    session.event_sequence += 1
                    event_id = session.event_sequence
                    for cluster in unmatched_clusters:
                        owner = self._nearest_person(
                            cluster["bbox_xyxy"], people_list
                        )
                        session.new_events.append(
                            {
                                "event_id": event_id,
                                "type": "paper_count_increased",
                                "reason": "paper_count_increased",
                                "previous_count": previous_count,
                                "current_count": new_stable,
                                "added_count": new_stable - previous_count,
                                "bbox_xyxy": list(cluster["bbox_xyxy"]),
                                "confidence": float(cluster["confidence"]),
                                "owner_track_id": owner.get("track_id"),
                                "owner_person_id": owner.get("person_id"),
                                "frame_id": int(frame_id),
                                "timestamp_ms": int(timestamp_ms),
                            }
                        )

        if not has_increase and session.stable_count == baseline and observed_count == baseline:
            # Refresh positions only while the known-good count is visible.
            session.baseline_clusters = [dict(cluster) for cluster in clusters]

        suspicious_clusters: list[dict[str, Any]] = []
        if (session.stable_count or 0) > baseline:
            suspicious_clusters = unmatched_clusters
        elif has_increase:
            suspicious_clusters = unmatched_clusters

        session.observations = self._decorate_observations(
            clusters,
            suspicious_clusters,
            people_list,
        )
        session.active_alerts = [
            {
                "source": "paper_count",
                "label": "cheat_sheet",
                "reason": "paper_count_above_baseline",
                "risk_score": 1.0,
                **{
                    key: observation.get(key)
                    for key in (
                        "bbox_xyxy",
                        "confidence",
                        "owner_track_id",
                        "owner_person_id",
                    )
                },
            }
            for observation in session.observations
            if observation["status"] == "suspicious_new_paper"
            and (session.stable_count or 0) > baseline
        ]
        return self._state_payload(session)

    def get_state(self, session_id: str) -> dict[str, Any]:
        return self._state_payload(self._get_session(session_id))

    def cluster_papers(
        self,
        paper_detections: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Collapse duplicate/partial YOLO boxes into physical paper regions."""

        items: list[dict[str, Any]] = []
        for payload in paper_detections:
            bbox = self._valid_bbox(payload.get("bbox_xyxy"))
            if bbox is None:
                continue
            items.append(
                {
                    "bbox_xyxy": bbox,
                    "confidence": float(payload.get("confidence", 0.0)),
                    "class_names": [
                        str(payload.get("class_name", "paper_unknown"))
                    ],
                    "source_box_count": 1,
                }
            )
        if not items:
            return []

        parents = list(range(len(items)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parents[root_right] = root_left

        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                if self._boxes_refer_to_same_region(
                    items[left]["bbox_xyxy"],
                    items[right]["bbox_xyxy"],
                    overlap_threshold=self.duplicate_overlap_threshold,
                    center_ratio=self.duplicate_center_distance_ratio,
                ):
                    union(left, right)

        groups: dict[int, list[dict[str, Any]]] = {}
        for index, item in enumerate(items):
            groups.setdefault(find(index), []).append(item)

        clusters = []
        for group in groups.values():
            boxes = [item["bbox_xyxy"] for item in group]
            class_names = sorted(
                {
                    name
                    for item in group
                    for name in item["class_names"]
                }
            )
            clusters.append(
                {
                    "bbox_xyxy": [
                        min(box[0] for box in boxes),
                        min(box[1] for box in boxes),
                        max(box[2] for box in boxes),
                        max(box[3] for box in boxes),
                    ],
                    "confidence": max(
                        float(item["confidence"]) for item in group
                    ),
                    "class_names": class_names,
                    "source_box_count": len(group),
                }
            )
        return sorted(clusters, key=lambda item: item["bbox_xyxy"][0])

    def _decorate_observations(
        self,
        clusters: list[dict[str, Any]],
        suspicious_clusters: list[dict[str, Any]],
        people: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        suspicious_ids = {id(cluster) for cluster in suspicious_clusters}
        # Matching helpers return original current-cluster objects, so identity
        # is a safe frame-local marker and is not a persistent paper identity.
        observations = []
        for index, cluster in enumerate(clusters, 1):
            owner = self._nearest_person(cluster["bbox_xyxy"], people)
            suspicious = id(cluster) in suspicious_ids
            observations.append(
                {
                    "observation_index": index,
                    "bbox_xyxy": list(cluster["bbox_xyxy"]),
                    "confidence": float(cluster["confidence"]),
                    "class_names": list(cluster["class_names"]),
                    "source_box_count": int(cluster["source_box_count"]),
                    "status": (
                        "suspicious_new_paper" if suspicious else "baseline_paper"
                    ),
                    "owner_track_id": owner.get("track_id"),
                    "owner_person_id": owner.get("person_id"),
                }
            )
        return observations

    def _unmatched_current(
        self,
        previous: Iterable[dict[str, Any]],
        current: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        previous_list = list(previous)
        pairs: list[tuple[float, int, int]] = []
        for old_index, old in enumerate(previous_list):
            old_bbox = old.get("bbox_xyxy", [])
            for new_index, new in enumerate(current):
                new_bbox = new["bbox_xyxy"]
                if not self._boxes_refer_to_same_region(
                    old_bbox,
                    new_bbox,
                    overlap_threshold=self.match_overlap_threshold,
                    center_ratio=self.match_center_distance_ratio,
                ):
                    continue
                score = self._intersection_over_smaller(old_bbox, new_bbox)
                distance = self._normalized_center_distance(old_bbox, new_bbox)
                pairs.append((score - 0.05 * distance, old_index, new_index))
        matched_old: set[int] = set()
        matched_new: set[int] = set()
        for _, old_index, new_index in sorted(pairs, reverse=True):
            if old_index in matched_old or new_index in matched_new:
                continue
            matched_old.add(old_index)
            matched_new.add(new_index)
        return [
            cluster
            for index, cluster in enumerate(current)
            if index not in matched_new
        ]

    @staticmethod
    def _nearest_person(
        paper_bbox: BBox,
        people: list[dict[str, Any]],
    ) -> dict[str, Any]:
        candidates = [person for person in people if person.get("is_present", True)]
        if not candidates:
            return {"track_id": None, "person_id": None}
        px = (paper_bbox[0] + paper_bbox[2]) / 2.0
        py = (paper_bbox[1] + paper_bbox[3]) / 2.0

        def distance(person: dict[str, Any]) -> tuple[float, float]:
            x1, y1, x2, y2 = person["bbox_xyxy"]
            inside = (x1 <= px <= x2) and (y1 <= py <= y2)
            dx = max(x1 - px, 0.0, px - x2)
            dy = max(y1 - py, 0.0, py - y2)
            center_distance = hypot(px - (x1 + x2) / 2.0, py - (y1 + y2) / 2.0)
            base_dist = hypot(dx, dy)
            if inside:
                base_dist -= 10000.0  # Give priority when paper is inside person ROI
            return base_dist, center_distance

        nearest = min(candidates, key=distance)
        return {
            "track_id": nearest.get("track_id"),
            "person_id": nearest.get("person_id"),
        }

    def _boxes_refer_to_same_region(
        self,
        left: BBox,
        right: BBox,
        *,
        overlap_threshold: float,
        center_ratio: float,
    ) -> bool:
        return (
            self._intersection_over_smaller(left, right) >= overlap_threshold
            or self._normalized_center_distance(left, right) <= center_ratio
        )

    @staticmethod
    def _intersection_over_smaller(left: BBox, right: BBox) -> float:
        x1 = max(left[0], right[0])
        y1 = max(left[1], right[1])
        x2 = min(left[2], right[2])
        y2 = min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        left_area = max(1, left[2] - left[0]) * max(1, left[3] - left[1])
        right_area = max(1, right[2] - right[0]) * max(1, right[3] - right[1])
        return intersection / min(left_area, right_area)

    @staticmethod
    def _normalized_center_distance(left: BBox, right: BBox) -> float:
        left_center = ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0)
        right_center = (
            (right[0] + right[2]) / 2.0,
            (right[1] + right[3]) / 2.0,
        )
        distance = hypot(
            left_center[0] - right_center[0],
            left_center[1] - right_center[1],
        )
        left_diagonal = hypot(left[2] - left[0], left[3] - left[1])
        right_diagonal = hypot(right[2] - right[0], right[3] - right[1])
        return distance / max(left_diagonal, right_diagonal, 1.0)

    @staticmethod
    def _valid_bbox(value: Any) -> BBox | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            bbox = [int(round(float(coordinate))) for coordinate in value]
        except (TypeError, ValueError):
            return None
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        return bbox

    @staticmethod
    def _mode_count(values: list[int]) -> int:
        if not values:
            return 0
        frequencies = Counter(values)
        return max(frequencies, key=lambda count: (frequencies[count], count))

    @staticmethod
    def _best_setup_snapshot(
        session: _CountSession,
        baseline_count: int,
    ) -> list[dict[str, Any]]:
        for count, snapshot in reversed(
            list(zip(session.setup_counts, session.setup_snapshots))
        ):
            if count == baseline_count:
                return [dict(cluster) for cluster in snapshot]
        return []

    def _get_session(self, session_id: str) -> _CountSession:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise KeyError(f"Unknown paper-count session: {session_id}") from error

    def _state_payload(self, session: _CountSession) -> dict[str, Any]:
        observed_count = len(session.observations)
        return {
            "mode": "count_only",
            "monitoring_armed": session.armed,
            "baseline_count": session.baseline_count,
            "observed_count": observed_count,
            "stable_count": session.stable_count,
            "candidate_count": session.candidate_count,
            "candidate_streak": session.candidate_streak,
            "confirmation_frames": self.confirmation_frames,
            "papers": [dict(item) for item in session.observations],
            "active_alerts": [dict(item) for item in session.active_alerts],
            "new_events": [dict(item) for item in session.new_events],
        }

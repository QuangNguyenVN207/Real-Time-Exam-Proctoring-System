"""Bridge face verification to the shared person/paper tracking session."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol


class FaceIdentityVerifier(Protocol):
    def identify(self, frame: Any, bbox: list[int]) -> tuple[str, float] | None: ...

    def verify_assigned_identity(
        self,
        frame: Any,
        bbox: list[int],
        expected_student_id: str,
        timestamp: float,
    ) -> dict[str, Any] | None: ...


class IdentityGuard:
    """Keep a face identity attached to the existing stable tracking ID.

    Face recognition is authoritative for long-term identity. IoU/appearance
    tracking remains the inexpensive short-term association layer between face
    scans. Multiple matching scans are required before assigning or alerting so
    one blurred frame cannot silently change a person's identity.
    """

    def __init__(
        self,
        face_verifier: FaceIdentityVerifier,
        tracking_manager: Any,
        *,
        scan_every_n_frames: int = 5,
        assignment_confirmations: int = 3,
        mismatch_confirmations: int = 3,
    ) -> None:
        if scan_every_n_frames < 1:
            raise ValueError("scan_every_n_frames must be at least 1")
        if assignment_confirmations < 1:
            raise ValueError("assignment_confirmations must be at least 1")
        if mismatch_confirmations < 1:
            raise ValueError("mismatch_confirmations must be at least 1")
        self._face = face_verifier
        self._tracking = tracking_manager
        self.scan_every_n_frames = scan_every_n_frames
        self.assignment_confirmations = assignment_confirmations
        self.mismatch_confirmations = mismatch_confirmations
        self._frames_seen: dict[str, int] = defaultdict(int)
        self._assignment_candidates: dict[tuple[str, int], tuple[str, int]] = {}
        self._mismatch_counts: dict[tuple[str, int], int] = defaultdict(int)

    def sync(
        self,
        session_id: str,
        frame: Any,
        timestamp: float,
    ) -> list[dict[str, Any]]:
        """Auto-assign and verify current tracks using the shared manager."""

        self._frames_seen[session_id] += 1
        if (self._frames_seen[session_id] - 1) % self.scan_every_n_frames:
            return []

        try:
            packet = self._tracking.get_packet(session_id)
            alerts: list[dict[str, Any]] = []
            visible_keys: set[tuple[str, int]] = set()
            for track in packet.tracks:
                if not track.is_present:
                    continue
                key = (session_id, track.track_id)
                visible_keys.add(key)
                bbox = track.bbox.to_list()
                if track.student_id is None:
                    self._mismatch_counts.pop(key, None)
                    self._try_auto_assign(
                        session_id,
                        track.track_id,
                        bbox,
                        frame,
                    )
                    continue

                self._assignment_candidates.pop(key, None)
                alert = self._face.verify_assigned_identity(
                    frame,
                    bbox,
                    track.student_id,
                    timestamp,
                )
                if alert is None:
                    self._mismatch_counts.pop(key, None)
                    continue

                count = self._mismatch_counts[key] + 1
                self._mismatch_counts[key] = count
                if count < self.mismatch_confirmations:
                    continue
                details = alert.setdefault("details", {})
                details["track_id"] = track.track_id
                details["confirmation_count"] = count
                alerts.append(alert)

            self._drop_inactive(session_id, visible_keys)
            return alerts
        except Exception:
            # A damaged frame or optional face model failure must never stop the
            # camera/server worker. Module-specific methods follow the same rule.
            return []

    def cleanup_session(self, session_id: str) -> None:
        self._frames_seen.pop(session_id, None)
        for state in (self._assignment_candidates, self._mismatch_counts):
            for key in tuple(state):
                if key[0] == session_id:
                    state.pop(key, None)

    def _try_auto_assign(
        self,
        session_id: str,
        track_id: int,
        bbox: list[int],
        frame: Any,
    ) -> None:
        key = (session_id, track_id)
        match = self._face.identify(frame, bbox)
        if match is None:
            self._assignment_candidates.pop(key, None)
            return

        student_id, _score = match
        previous_student, previous_count = self._assignment_candidates.get(
            key,
            ("", 0),
        )
        count = previous_count + 1 if previous_student == student_id else 1
        self._assignment_candidates[key] = (student_id, count)
        if count < self.assignment_confirmations:
            return

        try:
            self._tracking.assign_student(
                session_id,
                track_id=track_id,
                student_id=student_id,
            )
            self._assignment_candidates.pop(key, None)
        except Exception:
            # The same student may already be visible on another track. Waiting
            # for a later scan is safer than guessing or changing either ID.
            return

    def _drop_inactive(
        self,
        session_id: str,
        visible_keys: set[tuple[str, int]],
    ) -> None:
        for state in (self._assignment_candidates, self._mismatch_counts):
            for key in tuple(state):
                if key[0] == session_id and key not in visible_keys:
                    state.pop(key, None)

"""Session, student-ID assignment, and persistence management."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from ...settings import DEFAULT_MAX_MISSED_FRAMES, DEFAULT_MIN_IOU
from ..schemas import PersonDetection, TrackPacket, TrackedPerson
from ..tracker import IoUPersonTracker


STATE_VERSION = 1


class SessionNotFoundError(KeyError):
    """Raised when a request references an unknown tracking session."""


class SessionAlreadyExistsError(ValueError):
    """Raised when a fresh session would overwrite saved state."""


class AssignmentError(ValueError):
    """Raised when an ID assignment would produce an invalid session state."""


@dataclass(slots=True)
class _Session:
    session_id: str
    tracker: IoUPersonTracker
    assignments: dict[int, str] = field(default_factory=dict)
    last_frame_id: int | None = None
    last_timestamp_ms: int | None = None
    last_persisted_frame_id: int | None = None


class TrackingManager:
    """Thread-safe owner of per-session tracker and assignment state."""

    def __init__(
        self,
        storage_root: Path,
        *,
        max_tracks: int = 2,
        min_iou: float = DEFAULT_MIN_IOU,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        bbox_smoothing_alpha: float = 1.0,
        state_persist_interval_frames: int = 30,
    ) -> None:
        if state_persist_interval_frames < 1:
            raise ValueError("state_persist_interval_frames must be at least 1")

        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._max_tracks = max_tracks
        self._min_iou = min_iou
        self._max_missed_frames = max_missed_frames
        self._bbox_smoothing_alpha = bbox_smoothing_alpha
        self._state_persist_interval_frames = state_persist_interval_frames
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()

    def create_session(
        self,
        session_id: str,
        *,
        restore_existing: bool = False,
    ) -> None:
        """Create fresh state, restoring only when explicitly requested."""

        self._validate_session_id(session_id)
        with self._lock:
            if session_id in self._sessions:
                if restore_existing:
                    return
                raise SessionAlreadyExistsError(
                    f"Tracking session is already active: {session_id}"
                )

            state_path = self._state_path(session_id)
            if state_path.exists():
                if restore_existing:
                    self._sessions[session_id] = self._load_session(
                        session_id,
                        state_path,
                    )
                    return
                raise SessionAlreadyExistsError(
                    f"Saved tracking session already exists: {session_id}. "
                    "Request an explicit restore or use a new session ID."
                )

            self._sessions[session_id] = _Session(
                session_id=session_id,
                tracker=self._new_tracker(),
            )
            self._persist(session_id)

    def restore_session(self, session_id: str) -> None:
        """Restore a specifically requested session from persisted state."""

        self._validate_session_id(session_id)
        with self._lock:
            if session_id in self._sessions:
                return
            state_path = self._state_path(session_id)
            if not state_path.exists():
                raise SessionNotFoundError(
                    f"No saved tracking session exists: {session_id}"
                )
            self._sessions[session_id] = self._load_session(
                session_id,
                state_path,
            )

    def close_session(self, session_id: str) -> None:
        """Close active state while retaining its persisted restore snapshot."""

        with self._lock:
            session = self._get_session(session_id)
            self._persist(session.session_id)
            self._sessions.pop(session_id)

    @staticmethod
    def generate_session_id(prefix: str = "session") -> str:
        """Return a filesystem-safe, practically unique session ID."""

        safe_prefix = "".join(
            character
            for character in prefix
            if character.isalnum() or character in "_-"
        ).strip("_-")
        safe_prefix = safe_prefix or "session"
        return f"{safe_prefix}_{int(time() * 1000)}_{uuid4().hex[:8]}"

    def process_detections(
        self,
        session_id: str,
        *,
        frame_id: int,
        timestamp_ms: int,
        detections: list[PersonDetection],
    ) -> TrackPacket:
        with self._lock:
            session = self._get_session(session_id)
            tracks = session.tracker.update(detections)
            session.last_frame_id = frame_id
            session.last_timestamp_ms = timestamp_ms
            if self._should_persist(session):
                self._persist(session_id)
            return self._packet(session, tracks, frame_id, timestamp_ms)

    def assign_student(self, session_id: str, *, track_id: int, student_id: str) -> TrackPacket:
        """Assign a student, reusing the previous track ID after a disappearance."""

        student_id = student_id.strip()
        if not student_id:
            raise AssignmentError("student_id cannot be empty")

        with self._lock:
            session = self._get_session(session_id)
            if not session.tracker.is_track_present(track_id):
                raise AssignmentError(
                    f"track_id {track_id} is not present in session {session_id}"
                )

            existing_track_id = next(
                (
                    key
                    for key, value in session.assignments.items()
                    if value == student_id and key != track_id
                ),
                None,
            )

            if existing_track_id is not None:
                if session.tracker.is_track_present(existing_track_id):
                    raise AssignmentError(
                        f"student_id {student_id!r} is already visible on "
                        f"track_id {existing_track_id}"
                    )

                session.tracker.remap_track(track_id, existing_track_id)
                session.assignments.pop(track_id, None)
                track_id = existing_track_id

            session.assignments[track_id] = student_id
            self._persist(session_id)
            return self._current_packet(session)

    def unassign_student(self, session_id: str, *, track_id: int) -> TrackPacket:
        with self._lock:
            session = self._get_session(session_id)
            session.assignments.pop(track_id, None)
            self._persist(session_id)
            return self._current_packet(session)

    def get_packet(self, session_id: str) -> TrackPacket:
        with self._lock:
            return self._current_packet(self._get_session(session_id))

    def get_pose_gaze_input(self, session_id: str) -> dict[str, Any]:
        """Return only currently visible, assigned tracks for landmark extraction."""

        packet = self.get_packet(session_id)
        ready_tracks = [
            track.to_dict()
            for track in packet.tracks
            if track.is_present and track.student_id
        ]
        return {
            "session_id": session_id,
            "frame_id": packet.frame_id,
            "timestamp_ms": packet.timestamp_ms,
            "ready": len(ready_tracks) == self._max_tracks,
            "required_students": self._max_tracks,
            "assigned_visible_students": len(ready_tracks),
            "tracks": ready_tracks,
        }

    def generate_final_output(self, session_id: str) -> Path:
        """Write the current assigned tracks as the pose/gaze handoff JSON."""

        with self._lock:
            session = self._get_session(session_id)
            packet = self._current_packet(session)
            tracks_out = [
                {
                    "track_id": track.track_id,
                    "student_id": track.student_id,
                    "bbox_xyxy": track.bbox.to_list(),
                    "track_confidence": round(track.confidence, 2),
                }
                for track in packet.tracks
                if track.student_id
            ]
            payload = {
                "session_id": session_id,
                "frame_id": packet.frame_id,
                "timestamp_ms": packet.timestamp_ms,
                "frame": "numpy.ndarray BGR",
                "tracks": tracks_out,
            }

            output_path = self.storage_root / session_id / "pose_gaze_input.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return output_path

    def _new_tracker(self) -> IoUPersonTracker:
        return IoUPersonTracker(
            max_tracks=self._max_tracks,
            min_iou=self._min_iou,
            max_missed_frames=self._max_missed_frames,
            bbox_smoothing_alpha=self._bbox_smoothing_alpha,
        )

    def _current_packet(self, session: _Session) -> TrackPacket:
        return self._packet(
            session,
            session.tracker.snapshot(),
            session.last_frame_id if session.last_frame_id is not None else -1,
            session.last_timestamp_ms if session.last_timestamp_ms is not None else 0,
        )

    @staticmethod
    def _packet(
        session: _Session,
        tracks: list[TrackedPerson],
        frame_id: int,
        timestamp_ms: int,
    ) -> TrackPacket:
        enriched_tracks = tuple(
            TrackedPerson(
                track_id=track.track_id,
                bbox=track.bbox,
                confidence=track.confidence,
                age_frames=track.age_frames,
                missed_frames=track.missed_frames,
                is_present=track.is_present,
                student_id=session.assignments.get(track.track_id),
            )
            for track in tracks
        )
        return TrackPacket(session.session_id, frame_id, timestamp_ms, enriched_tracks)

    def _get_session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise SessionNotFoundError(
                f"Unknown tracking session: {session_id}"
            ) from error

    def _load_session(self, session_id: str, state_path: Path) -> _Session:
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot restore tracking state: {state_path}") from error

        if payload.get("session_id") != session_id:
            raise ValueError(
                f"Persisted session_id does not match requested session: {session_id}"
            )

        tracker = self._new_tracker()
        tracker_payload = payload.get("tracker")
        if tracker_payload is not None:
            if not isinstance(tracker_payload, dict):
                raise ValueError("tracker state must be an object")
            tracker.restore_state(tracker_payload)

        assignments: dict[int, str] = {}
        raw_assignments = payload.get("assignments", [])
        if not isinstance(raw_assignments, list):
            raise ValueError("assignments must be a list")
        for item in raw_assignments:
            track_id = int(item["track_id"])
            student_id = str(item["student_id"]).strip()
            if track_id < 1 or not student_id:
                raise ValueError("Invalid persisted assignment")
            assignments[track_id] = student_id

        last_frame_id = payload.get("last_frame_id")
        last_timestamp_ms = payload.get("last_timestamp_ms")
        return _Session(
            session_id=session_id,
            tracker=tracker,
            assignments=assignments,
            last_frame_id=int(last_frame_id) if last_frame_id is not None else None,
            last_timestamp_ms=(
                int(last_timestamp_ms) if last_timestamp_ms is not None else None
            ),
            last_persisted_frame_id=(
                int(last_frame_id) if last_frame_id is not None else None
            ),
        )

    def _should_persist(self, session: _Session) -> bool:
        if session.last_frame_id is None:
            return False
        if session.last_persisted_frame_id is None:
            return True
        frame_delta = session.last_frame_id - session.last_persisted_frame_id
        return frame_delta < 0 or frame_delta >= self._state_persist_interval_frames

    def _persist(self, session_id: str) -> None:
        session = self._sessions[session_id]
        state_path = self._state_path(session_id)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_version": STATE_VERSION,
            "session_id": session_id,
            "updated_at": time(),
            "last_frame_id": session.last_frame_id,
            "last_timestamp_ms": session.last_timestamp_ms,
            "tracker": session.tracker.export_state(),
            "assignments": [
                {"track_id": key, "student_id": value}
                for key, value in sorted(session.assignments.items())
            ],
        }

        temporary_path = state_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(state_path)
        if session.last_frame_id is not None:
            session.last_persisted_frame_id = session.last_frame_id

    def _state_path(self, session_id: str) -> Path:
        return self.storage_root / session_id / "tracking_state.json"

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        if not session_id or any(character not in allowed for character in session_id):
            raise ValueError(
                "session_id may contain only letters, digits, '_' and '-'"
            )

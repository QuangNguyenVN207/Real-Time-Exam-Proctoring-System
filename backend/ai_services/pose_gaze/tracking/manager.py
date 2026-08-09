"""Session, student-ID assignment, and handoff management for Module 1."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from .paper_tracking import (
    IoUPaperTracker,
    PaperAssessment,
    PaperAuthorizationPolicy,
    PaperDetection,
)
from .schemas import PersonDetection, TrackPacket, TrackedPerson
from .tracker import IoUPersonTracker


STATE_VERSION = 2


class SessionNotFoundError(KeyError):
    """Raised when a request references an unknown tracking session."""


class SessionAlreadyExistsError(ValueError):
    """Raised when a fresh session would overwrite active or saved state."""


class AssignmentError(ValueError):
    """Raised when an ID assignment would produce an invalid session state."""


@dataclass(slots=True)
class _Session:
    session_id: str
    tracker: IoUPersonTracker
    paper_tracker: IoUPaperTracker
    paper_policy: PaperAuthorizationPolicy
    assignments: dict[int, str] = field(default_factory=dict)
    manual_paper_ids: set[int] = field(default_factory=set)
    last_frame_id: int | None = None
    last_timestamp_ms: int | None = None
    paper_assessments: list[PaperAssessment] = field(default_factory=list)
    last_persisted_frame_id: int | None = None


class TrackingManager:
    """Thread-safe owner of per-session tracker and assignment state."""

    def __init__(
        self,
        storage_root: Path,
        *,
        max_tracks: int = 2,
        min_iou: float = 0.20,
        max_missed_frames: int = 15,
        person_appearance_match_threshold: float = 0.78,
        paper_registration_frames: int = 5,
        paper_alert_confirm_frames: int = 3,
        paper_max_missed_frames: int = 12,
        paper_auto_register_first: bool = True,
        paper_appearance_match_threshold: float = 0.86,
        state_persist_interval_frames: int = 30,
    ) -> None:
        if state_persist_interval_frames < 1:
            raise ValueError("state_persist_interval_frames must be at least 1")
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._max_tracks = max_tracks
        self._min_iou = min_iou
        self._max_missed_frames = max_missed_frames
        self._person_appearance_match_threshold = (
            person_appearance_match_threshold
        )
        self._paper_registration_frames = paper_registration_frames
        self._paper_alert_confirm_frames = paper_alert_confirm_frames
        self._paper_max_missed_frames = paper_max_missed_frames
        self._paper_auto_register_first = paper_auto_register_first
        self._paper_appearance_match_threshold = (
            paper_appearance_match_threshold
        )
        self._state_persist_interval_frames = state_persist_interval_frames
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()

    def create_session(
        self,
        session_id: str,
        *,
        restore_existing: bool = False,
    ) -> None:
        """Create fresh state, restoring saved state only when requested."""

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

            self._sessions[session_id] = self._new_session(session_id)
            self._persist(session_id)

    def ensure_session(self, session_id: str) -> None:
        """Idempotently ensure a realtime pipeline session is active."""

        self._validate_session_id(session_id)
        with self._lock:
            if session_id in self._sessions:
                return
            state_path = self._state_path(session_id)
            if state_path.exists():
                self._sessions[session_id] = self._load_session(
                    session_id,
                    state_path,
                )
            else:
                self._sessions[session_id] = self._new_session(session_id)
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
        """Close active state while retaining its persisted snapshot."""

        with self._lock:
            session = self._get_session(session_id)
            self._persist(session.session_id)
            self._sessions.pop(session_id)

    @staticmethod
    def generate_session_id(prefix: str = "session") -> str:
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
            # Learn only identities explicitly confirmed by the proctor.
            # Temporary detector IDs must never enter the identity archive.
            for track in tracks:
                if track.is_present and track.track_id in session.assignments:
                    session.tracker.register_identity(track.track_id)
            tracks = session.tracker.snapshot()
            session.last_frame_id = frame_id
            session.last_timestamp_ms = timestamp_ms
            if self._should_persist(session):
                self._persist(session_id)
            return self._packet(session, tracks, frame_id, timestamp_ms)

    def process_paper_detections(
        self,
        session_id: str,
        *,
        detections: list[PaperDetection],
        supports_test_paper: bool,
    ) -> dict:
        """Update stable paper IDs and evaluate authorized-vs-new paper tracks.

        Call this only when object inference actually ran. Reusing cached
        detections on skipped frames would incorrectly advance confirmation
        counters.
        """

        with self._lock:
            session = self._get_session(session_id)
            session.paper_tracker.set_model_capabilities(
                supports_test_paper=supports_test_paper
            )
            session.paper_policy.set_model_capabilities(
                supports_test_paper=supports_test_paper
            )
            people = session.tracker.snapshot()
            papers = session.paper_tracker.update(
                detections,
                people=people,
            )
            papers = session.paper_tracker.associate_owners(people)
            session.paper_assessments = session.paper_policy.evaluate(papers)
            for owner_track_id, paper_id in (
                session.paper_policy.authorized_mapping().items()
            ):
                if session.paper_tracker.has_track(paper_id):
                    session.paper_tracker.assign_owner(
                        paper_id=paper_id,
                        owner_track_id=owner_track_id,
                    )
                    session.paper_tracker.register_identity(
                        paper_id=paper_id,
                        owner_track_id=owner_track_id,
                    )
            self._persist(session_id)
            return self._paper_payload(session)

    def get_paper_state(self, session_id: str) -> dict:
        with self._lock:
            return self._paper_payload(self._get_session(session_id))

    def assign_paper_id(
        self,
        session_id: str,
        *,
        current_paper_id: int,
        stable_paper_id: int,
    ) -> dict:
        """Replace a temporary paper ID with the ID selected by the proctor.

        Reusing a known stable ID after a paper was lost restores the original
        authorization and keeps downstream alert references consistent.
        """

        if stable_paper_id < 1:
            raise AssignmentError("stable paper_id must be at least 1")
        with self._lock:
            session = self._get_session(session_id)
            if not session.paper_tracker.is_present(current_paper_id):
                raise AssignmentError(
                    f"paper_id {current_paper_id} is not visible in session "
                    f"{session_id}"
                )
            if (
                current_paper_id != stable_paper_id
                and session.paper_tracker.is_present(stable_paper_id)
            ):
                raise AssignmentError(
                    f"paper_id {stable_paper_id} is already visible in session "
                    f"{session_id}"
                )

            try:
                session.paper_tracker.remap_track(
                    current_paper_id=current_paper_id,
                    target_paper_id=stable_paper_id,
                )
            except ValueError as error:
                raise AssignmentError(str(error)) from error
            session.paper_policy.remap_paper_id(
                current_paper_id=current_paper_id,
                target_paper_id=stable_paper_id,
            )
            session.manual_paper_ids.discard(current_paper_id)
            session.manual_paper_ids.add(stable_paper_id)
            session.paper_tracker.register_identity(
                paper_id=stable_paper_id,
            )
            session.paper_assessments = session.paper_policy.evaluate(
                session.paper_tracker.snapshot()
            )
            self._persist(session_id)
            return self._paper_payload(session)

    def register_exam_paper(
        self,
        session_id: str,
        *,
        owner_track_id: int,
        paper_id: int,
        replace: bool = False,
    ) -> dict:
        with self._lock:
            session = self._get_session(session_id)
            if not session.tracker.has_track(owner_track_id):
                raise AssignmentError(
                    f"owner track_id {owner_track_id} is not active in session {session_id}"
                )
            if not session.paper_tracker.has_track(paper_id):
                raise AssignmentError(
                    f"paper_id {paper_id} is not active in session {session_id}"
                )
            session.paper_tracker.assign_owner(
                paper_id=paper_id,
                owner_track_id=owner_track_id,
            )
            session.paper_policy.register(
                owner_track_id=owner_track_id,
                paper_id=paper_id,
                replace=replace,
            )
            session.paper_tracker.register_identity(
                paper_id=paper_id,
                owner_track_id=owner_track_id,
            )
            session.paper_assessments = session.paper_policy.evaluate(
                session.paper_tracker.snapshot()
            )
            self._persist(session_id)
            return self._paper_payload(session)

    def unregister_exam_paper(self, session_id: str, *, paper_id: int) -> dict:
        with self._lock:
            session = self._get_session(session_id)
            session.paper_policy.unregister(paper_id=paper_id)
            session.paper_assessments = session.paper_policy.evaluate(
                session.paper_tracker.snapshot()
            )
            self._persist(session_id)
            return self._paper_payload(session)

    def arm_paper_monitoring(self, session_id: str) -> dict:
        with self._lock:
            session = self._get_session(session_id)
            session.paper_policy.arm()
            self._persist(session_id)
            return self._paper_payload(session)

    def disarm_paper_monitoring(self, session_id: str) -> dict:
        with self._lock:
            session = self._get_session(session_id)
            session.paper_policy.disarm()
            self._persist(session_id)
            return self._paper_payload(session)

    def assign_student(self, session_id: str, *, track_id: int, student_id: str) -> TrackPacket:
        """Attach a proctor-assigned person ID to a temporary tracker ID.

        Reusing a known ``student_id`` for a newly detected box restores the
        original numeric track ID. Once assigned, the current appearance is
        archived so later re-entry can restore this identity automatically.
        """

        student_id = student_id.strip()
        if not student_id:
            raise AssignmentError("student_id cannot be empty")
        with self._lock:
            session = self._get_session(session_id)
            if not session.tracker.is_present(track_id):
                raise AssignmentError(
                    f"track_id {track_id} is not visible in session {session_id}"
                )

            assigned_to_current_track = session.assignments.get(track_id)
            if (
                assigned_to_current_track is not None
                and assigned_to_current_track != student_id
            ):
                raise AssignmentError(
                    f"track_id {track_id} is already assigned to "
                    f"person_id {assigned_to_current_track!r}; unassign it first"
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
                if session.tracker.is_present(existing_track_id):
                    raise AssignmentError(
                        f"person_id {student_id!r} is already visible on "
                        f"track_id {existing_track_id}"
                    )

                try:
                    session.tracker.remap_track(track_id, existing_track_id)
                except ValueError as error:
                    raise AssignmentError(str(error)) from error
                session.paper_tracker.remap_owner(
                    current_track_id=track_id,
                    target_track_id=existing_track_id,
                )
                session.paper_policy.remap_owner(
                    current_track_id=track_id,
                    target_track_id=existing_track_id,
                )
                session.assignments.pop(track_id, None)
                track_id = existing_track_id
            else:
                session.assignments[track_id] = student_id

            session.assignments[track_id] = student_id
            session.tracker.register_identity(track_id)
            session.paper_assessments = session.paper_policy.evaluate(
                session.paper_tracker.snapshot()
            )
            self._persist(session_id)
            return self._current_packet(session)

    def unassign_student(self, session_id: str, *, track_id: int) -> TrackPacket:
        with self._lock:
            session = self._get_session(session_id)
            session.assignments.pop(track_id, None)
            session.tracker.unregister_identity(track_id)
            self._persist(session_id)
            return self._current_packet(session)

    def get_packet(self, session_id: str) -> TrackPacket:
        with self._lock:
            return self._current_packet(self._get_session(session_id))

    def get_pose_gaze_input(self, session_id: str) -> dict:
        """Return only currently visible, assigned tracks for the next module stage."""

        packet = self.get_packet(session_id)
        ready_tracks = [track.to_dict() for track in packet.tracks if track.is_present and track.student_id]
        return {
            "session_id": session_id,
            "frame_id": packet.frame_id,
            "timestamp_ms": packet.timestamp_ms,
            "ready": len(ready_tracks) == self._max_tracks,
            "required_students": self._max_tracks,
            "assigned_visible_students": len(ready_tracks),
            "tracks": ready_tracks,
        }

    def _new_tracker(self) -> IoUPersonTracker:
        return IoUPersonTracker(
            max_tracks=self._max_tracks,
            min_iou=self._min_iou,
            max_missed_frames=self._max_missed_frames,
            appearance_match_threshold=(
                self._person_appearance_match_threshold
            ),
        )

    def _new_session(self, session_id: str) -> _Session:
        return _Session(
            session_id=session_id,
            tracker=self._new_tracker(),
            paper_tracker=IoUPaperTracker(
                max_missed_frames=self._paper_max_missed_frames,
                appearance_match_threshold=(
                    self._paper_appearance_match_threshold
                ),
            ),
            paper_policy=PaperAuthorizationPolicy(
                registration_frames=self._paper_registration_frames,
                alert_confirm_frames=self._paper_alert_confirm_frames,
                auto_register_first_paper=self._paper_auto_register_first,
            ),
        )

    def _current_packet(self, session: _Session) -> TrackPacket:
        return self._packet(
            session,
            session.tracker.snapshot(),
            session.last_frame_id if session.last_frame_id is not None else -1,
            session.last_timestamp_ms if session.last_timestamp_ms is not None else 0,
        )

    @staticmethod
    def _packet(session: _Session, tracks: list[TrackedPerson], frame_id: int, timestamp_ms: int) -> TrackPacket:
        enriched_tracks = tuple(
            TrackedPerson(
                track_id=track.track_id,
                bbox=track.bbox,
                confidence=track.confidence,
                age_frames=track.age_frames,
                missed_frames=track.missed_frames,
                is_present=track.is_present,
                student_id=session.assignments.get(track.track_id),
                appearance_identity_registered=(
                    session.tracker.has_registered_identity(track.track_id)
                ),
            )
            for track in tracks
        )
        return TrackPacket(session.session_id, frame_id, timestamp_ms, enriched_tracks)

    def _get_session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise SessionNotFoundError(f"Unknown tracking session: {session_id}") from error

    def _load_session(self, session_id: str, state_path: Path) -> _Session:
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot restore tracking state: {state_path}") from error

        if payload.get("session_id") != session_id:
            raise ValueError(
                f"Persisted session_id does not match requested session: {session_id}"
            )

        session = self._new_session(session_id)
        tracker_payload = payload.get("tracker")
        if tracker_payload is not None:
            if not isinstance(tracker_payload, dict):
                raise ValueError("tracker state must be an object")
            session.tracker.restore_state(tracker_payload)

        paper_tracker_payload = payload.get("paper_tracker")
        if paper_tracker_payload is not None:
            if not isinstance(paper_tracker_payload, dict):
                raise ValueError("paper_tracker state must be an object")
            session.paper_tracker.restore_state(paper_tracker_payload)

        paper_policy_payload = payload.get("paper_policy")
        if paper_policy_payload is not None:
            if not isinstance(paper_policy_payload, dict):
                raise ValueError("paper_policy state must be an object")
            session.paper_policy.restore_state(paper_policy_payload)
        else:
            # Compatibility with the first paper-tracking state format.
            for item in payload.get("authorized_papers", []):
                session.paper_policy.register(
                    owner_track_id=int(item["owner_track_id"]),
                    paper_id=int(item["paper_id"]),
                )
            if payload.get("paper_monitoring_armed", False):
                session.paper_policy.arm()

        raw_assignments = payload.get("assignments", [])
        if not isinstance(raw_assignments, list):
            raise ValueError("assignments must be a list")
        for item in raw_assignments:
            track_id = int(item["track_id"])
            student_id = str(
                item.get("student_id", item.get("person_id", ""))
            ).strip()
            if track_id < 1 or not student_id:
                raise ValueError("Invalid persisted assignment")
            session.assignments[track_id] = student_id

        raw_manual_paper_ids = payload.get("manual_paper_ids", [])
        if not isinstance(raw_manual_paper_ids, list):
            raise ValueError("manual_paper_ids must be a list")
        session.manual_paper_ids = {
            int(paper_id) for paper_id in raw_manual_paper_ids
        }
        last_frame_id = payload.get("last_frame_id")
        last_timestamp_ms = payload.get("last_timestamp_ms")
        session.last_frame_id = (
            int(last_frame_id) if last_frame_id is not None else None
        )
        session.last_timestamp_ms = (
            int(last_timestamp_ms) if last_timestamp_ms is not None else None
        )
        session.last_persisted_frame_id = session.last_frame_id
        session.paper_assessments = session.paper_policy.evaluate(
            session.paper_tracker.snapshot()
        )
        return session

    def _should_persist(self, session: _Session) -> bool:
        if session.last_frame_id is None:
            return False
        if session.last_persisted_frame_id is None:
            return True
        frame_delta = session.last_frame_id - session.last_persisted_frame_id
        return frame_delta < 0 or frame_delta >= self._state_persist_interval_frames

    @staticmethod
    def _paper_payload(session: _Session) -> dict:
        assessments = []
        for assessment in session.paper_assessments:
            payload = assessment.to_dict()
            payload["paper_id_assigned"] = (
                assessment.paper.paper_id in session.manual_paper_ids
            )
            payload["paper_id_source"] = (
                "manual"
                if payload["paper_id_assigned"]
                else "temporary"
            )
            payload["appearance_identity_registered"] = (
                session.paper_tracker.has_registered_identity(
                    assessment.paper.paper_id
                )
            )
            payload["owner_person_id"] = session.assignments.get(
                assessment.paper.owner_track_id
            )
            assessments.append(payload)
        suspicious = [
            assessment
            for assessment in assessments
            if assessment["is_present"] and assessment["status"] == "suspicious"
        ]
        return {
            "session_id": session.session_id,
            "frame_id": session.last_frame_id if session.last_frame_id is not None else -1,
            "timestamp_ms": (
                session.last_timestamp_ms
                if session.last_timestamp_ms is not None
                else 0
            ),
            "monitoring_armed": session.paper_policy.armed,
            "authorized_papers": [
                {
                    "owner_track_id": owner,
                    "owner_person_id": session.assignments.get(owner),
                    "paper_id": paper_id,
                    "paper_id_assigned": paper_id in session.manual_paper_ids,
                }
                for owner, paper_id in sorted(
                    session.paper_policy.authorized_mapping().items()
                )
            ],
            "papers": assessments,
            "alerts": suspicious,
            "risk_score": max(
                (assessment["risk_score"] for assessment in assessments),
                default=0.0,
            ),
        }

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
            "paper_tracker": session.paper_tracker.export_state(),
            "paper_policy": session.paper_policy.export_state(),
            "assignments": [
                {
                    "track_id": key,
                    "person_id": value,
                    "student_id": value,
                }
                for key, value in sorted(session.assignments.items())
            ],
            "paper_monitoring_armed": session.paper_policy.armed,
            "manual_paper_ids": sorted(session.manual_paper_ids),
            "authorized_papers": [
                {
                    "owner_track_id": owner,
                    "owner_person_id": session.assignments.get(owner),
                    "paper_id": paper_id,
                    "paper_id_assigned": paper_id in session.manual_paper_ids,
                }
                for owner, paper_id in sorted(
                    session.paper_policy.authorized_mapping().items()
                )
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
        if not session_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in session_id):
            raise ValueError("session_id may contain only letters, digits, '_' and '-'")

    def generate_final_output(self, session_id: str) -> Path:
        """Xuất file JSON với định dạng toạ độ, confidence đúng yêu cầu khi kết thúc."""
        with self._lock:
            session = self._get_session(session_id)
            packet = self._current_packet(session)

            tracks_out = []
            for track in packet.tracks:
                if track.student_id:  # Chỉ lấy những người đã được xác nhận ID
                    tracks_out.append({
                        "track_id": track.track_id,
                        "student_id": track.student_id,
                        "bbox_xyxy": track.bbox.to_list(),
                        "track_confidence": round(track.confidence, 2)
                    })

            payload = {
                "session_id": session_id,
                "frame_id": packet.frame_id,
                "timestamp_ms": packet.timestamp_ms,
                "frame": "numpy.ndarray BGR",
                "tracks": tracks_out
            }

            output_path = self.storage_root / session_id / "pose_gaze_input.json"
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return output_path

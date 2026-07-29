"""Session, student-ID assignment, and handoff management for Module 1."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import time

from .paper_tracking import (
    IoUPaperTracker,
    PaperAssessment,
    PaperAuthorizationPolicy,
    PaperDetection,
)
from .schemas import PersonDetection, TrackPacket, TrackedPerson
from .tracker import IoUPersonTracker


class SessionNotFoundError(KeyError):
    """Raised when a request references an unknown tracking session."""


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
    ) -> None:
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
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()

    def create_session(self, session_id: str) -> None:
        self._validate_session_id(session_id)
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = _Session(
                    session_id=session_id,
                    tracker=IoUPersonTracker(
                        max_tracks=self._max_tracks,
                        min_iou=self._min_iou,
                        max_missed_frames=self._max_missed_frames,
                        appearance_match_threshold=(
                            self._person_appearance_match_threshold
                        ),
                    ),
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
                self._persist(session_id)

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
        state_path = self.storage_root / session_id / "tracking_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": session_id,
            "updated_at": time(),
            "assignments": [
                {
                    "track_id": key,
                    "person_id": value,
                    "student_id": value,
                }
                for key, value in session.assignments.items()
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
        state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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

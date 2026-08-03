"""Session, student-ID assignment, and handoff management for Module 1."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import time

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
    assignments: dict[int, str] = field(default_factory=dict)
    last_frame_id: int | None = None
    last_timestamp_ms: int | None = None


class TrackingManager:
    """Thread-safe owner of per-session tracker and assignment state."""

    def __init__(
        self,
        storage_root: Path,
        *,
        max_tracks: int = 2,
        min_iou: float = 0.20,
        max_missed_frames: int = 15,
    ) -> None:
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._max_tracks = max_tracks
        self._min_iou = min_iou
        self._max_missed_frames = max_missed_frames
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
            session.last_frame_id = frame_id
            session.last_timestamp_ms = timestamp_ms
            return self._packet(session, tracks, frame_id, timestamp_ms)

    def assign_student_old(self, session_id: str, *, track_id: int, student_id: str) -> TrackPacket:
        student_id = student_id.strip()
        if not student_id:
            raise AssignmentError("student_id cannot be empty")
        with self._lock:
            session = self._get_session(session_id)
            if not session.tracker.has_track(track_id):
                raise AssignmentError(f"track_id {track_id} is not active in session {session_id}")
            existing_track = next(
                (key for key, value in session.assignments.items() if value == student_id and key != track_id),
                None,
            )
            if existing_track is not None:
                raise AssignmentError(f"student_id {student_id!r} is already assigned to track_id {existing_track}")
            session.assignments[track_id] = student_id
            self._persist(session_id)
            return self._current_packet(session)

    def assign_student(self, session_id: str, *, track_id: int, student_id: str) -> TrackPacket:
        student_id = student_id.strip()
        if not student_id:
            raise AssignmentError("student_id cannot be empty")
        with self._lock:
            session = self._get_session(session_id)
            if not session.tracker.has_track(track_id):
                raise AssignmentError(f"track_id {track_id} is not active in session {session_id}")
            
            # KIỂM TRA LỊCH SỬ: Xem student_id này đã từng được gán cho track_id nào chưa
            existing_track_id = next(
                (key for key, value in session.assignments.items() if value == student_id and key != track_id),
                None,
            )
            
            if existing_track_id is not None:
                # Nếu track cũ vẫn còn đang hiển thị trên màn hình, báo lỗi (2 người ko thể chung ID)
                if session.tracker.has_track(existing_track_id):
                    raise AssignmentError(f"Sinh viên {student_id!r} đang xuất hiện đồng thời ở track {existing_track_id}")
                
                # NẾU TRACK CŨ ĐÃ MẤT DẤU (Re-tracking):
                # Yêu cầu Tracker đổi ID của track mới này về lại ID cũ. (số lượng ID không tăng thêm)
                session.tracker.remap_track(track_id, existing_track_id)
                track_id = existing_track_id  # Cập nhật biến để gán tiếp
            else:
                # Nếu là ID mới hoàn toàn, lưu vào danh sách
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
            )
            for track in tracks
        )
        return TrackPacket(session.session_id, frame_id, timestamp_ms, enriched_tracks)

    def _get_session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise SessionNotFoundError(f"Unknown tracking session: {session_id}") from error

    def _persist(self, session_id: str) -> None:
        session = self._sessions[session_id]
        state_path = self.storage_root / session_id / "tracking_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": session_id,
            "updated_at": time(),
            "assignments": [{"track_id": key, "student_id": value} for key, value in session.assignments.items()],
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
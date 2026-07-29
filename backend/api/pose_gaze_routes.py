"""REST API for person tracking and manual student-ID assignment."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.ai_services.pose_gaze.tracking.manager import AssignmentError, SessionNotFoundError, TrackingManager
from backend.ai_services.pose_gaze.tracking.paper_tracking import paper_detection_from_dict
from backend.ai_services.pose_gaze.tracking.schemas import detection_from_dict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRACKING_MANAGER = TrackingManager(_PROJECT_ROOT / "data" / "sessions")
router = APIRouter(prefix="/api/pose-gaze", tags=["pose-gaze-tracking"])


class CreateSessionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class DetectionRequest(BaseModel):
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)
    class_name: str = "person"
    appearance_fingerprint: list[float] | None = Field(
        default=None,
        min_length=8,
        max_length=512,
    )


class ProcessDetectionsRequest(BaseModel):
    frame_id: int = Field(ge=0)
    timestamp_ms: int | None = Field(default=None, ge=0)
    detections: list[DetectionRequest]


class AssignmentRequest(BaseModel):
    person_id: str | None = Field(default=None, min_length=1, max_length=128)
    # Backward-compatible input name used by the original pose/gaze contract.
    student_id: str | None = Field(default=None, min_length=1, max_length=128)

    def resolved_person_id(self) -> str:
        if self.person_id and self.student_id and self.person_id != self.student_id:
            raise AssignmentError("person_id and student_id must match when both are provided")
        person_id = self.person_id or self.student_id
        if person_id is None:
            raise AssignmentError("person_id is required")
        return person_id


class PaperDetectionRequest(BaseModel):
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)
    class_name: str = Field(pattern=r"^(cheat_sheet|test_paper|paper_unknown)$")
    appearance_fingerprint: list[float] | None = Field(
        default=None,
        min_length=8,
        max_length=512,
    )


class ProcessPaperDetectionsRequest(BaseModel):
    supports_test_paper: bool = False
    detections: list[PaperDetectionRequest]


class PaperIdAssignmentRequest(BaseModel):
    stable_paper_id: int = Field(ge=1)


class PaperAuthorizationRequest(BaseModel):
    owner_track_id: int = Field(ge=1)
    replace_existing: bool = False


def _packet_response(packet) -> dict:
    return packet.to_dict()


def _translate_error(error: Exception) -> HTTPException:
    if isinstance(error, SessionNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error))


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
def create_session(request: CreateSessionRequest) -> dict:
    try:
        _TRACKING_MANAGER.create_session(request.session_id)
        return _packet_response(_TRACKING_MANAGER.get_packet(request.session_id))
    except ValueError as error:
        raise _translate_error(error) from error


@router.post("/sessions/{session_id}/detections")
def process_detections(session_id: str, request: ProcessDetectionsRequest) -> dict:
    try:
        packet = _TRACKING_MANAGER.process_detections(
            session_id,
            frame_id=request.frame_id,
            timestamp_ms=request.timestamp_ms if request.timestamp_ms is not None else int(time.time() * 1000),
            detections=[detection_from_dict(detection.model_dump()) for detection in request.detections],
        )
        return _packet_response(packet)
    except (SessionNotFoundError, ValueError) as error:
        raise _translate_error(error) from error


@router.get("/sessions/{session_id}/tracks")
def get_tracks(session_id: str) -> dict:
    try:
        return _packet_response(_TRACKING_MANAGER.get_packet(session_id))
    except SessionNotFoundError as error:
        raise _translate_error(error) from error


@router.put("/sessions/{session_id}/tracks/{track_id}/assignment")
def assign_student(session_id: str, track_id: int, request: AssignmentRequest) -> dict:
    try:
        return _packet_response(
            _TRACKING_MANAGER.assign_student(
                session_id,
                track_id=track_id,
                student_id=request.resolved_person_id(),
            )
        )
    except (SessionNotFoundError, AssignmentError) as error:
        raise _translate_error(error) from error


@router.delete("/sessions/{session_id}/tracks/{track_id}/assignment")
def unassign_student(session_id: str, track_id: int) -> dict:
    try:
        return _packet_response(_TRACKING_MANAGER.unassign_student(session_id, track_id=track_id))
    except SessionNotFoundError as error:
        raise _translate_error(error) from error


@router.get("/sessions/{session_id}/pose-gaze-input")
def get_pose_gaze_input(session_id: str) -> dict:
    try:
        return _TRACKING_MANAGER.get_pose_gaze_input(session_id)
    except SessionNotFoundError as error:
        raise _translate_error(error) from error


@router.post("/sessions/{session_id}/paper-detections")
def process_paper_detections(
    session_id: str,
    request: ProcessPaperDetectionsRequest,
) -> dict:
    """Receive paper boxes only on frames where object inference actually ran."""

    try:
        return _TRACKING_MANAGER.process_paper_detections(
            session_id,
            detections=[
                paper_detection_from_dict(detection.model_dump())
                for detection in request.detections
            ],
            supports_test_paper=request.supports_test_paper,
        )
    except (SessionNotFoundError, ValueError) as error:
        raise _translate_error(error) from error


@router.get("/sessions/{session_id}/papers")
def get_papers(session_id: str) -> dict:
    try:
        return _TRACKING_MANAGER.get_paper_state(session_id)
    except SessionNotFoundError as error:
        raise _translate_error(error) from error


@router.put("/sessions/{session_id}/papers/{current_paper_id}/identity")
def assign_paper_id(
    session_id: str,
    current_paper_id: int,
    request: PaperIdAssignmentRequest,
) -> dict:
    try:
        return _TRACKING_MANAGER.assign_paper_id(
            session_id,
            current_paper_id=current_paper_id,
            stable_paper_id=request.stable_paper_id,
        )
    except (SessionNotFoundError, AssignmentError) as error:
        raise _translate_error(error) from error


@router.put("/sessions/{session_id}/papers/{paper_id}/authorization")
def authorize_exam_paper(
    session_id: str,
    paper_id: int,
    request: PaperAuthorizationRequest,
) -> dict:
    try:
        return _TRACKING_MANAGER.register_exam_paper(
            session_id,
            owner_track_id=request.owner_track_id,
            paper_id=paper_id,
            replace=request.replace_existing,
        )
    except (SessionNotFoundError, AssignmentError, ValueError) as error:
        raise _translate_error(error) from error


@router.delete("/sessions/{session_id}/papers/{paper_id}/authorization")
def unauthorize_exam_paper(session_id: str, paper_id: int) -> dict:
    try:
        return _TRACKING_MANAGER.unregister_exam_paper(
            session_id,
            paper_id=paper_id,
        )
    except SessionNotFoundError as error:
        raise _translate_error(error) from error


@router.post("/sessions/{session_id}/paper-monitoring/arm")
def arm_paper_monitoring(session_id: str) -> dict:
    try:
        return _TRACKING_MANAGER.arm_paper_monitoring(session_id)
    except SessionNotFoundError as error:
        raise _translate_error(error) from error


@router.post("/sessions/{session_id}/paper-monitoring/disarm")
def disarm_paper_monitoring(session_id: str) -> dict:
    try:
        return _TRACKING_MANAGER.disarm_paper_monitoring(session_id)
    except SessionNotFoundError as error:
        raise _translate_error(error) from error

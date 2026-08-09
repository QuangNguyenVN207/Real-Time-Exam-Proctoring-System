"""Tracking, assignment, and identity-aware paper monitoring."""

from .paper_pipeline import PoseGazePaperPipeline
from .tracking.manager import AssignmentError, SessionNotFoundError, TrackingManager
from .tracking.paper_tracking import (
    IoUPaperTracker,
    PaperAssessment,
    PaperAuthorizationPolicy,
    PaperDetection,
    TrackedPaper,
)
from .tracking.schemas import BoundingBox, PersonDetection, TrackPacket, TrackedPerson
from .tracking.tracker import IoUPersonTracker

__all__ = [
    "AssignmentError",
    "BoundingBox",
    "IoUPaperTracker",
    "IoUPersonTracker",
    "PaperAssessment",
    "PaperAuthorizationPolicy",
    "PaperDetection",
    "PersonDetection",
    "PoseGazePaperPipeline",
    "SessionNotFoundError",
    "TrackPacket",
    "TrackedPaper",
    "TrackedPerson",
    "TrackingManager",
]

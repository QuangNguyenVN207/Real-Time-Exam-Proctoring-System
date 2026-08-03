"""Person tracking and per-track holistic extraction for pose/gaze analysis."""

from .tracking.manager import AssignmentError, SessionNotFoundError, TrackingManager
from .tracking.schemas import BoundingBox, PersonDetection, TrackPacket, TrackedPerson
from .tracking.tracker import IoUPersonTracker

__all__ = [
    "AssignmentError",
    "BoundingBox",
    "IoUPersonTracker",
    "PersonDetection",
    "SessionNotFoundError",
    "TrackPacket",
    "TrackedPerson",
    "TrackingManager",
]

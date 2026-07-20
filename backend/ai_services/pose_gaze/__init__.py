"""Tracking and assignment foundation for the pose/gaze module."""

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

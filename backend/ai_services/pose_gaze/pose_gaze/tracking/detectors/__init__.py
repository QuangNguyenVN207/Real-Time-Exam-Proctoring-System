"""Public exports for person detector adapters."""

from .detectors import (
    OpenCVHOGPersonDetector,
    PersonDetector,
    UltralyticsPersonDetector,
    suppress_duplicate_detections,
)

__all__ = [
    "OpenCVHOGPersonDetector",
    "PersonDetector",
    "UltralyticsPersonDetector",
    "suppress_duplicate_detections",
]

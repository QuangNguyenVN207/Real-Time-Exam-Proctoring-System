"""Public exports for person detector adapters."""

from .detectors import (
    OpenCVHOGPersonDetector,
    PersonDetector,
    UltralyticsPersonDetector,
)

__all__ = [
    "OpenCVHOGPersonDetector",
    "PersonDetector",
    "UltralyticsPersonDetector",
]

"""Public configuration defaults shared by tracking and holistic packages."""

from .settings import (
    DEFAULT_HOLISTIC_CONFIDENCE,
    DEFAULT_HOLISTIC_SOFT_CONFIDENCE,
    DEFAULT_MAX_MISSED_FRAMES,
    DEFAULT_MIN_IOU,
    DEFAULT_PERSON_CONFIDENCE,
    PROJECT_ROOT,
    configure_mediapipe_logging,
)

__all__ = [
    "DEFAULT_HOLISTIC_CONFIDENCE",
    "DEFAULT_HOLISTIC_SOFT_CONFIDENCE",
    "DEFAULT_MAX_MISSED_FRAMES",
    "DEFAULT_MIN_IOU",
    "DEFAULT_PERSON_CONFIDENCE",
    "PROJECT_ROOT",
    "configure_mediapipe_logging",
]

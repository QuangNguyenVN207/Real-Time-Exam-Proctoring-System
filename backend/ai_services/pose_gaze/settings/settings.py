"""Shared defaults for person tracking and holistic landmark extraction."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_PERSON_CONFIDENCE = 0.50
DEFAULT_MIN_IOU = 0.30
DEFAULT_MAX_MISSED_FRAMES = 30

DEFAULT_HOLISTIC_CONFIDENCE = 0.50
DEFAULT_HOLISTIC_SOFT_CONFIDENCE = 0.20


def configure_mediapipe_logging() -> None:
    """Reduce native MediaPipe/TFLite diagnostics before importing MediaPipe.

    ``setdefault`` preserves an explicit logging choice made by an application.
    Native fatal errors and Python exceptions are still surfaced.
    """

    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

"""Shared defaults for person tracking and holistic landmark extraction."""

from __future__ import annotations

import os
from pathlib import Path


# settings.py lives at <repo>/backend/ai_services/pose_gaze/pose_gaze/settings.
# The repository root contains the shared weights/ and test_data_tracking/
# directories; parents[4] points only to <repo>/backend.
PROJECT_ROOT = Path(__file__).resolve().parents[5]

DEFAULT_PERSON_CONFIDENCE = 0.50
DEFAULT_MIN_IOU = 0.30
DEFAULT_MAX_MISSED_FRAMES = 30

DEFAULT_HOLISTIC_CONFIDENCE = 0.30
DEFAULT_HOLISTIC_SOFT_CONFIDENCE = 0.15


def configure_mediapipe_logging() -> None:
    """Reduce native MediaPipe/TFLite diagnostics before importing MediaPipe.

    ``setdefault`` preserves an explicit logging choice made by an application.
    Native fatal errors and Python exceptions are still surfaced.
    """

    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

"""Small OpenCV helpers shared by interactive webcam smoke tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.core.config import settings


def configure_webcam_capture(
    capture: Any,
    *,
    max_width: int = settings.webcam_max_width,
    max_height: int = settings.webcam_max_height,
) -> None:
    """Request a low-latency camera stream without assuming driver support."""

    try:
        import cv2

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except (AttributeError, TypeError):
        # Some camera backends do not expose one or more CAP_PROP settings.
        pass


def resize_live_frame(
    frame: np.ndarray,
    *,
    max_width: int = settings.webcam_max_width,
    max_height: int = settings.webcam_max_height,
) -> np.ndarray:
    """Downscale oversized webcam frames while preserving their aspect ratio."""

    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return frame
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame

    import cv2

    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )

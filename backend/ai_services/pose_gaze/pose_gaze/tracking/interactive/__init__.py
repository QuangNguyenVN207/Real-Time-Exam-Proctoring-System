"""Public exports for non-blocking webcam interaction."""

from .interactive import (
    WebcamInteractionController,
    pump_keyboard_until_frame_deadline,
)

__all__ = [
    "WebcamInteractionController",
    "pump_keyboard_until_frame_deadline",
]

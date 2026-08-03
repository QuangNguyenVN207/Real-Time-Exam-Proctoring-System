"""Public exports for the reusable webcam tracking module."""

from .webcam import (
    PersonTrackingConfig,
    PersonTrackingModule,
    ProcessingRateController,
)

__all__ = [
    "PersonTrackingConfig",
    "PersonTrackingModule",
    "ProcessingRateController",
]

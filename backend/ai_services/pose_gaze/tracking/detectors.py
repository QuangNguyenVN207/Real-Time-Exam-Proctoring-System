"""Optional person detector adapters used by the tracking foundation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .schemas import BoundingBox, PersonDetection


class PersonDetector(Protocol):
    def detect(self, frame) -> list[PersonDetection]:
        """Return people in the supplied OpenCV BGR frame."""


class OpenCVHOGPersonDetector:
    """CPU-only fallback detector; use YOLO in production when available."""

    def __init__(self, *, confidence_threshold: float = 0.35) -> None:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install opencv-python to use OpenCVHOGPersonDetector") from error
        self._cv2 = cv2
        self._confidence_threshold = confidence_threshold
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame) -> list[PersonDetection]:
        boxes, weights = self._hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        detections: list[PersonDetection] = []
        for (x, y, width, height), weight in zip(boxes, weights):
            confidence = min(1.0, max(0.0, float(weight)))
            if confidence >= self._confidence_threshold:
                detections.append(PersonDetection(BoundingBox(x, y, x + width, y + height), confidence))
        return detections


class UltralyticsPersonDetector:
    """YOLO adapter. The supplied model must expose a class named ``person``."""

    def __init__(self, model_path: str | Path, *, confidence_threshold: float = 0.35, device: str | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install ultralytics to use UltralyticsPersonDetector") from error
        self._model = YOLO(str(model_path))
        self._confidence_threshold = confidence_threshold
        self._device = device

    def detect(self, frame) -> list[PersonDetection]:
        result = self._model(frame, conf=self._confidence_threshold, device=self._device, verbose=False)[0]
        if result.boxes is None:
            return []
        detections: list[PersonDetection] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = str(result.names[class_id])
            if class_name != "person":
                continue
            detections.append(
                PersonDetection(
                    bbox=BoundingBox.from_xyxy(box.xyxy[0].tolist()),
                    confidence=float(box.conf[0]),
                    class_name=class_name,
                )
            )
        return detections

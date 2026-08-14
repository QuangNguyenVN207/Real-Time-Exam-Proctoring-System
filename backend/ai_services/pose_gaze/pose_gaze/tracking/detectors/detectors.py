"""Optional person detector adapters used by the tracking foundation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ...settings import DEFAULT_PERSON_CONFIDENCE
from ..schemas import BoundingBox, PersonDetection


class PersonDetector(Protocol):
    def detect(self, frame) -> list[PersonDetection]:
        """Return people in the supplied OpenCV BGR frame."""


class OpenCVHOGPersonDetector:
    """CPU-only fallback detector; use YOLO in production when available."""

    def __init__(self, *, confidence_threshold: float = DEFAULT_PERSON_CONFIDENCE) -> None:
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

    def __init__(self, model_path: str | Path, *, confidence_threshold: float = DEFAULT_PERSON_CONFIDENCE, device: str | None = None, nms_iou: float = 0.6, max_det: int = 20) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install ultralytics to use UltralyticsPersonDetector") from error
        self._model = YOLO(str(model_path))
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._nms_iou = nms_iou
        self._max_det = max_det

    def detect(self, frame) -> list[PersonDetection]:
        result = self._model(
            frame,
            conf=self._confidence_threshold,
            iou=self._nms_iou,
            max_det=self._max_det,
            device=self._device,
            verbose=False,
        )[0]
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
        return suppress_duplicate_detections(detections)


def suppress_duplicate_detections(
    detections: list[PersonDetection],
    *,
    iou_threshold: float = 0.70,
    center_distance_ratio: float = 0.20,
    containment_ratio: float = 0.80,
) -> list[PersonDetection]:
    """Keep strongest detection when boxes strongly indicate one person."""

    kept: list[PersonDetection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for existing in kept:
            left = max(candidate.bbox.x1, existing.bbox.x1)
            top = max(candidate.bbox.y1, existing.bbox.y1)
            right = min(candidate.bbox.x2, existing.bbox.x2)
            bottom = min(candidate.bbox.y2, existing.bbox.y2)
            intersection = max(0.0, right - left) * max(0.0, bottom - top)
            smaller_area = min(candidate.bbox.area, existing.bbox.area)
            candidate_center = ((candidate.bbox.x1 + candidate.bbox.x2) / 2, (candidate.bbox.y1 + candidate.bbox.y2) / 2)
            existing_center = ((existing.bbox.x1 + existing.bbox.x2) / 2, (existing.bbox.y1 + existing.bbox.y2) / 2)
            center_distance = ((candidate_center[0] - existing_center[0]) ** 2 + (candidate_center[1] - existing_center[1]) ** 2) ** 0.5
            scale = max(candidate.bbox.width, candidate.bbox.height, existing.bbox.width, existing.bbox.height)
            contained = smaller_area > 0 and intersection / smaller_area >= containment_ratio
            near_center = center_distance <= center_distance_ratio * scale
            if candidate.bbox.iou(existing.bbox) >= iou_threshold or (contained and near_center):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept

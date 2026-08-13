"""Face smoothing and two-dimensional pose-anchor mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


Point2D = tuple[float, float]


def _finite_point(point: Point2D) -> bool:
    return len(point) == 2 and all(math.isfinite(value) for value in point)


@dataclass(slots=True)
class OneEuroFilter2D:
    """One Euro filter for a point stream with millisecond timestamps."""

    min_cutoff: float = 1.0
    beta: float = 0.007
    derivative_cutoff: float = 1.0
    _value: Point2D | None = None
    _derivative: Point2D = (0.0, 0.0)
    _timestamp_ms: float | None = None

    def reset(self) -> None:
        self._value = None
        self._derivative = (0.0, 0.0)
        self._timestamp_ms = None

    def update(self, point: Point2D, timestamp_ms: float) -> Point2D:
        if not _finite_point(point) or not math.isfinite(timestamp_ms):
            return self._value if self._value is not None else (0.0, 0.0)
        if self._value is None or self._timestamp_ms is None:
            self._value = point
            self._timestamp_ms = timestamp_ms
            return point

        elapsed = max((timestamp_ms - self._timestamp_ms) / 1000.0, 1e-6)
        derivative_alpha = self._alpha(self.derivative_cutoff, elapsed)
        raw_derivative = (
            (point[0] - self._value[0]) / elapsed,
            (point[1] - self._value[1]) / elapsed,
        )
        self._derivative = tuple(
            previous + derivative_alpha * (current - previous)
            for previous, current in zip(self._derivative, raw_derivative)
        )
        speed = math.hypot(*self._derivative)
        value_alpha = self._alpha(self.min_cutoff + self.beta * speed, elapsed)
        self._value = tuple(
            previous + value_alpha * (current - previous)
            for previous, current in zip(self._value, point)
        )
        self._timestamp_ms = timestamp_ms
        return self._value

    @staticmethod
    def _alpha(cutoff: float, elapsed: float) -> float:
        cutoff = max(cutoff, 1e-6)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / elapsed)


@dataclass(frozen=True, slots=True)
class KabschTransform2D:
    scale: float
    rotation: tuple[tuple[float, float], tuple[float, float]]
    translation: Point2D
    residual: float

    def apply(self, point: Point2D) -> Point2D | None:
        if not _finite_point(point):
            return None
        rotated = (
            self.rotation[0][0] * point[0] + self.rotation[0][1] * point[1],
            self.rotation[1][0] * point[0] + self.rotation[1][1] * point[1],
        )
        mapped = (
            self.scale * rotated[0] + self.translation[0],
            self.scale * rotated[1] + self.translation[1],
        )
        return mapped if _finite_point(mapped) else None


@dataclass(frozen=True, slots=True)
class AffineTransform2D:
    matrix: tuple[tuple[float, float, float], tuple[float, float, float]]
    residual: float

    def apply(self, point: Point2D) -> Point2D | None:
        if not _finite_point(point):
            return None
        mapped = (
            self.matrix[0][0] * point[0]
            + self.matrix[0][1] * point[1]
            + self.matrix[0][2],
            self.matrix[1][0] * point[0]
            + self.matrix[1][1] * point[1]
            + self.matrix[1][2],
        )
        return mapped if _finite_point(mapped) else None


def fit_affine_2d(
    source: Sequence[Point2D],
    target: Sequence[Point2D],
) -> AffineTransform2D | None:
    """Fit diagnostic-only affine mapping from four corresponding points."""

    if len(source) != 4 or len(target) != 4:
        return None
    if not all(_finite_point(point) for point in (*source, *target)):
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    matrix, _ = cv2.estimateAffine2D(
        np.asarray(source, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        method=cv2.LMEDS,
    )
    if matrix is None or not np.isfinite(matrix).all():
        return None
    transform = AffineTransform2D(
        tuple(tuple(float(value) for value in row) for row in matrix),
        0.0,
    )
    residual = sum(
        math.dist(transform.apply(point) or (0.0, 0.0), expected)
        for point, expected in zip(source, target)
    ) / len(source)
    return AffineTransform2D(transform.matrix, residual)


def fit_kabsch_2d(
    source: Sequence[Point2D],
    target: Sequence[Point2D],
    *,
    min_baseline: float = 1e-6,
) -> KabschTransform2D | None:
    if len(source) != len(target) or len(source) < 2:
        return None
    if not all(_finite_point(point) for point in (*source, *target)):
        return None
    source_center = _centroid(source)
    target_center = _centroid(target)
    source_centered = [_subtract(point, source_center) for point in source]
    target_centered = [_subtract(point, target_center) for point in target]
    source_norm = sum(_squared_length(point) for point in source_centered)
    target_norm = sum(_squared_length(point) for point in target_centered)
    if source_norm < min_baseline or target_norm < min_baseline:
        return None

    cross = sum(
        source_point[0] * target_point[0] + source_point[1] * target_point[1]
        for source_point, target_point in zip(source_centered, target_centered)
    )
    cross_skew = sum(
        source_point[0] * target_point[1] - source_point[1] * target_point[0]
        for source_point, target_point in zip(source_centered, target_centered)
    )
    angle = math.atan2(cross_skew, cross)
    rotation = ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
    scale = math.sqrt(target_norm / source_norm)
    rotated_center = (
        rotation[0][0] * source_center[0] + rotation[0][1] * source_center[1],
        rotation[1][0] * source_center[0] + rotation[1][1] * source_center[1],
    )
    translation = (
        target_center[0] - scale * rotated_center[0],
        target_center[1] - scale * rotated_center[1],
    )
    transform = KabschTransform2D(scale, rotation, translation, 0.0)
    residual = sum(
        math.dist(transform.apply(point) or (0.0, 0.0), expected)
        for point, expected in zip(source, target)
    ) / len(source)
    return KabschTransform2D(scale, rotation, translation, residual)


def fit_pose_anchor_transform(
    pose_points: dict[int, Point2D],
    *,
    confidence_threshold: float = 0.0,
) -> tuple[KabschTransform2D, str] | None:
    """Fit face-frame coordinates into a canonical torso frame."""

    del confidence_threshold
    for indices, target, source in (
        ((11, 12), ((-0.5, 0.0), (0.5, 0.0)), "shoulders"),
        ((23, 24), ((-0.5, 1.0), (0.5, 1.0)), "hips"),
    ):
        anchors = [pose_points.get(index) for index in indices]
        if any(point is None for point in anchors):
            continue
        transform = fit_kabsch_2d(anchors, target)
        if transform is not None:
            return transform, source
    return None


def _centroid(points: Iterable[Point2D]) -> Point2D:
    points = tuple(points)
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _subtract(left: Point2D, right: Point2D) -> Point2D:
    return left[0] - right[0], left[1] - right[1]


def _squared_length(point: Point2D) -> float:
    return point[0] ** 2 + point[1] ** 2
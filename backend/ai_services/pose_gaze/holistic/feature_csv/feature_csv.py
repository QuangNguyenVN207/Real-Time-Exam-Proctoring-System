"""Stable, two-dimensional Holistic feature schema shared by CSV and XGBoost."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.ai_services.pose_gaze.holistic.landmark import (
    HAND_LANDMARK_INDICES,
    POSE_LANDMARK_INDICES,
    SELECTED_FACE_LANDMARK_INDICES,
    LandmarkPoint,
    TrackHolisticResult,
)
from backend.ai_services.pose_gaze.tracking.schemas import TrackedPerson


CSV_SCHEMA_VERSION = 1
POINT_FIELDS = ("x", "y", "visibility", "presence")

# Prefix, TrackHolisticResult attribute, and the only indexes that are ever
# materialized by the extractor. Keeping this definition in one module makes
# batch-training rows and realtime inference vectors byte-for-byte compatible.
LANDMARK_GROUPS = (
    ("pose", "pose_landmarks", tuple(sorted(POSE_LANDMARK_INDICES))),
    (
        "pose_world",
        "pose_world_landmarks",
        tuple(sorted(POSE_LANDMARK_INDICES)),
    ),
    (
        "left_hand",
        "left_hand_landmarks",
        tuple(sorted(HAND_LANDMARK_INDICES)),
    ),
    (
        "left_hand_world",
        "left_hand_world_landmarks",
        tuple(sorted(HAND_LANDMARK_INDICES)),
    ),
    (
        "right_hand",
        "right_hand_landmarks",
        tuple(sorted(HAND_LANDMARK_INDICES)),
    ),
    (
        "right_hand_world",
        "right_hand_world_landmarks",
        tuple(sorted(HAND_LANDMARK_INDICES)),
    ),
    (
        "face",
        "selected_face_landmarks",
        tuple(sorted(SELECTED_FACE_LANDMARK_INDICES)),
    ),
)


def _point_columns() -> tuple[str, ...]:
    return tuple(
        f"{prefix}_{index:03d}_{field}"
        for prefix, _, indices in LANDMARK_GROUPS
        for index in indices
        for field in POINT_FIELDS
    )


QUALITY_FEATURE_COLUMNS = tuple(
    [f"{prefix}_valid_ratio" for prefix, _, _ in LANDMARK_GROUPS]
    + ["all_landmarks_valid_ratio"]
)
MODEL_FEATURE_COLUMNS = _point_columns() + QUALITY_FEATURE_COLUMNS

# These columns mirror the supplied dataset note. Input manifest headers are
# matched case-insensitively, while output is normalized to snake_case.
ANNOTATION_COLUMNS = (
    "recording_session",
    "actor_ids",
    "action_actor_ids",
    "observed_looking",
    "observed_talking",
    "observed_looking_down",
    "observed_body_turn",
    "observed_hand_reach",
    "observed_sign_code",
    "observed_banned_object",
    "layout",
    "action_start_s",
    "action_end_s",
    "reviewed",
    "duration_s",
    "split_group",
    "exclude_from_training",
    "quality",
    "note",
)

CSV_METADATA_COLUMNS = (
    "schema_version",
    "split",
    "class_code",
    "label",
    "status",
    "error",
    "source_path",
    "source_filename",
    "sequence_id",
    "source_frame_index",
    "frame_id",
    "timestamp_ms",
    "session_id",
    "frame_width",
    "frame_height",
    "person_count",
    "track_id",
    "student_id",
    "track_confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "crop_bbox_x1",
    "crop_bbox_y1",
    "crop_bbox_x2",
    "crop_bbox_y2",
) + ANNOTATION_COLUMNS
CSV_FIELDNAMES = CSV_METADATA_COLUMNS + MODEL_FEATURE_COLUMNS


def _point_values(
    points: tuple[LandmarkPoint, ...],
    *,
    prefix: str,
    indices: tuple[int, ...],
) -> tuple[dict[str, float | None], int]:
    indexed = {point.index: point for point in points}
    values: dict[str, float | None] = {}
    valid_count = 0
    for index in indices:
        point = indexed.get(index)
        if point is not None and point.x is not None and point.y is not None:
            valid_count += 1
        for field in POINT_FIELDS:
            values[f"{prefix}_{index:03d}_{field}"] = (
                getattr(point, field) if point is not None else None
            )
    return values, valid_count


def model_features_from_result(
    result: TrackHolisticResult | None,
) -> dict[str, float | None]:
    """Flatten one result into the exact feature order expected by XGBoost."""

    output: dict[str, float | None] = {
        column: None for column in MODEL_FEATURE_COLUMNS
    }
    if result is None:
        return output

    total_valid = 0
    total_expected = 0
    for prefix, attribute, indices in LANDMARK_GROUPS:
        values, valid_count = _point_values(
            getattr(result, attribute),
            prefix=prefix,
            indices=indices,
        )
        output.update(values)
        total_valid += valid_count
        total_expected += len(indices)
        output[f"{prefix}_valid_ratio"] = (
            valid_count / len(indices) if indices else 0.0
        )
    output["all_landmarks_valid_ratio"] = (
        total_valid / total_expected if total_expected else 0.0
    )
    return output


def build_csv_row(
    *,
    split: str,
    class_code: str,
    label: str,
    status: str,
    source_path: str,
    sequence_id: str | None,
    source_frame_index: int,
    frame_id: int,
    timestamp_ms: int,
    session_id: str,
    frame_width: int | None,
    frame_height: int | None,
    person_count: int,
    track: TrackedPerson | None = None,
    result: TrackHolisticResult | None = None,
    annotation: Mapping[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    """Build one complete CSV row without ever adding a depth coordinate."""

    row: dict[str, Any] = {column: None for column in CSV_FIELDNAMES}
    row.update(
        {
            "schema_version": CSV_SCHEMA_VERSION,
            "split": split,
            "class_code": class_code,
            "label": label,
            "status": status,
            "error": error,
            "source_path": source_path,
            "source_filename": source_path.replace("\\", "/").rsplit("/", 1)[-1],
            "sequence_id": sequence_id,
            "source_frame_index": source_frame_index,
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "session_id": session_id,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "person_count": person_count,
        }
    )
    if annotation is not None:
        for column in ANNOTATION_COLUMNS:
            if column in annotation:
                row[column] = annotation[column]

    if track is not None:
        row.update(
            {
                "track_id": track.track_id,
                "student_id": track.student_id,
                "track_confidence": track.confidence,
                "bbox_x1": track.bbox.x1,
                "bbox_y1": track.bbox.y1,
                "bbox_x2": track.bbox.x2,
                "bbox_y2": track.bbox.y2,
            }
        )
    if result is not None:
        row["student_id"] = result.student_id
        row.update(
            {
                "crop_bbox_x1": result.crop_bbox.x1,
                "crop_bbox_y1": result.crop_bbox.y1,
                "crop_bbox_x2": result.crop_bbox.x2,
                "crop_bbox_y2": result.crop_bbox.y2,
            }
        )
    row.update(model_features_from_result(result))
    return row


__all__ = [
    "ANNOTATION_COLUMNS",
    "CSV_FIELDNAMES",
    "CSV_METADATA_COLUMNS",
    "CSV_SCHEMA_VERSION",
    "LANDMARK_GROUPS",
    "MODEL_FEATURE_COLUMNS",
    "POINT_FIELDS",
    "QUALITY_FEATURE_COLUMNS",
    "build_csv_row",
    "model_features_from_result",
]

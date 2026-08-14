"""Canonical landmark feature-column contract and batch CSV adapter."""

from collections.abc import Mapping
from typing import Any

from ..landmark import (
    HAND_LANDMARK_INDICES,
    POSE_LANDMARK_INDICES,
    SELECTED_FACE_LANDMARK_INDICES,
    LandmarkPoint,
    TrackHolisticResult,
)

CSV_SCHEMA_VERSION = 3

# These fields are annotation metadata, not model inputs. They are kept in
# the batch output so the canonical extractor can be audited/rejoined with
# the manifest without changing the current model feature vector.
ANNOTATION_COLUMNS = (
    "recording_session", "actor_ids", "action_actor_ids",
    "observed_looking", "observed_talking", "observed_looking_down",
    "observed_body_turn", "observed_hand_reach", "observed_sign_code",
    "observed_banned_object", "layout", "action_start_s", "action_end_s",
    "reviewed", "duration_s", "split_group", "exclude_from_training",
    "quality", "note",
)

CSV_METADATA_COLUMNS = (
    "schema_version", "split", "class_code", "label", "status", "error",
    "source_path", "source_filename", "sequence_id", "source_frame_index",
    "frame_id", "timestamp_ms", "session_id", "frame_width", "frame_height",
    "person_count", "track_id", "student_id", "track_confidence",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "crop_bbox_x1",
    "crop_bbox_y1", "crop_bbox_x2", "crop_bbox_y2", "face_observed_mask",
    "face_predicted",
) + ANNOTATION_COLUMNS

_LANDMARK_GROUPS = (
    ("pose", "pose_landmarks", tuple(sorted(POSE_LANDMARK_INDICES)), False),
    ("pose_world", "pose_world_landmarks", tuple(sorted(POSE_LANDMARK_INDICES)), True),
    ("left_hand", "left_hand_landmarks", tuple(sorted(HAND_LANDMARK_INDICES)), False),
    ("left_hand_world", "left_hand_world_landmarks", tuple(sorted(HAND_LANDMARK_INDICES)), True),
    ("right_hand", "right_hand_landmarks", tuple(sorted(HAND_LANDMARK_INDICES)), False),
    ("right_hand_world", "right_hand_world_landmarks", tuple(sorted(HAND_LANDMARK_INDICES)), True),
    ("face", "selected_face_landmarks", tuple(sorted(SELECTED_FACE_LANDMARK_INDICES)), False),
)


def _point_values(points: tuple[LandmarkPoint, ...], prefix: str,
                  indices: tuple[int, ...], world: bool) -> tuple[dict[str, float | None], int]:
    indexed = {point.index: point for point in points}
    values: dict[str, float | None] = {}
    valid = 0
    for index in indices:
        point = indexed.get(index)
        # ``_world_points`` stores MediaPipe world coordinates in x/y; the
        # optional world_x/world_y fields are used only by pseudo-world face
        # mapping. Do not read the latter for pose/hand world groups.
        x = point.x if point else None
        y = point.y if point else None
        values[f"{prefix}_{index:03d}_x"] = x
        values[f"{prefix}_{index:03d}_y"] = y
        if x is not None and y is not None:
            valid += 1
    return values, valid


def model_features_from_result(result: TrackHolisticResult | None) -> dict[str, float | None]:
    output: dict[str, float | None] = {column: None for column in MODEL_FEATURE_COLUMNS}
    if result is None:
        return output
    total_valid = 0
    total_expected = 0
    for prefix, attribute, indices, world in _LANDMARK_GROUPS:
        values, valid = _point_values(getattr(result, attribute), prefix, indices, world)
        output.update(values)
        output[f"{prefix}_valid_ratio"] = valid / len(indices) if indices else 0.0
        total_valid += valid
        total_expected += len(indices)
    output["all_landmarks_valid_ratio"] = total_valid / total_expected if total_expected else 0.0
    return output


def build_csv_row(*, split: str, class_code: str, label: str, status: str,
                  source_path: str, sequence_id: str | None,
                  source_frame_index: int, frame_id: int, timestamp_ms: int,
                  session_id: str, frame_width: int | None, frame_height: int | None,
                  person_count: int, track: Any = None, result: TrackHolisticResult | None = None,
                  annotation: Mapping[str, Any] | None = None, error: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {column: None for column in CSV_FIELDNAMES}
    row.update({
        "schema_version": CSV_SCHEMA_VERSION, "split": split, "class_code": class_code,
        "label": label, "status": status, "error": error, "source_path": source_path,
        "source_filename": source_path.replace("\\", "/").rsplit("/", 1)[-1],
        "sequence_id": sequence_id, "source_frame_index": source_frame_index,
        "frame_id": frame_id, "timestamp_ms": timestamp_ms, "session_id": session_id,
        "frame_width": frame_width, "frame_height": frame_height, "person_count": person_count,
    })
    for column in ANNOTATION_COLUMNS:
        if annotation and column in annotation:
            row[column] = annotation[column]
    if track is not None:
        row.update({"track_id": track.track_id, "student_id": track.student_id,
                    "track_confidence": track.confidence, "bbox_x1": track.bbox.x1,
                    "bbox_y1": track.bbox.y1, "bbox_x2": track.bbox.x2, "bbox_y2": track.bbox.y2})
    if result is not None:
        row.update({"student_id": result.student_id, "crop_bbox_x1": result.crop_bbox.x1,
                    "crop_bbox_y1": result.crop_bbox.y1, "crop_bbox_x2": result.crop_bbox.x2,
                    "crop_bbox_y2": result.crop_bbox.y2, "face_observed_mask": bool(result.face_valid),
                    "face_predicted": bool(result.face_predicted)})
    row.update(model_features_from_result(result))
    return row

def _landmark_columns(prefix: str, indices: range | tuple[int, ...]) -> list[str]:
    return [f"{prefix}_{i:03d}_{field}" for i in indices for field in ("x", "y")]


MODEL_FEATURE_COLUMNS = tuple(
    _landmark_columns("pose", range(25))
    + _landmark_columns("pose_world", range(25))
    + _landmark_columns("left_hand", range(21))
    + _landmark_columns("left_hand_world", range(21))
    + _landmark_columns("right_hand", range(21))
    + _landmark_columns("right_hand_world", range(21))
    + _landmark_columns("face", (0, 1, 2, 7, 10, 13, 14, 17, 33, 37, 39, 40, 54, 58, 61, 67, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 98, 103, 109, 132, 133, 144, 145, 146, 152, 153, 154, 155, 163, 172, 178, 181, 185, 191, 234, 249, 263, 267, 269, 270, 284, 288, 291, 297, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 326, 327, 332, 338, 361, 362, 373, 374, 375, 380, 381, 382, 390, 397, 402, 405, 409, 415, 454))
    + ["pose_valid_ratio", "pose_world_valid_ratio", "left_hand_valid_ratio", "left_hand_world_valid_ratio", "right_hand_valid_ratio", "right_hand_world_valid_ratio", "face_valid_ratio", "all_landmarks_valid_ratio"]
)

CSV_FIELDNAMES = CSV_METADATA_COLUMNS + MODEL_FEATURE_COLUMNS


def is_training_row(row: dict) -> bool:
    return str(row.get("status", "ok")) == "ok" and str(row.get("face_predicted", "False")).lower() not in {"true", "1"}

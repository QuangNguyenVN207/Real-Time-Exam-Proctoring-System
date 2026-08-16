"""Pure, detector-free temporal and pairwise behavior features.

Coordinates are normalized Holistic landmark coordinates.  This module never
uses class labels to create cues and never assigns a cue to another track.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Any, Mapping


POSE_FALLBACK_COLUMNS = (
    "pose_fallback_dx", "pose_fallback_dy", "pose_fallback_displacement",
    "pose_fallback_valid", "pose_fallback_active", "pose_fallback_quality",
    "pose_fallback_anchor_ears", "pose_fallback_left", "pose_fallback_right",
    "pose_fallback_down_deep",
)

BEHAVIOR_COLUMNS = (
    "mouth_aspect_ratio", "mouth_open", "mouth_open_valid", "mouth_activity",
    "mouth_activity_valid", "mouth_velocity", "mouth_open_duration",
    "c6_mouth_prior", "yaw", "pitch", "roll", "yaw_valid", "pitch_valid",
    "roll_valid", "turn_velocity", "torso_lean", "torso_lean_valid",
    "head_pnp_yaw", "head_pnp_pitch", "head_pnp_roll",
    "head_pnp_relative_yaw", "head_pnp_relative_pitch", "head_pnp_relative_roll",
    "head_pnp_left", "head_pnp_right", "head_pnp_up", "head_pnp_down",
    "head_pnp_valid", "head_pnp_reprojection_error", "torso_roll_valid",
    "camera_motion_dx", "camera_motion_dy", "camera_motion_magnitude",
    "camera_motion_valid", "camera_stable",
    *POSE_FALLBACK_COLUMNS,
    "hand_openness", "hand_openness_valid", "finger_shape", "finger_shape_valid",
    "finger_shape_change", "wrist_velocity", "hand_rest", "hand_rest_valid",
    "hand_rest_duration", "hand_retract_speed", "hand_retract_displacement",
    "hand_retract_downward", "hand_lower_torso_hold", "hand_rest_to_retract",
    "hand_raise_speed", "hand_raise_displacement", "hand_upper_hold",
    "hand_rest_to_raise", "hand_signal_shape", "hand_signal_duration",
    "partner_facing_score", "partner_facing_valid", "mutual_look",
    "mutual_look_valid", "hand_to_partner_distance", "hand_shared_zone",
    "partner_hand_exchange", "partner_hand_exchange_valid",
    "partner_directed_downward_look_proxy", "c3_peek_proxy_valid",
    "reciprocal_mouth_activity", "reciprocal_mouth_valid",
)

_EPS = 1e-6
_POSE_FALLBACK_QUALITY_THRESHOLD = 0.30
_POSE_FALLBACK_BASELINE_SECONDS = 1.0
_POSE_FALLBACK_TURN_THRESHOLD = 0.08
_POSE_FALLBACK_DOWN_THRESHOLD = 0.10
_POSE_FALLBACK_DEEP_DISTANCE = 0.12
_HEAD_PNP_YAW_THRESHOLD = 0.12
_HEAD_PNP_PITCH_THRESHOLD = 0.10
_HEAD_PNP_REPROJECTION_THRESHOLD = 0.035
_CAMERA_MOTION_THRESHOLD = 0.035
_CAMERA_MOTION_AGREEMENT = 0.60
_POSE = {0, 11, 12, 15, 16, 23, 24}
_FINGERS = ((4, 2), (8, 5), (12, 9), (16, 13), (20, 17))

# Generic anthropometric face model in millimetres.  The model is used only
# for orientation; monocular translation is deliberately not exposed as a
# behavior feature because its scale is not identifiable without calibration.
_HEAD_PNP_MODEL = {
    1: (0.0, 0.0, 0.0),       # nose tip
    152: (0.0, -63.6, -12.5), # chin
    33: (-43.3, 32.7, -26.0), # left eye outer corner
    263: (43.3, 32.7, -26.0), # right eye outer corner
    61: (-28.9, -28.9, -24.1),
    291: (28.9, -28.9, -24.1),
}


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _point(row: Mapping[str, Any], prefix: str, index: int) -> tuple[float, float] | None:
    x = _finite(row.get(f"{prefix}_{index:03d}_x"))
    y = _finite(row.get(f"{prefix}_{index:03d}_y"))
    return (x, y) if x is not None and y is not None else None


def _dist(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) if a and b else 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _head_pnp_pose(row: Mapping[str, Any]) -> dict[str, float]:
    """Estimate head orientation from six face points with SolvePnP.

    Face coordinates are normalized image coordinates.  A unit focal length
    and principal point (0.5, 0.5) are therefore used as an explicit
    uncalibrated prototype.  The reprojection gate prevents this estimate from
    becoming evidence when the generic 3-D face model does not fit the frame.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"valid": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reprojection_error": 0.0}
    image_points = [_point(row, "face", index) for index in _HEAD_PNP_MODEL]
    if any(point is None for point in image_points):
        return {"valid": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reprojection_error": 0.0}
    object_points = np.asarray([_HEAD_PNP_MODEL[index] for index in _HEAD_PNP_MODEL], dtype=np.float64)
    image_array = np.asarray(image_points, dtype=np.float64)
    camera = np.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)
    try:
        solved, rvec, tvec = cv2.solvePnP(
            object_points, image_array, camera, distortion, flags=cv2.SOLVEPNP_SQPNP
        )
    except cv2.error:
        return {"valid": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reprojection_error": 0.0}
    if not solved:
        return {"valid": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reprojection_error": 0.0}
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera, distortion)
    error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_array, axis=1)))
    if not math.isfinite(error) or error > _HEAD_PNP_REPROJECTION_THRESHOLD:
        return {"valid": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reprojection_error": error}
    rotation, _ = cv2.Rodrigues(rvec)
    # Camera convention: negative yaw=left, positive yaw=right; positive
    # pitch=down in image coordinates.  These signs are validated by tests.
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    pitch = math.atan2(float(-rotation[2, 0]), math.hypot(float(rotation[2, 1]), float(rotation[2, 2])))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    return {"valid": 1.0, "yaw": yaw, "pitch": pitch, "roll": roll, "reprojection_error": error}


def _torso_roll(context: Mapping[str, Any]) -> tuple[float, bool]:
    left, right = context.get("shoulder_left"), context.get("shoulder_right")
    if not left or not right or _dist(left, right) <= _EPS:
        return 0.0, False
    return math.atan2(right[1] - left[1], right[0] - left[0]), True


def _camera_motion_by_frame(rows: list[dict[str, Any]], contexts: list[dict[str, Any]]) -> dict[tuple[str, int], tuple[float, float, float, bool]]:
    """Estimate common torso displacement as a conservative camera gate.

    The feature CSV has no background tracks, so this is intentionally a gate,
    not a camera-motion correction.  A large displacement shared by at least
    two actors invalidates SolvePnP for that frame; ordinary actor motion does
    not invalidate it unless it is common across the scene.
    """
    by_track: dict[str, tuple[tuple[str, int], tuple[float, float]]] = {}
    output: dict[tuple[str, int], tuple[float, float, float, bool]] = {}
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (str(row.get("clip_id", row.get("source_filename", ""))), int(_finite(row.get("frame_id")) or 0))
        context = contexts[index]
        center = context.get("shoulder_mid")
        track = str(row.get("track_id", ""))
        previous = by_track.get(track)
        if center and previous:
            grouped[key].append((center[0] - previous[1][0], center[1] - previous[1][1]))
        if center:
            by_track[track] = (key, center)
    for key, displacements in grouped.items():
        dx = median(value[0] for value in displacements)
        dy = median(value[1] for value in displacements)
        magnitude = math.hypot(dx, dy)
        agreeing = sum(
            math.hypot(px - dx, py - dy) <= max(0.015, magnitude * 0.5)
            for px, py in displacements
        ) / len(displacements)
        camera_motion = magnitude >= _CAMERA_MOTION_THRESHOLD and len(displacements) >= 2 and agreeing >= _CAMERA_MOTION_AGREEMENT
        output[key] = (dx, dy, magnitude, not camera_motion)
    return output


def _pose_quality(row: Mapping[str, Any], index: int) -> float:
    values = [_finite(row.get(f"pose_{index:03d}_{name}")) for name in ("visibility", "presence")]
    values = [value for value in values if value is not None]
    if values:
        return max(values)
    return _finite(row.get("pose_valid_ratio")) or 0.0


def _pose_fallback_context(row: Mapping[str, Any]) -> dict[str, Any]:
    """Compute a pose-only head coordinate in its own coordinate system.

    The coordinate is the head-anchor midpoint relative to the shoulder
    midpoint, normalized by shoulder width. It is deliberately not an angle
    and is never written into yaw/pitch/roll.
    """
    required = [0, 11, 12]
    if any(_point(row, "pose", index) is None for index in required):
        return {"valid": False, "quality": 0.0}
    anchor_source = "ears"
    left_anchor, right_anchor = _point(row, "pose", 7), _point(row, "pose", 8)
    if not left_anchor or not right_anchor:
        anchor_source = "eyes"
        left_anchor, right_anchor = _point(row, "pose", 3), _point(row, "pose", 6)
    if not left_anchor or not right_anchor:
        return {"valid": False, "quality": 0.0}
    indices = [0, 11, 12, 7 if anchor_source == "ears" else 3, 8 if anchor_source == "ears" else 6]
    quality = min(_pose_quality(row, index) for index in indices)
    shoulder_mid = (
        (_point(row, "pose", 11)[0] + _point(row, "pose", 12)[0]) / 2.0,
        (_point(row, "pose", 11)[1] + _point(row, "pose", 12)[1]) / 2.0,
    )
    head_mid = ((left_anchor[0] + right_anchor[0]) / 2.0, (left_anchor[1] + right_anchor[1]) / 2.0)
    scale = max(_dist(_point(row, "pose", 11), _point(row, "pose", 12)), 0.05)
    valid = quality >= _POSE_FALLBACK_QUALITY_THRESHOLD
    return {
        "valid": valid,
        "quality": quality,
        "anchor_ears": float(anchor_source == "ears"),
        "head_x": (head_mid[0] - shoulder_mid[0]) / scale,
        "head_y": (head_mid[1] - shoulder_mid[1]) / scale,
    }


def _pose_fallback_baselines(
    rows: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    by_track: Mapping[str, list[int]],
) -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}
    for track, indices in by_track.items():
        valid = [index for index in indices if contexts[index].get("valid")]
        if not valid:
            continue
        first_time = _frame_time(rows[valid[0]])
        selected = [index for index in valid if _frame_time(rows[index]) <= first_time + _POSE_FALLBACK_BASELINE_SECONDS] or valid[:1]
        baselines[track] = {
            "head_x": median(float(contexts[index]["head_x"]) for index in selected),
            "head_y": median(float(contexts[index]["head_y"]) for index in selected),
        }
    return baselines


def derive_pose_fallback_features(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Return only the independent pose fallback branch for one video.

    The input rows are not modified. This is used by the streaming rebuild so
    every legacy CSV value can be copied byte-for-byte while pose features are
    appended as a new branch.
    """
    contexts = [_pose_fallback_context(row) for row in rows]
    by_track: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_track[str(row.get("track_id", ""))].append(index)
    baselines = _pose_fallback_baselines(rows, contexts, by_track)
    output: list[dict[str, float]] = []
    for row, context in zip(rows, contexts):
        baseline = baselines.get(str(row.get("track_id", "")))
        valid = bool(context.get("valid") and baseline)
        dx = float(context["head_x"] - baseline["head_x"]) if valid else 0.0
        dy = float(context["head_y"] - baseline["head_y"]) if valid else 0.0
        displacement = math.hypot(dx, dy) if valid else 0.0
        face_left, face_right = _point(row, "face", 33), _point(row, "face", 263)
        face_nose = _point(row, "face", 1)
        face_mouth_left, face_mouth_right = _point(row, "face", 61), _point(row, "face", 291)
        face_active = not (
            face_nose and face_left and face_right and face_mouth_left and face_mouth_right
            and _dist(face_left, face_right) > _EPS and _dist(face_mouth_left, face_mouth_right) > _EPS
        )
        output.append({
            "pose_fallback_dx": dx,
            "pose_fallback_dy": dy,
            "pose_fallback_displacement": displacement,
            "pose_fallback_valid": float(valid),
            "pose_fallback_active": float(face_active),
            "pose_fallback_quality": float(context.get("quality", 0.0)),
            "pose_fallback_anchor_ears": float(context.get("anchor_ears", 0.0)) if valid else 0.0,
            "pose_fallback_left": float(valid and dx < -_POSE_FALLBACK_TURN_THRESHOLD),
            "pose_fallback_right": float(valid and dx > _POSE_FALLBACK_TURN_THRESHOLD),
            "pose_fallback_down_deep": float(valid and dy > _POSE_FALLBACK_DOWN_THRESHOLD and displacement > _POSE_FALLBACK_DEEP_DISTANCE),
        })
    return output


def _body_context(row: Mapping[str, Any]) -> dict[str, Any]:
    points = {index: _point(row, "pose", index) for index in _POSE}
    valid = [point for point in points.values() if point is not None]
    if len(valid) < 4:
        return {"valid": False}
    shoulder_mid = (
        (points[11][0] + points[12][0]) / 2.0,
        (points[11][1] + points[12][1]) / 2.0,
    ) if points[11] and points[12] else None
    hip_mid = (
        (points[23][0] + points[24][0]) / 2.0,
        (points[23][1] + points[24][1]) / 2.0,
    ) if points[23] and points[24] else None
    scale = _dist(points[11], points[12])
    if scale < _EPS:
        scale = _dist(points[23], points[24])
    scale = max(scale, 0.05)
    body_points = [points[index] for index in (0, 11, 12, 23, 24) if points[index]]
    xs, ys = zip(*body_points)
    return {
        "valid": bool(shoulder_mid and hip_mid),
        "shoulder_mid": shoulder_mid,
        "shoulder_left": points[11],
        "shoulder_right": points[12],
        "hip_mid": hip_mid,
        "scale": scale,
        "bbox": (min(xs), min(ys), max(xs), max(ys)),
        "nose": points[0],
        "left_wrist": points[15],
        "right_wrist": points[16],
    }


def _distance_to_box(point: tuple[float, float] | None, box: tuple[float, float, float, float] | None) -> float:
    if not point or not box:
        return float("inf")
    x, y = point
    return math.hypot(max(box[0] - x, 0.0, x - box[2]), max(box[1] - y, 0.0, y - box[3]))


def _mouth(row: Mapping[str, Any]) -> tuple[float, bool]:
    upper, lower = _point(row, "face", 13), _point(row, "face", 14)
    left, right = _point(row, "face", 61), _point(row, "face", 291)
    width = _dist(left, right)
    return (_dist(upper, lower) / width, width > _EPS) if upper and lower and width > _EPS else (0.0, False)


def _hand_values(row: Mapping[str, Any], side: str, context: dict[str, Any], previous: Mapping[str, Any] | None, dt_s: float) -> dict[str, float]:
    prefix = f"{side}_hand"
    wrist = _point(row, prefix, 0)
    points = {index: _point(row, prefix, index) for index in range(21)}
    valid_points = [point for point in points.values() if point is not None]
    valid = bool(wrist and len(valid_points) >= 15 and context.get("valid"))
    if not valid:
        return {"valid": 0.0, "wrist": wrist, "openness": 0.0, "shape": 0.0, "velocity": 0.0}
    hand_scale = max(_dist(points[0], points[9]), 0.03)
    extended = []
    for tip, mcp in _FINGERS:
        extended.append(float(_dist(points[tip], points[0]) > _dist(points[mcp], points[0]) * 1.25))
    openness = sum(extended) / len(extended)
    shape = max(openness, 0.75 if sum(extended) == 1 else 0.0)
    previous_wrist = previous.get(f"_{side}_wrist") if previous else None
    velocity = _dist(wrist, previous_wrist) / max(dt_s, 1e-3) / max(context["scale"], 0.03) if previous_wrist else 0.0
    return {"valid": 1.0, "wrist": wrist, "openness": openness, "shape": shape, "velocity": velocity, "hand_scale": hand_scale}


def _frame_time(row: Mapping[str, Any]) -> float:
    value = _finite(row.get("timestamp_ms"))
    return value / 1000.0 if value is not None else 0.0


def derive_behavior_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add behavior features to rows belonging to one video.

    Rows remain metadata-preserving.  All added values are finite scalars;
    invalid measurements are zero with an explicit validity feature.
    """
    if not rows:
        return rows
    rows.sort(key=lambda row: (int(_finite(row.get("frame_id")) or 0), str(row.get("track_id", ""))))
    contexts = [_body_context(row) for row in rows]
    camera_motion = _camera_motion_by_frame(rows, contexts)
    pnp_by_track: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        pose = _head_pnp_pose(row)
        if pose["valid"] and not _truthy(row.get("face_predicted")):
            pnp_by_track[str(row.get("track_id", ""))].append(pose)
    pnp_baselines = {
        track: {
            name: median(pose[name] for pose in poses[: min(len(poses), 30)])
            for name in ("yaw", "pitch", "roll")
        }
        for track, poses in pnp_by_track.items() if poses
    }
    pose_fallback_contexts = [_pose_fallback_context(row) for row in rows]
    frame_tracks: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        frame_tracks[(str(row.get("frame_id", "")), int(_finite(row.get("timestamp_ms")) or 0))].append(index)
    by_track: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_track[str(row.get("track_id", ""))].append(index)
    pose_fallback_baselines = _pose_fallback_baselines(rows, pose_fallback_contexts, by_track)

    mouth_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        mouth, valid = _mouth(row)
        if valid:
            mouth_values[str(row.get("track_id", ""))].append(mouth)
    baselines = {track: median(values) if values else 0.0 for track, values in mouth_values.items()}

    states: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        context = contexts[index]
        track = str(row.get("track_id", ""))
        previous = states.get(track)
        previous_time = previous.get("time") if previous else None
        dt_s = max(_frame_time(row) - previous_time, 1 / 30) if previous_time is not None else 1 / 30
        values = {column: 0.0 for column in BEHAVIOR_COLUMNS}

        pnp = _head_pnp_pose(row)
        frame_key = (str(row.get("clip_id", row.get("source_filename", ""))), int(_finite(row.get("frame_id")) or 0))
        camera_dx, camera_dy, camera_magnitude, camera_stable = camera_motion.get(
            frame_key, (0.0, 0.0, 0.0, True)
        )
        camera_motion_valid = float(frame_key in camera_motion)
        pnp_valid = bool(
            pnp["valid"] and camera_stable and not _truthy(row.get("face_predicted"))
        )
        pnp_baseline = pnp_baselines.get(track, {})
        torso_roll, torso_roll_valid = _torso_roll(context)
        relative_yaw = pnp["yaw"] - pnp_baseline.get("yaw", pnp["yaw"]) if pnp_valid else 0.0
        relative_pitch = pnp["pitch"] - pnp_baseline.get("pitch", pnp["pitch"]) if pnp_valid else 0.0
        relative_roll = (pnp["roll"] - torso_roll) - pnp_baseline.get("roll", pnp["roll"]) if pnp_valid and torso_roll_valid else 0.0
        values.update({
            "head_pnp_yaw": pnp["yaw"] if pnp_valid else 0.0,
            "head_pnp_pitch": pnp["pitch"] if pnp_valid else 0.0,
            "head_pnp_roll": pnp["roll"] if pnp_valid else 0.0,
            "head_pnp_relative_yaw": relative_yaw,
            "head_pnp_relative_pitch": relative_pitch,
            "head_pnp_relative_roll": relative_roll,
            "head_pnp_left": float(pnp_valid and relative_yaw <= -_HEAD_PNP_YAW_THRESHOLD),
            "head_pnp_right": float(pnp_valid and relative_yaw >= _HEAD_PNP_YAW_THRESHOLD),
            "head_pnp_up": float(pnp_valid and relative_pitch <= -_HEAD_PNP_PITCH_THRESHOLD),
            "head_pnp_down": float(pnp_valid and relative_pitch >= _HEAD_PNP_PITCH_THRESHOLD),
            "head_pnp_valid": float(pnp_valid),
            "head_pnp_reprojection_error": pnp["reprojection_error"],
            "torso_roll_valid": float(torso_roll_valid),
            "camera_motion_dx": camera_dx,
            "camera_motion_dy": camera_dy,
            "camera_motion_magnitude": camera_magnitude,
            "camera_motion_valid": camera_motion_valid,
            "camera_stable": float(camera_stable),
        })

        mouth, mouth_valid = _mouth(row)
        baseline = baselines.get(track, 0.0)
        mouth_active = mouth_valid and baseline > 0 and (mouth > baseline + 0.08 or mouth / max(baseline, 0.02) > 1.35)
        mouth_velocity = (mouth - previous.get("mouth", mouth)) / dt_s if previous and mouth_valid and previous.get("mouth_valid") else 0.0
        mouth_duration = previous.get("mouth_duration", 0.0) + dt_s if mouth_active and previous and previous.get("mouth_active") else (dt_s if mouth_active else 0.0)
        values.update({
            "mouth_aspect_ratio": mouth,
            "mouth_open": float(mouth_active),
            "mouth_open_valid": float(mouth_valid),
            "mouth_activity": max(0.0, abs(mouth - baseline) / max(baseline, 0.02)) if mouth_valid and baseline else 0.0,
            "mouth_activity_valid": float(mouth_valid and baseline > 0),
            "mouth_velocity": mouth_velocity,
            "mouth_open_duration": mouth_duration,
            "c6_mouth_prior": _clamp((mouth_duration / 0.4) * min(1.0, max(0.0, (abs(mouth - baseline) / max(baseline, 0.02)) / 0.35))) if mouth_valid else 0.0,
        })

        nose = _point(row, "face", 1)
        face_left, face_right = _point(row, "face", 33), _point(row, "face", 263)
        mouth_left, mouth_right = _point(row, "face", 61), _point(row, "face", 291)
        face_width = _dist(face_left, face_right)
        face_height = _dist(mouth_left, mouth_right)
        yaw_valid = bool(nose and face_left and face_right and face_width > _EPS)
        pitch_valid = bool(nose and mouth_left and mouth_right and face_height > _EPS)
        roll_valid = bool(face_left and face_right and face_width > _EPS)
        yaw = ((nose[0] - (face_left[0] + face_right[0]) / 2.0) / face_width) if yaw_valid else 0.0
        pitch = ((nose[1] - (mouth_left[1] + mouth_right[1]) / 2.0) / face_height) if pitch_valid else 0.0
        roll = (math.atan2(face_right[1] - face_left[1], face_right[0] - face_left[0]) / math.pi) if roll_valid else 0.0
        turn_velocity = abs(yaw - previous.get("yaw", yaw)) / dt_s if previous and yaw_valid and previous.get("yaw_valid") else 0.0
        pose_head = pose_fallback_contexts[index]
        pose_baseline = pose_fallback_baselines.get(track)
        pose_valid = bool(pose_head.get("valid") and pose_baseline)
        pose_dx = float(pose_head["head_x"] - pose_baseline["head_x"]) if pose_valid else 0.0
        pose_dy = float(pose_head["head_y"] - pose_baseline["head_y"]) if pose_valid else 0.0
        pose_displacement = math.hypot(pose_dx, pose_dy) if pose_valid else 0.0
        pose_left = pose_valid and pose_dx < -_POSE_FALLBACK_TURN_THRESHOLD
        pose_right = pose_valid and pose_dx > _POSE_FALLBACK_TURN_THRESHOLD
        pose_down_deep = pose_valid and pose_dy > _POSE_FALLBACK_DOWN_THRESHOLD and pose_displacement > _POSE_FALLBACK_DEEP_DISTANCE
        shoulder_mid, hip_mid = context.get("shoulder_mid"), context.get("hip_mid")
        torso_lean = abs(hip_mid[0] - shoulder_mid[0]) / context["scale"] if context.get("valid") else 0.0
        values.update({
            "yaw": yaw, "pitch": pitch, "roll": roll,
            "yaw_valid": float(yaw_valid), "pitch_valid": float(pitch_valid),
            "roll_valid": float(roll_valid), "turn_velocity": turn_velocity,
            "torso_lean": torso_lean,
            "torso_lean_valid": float(context.get("valid", False)),
            "pose_fallback_dx": pose_dx,
            "pose_fallback_dy": pose_dy,
            "pose_fallback_displacement": pose_displacement,
            "pose_fallback_valid": float(pose_valid),
            "pose_fallback_active": float(not (yaw_valid and pitch_valid and roll_valid)),
            "pose_fallback_quality": float(pose_head.get("quality", 0.0)),
            "pose_fallback_anchor_ears": float(pose_head.get("anchor_ears", 0.0)) if pose_valid else 0.0,
            "pose_fallback_left": float(pose_left),
            "pose_fallback_right": float(pose_right),
            "pose_fallback_down_deep": float(pose_down_deep),
        })

        hands = [_hand_values(row, "left", context, previous, dt_s), _hand_values(row, "right", context, previous, dt_s)]
        valid_hands = [hand for hand in hands if hand["valid"]]
        hand = max(valid_hands, key=lambda value: value["velocity"], default={"valid": 0.0, "wrist": None, "openness": 0.0, "shape": 0.0, "velocity": 0.0})
        values.update({"hand_openness": hand["openness"], "hand_openness_valid": hand["valid"], "finger_shape": hand["shape"], "finger_shape_valid": hand["valid"], "wrist_velocity": hand["velocity"]})

        rest_flags, lower_flags, upper_flags = [], [], []
        for candidate in hands:
            wrist = candidate.get("wrist")
            if not wrist or not context.get("valid"):
                continue
            shoulder_y, hip_y = shoulder_mid[1], hip_mid[1]
            torso_height = max(abs(hip_y - shoulder_y), 0.05)
            rest_flags.append(shoulder_y + 0.12 * torso_height <= wrist[1] <= hip_y + 0.25 * torso_height and candidate["velocity"] < 0.15)
            lower_flags.append(wrist[1] > hip_y - 0.05 * torso_height)
            upper_flags.append(wrist[1] < shoulder_y + 0.45 * torso_height)
        rest = bool(any(rest_flags))
        lower = bool(any(lower_flags))
        upper = bool(any(upper_flags))
        rest_duration = previous.get("rest_duration", 0.0) + dt_s if rest and previous and previous.get("rest") else (dt_s if rest else 0.0)
        lower_duration = previous.get("lower_duration", 0.0) + dt_s if lower and previous and previous.get("lower") else (dt_s if lower else 0.0)
        upper_duration = previous.get("upper_duration", 0.0) + dt_s if upper and previous and previous.get("upper") else (dt_s if upper else 0.0)
        current_wrist = hand.get("wrist")
        previous_wrist = previous.get("wrist") if previous else None
        displacement = _dist(current_wrist, previous_wrist) / max(context.get("scale", 0.05), 0.05) if current_wrist and previous_wrist else 0.0
        dy = (current_wrist[1] - previous_wrist[1]) / max(context.get("scale", 0.05), 0.05) if current_wrist and previous_wrist else 0.0
        speed = hand["velocity"]
        retract = bool(previous and previous.get("rest_duration", 0.0) >= 0.6 and speed > 0.8 and displacement >= 0.35 and dy >= 0.20)
        raise_event = bool(previous and previous.get("rest_duration", 0.0) >= 0.6 and speed > 0.8 and displacement >= 0.30 and dy <= -0.20)
        shape_change = abs(hand["shape"] - previous.get("shape", hand["shape"])) if previous else 0.0
        signal = hand["shape"] if hand["valid"] and (upper or raise_event) else 0.0
        signal_duration = previous.get("signal_duration", 0.0) + dt_s if signal >= 0.75 and previous and previous.get("signal", 0.0) >= 0.75 else (dt_s if signal >= 0.75 else 0.0)
        values.update({
            "hand_rest": float(rest), "hand_rest_valid": float(bool(valid_hands)), "hand_rest_duration": rest_duration,
            "hand_retract_speed": speed if dy >= 0 else 0.0, "hand_retract_displacement": displacement if dy >= 0 else 0.0,
            "hand_retract_downward": max(0.0, dy), "hand_lower_torso_hold": lower_duration,
            "hand_rest_to_retract": float(retract), "hand_raise_speed": speed if dy <= 0 else 0.0,
            "hand_raise_displacement": displacement if dy <= 0 else 0.0, "hand_upper_hold": upper_duration,
            "hand_rest_to_raise": float(raise_event), "hand_signal_shape": signal,
            "hand_signal_duration": signal_duration, "finger_shape_change": shape_change,
        })

        states[track] = {"time": _frame_time(row), "mouth": mouth, "mouth_valid": mouth_valid, "mouth_active": mouth_active, "mouth_duration": mouth_duration, "rest": rest, "rest_duration": rest_duration, "lower": lower, "lower_duration": lower_duration, "upper": upper, "upper_duration": upper_duration, "wrist": current_wrist, "_left_wrist": hands[0].get("wrist"), "_right_wrist": hands[1].get("wrist"), "shape": hand["shape"], "signal": signal, "yaw": yaw, "yaw_valid": yaw_valid}
        row.update(values)

    # Pairwise pass after per-track features. Only same-frame reliable tracks participate.
    # Compute facing scores first. This avoids frame-order asymmetry when the
    # partner row appears later in the list.
    facing_scores: dict[int, tuple[float, int, float]] = {}
    for index, row in enumerate(rows):
        frame_key = (str(row.get("frame_id", "")), int(_finite(row.get("timestamp_ms")) or 0))
        candidates = [candidate for candidate in frame_tracks[frame_key] if candidate != index and contexts[candidate].get("valid")]
        own = contexts[index]
        if not candidates or not own.get("valid"):
            continue
        partner_index = min(candidates, key=lambda candidate: _dist(own.get("shoulder_mid"), contexts[candidate].get("shoulder_mid")))
        partner = contexts[partner_index]
        direction = 1.0 if partner["shoulder_mid"][0] >= own["shoulder_mid"][0] else -1.0
        nose = own.get("nose")
        facing_raw = ((nose[0] - own["shoulder_mid"][0]) * direction / own["scale"]) if nose else -1.0
        # Neutral nose position is not evidence of looking at partner.
        facing = _clamp((facing_raw - 0.02) / 0.10)
        facing_scores[index] = (facing, partner_index, direction)

    for index, row in enumerate(rows):
        if index not in facing_scores:
            continue
        facing, partner_index, direction = facing_scores[index]
        own = contexts[index]
        partner = contexts[partner_index]
        partner_facing = facing_scores.get(partner_index, (0.0, index, 0.0))[0]
        mutual = facing >= 0.55 and partner_facing >= 0.55
        wrist_points = [_point(row, "left_hand", 0), _point(row, "right_hand", 0)]
        wrist_points = [point for point in wrist_points if point]
        distance = min((_distance_to_box(point, partner["bbox"]) / own["scale"] for point in wrist_points), default=0.0)
        shared = float(any(direction * (point[0] - own["shoulder_mid"][0]) > 0 and direction * (point[0] - partner["shoulder_mid"][0]) < 0 for point in wrist_points))
        exchange = float(shared and distance < 0.45 and (float(row.get("wrist_velocity", 0.0)) > 0.25 or float(row.get("hand_rest_to_raise", 0.0)) > 0.0 or float(row.get("hand_rest_to_retract", 0.0)) > 0.0))
        partner_mouth = float(rows[partner_index].get("mouth_open", 0.0))
        reciprocal = float(row.get("mouth_open", 0.0) or partner_mouth)
        row.update({
            "partner_facing_score": facing, "partner_facing_valid": 1.0,
            "mutual_look": float(mutual), "mutual_look_valid": 1.0,
            "hand_to_partner_distance": distance, "hand_shared_zone": shared,
            "partner_hand_exchange": exchange, "partner_hand_exchange_valid": 1.0,
            "partner_directed_downward_look_proxy": float(mutual and float(row.get("pitch", 0.0)) > 0.30 and float(row.get("torso_lean", 0.0)) > 0.30),
            "c3_peek_proxy_valid": 1.0, "reciprocal_mouth_activity": reciprocal,
            "reciprocal_mouth_valid": 1.0,
        })
        if mutual and reciprocal:
            row["c6_mouth_prior"] = max(float(row.get("c6_mouth_prior", 0.0)), 0.5)
    return rows


__all__ = ["BEHAVIOR_COLUMNS", "POSE_FALLBACK_COLUMNS", "derive_behavior_features", "derive_pose_fallback_features"]

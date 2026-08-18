"""Actor-only behavior classifier for c2/c3/c7.

Contract:
* Manifest supplies actor truth through action_actor_ids and class_code.
* Full video is scanned; action_start_s/action_end_s define truth onset/end,
  never an input crop.
* XGBoost predicts c2/c3/c5/c7 per actor-frame.
* Final decision is one state per (video, actor_id), selected from strongest
  actor evidence. No video-level label or metric is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .causal_stream import CausalActorWindow, CausalSpecialistState

TARGET_CLASSES = ("c2", "c3", "c7")
MODEL_CLASSES = ("c2", "c3", "c5", "c7")
EXCLUDE_C7 = False
EXTENDED_SUSPICIOUS = False
FRAME_FLAG_THRESHOLD = 0.5
CAUSAL_WARMUP_FRAMES = 15

# Selected at runtime from numeric actor-frame geometry columns. Labels,
# identity, time, semantic flags, and video metadata are excluded.
FEATURES = ()
LEGACY_C7_FORMULA = True
ACTOR_BASE_FEATURES = (
    "near_midpoint_pre_cross", "pair_hand_distance", "own_side_distance",
    "hand_speed", "hand_raise", "hand_motion", "peer_face_displacement",
    "peer_torso_displacement", "delta_nose_down", "delta_mouth_down",
    "delta_eye_down",
)
POSE_C3_FEATURES = (
    "c3_face_roll", "c3_ear_roll", "c3_nose_eye_lateral", "c3_nose_eye_vertical",
    "c3_nose_ear_lateral", "c3_nose_ear_vertical", "c3_shoulder_roll",
    "c3_shoulder_dy", "c3_shoulder_dx", "c3_head_shoulder_roll",
    "c3_head_toward_peer", "c3_shoulder_toward_peer", "c3_face_roll_delta",
    "c3_ear_roll_delta", "c3_shoulder_roll_delta", "c3_shoulder_dy_delta",
    "c3_shoulder_dx_delta", "c3_head_shoulder_roll_delta",
    "c3_face_roll_velocity", "c3_ear_roll_velocity",
    "c3_shoulder_roll_velocity", "c3_head_toward_peer_velocity",
    "head_pnp_relative_yaw", "head_pnp_relative_pitch", "head_pnp_relative_roll",
    "head_pnp_left", "head_pnp_right", "head_pnp_up", "head_pnp_down",
    "head_pnp_valid", "camera_stable",
)
FACE_C3_FEATURES = (
    "face_c3_roll", "face_c3_cheek_roll",
    "face_c3_nose_eye_along", "face_c3_nose_eye_perp",
    "face_c3_nose_cheek_along", "face_c3_nose_cheek_perp",
    "face_c3_mouth_slope", "face_c3_roll_delta", "face_c3_cheek_roll_delta",
    "face_c3_nose_eye_along_delta", "face_c3_nose_eye_perp_delta",
    "face_c3_nose_cheek_along_delta", "face_c3_nose_cheek_perp_delta",
    "face_c3_mouth_slope_delta",
)
C3_FEATURES = (*POSE_C3_FEATURES, *FACE_C3_FEATURES, "face_quality_mask", "pose_quality_mask")
# Contract-faithful A/B branch.  This is deliberately separate from the
# historical C3 specialist so the old benchmark remains reproducible.
C3_POSE_ONLY_BASE_FEATURES = (
    "c3_pose_head_peer_delta", "c3_pose_torso_peer_delta",
    "c3_pose_head_peer_velocity", "c3_pose_torso_peer_velocity",
    "c3_pose_head_valid", "c3_pose_torso_valid", "c3_pose_peer_valid",
    "c3_pose_required_valid",
)
C3_POSE_ONLY_FEATURES = C3_POSE_ONLY_BASE_FEATURES
STRICT_C3_SUSPICIOUS_FEATURES = (
    "delta_nose_down", "hand_motion", "finger_motion", "hand_finger_motion",
    "own_side_distance", "strict_hand_below_hip", "strict_hand_quality",
    "strict_head_down_delta", "strict_midpoint_hit",
    "strict_own_side_outside_midpoint",
)
C2_CUE_FEATURES = (
    "near_midpoint_pre_cross", "pair_hand_distance", "own_side_distance",
    "crossing_indicator", "pair_convergence", "hand_direction",
    "head_turn_to_hand_approach_frames",
    "hand_speed", "finger_speed", "finger_motion", "hand_finger_motion",
    "hand_quality_mask", "pose_quality_mask",
)
C7_CUE_FEATURES = (
    "finger_speed", "finger_motion", "hand_finger_motion",
    "finger_extension_mean", "finger_extension_std", "finger_shape_change",
    "raise_displacement", "raise_speed", "own_side_distance",
    "hand_visibility", "hand_speed", "hand_motion",
    "hand_quality_mask", "c7_not_midpoint", "c7_not_shared_zone",
    "c7_shared_zone", "c7_toward_midpoint", "c7_raise_directional",
)
C7_CUE_FEATURES_LEGACY = (
    "finger_speed", "finger_motion", "hand_finger_motion",
    "finger_extension_mean", "finger_extension_std", "finger_shape_change",
    "raise_displacement", "raise_speed", "own_side_distance",
    "hand_visibility", "hand_speed", "hand_motion",
    "hand_quality_mask", "c7_not_midpoint", "c7_not_shared_zone",
)
HAND_JITTER_C7_CUE_FEATURES = (
    "c7_selected_hand_valid", "c7_selected_hand_coherence",
    "c7_selected_hand_jitter_speed", "c7_selected_hand_gap_frames",
    "c7_selected_hand_raise_displacement", "c7_selected_hand_raise_speed",
    "c7_selected_hand_shape_change", "c7_selected_hand_own_side_distance",
    "c7_selected_hand_near_midpoint", "c7_selected_hand_shared_zone",
    "c7_selected_hand_finger_valid", "c7_selected_hand_wrist_valid",
)
# Controlled normalized A/B branch: actor-scale or baseline-relative cues
# only. Absolute image/pixel layout coordinates are intentionally excluded.
NORMALIZED_FEATURES = {
    "delta_nose_down", "delta_mouth_down", "delta_eye_down", "head_down_valid",
    "peer_face_displacement", "peer_face_toward",
    "peer_torso_displacement", "peer_torso_toward",
    "hand_speed", "hand_motion",
    "face_valid", "mouth_valid", "left_hand_valid", "right_hand_valid",
    "pose_valid_ratio",
    "face_quality_mask", "pose_quality_mask", "hand_quality_mask",
    *C3_FEATURES,
    "head_pnp_relative_yaw", "head_pnp_relative_pitch", "head_pnp_relative_roll",
    "head_pnp_left", "head_pnp_right", "head_pnp_up", "head_pnp_down",
    "head_pnp_valid", "camera_stable",
    "finger_motion", "finger_speed", "hand_finger_motion",
}
BLOCKED_FEATURES = {
    "clip_id", "filename", "split", "split_group", "class_code", "actor_id",
    "track_id", "spatial_role", "mapping_confidence", "is_action_actor",
    "source_frame_index", "frame_id", "timestamp_ms", "dt_ms", "baseline_source",
    "selected_exchange_point", "head_down_candidate",
    "source_filename", "source_path", "sequence_id", "target_state", "protocol",
    "interval_used_for_features", "video_class_code", "action_actor_id",
    "peer_actor_id", "actor_role", "track_side",
    "source_actor", "within_action_interval", "excluded_source", "action_start_s",
    "action_end_s", "manifest_class_code", "truth", "actor_truth", "actor_label",
    "relative_frame_progress",
    # Provenance/layout fields are not behavior evidence.  In particular,
    # paired_valid_source_frame is derived from the audit procedure and
    # actor_side is an identity/layout assignment.
    "paired_valid_source_frame", "actor_side",
    "interaction_peer_ids",
    "c7_selected_hand_left", "c7_selected_hand_right",
    "c7_shared_zone", "c7_toward_midpoint", "c7_raise_directional",
    "c7_selected_hand_valid", "c7_selected_hand_coherence",
    "c7_selected_hand_jitter_speed", "c7_selected_hand_gap_frames",
    "c7_selected_hand_raise_displacement", "c7_selected_hand_raise_speed",
    "c7_selected_hand_shape_change", "c7_selected_hand_own_side_distance",
    "c7_selected_hand_near_midpoint", "c7_selected_hand_shared_zone",
    "c7_selected_hand_finger_valid", "c7_selected_hand_wrist_valid",
    "peer_face_displacement", "peer_face_toward", "peer_torso_displacement",
    "peer_torso_toward", "hand_speed", "hand_motion", "hand_raise",
    *C3_FEATURES,
}


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


HEAD_PNP_MODEL = {
    1: (0.0, 0.0, 0.0),
    152: (0.0, -63.6, -12.5),
    33: (-43.3, 32.7, -26.0),
    263: (43.3, 32.7, -26.0),
    61: (-28.9, -28.9, -24.1),
    291: (28.9, -28.9, -24.1),
}
HEAD_PNP_REPROJECTION_THRESHOLD = 0.035
HEAD_PNP_DIRECTION_DEG = 25.0
HEAD_PNP_EMA_ALPHA = 0.4
CAMERA_MOTION_THRESHOLD = 0.035
CAMERA_MOTION_AGREEMENT = 0.60
HEAD_FRAME_WIDTH = 1920.0
HEAD_FRAME_HEIGHT = 1080.0
HEAD_PNP_FEATURES = (
    "head_pnp_yaw", "head_pnp_pitch", "head_pnp_roll",
    "head_pnp_relative_yaw", "head_pnp_relative_pitch", "head_pnp_relative_roll",
    "head_pnp_left", "head_pnp_right", "head_pnp_up", "head_pnp_down",
    "head_pnp_valid", "head_pnp_reprojection_error", "camera_stable",
)


def _head_pnp_pose(points):
    """Estimate orientation from selected face points, never translation."""
    if any(index not in points for index in HEAD_PNP_MODEL):
        return {"valid": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reprojection_error": 0.0}
    try:
        import cv2
    except ImportError:
        return {"valid": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reprojection_error": 0.0}
    object_points = np.asarray([HEAD_PNP_MODEL[index] for index in HEAD_PNP_MODEL], dtype=np.float64)
    # Match the paper: use absolute pixel coordinates and an approximate
    # intrinsics matrix derived from the fixed 1920x1080 frame dimensions.
    try:
        image_points = np.asarray([points[index] for index in HEAD_PNP_MODEL], dtype=np.float64)
    except (TypeError, ValueError, KeyError):
        return {"valid": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reprojection_error": 0.0}
    if image_points.shape != object_points.shape or not np.isfinite(image_points).all():
        return {"valid": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reprojection_error": 0.0}
    camera = np.asarray([[HEAD_FRAME_WIDTH, 0.0, HEAD_FRAME_WIDTH / 2.0],
                         [0.0, HEAD_FRAME_WIDTH, HEAD_FRAME_HEIGHT / 2.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)
    try:
        solved, rvec, tvec = cv2.solvePnP(object_points, image_points, camera, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return {"valid": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reprojection_error": 0.0}
    if not solved:
        return {"valid": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reprojection_error": 0.0}
    try:
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera, distortion)
        error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)) / HEAD_FRAME_WIDTH)
    except (cv2.error, ValueError, FloatingPointError):
        return {"valid": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reprojection_error": 0.0}
    if not math.isfinite(error) or error > HEAD_PNP_REPROJECTION_THRESHOLD:
        return {"valid": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reprojection_error": error}
    rotation, _ = cv2.Rodrigues(rvec)
    return {
        "valid": 1.0,
        "yaw": math.atan2(float(rotation[1, 0]), float(rotation[0, 0])),
        "pitch": math.atan2(float(-rotation[2, 0]), math.hypot(float(rotation[2, 1]), float(rotation[2, 2]))),
        "roll": math.atan2(float(rotation[2, 1]), float(rotation[2, 2])),
        "reprojection_error": error,
    }


def _head_pnp_features(rows):
    """Add torso-relative SolvePnP directions and a common-motion camera gate."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["clip_id"], row["actor_id"])].append(row)
    frame_displacements = defaultdict(list)
    previous_by_track = {}
    for row in sorted(rows, key=lambda item: (item["clip_id"], int(item.get("source_frame_index", 0)), item.get("track_id", ""))):
        left, right = _point(row, "pose", 11), _point(row, "pose", 12)
        if left and right:
            key = (row["clip_id"], int(row.get("source_frame_index", 0)))
            track = (row["clip_id"], str(row.get("track_id", "")))
            center = (
                ((left[0] + right[0]) / 2.0) / HEAD_FRAME_WIDTH,
                ((left[1] + right[1]) / 2.0) / HEAD_FRAME_HEIGHT,
            )
            if track in previous_by_track:
                previous = previous_by_track[track]
                frame_displacements[key].append((center[0] - previous[0], center[1] - previous[1]))
            previous_by_track[track] = center
    camera_gate = {}
    for key, values in frame_displacements.items():
        dx = float(np.median([value[0] for value in values]))
        dy = float(np.median([value[1] for value in values]))
        magnitude = math.hypot(dx, dy)
        agreement = sum(math.hypot(px - dx, py - dy) <= max(0.015, magnitude * 0.5) for px, py in values) / len(values)
        camera_gate[key] = (dx, dy, magnitude, not (len(values) >= 2 and magnitude >= CAMERA_MOTION_THRESHOLD and agreement >= CAMERA_MOTION_AGREEMENT))
    for group in grouped.values():
        group.sort(key=lambda row: int(row.get("source_frame_index", 0)))
        observations = []
        smoothed_by_row = {}
        ema = None
        for row in group:
            pose = _head_pnp_pose(row.get("_selected_face_points", {}))
            if pose["valid"] and str(row.get("face_predicted", "False")).lower() not in {"true", "1"}:
                if ema is None:
                    ema = dict(pose)
                else:
                    for name in ("yaw", "pitch", "roll"):
                        ema[name] = (HEAD_PNP_EMA_ALPHA * pose[name]) + ((1.0 - HEAD_PNP_EMA_ALPHA) * ema[name])
                    ema["reprojection_error"] = pose["reprojection_error"]
                smoothed = dict(ema)
                observations.append(smoothed)
                smoothed_by_row[id(row)] = smoothed
        baseline = {name: float(np.median([item[name] for item in observations[:30]])) for name in ("yaw", "pitch", "roll")} if observations else {}
        for row in group:
            pose = _head_pnp_pose(row.get("_selected_face_points", {}))
            pose = smoothed_by_row.get(id(row), pose)
            frame_key = (row["clip_id"], int(row.get("source_frame_index", 0)))
            camera_dx, camera_dy, camera_magnitude, camera_stable = camera_gate.get(frame_key, (0.0, 0.0, 0.0, True))
            valid = bool(pose["valid"] and camera_stable and str(row.get("face_predicted", "False")).lower() not in {"true", "1"})
            left, right = _point(row, "pose", 11), _point(row, "pose", 12)
            torso_roll = math.atan2(right[1] - left[1], right[0] - left[0]) if left and right else 0.0
            relative_yaw = pose["yaw"] - baseline.get("yaw", pose["yaw"]) if valid else 0.0
            relative_pitch = pose["pitch"] - baseline.get("pitch", pose["pitch"]) if valid else 0.0
            relative_roll = pose["roll"] - torso_roll - baseline.get("roll", pose["roll"]) if valid and left and right else 0.0
            row.update({
                "head_pnp_yaw": pose["yaw"] if valid else 0.0,
                "head_pnp_pitch": pose["pitch"] if valid else 0.0,
                "head_pnp_roll": pose["roll"] if valid else 0.0,
                "head_pnp_relative_yaw": relative_yaw,
                "head_pnp_relative_pitch": relative_pitch,
                "head_pnp_relative_roll": relative_roll,
                "head_pnp_left": float(valid and relative_yaw <= -math.radians(HEAD_PNP_DIRECTION_DEG)),
                "head_pnp_right": float(valid and relative_yaw >= math.radians(HEAD_PNP_DIRECTION_DEG)),
                "head_pnp_up": float(valid and relative_pitch <= -math.radians(HEAD_PNP_DIRECTION_DEG)),
                "head_pnp_down": float(valid and relative_pitch >= math.radians(HEAD_PNP_DIRECTION_DEG)),
                "head_pnp_valid": float(valid),
                "head_pnp_reprojection_error": pose["reprojection_error"],
                "camera_motion_dx": camera_dx,
                "camera_motion_dy": camera_dy,
                "camera_motion_magnitude": camera_magnitude,
                "camera_stable": float(camera_stable),
            })
    return rows


def load_manifest(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = {}
        for row in csv.DictReader(handle):
            clip_id = row["clip_id"]
            try:
                actors = json.loads(row.get("action_actor_ids") or "[]")
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid action_actor_ids for {clip_id}") from exc
            row["action_actor_ids_parsed"] = {str(actor) for actor in actors}
            try:
                pairs = json.loads(row.get("interaction_pairs") or "[]")
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid interaction_pairs for {clip_id}") from exc
            peers = defaultdict(set)
            for pair in pairs:
                source, peer = str(pair.get("source", "")), str(pair.get("peer", ""))
                if source and peer and source != peer:
                    # A declared relation is a two-actor prediction event even
                    # when the manifest records only one direction.
                    peers[source].add(peer)
                    peers[peer].add(source)
            row["interaction_peers_parsed"] = peers
            manifest[clip_id] = row
    return manifest


def load_actor_mapping(path: Path | None):
    if path is None:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (row.get("source_filename", row.get("filename", "")), row["track_id"]): row["actor_id"]
            for row in csv.DictReader(handle)
        }


def attach_selected_face_points(rows, json_root: Path | None):
    """Attach selected face frame points as transient, non-feature metadata."""
    if json_root is None:
        return rows
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["clip_id"]].append(row)
    for clip_id, clip_rows in grouped.items():
        path = json_root / f"{clip_id}.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        lookup = {}
        for frame in payload.get("frames", []):
            source_frame = int(frame.get("source_frame_index", 0))
            for track in frame.get("tracks", []):
                points = {
                    int(point["index"]): (
                        float(point["frame_x"]), float(point["frame_y"])
                    )
                    for point in track.get("selected_face_landmarks", [])
                    if point.get("frame_x") is not None and point.get("frame_y") is not None
                }
                lookup[(source_frame, str(track.get("track_id")))] = points
        for row in clip_rows:
            row["_selected_face_points"] = lookup.get(
                (int(row.get("source_frame_index", 0)), str(row.get("track_id", ""))), {}
            )
    return rows


def _face_axis(first, second, point):
    if not first or not second or not point:
        return 0.0, 0.0, 0.0
    vx, vy = second[0] - first[0], second[1] - first[1]
    length = max(math.hypot(vx, vy), 1.0)
    ux, uy = vx / length, vy / length
    px, py = -uy, ux
    mid_x, mid_y = (first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0
    dx, dy = point[0] - mid_x, point[1] - mid_y
    return math.atan2(vy, vx), (dx * ux + dy * uy) / length, (dx * px + dy * py) / length


def _circular_delta(value, baseline):
    return math.atan2(math.sin(value - baseline), math.cos(value - baseline))


def derive_face_c3_features(rows):
    """Derive normalized face geometry from selected face landmarks only."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["clip_id"], row["actor_id"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row.get("source_frame_index", 0)))
        observations = []
        for row in group:
            points = row.get("_selected_face_points", {})
            try:
                nose, eye_left, eye_right = points[1], points[33], points[263]
                cheek_left, cheek_right = points[234], points[454]
                mouth_left, mouth_right = points[61], points[291]
            except KeyError:
                observations.append((row, None))
                continue
            face_roll, nose_eye_along, nose_eye_perp = _face_axis(eye_left, eye_right, nose)
            cheek_roll, nose_cheek_along, nose_cheek_perp = _face_axis(
                cheek_left, cheek_right, nose
            )
            mouth_slope = math.atan2(
                mouth_right[1] - mouth_left[1], mouth_right[0] - mouth_left[0]
            )
            observations.append((row, {
                "face_c3_valid": 1.0,
                "face_c3_roll": face_roll,
                "face_c3_cheek_roll": cheek_roll,
                "face_c3_nose_eye_along": nose_eye_along,
                "face_c3_nose_eye_perp": nose_eye_perp,
                "face_c3_nose_cheek_along": nose_cheek_along,
                "face_c3_nose_cheek_perp": nose_cheek_perp,
                "face_c3_mouth_slope": mouth_slope,
            }))
        valid = [item for _, item in observations if item is not None]
        baseline = {
            name: float(np.median([item[name] for item in valid[:30]]))
            for name in valid[0]
        } if valid else {}
        for row, item in observations:
            if item is None:
                for name in ("face_c3_valid", *FACE_C3_FEATURES):
                    row[name] = 0.0
                row["face_quality_mask"] = 0.0
                continue
            row.update(item)
            row["face_quality_mask"] = float(number(row.get("face_valid")) > 0.0)
            for name in ("face_c3_roll", "face_c3_cheek_roll", "face_c3_mouth_slope"):
                row[f"{name}_delta"] = _circular_delta(item[name], baseline[name])
            for name in (
                "face_c3_nose_eye_along", "face_c3_nose_eye_perp",
                "face_c3_nose_cheek_along", "face_c3_nose_cheek_perp",
            ):
                row[f"{name}_delta"] = item[name] - baseline[name]
    return rows


def derive_finger_motion(rows):
    """Add actor-scale-normalized fingertip motion for c7 hand evidence."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["clip_id"], row["actor_id"])].append(row)
    finger_indices = (4, 8, 12, 16, 20)
    for group in grouped.values():
        group.sort(key=lambda row: int(row.get("source_frame_index", 0)))
        previous = {}
        for row in group:
            left = _point(row, "pose", 11)
            right = _point(row, "pose", 12)
            scale = max(
                math.hypot(left[0] - right[0], left[1] - right[1])
                if left and right else 1.0,
                1.0,
            )
            motions = []
            for side in ("left_hand", "right_hand"):
                for index in finger_indices:
                    point = _point(row, side, index)
                    key = (side, index)
                    if point and key in previous:
                        motions.append(math.hypot(point[0] - previous[key][0], point[1] - previous[key][1]) / scale)
                    if point:
                        previous[key] = point
            row["finger_speed"] = max(motions, default=0.0)
            row["finger_motion"] = sum(motions) / len(motions) if motions else 0.0
            row["hand_finger_motion"] = max(row["finger_speed"], row["hand_motion"])
    return rows


def derive_hand_shape_and_pair_cues(rows):
    """Add landmark-only hand shape, raise and pair convergence cues.

    These are frame/actor geometry only.  No class, interval, or annotation
    field is read, and crossing is retained separately from pre-crossing.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["clip_id"], row["actor_id"])].append(row)
    finger_indices = (4, 8, 12, 16, 20)
    for group in grouped.values():
        group.sort(key=lambda row: int(row.get("source_frame_index", 0)))
        previous_shape = None
        previous_pair = None
        previous_wrist_y = None
        previous_own_side = None
        wrist_baseline = []
        head_onset_frame = None
        head_values = [
            abs(number(row.get("c3_face_roll_delta")))
            + abs(number(row.get("c3_ear_roll_delta")))
            + abs(number(row.get("c3_shoulder_roll_delta")))
            for row in group[:30]
        ]
        head_threshold = max(0.08, float(np.median(head_values) + 2.0 * np.std(head_values))) if head_values else 0.08
        for row in group:
            scale_points = [_point(row, "pose", 11), _point(row, "pose", 12)]
            scale = max(
                math.hypot(scale_points[0][0] - scale_points[1][0], scale_points[0][1] - scale_points[1][1])
                if all(scale_points) else 1.0,
                1.0,
            )
            extensions = []
            visible = 0
            wrist_points = []
            for side in ("left_hand", "right_hand"):
                wrist = _point(row, side, 0)
                palm = _point(row, side, 9) or _point(row, side, 5)
                if wrist:
                    visible += 1
                    wrist_points.append(wrist)
                if not wrist or not palm:
                    continue
                palm_scale = max(math.hypot(palm[0] - wrist[0], palm[1] - wrist[1]), scale * 0.05)
                for index in finger_indices:
                    tip = _point(row, side, index)
                    if tip:
                        extensions.append(math.hypot(tip[0] - wrist[0], tip[1] - wrist[1]) / palm_scale)
            shape = np.asarray(extensions, dtype=np.float32)
            row["finger_extension_mean"] = float(np.mean(shape)) if len(shape) else 0.0
            row["finger_extension_std"] = float(np.std(shape)) if len(shape) else 0.0
            row["finger_shape_change"] = float(np.mean(np.abs(shape - previous_shape))) if previous_shape is not None and len(shape) == len(previous_shape) else 0.0
            previous_shape = shape if len(shape) else previous_shape
            row["hand_visibility"] = visible / 2.0
            row["hand_quality_mask"] = float(visible > 0)
            wrist_y = float(np.mean([point[1] for point in wrist_points])) if wrist_points else 0.0
            # c7 raise is a change from the actor's own normal prefix, not an
            # absolute image height.  With camera y increasing downward, only
            # a negative y delta is upward; downward motion must not qualify.
            if wrist_points and len(wrist_baseline) < 30:
                wrist_baseline.append(wrist_y)
            baseline_wrist_y = float(np.median(wrist_baseline)) if wrist_baseline else wrist_y
            if LEGACY_C7_FORMULA:
                row["raise_displacement"] = -wrist_y / scale
                row["raise_speed"] = abs(wrist_y - previous_wrist_y) / scale if previous_wrist_y is not None else 0.0
            else:
                row["raise_displacement"] = max(0.0, (baseline_wrist_y - wrist_y) / scale) if wrist_points else 0.0
                row["raise_speed"] = max(0.0, (previous_wrist_y - wrist_y) / scale) if previous_wrist_y is not None and wrist_points else 0.0
            row["c7_raise_directional"] = float(row["raise_speed"] > 0.0)
            previous_wrist_y = wrist_y if wrist_points else previous_wrist_y
            pair_distance = number(row.get("pair_hand_distance"))
            row["pair_convergence"] = max(0.0, previous_pair - pair_distance) if previous_pair is not None and pair_distance else 0.0
            row["crossing_indicator"] = float(number(row.get("own_side_distance")) < 0.0)
            own_side = number(row.get("own_side_distance"))
            row["hand_direction"] = own_side - previous_own_side if previous_own_side is not None else 0.0
            head_signal = (
                abs(number(row.get("c3_face_roll_delta")))
                + abs(number(row.get("c3_ear_roll_delta")))
                + abs(number(row.get("c3_shoulder_roll_delta")))
            )
            if head_onset_frame is None and head_signal >= head_threshold:
                head_onset_frame = int(row.get("source_frame_index", 0))
            approaching = row["pair_convergence"] > 0.0 or row["hand_direction"] < 0.0
            row["head_turn_to_hand_approach_frames"] = (
                max(0, int(row.get("source_frame_index", 0)) - head_onset_frame)
                if head_onset_frame is not None and approaching else 0.0
            )
            # Shared-zone is an event, not merely the absence of midpoint
            # crossing.  It requires directed travel toward the fixed pair
            # midpoint, reduced own-side distance, and hand near-contact.
            normalized_pair_distance = pair_distance / scale if pair_distance else float("inf")
            toward_midpoint = previous_own_side is not None and own_side < previous_own_side
            near_contact = normalized_pair_distance <= 1.0
            shared_zone = bool(toward_midpoint and row["hand_direction"] < 0.0 and near_contact)
            row["c7_toward_midpoint"] = float(toward_midpoint)
            row["c7_shared_zone"] = float(shared_zone)
            row["c7_not_midpoint"] = float(number(row.get("near_midpoint_pre_cross")) < 1.0)
            row["c7_not_shared_zone"] = (
                float(number(row.get("near_midpoint_pre_cross")) < 1.0
                      and number(row.get("crossing_indicator")) < 1.0)
                if LEGACY_C7_FORMULA else float(not shared_zone)
            )
            previous_own_side = own_side
            previous_pair = pair_distance if pair_distance else previous_pair
    return rows


def derive_hand_jitter_aware_cues(rows):
    """Derive same-hand c7 cues while treating missing/jittery hands as unknown.

    This is intentionally separate from the legacy all-hand cues.  A hand can
    disappear for several frames; those frames do not become zero motion or
    c5 evidence.  Velocity uses elapsed source-frame distance after the last
    valid observation, and the actor selects one stable left/right stream for
    all c7 terms.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["clip_id"], row["actor_id"])].append(row)
    finger_indices = (4, 8, 12, 16, 20)
    for group in grouped.values():
        group.sort(key=lambda row: int(row.get("source_frame_index", 0)))
        state = {}
        for side in ("left", "right"):
            state[side] = {"previous": None, "previous_frame": None, "shapes": [], "baseline_y": []}
            for row in group[:30]:
                wrist = _point(row, f"{side}_hand", 0)
                if wrist:
                    state[side]["baseline_y"].append(wrist[1])
                palm = _point(row, f"{side}_hand", 9) or _point(row, f"{side}_hand", 5)
                tips = [_point(row, f"{side}_hand", index) for index in finger_indices]
                if wrist and palm and all(point is not None for point in tips):
                    palm_scale = max(math.hypot(palm[0] - wrist[0], palm[1] - wrist[1]), 1.0)
                    state[side]["shapes"].append(np.asarray([
                        math.hypot(point[0] - wrist[0], point[1] - wrist[1]) / palm_scale
                        for point in tips
                    ], dtype=np.float32))
        for row in group:
            pose_left, pose_right = _point(row, "pose", 11), _point(row, "pose", 12)
            scale = max(math.hypot(pose_left[0] - pose_right[0], pose_left[1] - pose_right[1]) if pose_left and pose_right else 1.0, 1.0)
            midpoint = number(row.get("pair_mid_x_0"))
            margin = number(row.get("pair_margin_10pct"))
            actor_side = number(row.get("actor_side"))
            for side in ("left", "right"):
                prefix = f"{side}_hand"
                wrist = _point(row, prefix, 0)
                palm = _point(row, prefix, 9) or _point(row, prefix, 5)
                tips = [_point(row, prefix, index) for index in finger_indices]
                finger_valid = int(all(point is not None for point in tips))
                shape = None
                if wrist and palm and finger_valid:
                    palm_scale = max(math.hypot(palm[0] - wrist[0], palm[1] - wrist[1]), scale * 0.05)
                    shape = np.asarray([math.hypot(point[0] - wrist[0], point[1] - wrist[1]) / palm_scale for point in tips], dtype=np.float32)
                item = state[side]
                baseline_y = float(np.median(item["baseline_y"])) if item["baseline_y"] else (wrist[1] if wrist else 0.0)
                baseline_shape = np.median(np.asarray(item["shapes"], dtype=np.float32), axis=0) if item["shapes"] else None
                previous = item["previous"]
                previous_frame = item["previous_frame"]
                gap = int(row.get("source_frame_index", 0)) - previous_frame if previous_frame is not None else 0
                dx = (wrist[0] - previous[0][0]) if wrist and previous else 0.0
                dy = (wrist[1] - previous[0][1]) if wrist and previous else 0.0
                elapsed = max(gap, 1)
                raise_disp = max(0.0, (baseline_y - wrist[1]) / scale) if wrist else 0.0
                raise_speed = max(0.0, -dy / (scale * elapsed)) if wrist and previous else 0.0
                jitter_speed = math.hypot(dx, dy) / (scale * elapsed) if wrist and previous else 0.0
                coherence_values = []
                if wrist and previous:
                    for tip_index in finger_indices:
                        current_tip = _point(row, prefix, tip_index)
                        previous_tip = previous[1].get(tip_index)
                        if current_tip and previous_tip:
                            tip_dx = current_tip[0] - previous_tip[0]
                            tip_dy = current_tip[1] - previous_tip[1]
                            wrist_norm = max(math.hypot(dx, dy), 1e-6)
                            coherence_values.append((tip_dx * dx + tip_dy * dy) / (max(math.hypot(tip_dx, tip_dy), 1e-6) * wrist_norm))
                coherence = float(np.mean([value >= 0.25 for value in coherence_values])) if coherence_values else 0.0
                shape_change = float(np.mean(np.abs(shape - baseline_shape))) if shape is not None and baseline_shape is not None else 0.0
                selected_point = None
                if midpoint and actor_side:
                    candidates = [point for point in tips if point is not None]
                    selected_point = min(candidates, key=lambda point: abs(point[0] - midpoint)) if candidates else wrist
                own_distance = actor_side * (selected_point[0] - midpoint) if selected_point and midpoint and actor_side else 0.0
                near_midpoint = float(0.0 <= own_distance <= margin) if margin else 0.0
                row[f"{side}_hand_valid"] = float(wrist is not None)
                row[f"{side}_hand_raise_displacement"] = raise_disp
                row[f"{side}_hand_raise_speed"] = raise_speed
                row[f"{side}_hand_jitter_speed"] = jitter_speed
                row[f"{side}_hand_gap_frames"] = float(max(gap - 1, 0)) if wrist else float(max(gap, 0))
                row[f"{side}_hand_coherence"] = coherence
                row[f"{side}_hand_shape_change_baseline"] = shape_change
                row[f"{side}_hand_own_side_distance"] = own_distance
                row[f"{side}_hand_near_midpoint"] = near_midpoint
                row[f"{side}_hand_finger_valid"] = float(finger_valid)
                row[f"{side}_hand_wrist_valid"] = float(wrist is not None)
                if wrist:
                    item["previous"] = (wrist, {index: point for index, point in zip(finger_indices, tips) if point is not None})
                    item["previous_frame"] = int(row.get("source_frame_index", 0))
        # Shared-zone is computed for each candidate hand against an explicit
        # peer hand, never from the other hand of the same actor.
        by_frame = defaultdict(dict)
        for row in group:
            by_frame[int(row.get("source_frame_index", 0))][row["actor_id"]] = row
        for side in ("left", "right"):
            previous_own = None
            for row in group:
                frame_rows = by_frame[int(row.get("source_frame_index", 0))]
                try:
                    peers = json.loads(row.get("interaction_peer_ids") or "[]")
                except json.JSONDecodeError:
                    peers = []
                point = _point(row, f"{side}_hand", 0)
                peer_points = []
                for peer in peers:
                    peer_row = frame_rows.get(str(peer))
                    if peer_row:
                        for peer_side in ("left", "right"):
                            peer_point = _point(peer_row, f"{peer_side}_hand", 0)
                            if peer_point:
                                peer_points.append(peer_point)
                pair_distance = min((math.hypot(point[0] - other[0], point[1] - other[1]) for other in peer_points), default=float("inf")) if point else float("inf")
                scale = max(number(row.get("pair_margin_10pct")) / 0.10, 1.0)
                own = number(row.get(f"{side}_hand_own_side_distance"))
                toward = previous_own is not None and own < previous_own
                row[f"{side}_hand_shared_zone"] = float(toward and pair_distance / scale <= 1.0)
                previous_own = own if point else previous_own
        # Select one actor hand from valid, coherent, baseline-relative signal;
        # this selection uses no class, interval, or model score.
        hand_scores = {}
        for side in ("left", "right"):
            hand_scores[side] = max(
                (number(row.get(f"{side}_hand_raise_displacement")) + number(row.get(f"{side}_hand_shape_change_baseline")))
                for row in group
            )
        selected_side = "left" if hand_scores["left"] >= hand_scores["right"] else "right"
        for row in group:
            row["c7_selected_hand_left"] = float(selected_side == "left")
            row["c7_selected_hand_right"] = float(selected_side == "right")
            prefix = f"{selected_side}_hand"
            row["c7_selected_hand_valid"] = number(row.get(f"{prefix}_valid"))
            row["c7_selected_hand_coherence"] = number(row.get(f"{prefix}_coherence"))
            row["c7_selected_hand_jitter_speed"] = number(row.get(f"{prefix}_jitter_speed"))
            row["c7_selected_hand_gap_frames"] = number(row.get(f"{prefix}_gap_frames", 0))
            row["c7_selected_hand_raise_displacement"] = number(row.get(f"{prefix}_raise_displacement"))
            row["c7_selected_hand_raise_speed"] = number(row.get(f"{prefix}_raise_speed"))
            row["c7_selected_hand_shape_change"] = number(row.get(f"{prefix}_shape_change_baseline"))
            row["c7_selected_hand_own_side_distance"] = number(row.get(f"{prefix}_own_side_distance"))
            row["c7_selected_hand_near_midpoint"] = number(row.get(f"{prefix}_near_midpoint"))
            row["c7_selected_hand_shared_zone"] = number(row.get(f"{prefix}_shared_zone"))
            row["c7_selected_hand_finger_valid"] = number(row.get(f"{prefix}_finger_valid"))
            row["c7_selected_hand_wrist_valid"] = number(row.get(f"{prefix}_wrist_valid"))
    return rows


def actor_truth(row, manifest):
    item = manifest.get(row["clip_id"])
    if item is None:
        raise ValueError(f"manifest row missing for clip_id={row['clip_id']}")
    action_class = item.get("class_code", "c5")
    start = number(item.get("action_start_s"))
    end = number(item.get("action_end_s"))
    timestamp = number(row.get("timestamp_ms")) / 1000.0
    in_interval = start <= timestamp <= end
    source_actor = row["actor_id"] in item["action_actor_ids_parsed"]
    public_class = "suspicious_activity" if EXTENDED_SUSPICIOUS and action_class in {"c1", "c4"} else action_class
    is_target_source = public_class in TARGET_CLASSES and source_actor
    # c1/c4/c6 source actors are excluded, never silently relabeled as c5.
    excluded_source = public_class not in TARGET_CLASSES and source_actor
    return {
        "actor_truth": public_class if is_target_source else "c5",
        "truth": public_class if is_target_source and in_interval else "c5",
        "source_actor": int(source_actor),
        "within_action_interval": int(in_interval),
        "excluded_source": int(excluded_source),
        "action_start_s": item.get("action_start_s", ""),
        "action_end_s": item.get("action_end_s", ""),
        "manifest_class_code": action_class,
        "interaction_peer_ids": json.dumps(sorted(item["interaction_peers_parsed"].get(row["actor_id"], set()))),
    }


def features(row):
    return [number(row.get(name)) for name in FEATURES]


def select_features(rows, allowed_features=None):
    if not rows:
        raise ValueError("no rows available for feature selection")
    fields = []
    for name in rows[0]:
        if allowed_features is not None and name not in allowed_features:
            continue
        if allowed_features is None and (name in BLOCKED_FEATURES or name.endswith("_candidate")):
            continue
        values = [rows[index].get(name, "") for index in range(min(len(rows), 200))]
        numeric_count = sum(1 for value in values if value not in (None, "") and _is_number(value))
        if numeric_count >= max(1, int(len(values) * 0.98)):
            fields.append(name)
    if not fields:
        raise ValueError("no numeric actor-frame geometry features selected")
    # These are an explicit geometry family, not labels or provenance. Keep
    # directional/validity outputs in the schema even when a split has a
    # degenerate all-zero gate and generic discovery would drop them.
    if allowed_features is None:
        fields.extend(name for name in HEAD_PNP_FEATURES if name in rows[0] and name not in fields)
    return tuple(fields)


def _is_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _point(row, prefix, index):
    if row.get(f"{prefix}_{index}_valid") != "1":
        return None
    x = number(row.get(f"{prefix}_{index}_frame_x", row.get(f"{prefix}_{index}_x")))
    y = number(row.get(f"{prefix}_{index}_frame_y", row.get(f"{prefix}_{index}_y")))
    return x, y


def derive_behavior_motion(rows):
    """Add actor-local motion/peer cues from raw landmarks only."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["clip_id"], row["actor_id"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row.get("source_frame_index", 0)))
        last_frame = max(1, int(group[-1].get("source_frame_index", 0)))
        side = number(group[0].get("actor_side"))
        if side == 0:
            side = -1.0 if number(group[0].get("bbox_x1")) < 0 else 1.0
        observations = []
        for row in group:
            nose = _point(row, "pose", 0)
            eye_left = _point(row, "pose", 1)
            eye_right = _point(row, "pose", 4)
            ear_left = _point(row, "pose", 7)
            ear_right = _point(row, "pose", 8)
            left = _point(row, "pose", 11)
            right = _point(row, "pose", 12)
            wrist = _point(row, "pose", 16) or _point(row, "pose", 15)
            scale = math.hypot(left[0] - right[0], left[1] - right[1]) if left and right else 1.0
            scale = max(scale, 1.0)
            torso_x = (left[0] + right[0]) / 2.0 if left and right else None
            nose_lateral = (nose[0] - torso_x) / scale if nose and torso_x is not None else None
            def axis_features(first, second, point):
                if not first or not second or not point:
                    return 0.0, 0.0, 0.0
                vx, vy = second[0] - first[0], second[1] - first[1]
                length = max(math.hypot(vx, vy), 1.0)
                ux, uy = vx / length, vy / length
                px, py = -uy, ux
                midx, midy = (first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0
                dx, dy = point[0] - midx, point[1] - midy
                return math.atan2(vy, vx), (dx * ux + dy * uy) / scale, (dx * px + dy * py) / scale
            face_roll, nose_eye_lateral, nose_eye_vertical = axis_features(eye_left, eye_right, nose)
            ear_roll, nose_ear_lateral, nose_ear_vertical = axis_features(ear_left, ear_right, nose)
            shoulder_roll = math.atan2((right[1] - left[1]) if left and right else 0.0, (right[0] - left[0]) if left and right else 1.0)
            shoulder_dy = ((right[1] - left[1]) / scale) if left and right else 0.0
            shoulder_dx = ((right[0] - left[0]) / scale) if left and right else 0.0
            observations.append((row, nose_lateral, torso_x, wrist, scale, face_roll, ear_roll, nose_eye_lateral, nose_eye_vertical, nose_ear_lateral, nose_ear_vertical, shoulder_roll, shoulder_dy, shoulder_dx))
        valid_nose = [item[1] for item in observations[:30] if item[1] is not None]
        valid_torso = [item[2] for item in observations[:30] if item[2] is not None]
        baseline_nose = float(np.median(valid_nose)) if valid_nose else 0.0
        baseline_torso = float(np.median(valid_torso)) if valid_torso else 0.0
        baseline_face_roll = float(np.median([item[5] for item in observations[:30]]))
        baseline_ear_roll = float(np.median([item[6] for item in observations[:30]]))
        baseline_shoulder_roll = float(np.median([item[11] for item in observations[:30]]))
        baseline_shoulder_dy = float(np.median([item[12] for item in observations[:30]]))
        baseline_shoulder_dx = float(np.median([item[13] for item in observations[:30]]))
        previous_wrist = None
        previous_c3 = None
        for row, nose_lateral, torso_x, wrist, scale, face_roll, ear_roll, nose_eye_lateral, nose_eye_vertical, nose_ear_lateral, nose_ear_vertical, shoulder_roll, shoulder_dy, shoulder_dx in observations:
            toward_nose = side * ((nose_lateral - baseline_nose) if nose_lateral is not None else 0.0)
            toward_torso = side * ((torso_x - baseline_torso) / scale if torso_x is not None else 0.0)
            speed = 0.0
            if wrist and previous_wrist:
                speed = math.hypot(wrist[0] - previous_wrist[0], wrist[1] - previous_wrist[1]) / scale
            if wrist:
                previous_wrist = wrist
            row["peer_face_displacement"] = abs(toward_nose)
            row["peer_face_toward"] = toward_nose
            row["peer_torso_displacement"] = abs(toward_torso)
            row["peer_torso_toward"] = toward_torso
            row["hand_speed"] = speed
            row["hand_motion"] = speed
            row["hand_raise"] = abs((wrist[1] / scale) if wrist else 0.0)
            row["relative_frame_progress"] = int(row.get("source_frame_index", 0)) / last_frame
            row["c3_face_roll"] = face_roll
            row["c3_ear_roll"] = ear_roll
            row["c3_nose_eye_lateral"] = nose_eye_lateral
            row["c3_nose_eye_vertical"] = nose_eye_vertical
            row["c3_nose_ear_lateral"] = nose_ear_lateral
            row["c3_nose_ear_vertical"] = nose_ear_vertical
            row["c3_shoulder_roll"] = shoulder_roll
            row["c3_shoulder_dy"] = shoulder_dy
            row["c3_shoulder_dx"] = shoulder_dx
            row["c3_head_shoulder_roll"] = face_roll - shoulder_roll
            row["c3_head_toward_peer"] = side * nose_eye_lateral
            row["c3_shoulder_toward_peer"] = side * shoulder_dx
            row["c3_face_roll_delta"] = face_roll - baseline_face_roll
            row["c3_ear_roll_delta"] = ear_roll - baseline_ear_roll
            row["c3_shoulder_roll_delta"] = shoulder_roll - baseline_shoulder_roll
            row["c3_shoulder_dy_delta"] = shoulder_dy - baseline_shoulder_dy
            row["c3_shoulder_dx_delta"] = shoulder_dx - baseline_shoulder_dx
            row["c3_head_shoulder_roll_delta"] = (face_roll - shoulder_roll) - (baseline_face_roll - baseline_shoulder_roll)
            if previous_c3 is None:
                row["c3_face_roll_velocity"] = 0.0
                row["c3_ear_roll_velocity"] = 0.0
                row["c3_shoulder_roll_velocity"] = 0.0
                row["c3_head_toward_peer_velocity"] = 0.0
            else:
                row["c3_face_roll_velocity"] = _circular_delta(face_roll, previous_c3[0])
                row["c3_ear_roll_velocity"] = _circular_delta(ear_roll, previous_c3[1])
                row["c3_shoulder_roll_velocity"] = _circular_delta(shoulder_roll, previous_c3[2])
                row["c3_head_toward_peer_velocity"] = side * (nose_eye_lateral - previous_c3[3])
            previous_c3 = (face_roll, ear_roll, shoulder_roll, nose_eye_lateral)
            row["pose_quality_mask"] = float(number(row.get("pose_valid_ratio")) > 0.0)
    return rows


def derive_strict_c2_c3_suspicious_cues(rows, baseline_frames=CAUSAL_WARMUP_FRAMES):
    """Current/past-only gates shared by C2, C3 and suspicious activity.

    These are geometry gates, never annotation, identity, or future-window
    features.  The existing C2 midpoint flag retains its finger-first contract.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["clip_id"], row["actor_id"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row.get("source_frame_index", 0)))
        baseline = []
        for row in group:
            left, right, nose = _point(row, "pose", 11), _point(row, "pose", 12), _point(row, "pose", 0)
            scale = max(math.hypot(left[0] - right[0], left[1] - right[1]), 1.0) if left and right else 1.0
            shoulder_y = ((left[1] + right[1]) / 2.0) if left and right else 0.0
            head_depth = (nose[1] - shoulder_y) / scale if nose and left and right else 0.0
            if len(baseline) < baseline_frames and nose and left and right:
                baseline.append(head_depth)
            base_depth = float(np.median(baseline)) if baseline else head_depth
            lower = []
            for wrist_index, hip_index in ((15, 23), (16, 24)):
                wrist, hip = _point(row, "pose", wrist_index), _point(row, "pose", hip_index)
                if wrist and hip:
                    lower.append((wrist[1] - hip[1]) / scale)
            margin = number(row.get("pair_margin_10pct"))
            own_distance = number(row.get("own_side_distance"))
            row.update({
                "strict_head_down_delta": max(0.0, head_depth - base_depth),
                "strict_hand_below_hip": max(lower, default=0.0),
                "strict_hand_quality": float(number(row.get("hand_quality_mask")) > 0.0),
                "strict_midpoint_hit": float(number(row.get("near_midpoint_pre_cross")) >= 1.0),
                # An own-side suspicious cue must be outside C2's midpoint margin.
                "strict_own_side_outside_midpoint": float(own_distance > margin),
            })
    return rows


def _declared_peer_ids(row):
    try:
        value = json.loads(row.get("interaction_peer_ids") or "[]")
    except (TypeError, json.JSONDecodeError):
        return ()
    return tuple(str(item) for item in value)


def _bbox_center_x(row):
    try:
        return (float(row["bbox_x1"]) + float(row["bbox_x2"])) / 2.0
    except (KeyError, TypeError, ValueError):
        return None


def _c3_pose_point(row, index):
    if str(row.get(f"pose_{index}_valid", "0")) != "1":
        return None
    try:
        return float(row[f"pose_{index}_frame_x"]), float(row[f"pose_{index}_frame_y"])
    except (KeyError, TypeError, ValueError):
        return None


def derive_c3_pose_contract(rows, baseline_frames=CAUSAL_WARMUP_FRAMES):
    """Derive the causal, pose-only C3 contract for an explicit peer.

    The historical C3 specialist is intentionally untouched.  This branch
    uses nose/eye/shoulder/hip geometry only: baseline-relative signed head
    and torso displacement toward the declared peer, with validity and
    current-frame peer direction carried as explicit features.

    Temporary live-camera fallback: when no declared peer is visible, use the
    screen-right direction so a one-person camera can exercise the C3 path.
    This is deliberately not a benchmark rule: without a peer, a left/right
    turn cannot be interpreted as ``toward another actor``.
    """
    grouped = defaultdict(list)
    by_frame = defaultdict(dict)
    for row in rows:
        key = (row["clip_id"], row["actor_id"])
        grouped[key].append(row)
        by_frame[(row["clip_id"], int(row.get("source_frame_index", 0)))][row["actor_id"]] = row

    for (clip_id, actor_id), group in grouped.items():
        group.sort(key=lambda row: int(row.get("source_frame_index", 0)))
        observations = []
        for row in group:
            nose = _c3_pose_point(row, 0)
            eye_left = _c3_pose_point(row, 1)
            eye_right = _c3_pose_point(row, 4)
            shoulder_left = _c3_pose_point(row, 11)
            shoulder_right = _c3_pose_point(row, 12)
            hip_left = _c3_pose_point(row, 23)
            hip_right = _c3_pose_point(row, 24)
            shoulder_mid = (
                (shoulder_left[0] + shoulder_right[0]) / 2.0,
                (shoulder_left[1] + shoulder_right[1]) / 2.0,
            ) if shoulder_left and shoulder_right else None
            hip_mid = (
                (hip_left[0] + hip_right[0]) / 2.0,
                (hip_left[1] + hip_right[1]) / 2.0,
            ) if hip_left and hip_right else None
            shoulder_scale = (
                math.hypot(shoulder_right[0] - shoulder_left[0], shoulder_right[1] - shoulder_left[1])
                if shoulder_left and shoulder_right else 0.0
            )
            torso_scale = (
                math.hypot(shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
                if shoulder_mid and hip_mid else 0.0
            )
            frame_index = int(row.get("source_frame_index", 0))
            peers = _declared_peer_ids(row)
            peer_id = peers[0] if peers else None
            peer_row = by_frame.get((clip_id, frame_index), {}).get(peer_id or "")
            peer_x = _bbox_center_x(peer_row) if peer_row else None
            required_valid = int(all((nose, eye_left, eye_right, shoulder_left, shoulder_right, hip_left, hip_right)))
            head_valid = int(all((nose, eye_left, eye_right, shoulder_left, shoulder_right)) and shoulder_scale > 1.0)
            torso_valid = int(all((shoulder_left, shoulder_right, hip_left, hip_right)) and torso_scale > 1.0)
            # TEMPORARY live demo: permit one actor.  Do not copy this fallback
            # into the benchmark; its C3 truth contract requires a peer.
            peer_valid = int(shoulder_mid is not None)
            head_lateral = ((nose[0] - shoulder_mid[0]) / shoulder_scale) if head_valid else None
            torso_lateral = ((shoulder_mid[0] - hip_mid[0]) / torso_scale) if torso_valid else None
            direction = (
                1.0 if peer_x is None or peer_x > shoulder_mid[0] else -1.0
            ) if peer_valid else None
            observations.append({
                "row": row, "frame": frame_index,
                "head_lateral": head_lateral, "torso_lateral": torso_lateral,
                "direction": direction, "head_valid": head_valid,
                "torso_valid": torso_valid, "peer_valid": peer_valid,
                "required_valid": required_valid,
            })

        def baseline(name):
            values = [item[name] for item in observations[:baseline_frames] if item[name] is not None]
            return float(np.median(values)) if values else None

        baseline_head = baseline("head_lateral")
        baseline_torso = baseline("torso_lateral")
        previous_head = None
        previous_torso = None
        previous_frame = None
        for item in observations:
            row = item["row"]
            head_delta = 0.0
            torso_delta = 0.0
            if item["head_lateral"] is not None and baseline_head is not None and item["direction"] is not None:
                head_delta = item["direction"] * (item["head_lateral"] - baseline_head)
            if item["torso_lateral"] is not None and baseline_torso is not None and item["direction"] is not None:
                torso_delta = item["direction"] * (item["torso_lateral"] - baseline_torso)
            dt = max(1.0, float(item["frame"] - previous_frame)) if previous_frame is not None else 1.0
            head_velocity = (head_delta - previous_head) / dt if previous_head is not None and item["head_valid"] and item["peer_valid"] else 0.0
            torso_velocity = (torso_delta - previous_torso) / dt if previous_torso is not None and item["torso_valid"] and item["peer_valid"] else 0.0
            row.update({
                "c3_pose_head_peer_delta": head_delta,
                "c3_pose_torso_peer_delta": torso_delta,
                "c3_pose_head_peer_velocity": head_velocity,
                "c3_pose_torso_peer_velocity": torso_velocity,
                "c3_pose_head_valid": float(item["head_valid"]),
                "c3_pose_torso_valid": float(item["torso_valid"]),
                "c3_pose_peer_valid": float(item["peer_valid"]),
                "c3_pose_required_valid": float(item["required_valid"]),
            })
            if item["head_valid"] and item["peer_valid"]:
                previous_head = head_delta
            if item["torso_valid"] and item["peer_valid"]:
                previous_torso = torso_delta
            previous_frame = item["frame"]
    return rows


def fit_model(rows, c7_weight=3.0):
    usable = [row for row in rows if not row["excluded_source"]]
    counts = defaultdict(int)
    for row in usable:
        counts[row["truth"]] += 1
    if set(counts) != set(MODEL_CLASSES):
        raise ValueError(f"training classes incomplete: {dict(counts)}")
    weights = np.asarray([
        (float(c7_weight) if row["truth"] == "c7" else 1.0)
        / math.sqrt(counts[row["truth"]])
        for row in usable
    ], dtype=np.float32)
    mapping = {name: index for index, name in enumerate(MODEL_CLASSES)}
    matrix = xgb.DMatrix(
        np.asarray([features(row) for row in usable], dtype=np.float32),
        label=np.asarray([mapping[row["truth"]] for row in usable]),
        weight=weights,
    )
    model = xgb.train(
        {
            "objective": "multi:softprob",
            "num_class": len(MODEL_CLASSES),
            "tree_method": "hist",
            "device": "cpu",
            "max_depth": 4,
            "min_child_weight": 3,
            "eta": 0.04,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "seed": 20260811,
        },
        matrix,
        num_boost_round=350,
    )
    return model, dict(counts)


def actor_evidence(rows, probabilities):
    grouped = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        grouped[(row["clip_id"], row["actor_id"])].append((row, probability))
    output = []
    for (video, actor_id), group in grouped.items():
        # Max margin over full video. This permits intermittent actions and
        # compares each positive frame against trained c5 evidence.
        scores = {
            name: max(float(probability[index]) for _, probability in group)
            for index, name in enumerate(MODEL_CLASSES)
        }
        margins = {
            name: max(float(probability[index] - probability[2]) for _, probability in group)
            for index, name in enumerate(MODEL_CLASSES)
            if name in TARGET_CLASSES
        }
        output.append({
            "video": video,
            "actor_id": actor_id,
            "truth": group[0][0]["actor_label"],
            "predicted_class": "c5",
            "c2_score": scores["c2"],
            "c3_score": scores["c3"],
            "c5_score": scores["c5"],
            "c7_score": scores.get("c7", 0.0),
            "c2_margin": margins["c2"],
            "c3_margin": margins["c3"],
            "c7_margin": margins.get("c7", 0.0),
            "frames_scanned": len(group),
        })
    return output


def frame_evidence_predictions(rows, probabilities):
    """Assign one actor class from the strongest qualified frame.

    A positive frame is sufficient by contract.  Scores are therefore
    reduced with max over the full video, and c5 is the learned reference
    score rather than an assumed zero.
    """
    actor_rows = actor_evidence(rows, probabilities)
    for row in actor_rows:
        eligible = [
            name for name in TARGET_CLASSES
            if row[f"{name}_margin"] >= 0.0
        ]
        row["predicted_class"] = (
            max(eligible, key=lambda name: row[f"{name}_margin"])
            if eligible else "c5"
        )
    return actor_rows


def fit_actor_thresholds(actor_rows):
    thresholds = {}
    for name in TARGET_CLASSES:
        values = sorted(row[f"{name}_margin"] for row in actor_rows)
        candidates = sorted(set(values))
        best = (0.0, 1.0)
        for threshold in candidates:
            truth = [row["truth"] == name for row in actor_rows]
            predicted = [row[f"{name}_margin"] >= threshold for row in actor_rows]
            tp = sum(actual and guess for actual, guess in zip(truth, predicted))
            fp = sum(not actual and guess for actual, guess in zip(truth, predicted))
            fn = sum(actual and not guess for actual, guess in zip(truth, predicted))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            # Prefer a higher threshold on ties to suppress c5 false alerts.
            candidate = (f1, threshold)
            if candidate[0] > best[0] or candidate[0] == best[0] and candidate[1] > best[1]:
                best = candidate
        thresholds[name] = best[1]
    return thresholds


def resolve_actor_predictions(actor_rows, thresholds):
    for row in actor_rows:
        eligible = [
            name for name in TARGET_CLASSES
            if row[f"{name}_margin"] >= thresholds[name]
        ]
        row["predicted_class"] = max(
            eligible, key=lambda name: row[f"{name}_margin"]
        ) if eligible else "c5"
    return actor_rows


def actor_metrics(actor_rows):
    labels = (*TARGET_CLASSES, "c5")
    truth = [row["truth"] for row in actor_rows]
    predicted = [row["predicted_class"] for row in actor_rows]
    report = classification_report(
        truth, predicted, labels=list(labels), output_dict=True, zero_division=0
    )
    return {
        # This is the acceptance metric: all modeled actor classes, including
        # normal c5.  Target-only F1 is intentionally not reported as primary.
        "actor_macro_f1_c2_c3_c5" if EXCLUDE_C7 else "actor_macro_f1_c2_c3_c5_c7": float(
            f1_score(truth, predicted, labels=list(labels), average="macro", zero_division=0)
        ),
        "actor_macro_f1": float(
            f1_score(truth, predicted, labels=list(labels), average="macro", zero_division=0)
        ),
        "actor_metrics": {name: report.get(name, {}) for name in labels},
        "actor_confusion_matrix": confusion_matrix(truth, predicted, labels=list(labels)).tolist(),
    }


def aggregate_actor_rows(rows, selected_features=None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["clip_id"], row["actor_id"])].append(row)
    actor_rows = []
    feature_names = []
    statistics = ("mean", "std", "max", "q95", "min")
    base_features = selected_features or FEATURES or ACTOR_BASE_FEATURES
    for base in base_features:
        for statistic in statistics:
            feature_names.append(f"{base}__{statistic}")
    for (video, actor_id), group in grouped.items():
        actor = {
            "video": video,
            "actor_id": actor_id,
            "truth": group[0]["actor_truth"],
            "interaction_peer_ids": group[0]["interaction_peer_ids"],
            "source_actor": group[0]["source_actor"],
            "manifest_class_code": group[0]["manifest_class_code"],
            "frames_scanned": len(group),
        }
        for base in base_features:
            values = np.asarray([number(row.get(base)) for row in group], dtype=np.float32)
            actor[f"{base}__mean"] = float(np.mean(values))
            actor[f"{base}__std"] = float(np.std(values))
            actor[f"{base}__max"] = float(np.max(values))
            actor[f"{base}__q95"] = float(np.quantile(values, 0.95))
            actor[f"{base}__min"] = float(np.min(values))
        actor_rows.append(actor)
    return actor_rows, feature_names


def causal_aggregate_rows(
    rows,
    selected_features=None,
    warmup_frames=CAUSAL_WARMUP_FRAMES,
    window_frames=90,
):
    """Build one actor rolling-window row per frame using only current/past values."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["clip_id"], row["actor_id"])].append(row)
    feature_names = []
    statistics = ("mean", "std", "max", "q95", "min")
    base_features = selected_features or FEATURES or ACTOR_BASE_FEATURES
    for base in base_features:
        for statistic in statistics:
            feature_names.append(f"{base}__{statistic}")
    prefix_rows = []
    for (video, actor_id), group in grouped.items():
        ordered = sorted(group, key=lambda row: int(row.get("source_frame_index", 0)))
        window = CausalActorWindow(actor_id, base_features, max_frames=window_frames)
        for row in ordered:
            state = window.update(
                frame_index=int(row.get("source_frame_index", 0)),
                timestamp_ms=int(number(row.get("timestamp_ms", 0))),
                features=row,
            )
            prefix = {
                "video": video,
                "actor_id": actor_id,
                "truth": row["actor_truth"],
                "actor_truth": row["actor_truth"],
                "interaction_peer_ids": row["interaction_peer_ids"],
                "source_actor": row["source_actor"],
                "manifest_class_code": row["manifest_class_code"],
                "source_frame_index": row.get("source_frame_index", ""),
                "timestamp_ms": row.get("timestamp_ms", ""),
                "near_midpoint_pre_cross": row.get("near_midpoint_pre_cross", "0"),
                "current_pair_hand_distance": row.get("pair_hand_distance", ""),
                "current_pair_margin_10pct": row.get("pair_margin_10pct", ""),
                "current_hand_quality_mask": row.get("hand_quality_mask", "0"),
                "actor_label": row.get("actor_label", row["actor_truth"]),
                "prefix_frames": state.window_size,
                "warmup_ready": int(state.window_size >= warmup_frames),
                "window_start_frame": state.window_start_frame,
            }
            prefix.update(state.features)
            prefix_rows.append(prefix)
    return prefix_rows, feature_names


def fit_actor_model(train_rows, feature_names, c7_weight=3.0):
    counts = defaultdict(int)
    for row in train_rows:
        counts[row["truth"]] += 1
    if set(counts) != set(MODEL_CLASSES):
        raise ValueError(f"actor training classes incomplete: {dict(counts)}")
    mapping = {name: index for index, name in enumerate(MODEL_CLASSES)}
    class_weight = {"c2": 1.0, "c3": 1.0, "c5": 0.8, "c7": float(c7_weight)}
    weights = np.asarray(
        [class_weight[row["truth"]] / math.sqrt(counts[row["truth"]]) for row in train_rows], dtype=np.float32
    )
    matrix = xgb.DMatrix(
        np.asarray([[row[name] for name in feature_names] for row in train_rows], dtype=np.float32),
        label=np.asarray([mapping[row["truth"]] for row in train_rows]),
        weight=weights,
    )
    model = xgb.train(
        {
            "objective": "multi:softprob", "num_class": len(MODEL_CLASSES),
            "tree_method": "hist", "device": "cpu", "max_depth": 3,
            "min_child_weight": 2, "eta": 0.035, "subsample": 0.9,
            "colsample_bytree": 0.8, "seed": 20260811,
        },
        matrix,
        num_boost_round=300,
    )
    return model, dict(counts)


def fit_binary_actor_model(train_rows, feature_names, positive_class, positive_weight):
    labels = np.asarray(
        [int(row["truth"] == positive_class) for row in train_rows], dtype=np.float32
    )
    counts = {
        0: max(1, int((labels == 0).sum())),
        1: max(1, int((labels == 1).sum())),
    }
    weights = np.asarray([
        (float(positive_weight) if label else 1.0) / math.sqrt(counts[int(label)])
        for label in labels
    ], dtype=np.float32)
    matrix = xgb.DMatrix(
        np.asarray([[row[name] for name in feature_names] for row in train_rows], dtype=np.float32),
        label=labels,
        weight=weights,
    )
    model = xgb.train(
        {
            "objective": "binary:logistic", "tree_method": "hist", "device": "cpu",
            "max_depth": 3, "min_child_weight": 2, "eta": 0.035,
            "subsample": 0.9, "colsample_bytree": 0.9, "seed": 20260811,
        },
        matrix,
        num_boost_round=300,
    )
    return model, counts


def explicit_pair_keys(rows):
    """Return exact declared actor pairs, never all actors in one video."""
    pairs = set()
    observed = {(row.get("video") or row["clip_id"], row["actor_id"]) for row in rows}
    for row in rows:
        video = row.get("video") or row["clip_id"]
        try:
            peers = json.loads(row.get("interaction_peer_ids") or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid interaction peers for {video}/{row['actor_id']}") from exc
        for peer in peers:
            other = (video, str(peer))
            if other in observed:
                pairs.add(frozenset(((video, row["actor_id"]), other)))
    return pairs


def pair_c2_event_keys(test_rows, c2_probabilities, threshold):
    """Return actor keys covered by a pair-level c2 exchange event.

    The hand cue is observed on one actor, but c2 is a source/peer exchange
    event.  Therefore a qualified midpoint approach propagates to every
    actor in that two-person pair.  This is prediction structure, not truth:
    no manifest class or action_actor_ids is consulted here.
    """
    by_key = {(row["video"], row["actor_id"]): (row, float(probability))
              for row, probability in zip(test_rows, c2_probabilities)}
    keys = set()
    for pair in explicit_pair_keys(test_rows):
        entries = [by_key[key] for key in pair]
        near_midpoint = any(number(row.get("near_midpoint_pre_cross__max")) >= 1.0 for row, _ in entries)
        learned_c2 = any(score >= threshold for _, score in entries)
        if near_midpoint and learned_c2:
            keys.update(pair)
    return keys


def pair_c7_event_keys(test_rows, c7_probabilities, threshold):
    """Return both endpoints of an explicit pair-wise c7 source event.

    c7, like c2, is a source/peer event.  A qualified c7 cue on the declared
    source is therefore propagated only to its declared peer; an unrelated
    actor in the same video is never included.  The learned cue threshold is
    supplied by the train split, and manifest truth is deliberately ignored.
    """
    by_key = {(row["video"], row["actor_id"]): (row, float(probability))
              for row, probability in zip(test_rows, c7_probabilities)}
    keys = set()
    for pair in explicit_pair_keys(test_rows):
        entries = [by_key[key] for key in pair]
        learned_c7 = any(score >= threshold for _, score in entries)
        if learned_c7:
            keys.update(pair)
    return keys


def pair_propagated_scores(rows, probabilities):
    """Propagate a pair-event score only across explicit interaction pairs."""
    by_key = {(row["video"], row["actor_id"]): float(probability)
              for row, probability in zip(rows, probabilities)}
    scores = dict(by_key)
    for pair in explicit_pair_keys(rows):
        pair_score = max(by_key.get(key, 0.0) for key in pair)
        for key in pair:
            scores[key] = pair_score
    return [scores[(row["video"], row["actor_id"])] for row in rows]


def fit_binary_threshold(rows, probabilities, positive_class):
    """Fit a binary specialist threshold from train actors only."""
    candidates = sorted({float(value) for value in probabilities} | {0.5})
    truth = [row["truth"] == positive_class for row in rows]
    best = (0.0, 0.5)
    for threshold in candidates:
        predicted = [float(value) >= threshold for value in probabilities]
        tp = sum(actual and guess for actual, guess in zip(truth, predicted))
        fp = sum(not actual and guess for actual, guess in zip(truth, predicted))
        fn = sum(actual and not guess for actual, guess in zip(truth, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, threshold)
        if candidate[0] > best[0] or candidate[0] == best[0] and candidate[1] > best[1]:
            best = candidate
    return best[1]


def fit_c7_q7_thresholds(train_rows, hand_jitter_aware=False):
    """Calibrate geometric c7 gates from train c5 motion only.

    These are nuisance-motion ceilings, not labels/features at inference.  The
    c7 decision score is calibrated separately at actor level.
    """
    c5_rows = [row for row in train_rows if row["truth"] == "c5"]
    if not c5_rows:
        raise ValueError("c5 train frames are required to calibrate Q7")
    if hand_jitter_aware:
        names = {
            "raise_displacement": "c7_selected_hand_raise_displacement",
            "raise_speed": "c7_selected_hand_raise_speed",
            "shape_change": "c7_selected_hand_shape_change",
            "jitter_speed": "c7_selected_hand_jitter_speed",
        }
        thresholds = {
            key: float(np.quantile([number(row.get(value)) for row in c5_rows], 0.95))
            for key, value in names.items()
        }
        thresholds["coherence"] = 0.5
        return thresholds
    return {
        name: float(np.quantile(
            [number(row.get(name)) for row in c5_rows], 0.95
        ))
        for name in ("raise_displacement", "raise_speed", "finger_shape_change")
    }


def q7_frame(row, thresholds, hand_jitter_aware=False):
    """Implement Q7: own-side, non-exchange hand raise/sign evidence."""
    if hand_jitter_aware:
        visible_hand = number(row.get("c7_selected_hand_valid")) >= 1.0 and number(row.get("c7_selected_hand_finger_valid")) >= 1.0
        own_side = number(row.get("c7_selected_hand_own_side_distance")) > 0.0
        not_midpoint = number(row.get("c7_selected_hand_near_midpoint")) < 1.0
        not_shared_zone = number(row.get("c7_selected_hand_shared_zone")) < 1.0
        coherent = number(row.get("c7_selected_hand_coherence")) >= thresholds["coherence"]
        raise_displacement = number(row.get("c7_selected_hand_raise_displacement"))
        raise_speed = number(row.get("c7_selected_hand_raise_speed"))
        shape_change = number(row.get("c7_selected_hand_shape_change"))
        jitter_speed = number(row.get("c7_selected_hand_jitter_speed"))
        # Missing/reacquired observations are unknown for shape evidence. A
        # valid same-hand return can still qualify through coherent movement.
        shape_changed = (
            shape_change > thresholds["shape_change"]
            and number(row.get("c7_selected_hand_finger_valid")) >= 1.0
            and jitter_speed <= thresholds["jitter_speed"]
            and number(row.get("c7_selected_hand_gap_frames")) <= 0.0
        )
        raised = (
            raise_displacement > thresholds["raise_displacement"]
            and raise_speed > thresholds["raise_speed"]
            and jitter_speed > thresholds["jitter_speed"]
            and coherent
        )
        return visible_hand and own_side and not_midpoint and not_shared_zone and (raised or shape_changed)
    visible_hand = number(row.get("hand_quality_mask")) >= 1.0
    own_side = number(row.get("own_side_distance")) > 0.0
    not_midpoint = number(row.get("near_midpoint_pre_cross")) < 1.0
    not_shared_zone = number(row.get("c7_shared_zone")) < 1.0
    # Raise is deliberately conjunctive: a baseline-relative displacement and
    # an upward (not absolute) velocity are both required.
    raised = (
        number(row.get("raise_displacement")) > thresholds["raise_displacement"]
        and number(row.get("raise_speed")) > thresholds["raise_speed"]
    )
    shape_changed = number(row.get("finger_shape_change")) > thresholds["finger_shape_change"]
    return visible_hand and own_side and not_midpoint and not_shared_zone and (raised or shape_changed)


def q7_actor_scores(rows, probabilities, thresholds, hand_jitter_aware=False):
    """Compute S7 and propagate only within explicit interaction pairs."""
    scores = {}
    truth = {}
    frames = defaultdict(int)
    for row, probability in zip(rows, probabilities):
        key = (row["clip_id"], row["actor_id"])
        truth[key] = row["actor_label"]
        frames[key] += 1
        if q7_frame(row, thresholds, hand_jitter_aware=hand_jitter_aware):
            scores[key] = max(scores.get(key, -1.0), float(probability[3] - probability[2]))
        else:
            scores.setdefault(key, -1.0)
    actor_rows = [{
        "video": key[0], "actor_id": key[1], "truth": truth[key],
        "c7_q7_actor_score": score, "q7_qualified": int(score > -1.0),
        "frames_scanned": frames[key],
    } for key, score in scores.items()]
    by_key = {(row["video"], row["actor_id"]): row for row in actor_rows}
    # A qualified c7 event is pair-wise.  Its strongest actor score belongs to
    # both endpoints, but never to an unrelated actor in the same video.
    for pair in explicit_pair_keys(actor_rows):
        pair_score = max(by_key[key]["c7_q7_actor_score"] for key in pair)
        if pair_score > -1.0:
            for key in pair:
                by_key[key]["c7_q7_actor_score"] = pair_score
                by_key[key]["q7_qualified"] = 1
    return actor_rows


def fit_c7_actor_score_threshold(actor_rows):
    values = sorted({row["c7_q7_actor_score"] for row in actor_rows if row["q7_qualified"]})
    if not values:
        raise ValueError("no train actor has a Q7-qualified frame")
    best = (0.0, max(values))
    for threshold in values:
        truth = [row["truth"] == "c7" for row in actor_rows]
        predicted = [row["q7_qualified"] and row["c7_q7_actor_score"] >= threshold for row in actor_rows]
        tp = sum(actual and guess for actual, guess in zip(truth, predicted))
        fp = sum(not actual and guess for actual, guess in zip(truth, predicted))
        fn = sum(actual and not guess for actual, guess in zip(truth, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, threshold)
        if candidate[0] > best[0] or candidate[0] == best[0] and candidate[1] > best[1]:
            best = candidate
    return best[1]


def apply_q7_actor_decisions(predictions, q7_rows, threshold, cue_rows=None, cue_probabilities=None, cue_threshold=None):
    """Replace only the c7 branch; c2/c3 retain their earlier priority."""
    by_key = {(row["video"], row["actor_id"]): row for row in q7_rows}
    cue_by_key = {}
    if cue_rows is not None and cue_probabilities is not None:
        cue_by_key = {
            (row["video"], row["actor_id"]): float(probability)
            for row, probability in zip(cue_rows, cue_probabilities)
        }
    pair_cue_score = {}
    if cue_by_key and cue_threshold is not None:
        for pair in explicit_pair_keys(q7_rows):
            pair_cue_score.update({key: max(cue_by_key.get(member, 0.0) for member in pair) for key in pair})
    for row in predictions:
        evidence = by_key[(row["video"], row["actor_id"])]
        q7_qualified = evidence["q7_qualified"] and evidence["c7_q7_actor_score"] >= threshold
        cue_qualified = cue_threshold is None or pair_cue_score.get((row["video"], row["actor_id"]), 0.0) >= cue_threshold
        qualified = q7_qualified and cue_qualified
        if row["predicted_class"] == "c7" and not qualified:
            row["predicted_class"] = "c5"
        elif row["predicted_class"] == "c5" and qualified:
            row["predicted_class"] = "c7"
        row["c7_q7_actor_score"] = evidence["c7_q7_actor_score"]
        row["q7_qualified"] = evidence["q7_qualified"]
        row["q7_threshold"] = threshold
        row["pair_c7_cue_score"] = pair_cue_score.get((row["video"], row["actor_id"]), 0.0)
        if cue_threshold is not None:
            row["pair_c7_cue_threshold"] = cue_threshold
    return predictions


def specialist_priority_predictions(
    test_rows, c2_probabilities, c3_probabilities, c7_probabilities,
    thresholds=None,
):
    """Actor-level priority: pair c2, pair c7, then c3, then c5."""
    thresholds = thresholds or {"c2": 0.5, "c3": 0.5, "c7": 0.5}
    # Keep c2's historical fixed gate for benchmark compatibility.  c7 uses
    # its train-only specialist threshold, with the identical explicit-pair
    # propagation structure.
    pair_c2_keys = pair_c2_event_keys(test_rows, c2_probabilities, threshold=0.5)
    pair_c7_keys = set() if EXCLUDE_C7 else pair_c7_event_keys(test_rows, c7_probabilities, threshold=thresholds["c7"])
    output = []
    for row, c2, c3, c7 in zip(test_rows, c2_probabilities, c3_probabilities, c7_probabilities):
        scores = {"c2": float(c2), "c3": float(c3), "c7": float(c7)}
        key = (row["video"], row["actor_id"])
        # Max is too sensitive to a single landmark jump.  c3 validation says
        # the hand is nearly static, so use a high quantile for the sustained
        # activity gate and reserve max for diagnostics only.
        hand_activity = max(
            number(row.get("hand_motion__q95")),
            number(row.get("finger_motion__q95")),
            number(row.get("hand_finger_motion__q95")),
        )
        hand_is_static = hand_activity <= 0.08
        if key in pair_c2_keys:
            state = "c2"
        elif not EXCLUDE_C7 and key in pair_c7_keys:
            state = "c7"
        elif hand_is_static and scores["c3"] >= thresholds["c3"]:
            # Validated c3 pattern: face/torso cue while the hands are nearly
            # stationary. Generic finger probability must not turn this into c7.
            state = "c3"
        elif scores["c3"] >= thresholds["c3"]:
            state = "c3"
        elif not EXCLUDE_C7 and scores["c7"] >= thresholds["c7"]:
            state = "c7"
        else:
            state = "c5"
        output.append({
            "video": row["video"],
            "actor_id": row["actor_id"],
            "truth": row["truth"],
            "predicted_class": state,
            "c2_cue_score": scores["c2"],
            "c3_face_score": scores["c3"],
            "c7_hand_score": scores["c7"],
            "pair_c2_event": int(key in pair_c2_keys),
            "pair_c7_event": int(key in pair_c7_keys),
            "hand_activity_q95": hand_activity,
            "hand_is_static": int(hand_is_static),
            "frames_scanned": row["frames_scanned"],
        })
    return output


def attach_specialist_frame_evidence(
    predictions, test_rows, frame_probabilities, c3_frame_probabilities
):
    """Attach strongest and first-threshold frames without changing classes."""
    grouped = defaultdict(list)
    for row, probability, c3_probability in zip(
        test_rows, frame_probabilities, c3_frame_probabilities
    ):
        grouped[(row["clip_id"], row["actor_id"])].append(
            (row, probability, float(c3_probability))
        )
    pairs = explicit_pair_keys(test_rows)
    pair_by_key = {}
    for pair in pairs:
        for key in pair:
            pair_by_key[key] = pair

    for prediction in predictions:
        key = (prediction["video"], prediction["actor_id"])
        evidence_class = prediction["predicted_class"]
        chosen = None
        first_flag = None
        source_actor = prediction["actor_id"]
        if evidence_class == "c3":
            candidates = grouped.get(key, [])
            if candidates:
                chosen = max(candidates, key=lambda item: item[2])
                first_flag = next(
                    (
                        item for item in sorted(
                            candidates,
                            key=lambda item: int(item[0].get("source_frame_index", 0)),
                        ) if item[2] >= FRAME_FLAG_THRESHOLD
                    ),
                    None,
                )
        elif evidence_class == "c2":
            pair = pair_by_key.get(key)
            candidates = []
            if pair:
                for pair_key in pair:
                    for item in grouped.get(pair_key, []):
                        row, probability, _ = item
                        if number(row.get("near_midpoint_pre_cross")) >= 1.0:
                            candidates.append((float(probability[0]) - float(probability[2]), pair_key, item))
            if candidates:
                _, source_key, chosen = max(candidates, key=lambda item: item[0])
                source_actor = source_key[1]
                first_flag = next(
                    (
                        item for score, _, item in sorted(
                            candidates,
                            key=lambda item: int(item[2][0].get("source_frame_index", 0)),
                        ) if score >= FRAME_FLAG_THRESHOLD
                    ),
                    None,
                )
        prediction.update({
            "evidence_class": evidence_class if chosen is not None else "",
            "evidence_source_actor_id": source_actor,
            "evidence_frame_index": chosen[0].get("source_frame_index", "") if chosen else "",
            "evidence_timestamp_ms": chosen[0].get("timestamp_ms", "") if chosen else "",
            "evidence_score": (
                float(chosen[2]) if evidence_class == "c3" and chosen else
                float(chosen[1][0]) - float(chosen[1][2]) if evidence_class == "c2" and chosen else
                ""
            ),
            "first_flag_frame_index": first_flag[0].get("source_frame_index", "") if first_flag else "",
            "first_flag_timestamp_ms": first_flag[0].get("timestamp_ms", "") if first_flag else "",
            "frame_flag_threshold": FRAME_FLAG_THRESHOLD,
            "evidence_status": (
                "positive_frame" if first_flag is not None else
                "no_threshold_frame" if chosen is not None else "none"
            ),
        })
    return predictions


def fit_c3_frame_model(rows):
    labels = np.asarray([int(row["truth"] == "c3") for row in rows], dtype=np.float32)
    counts = {0: max(1, int((labels == 0).sum())), 1: max(1, int((labels == 1).sum()))}
    weights = np.asarray([1.0 / math.sqrt(counts[int(label)]) for label in labels], dtype=np.float32)
    matrix = xgb.DMatrix(
        np.asarray([[number(row.get(name)) for name in C3_FEATURES] for row in rows], dtype=np.float32),
        label=labels, weight=weights,
    )
    return xgb.train(
        {
            "objective": "binary:logistic", "tree_method": "hist", "device": "cpu",
            "max_depth": 3, "min_child_weight": 5, "eta": 0.04,
            "subsample": 0.9, "colsample_bytree": 0.9, "seed": 20260811,
        }, matrix, num_boost_round=250,
    )


def c3_frame_actor_evidence(rows, probabilities):
    grouped = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        grouped[(row["clip_id"], row["actor_id"])].append(float(probability))
    return {
        key: max(values) for key, values in grouped.items()
    }


def priority_state_predictions(rows, probabilities, c3_probabilities):
    """Experimental state: c3 tilt first, c2/c7 hand cues override later."""
    grouped = defaultdict(list)
    for row, probability, c3_probability in zip(rows, probabilities, c3_probabilities):
        grouped[(row["clip_id"], row["actor_id"])].append(
            (row, probability, float(c3_probability))
        )
    output = []
    for (video, actor_id), group in grouped.items():
        group.sort(key=lambda item: int(item[0].get("source_frame_index", 0)))
        state = "c5"
        best_score = 0.0
        for row, probability, c3_probability in group:
            c5_probability = float(probability[2])
            if c3_probability > 0.5 and c3_probability > best_score:
                state = "c3"
                best_score = c3_probability
            override_classes = ((0, "c2"),) if EXCLUDE_C7 else ((0, "c2"), (3, "c7"))
            overrides = [
                (name, float(probability[index]) - c5_probability)
                for index, name in override_classes
                if float(probability[index]) > c5_probability
            ]
            if overrides:
                name, score = max(overrides, key=lambda item: item[1])
                if score > best_score:
                    state = name
                    best_score = score
        output.append({
            "video": video,
            "actor_id": actor_id,
            "truth": group[0][0]["actor_label"],
            "predicted_class": state,
            "frames_scanned": len(group),
            "priority_best_score": best_score,
        })
    return output


def causal_specialist_replay(
    rows,
    c2_probabilities,
    c3_probabilities,
    *,
    c2_threshold=0.5,
    c3_threshold,
    suspicious_probabilities=None,
    suspicious_threshold=None,
    c3_gate=None,
    suspicious_gate=None,
    warmup_frames=CAUSAL_WARMUP_FRAMES,
):
    """Replay the same state machine intended for the live feed."""
    if suspicious_probabilities is None:
        suspicious_probabilities = np.zeros(len(rows), dtype=np.float32)
    usable = [
        (row, float(c2), float(c3), float(suspicious))
        for row, c2, c3, suspicious in zip(rows, c2_probabilities, c3_probabilities, suspicious_probabilities)
        if row["warmup_ready"]
    ]
    output = []
    encode = lambda key: f"{key[0]}::{key[1]}"
    for video in sorted({str(row["video"]) for row, _, _, _ in usable}):
        video_rows = [(row, c2, c3, suspicious) for row, c2, c3, suspicious in usable if str(row["video"]) == video]
        actor_keys = sorted({(video, str(row["actor_id"])) for row, _, _, _ in video_rows})
        actor_ids = [encode(key) for key in actor_keys]
        pairs = [
            tuple(encode(key) for key in pair)
            for pair in explicit_pair_keys([row for row, _, _, _ in video_rows])
            if len(pair) == 2
        ]
        state = CausalSpecialistState(
            tuple(actor_ids), c3_threshold=c3_threshold,
            c2_threshold=c2_threshold,
            suspicious_threshold=suspicious_threshold, c3_gate=c3_gate,
            suspicious_gate=suspicious_gate,
        )
        by_frame: dict[int, list[tuple[dict[str, object], float, float, float]]] = defaultdict(list)
        truth_by_actor: dict[str, str] = {}
        for row, c2, c3, suspicious in video_rows:
            by_frame[int(row["source_frame_index"])].append((row, c2, c3, suspicious))
            truth_by_actor[encode((video, str(row["actor_id"])))]= str(row["truth"])
        for frame_index in sorted(by_frame):
            current = by_frame[frame_index]
            scores = {
                encode((video, str(row["actor_id"]))): {
                    "c2": c2, "c3": c3, "suspicious_activity": suspicious,
                    "strict_hand_quality__mean": row.get("strict_hand_quality__mean", 0.0),
                    "hand_motion__q95": row.get("hand_motion__q95", 0.0),
                    "finger_motion__q95": row.get("finger_motion__q95", 0.0),
                    "c3_pose_head_peer_delta__max": row.get("c3_pose_head_peer_delta__max", 0.0),
                    "strict_head_down_delta__q95": row.get("strict_head_down_delta__q95", 0.0),
                    "strict_hand_below_hip__max": row.get("strict_hand_below_hip__max", 0.0),
                    "strict_own_side_outside_midpoint__max": row.get("strict_own_side_outside_midpoint__max", 0.0),
                }
                for row, c2, c3, suspicious in current
            }
            midpoint = {
                # C2 midpoint evidence must occur on the same observed frame
                # as the causal decision.  A rolling max can retain midpoint
                # evidence from an earlier frame while the current C2 score
                # comes from a different frame, creating synthetic exchange
                # evidence and suspicious_activity -> c2 false positives.
                encode((video, str(row["actor_id"]))): (
                    row.get("near_midpoint_pre_cross", 0.0)
                    if number(row.get("current_hand_quality_mask")) > 0.0
                    and number(row.get("current_pair_hand_distance")) > 0.0
                    and number(row.get("current_pair_margin_10pct")) > 0.0
                    else 0.0
                )
                for row, _, _, _ in current
            }
            timestamp = min(int(number(row.get("timestamp_ms", 0))) for row, _, _, _ in current)
            state.update(
                frame_index=frame_index,
                timestamp_ms=timestamp,
                scores_by_actor=scores,
                explicit_pairs=pairs,
                near_midpoint_by_actor=midpoint,
            )

        for actor_key, decision in state.decisions().items():
            _, actor_id = actor_key.split("::", 1)
            source_actor_id = decision.source_actor_id or ""
            if "::" in source_actor_id:
                source_actor_id = source_actor_id.split("::", 1)[1]
            output.append({
                "video": video,
                "actor_id": actor_id,
                "truth": truth_by_actor[actor_key],
                "predicted_class": decision.class_code,
                "evidence_class": decision.class_code if decision.evidence_frame_index is not None else "",
                "evidence_source_actor_id": source_actor_id,
                "evidence_frame_index": decision.evidence_frame_index if decision.evidence_frame_index is not None else "",
                "evidence_timestamp_ms": decision.evidence_timestamp_ms if decision.evidence_timestamp_ms is not None else "",
                "evidence_score": decision.evidence_score if decision.evidence_score is not None else "",
                "first_flag_frame_index": decision.first_flag_frame_index if decision.first_flag_frame_index is not None else "",
                "first_flag_timestamp_ms": decision.first_flag_timestamp_ms if decision.first_flag_timestamp_ms is not None else "",
                "first_flag_source_actor_id": source_actor_id,
                "warmup_frames": warmup_frames,
                "frames_scanned_after_warmup": sum(
                    1 for row, _, _, _ in video_rows
                    if str(row["actor_id"]) == actor_id
                ),
                "causal": True,
            })
    return output


def _fit_actor_max_threshold(rows, probabilities, positive_class):
    """Fit a threshold on the exact actor-level max-prefix statistic."""
    grouped = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        grouped[(row["video"], row["actor_id"])].append(float(probability))
    actor_rows = [
        (key, max(values), rows_by_key_truth(rows, key))
        for key, values in grouped.items()
    ]
    candidates = sorted({score for _, score, _ in actor_rows} | {0.5})
    best = (0.0, 0.5)
    for threshold in candidates:
        tp = sum(truth == positive_class and score >= threshold for _, score, truth in actor_rows)
        fp = sum(truth != positive_class and score >= threshold for _, score, truth in actor_rows)
        fn = sum(truth == positive_class and score < threshold for _, score, truth in actor_rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, threshold)
        if candidate[0] > best[0] or candidate[0] == best[0] and candidate[1] > best[1]:
            best = candidate
    return best[1]


def _fit_current_geometry_actor_threshold(rows, probabilities, positive_class):
    """Fit actor-max score only over qualified current-frame geometry."""
    all_keys = {(row["video"], row["actor_id"]) for row in rows}
    qualified = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        if (
            number(row.get("near_midpoint_pre_cross")) >= 1.0
            and number(row.get("current_hand_quality_mask")) > 0.0
            and number(row.get("current_pair_hand_distance")) > 0.0
            and number(row.get("current_pair_margin_10pct")) > 0.0
        ):
            qualified[(row["video"], row["actor_id"])].append(float(probability))
    actor_rows = [
        (key, max(qualified.get(key, [0.0])), rows_by_key_truth(rows, key))
        for key in all_keys
    ]
    candidates = sorted({score for _, score, _ in actor_rows} | {0.5})
    best = (0.0, 0.5)
    for threshold in candidates:
        tp = sum(truth == positive_class and score >= threshold for _, score, truth in actor_rows)
        fp = sum(truth != positive_class and score >= threshold for _, score, truth in actor_rows)
        fn = sum(truth == positive_class and score < threshold for _, score, truth in actor_rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, threshold)
        if candidate[0] > best[0] or candidate[0] == best[0] and candidate[1] > best[1]:
            best = candidate
    return best[1]


def rows_by_key_truth(rows, key):
    for row in rows:
        if (row["video"], row["actor_id"]) == key:
            return row["truth"]
    raise KeyError(key)


def run_causal_replay(train_frame_rows, test_frame_rows, output_dir, *, c7_weight=3.0, c3_pose_only=False):
    """Train and evaluate the causal live-feed specialist replay."""
    if not EXCLUDE_C7:
        raise ValueError("causal replay currently requires --exclude-c7")
    if EXTENDED_SUSPICIOUS:
        return run_extended_suspicious_causal_replay(
            train_frame_rows, test_frame_rows, output_dir
        )
    causal_c2_train, c2_feature_names = causal_aggregate_rows(
        train_frame_rows, C2_CUE_FEATURES, window_frames=90
    )
    c3_bases = C3_POSE_ONLY_FEATURES if c3_pose_only else C3_FEATURES
    causal_c3_train, c3_feature_names = causal_aggregate_rows(
        train_frame_rows, c3_bases, window_frames=90
    )
    causal_c2_test, _ = causal_aggregate_rows(test_frame_rows, C2_CUE_FEATURES)
    causal_c3_test, _ = causal_aggregate_rows(test_frame_rows, c3_bases, window_frames=90)
    train_c2 = [row for row in causal_c2_train if row["warmup_ready"]]
    train_c3 = [row for row in causal_c3_train if row["warmup_ready"]]
    test_c2 = [row for row in causal_c2_test if row["warmup_ready"]]
    test_c3 = [row for row in causal_c3_test if row["warmup_ready"]]

    c2_model, c2_counts = fit_binary_actor_model(
        train_c2, c2_feature_names, "c2", positive_weight=3.0
    )
    c3_model, c3_counts = fit_binary_actor_model(
        train_c3, c3_feature_names, "c3", positive_weight=3.0
    )
    c2_model.save_model(str(output_dir / "causal_c2_specialist.ubj"))
    c3_model.save_model(str(output_dir / "causal_c3_specialist.ubj"))
    (output_dir / "causal_c2_feature_names.json").write_text(
        json.dumps(c2_feature_names, indent=2), encoding="utf-8"
    )
    (output_dir / "causal_c3_feature_names.json").write_text(
        json.dumps(c3_feature_names, indent=2), encoding="utf-8"
    )

    train_c3_probabilities = c3_model.predict(
        xgb.DMatrix(np.asarray(
            [[row[name] for name in c3_feature_names] for row in train_c3],
            dtype=np.float32,
        ))
    )
    c3_threshold = (
        _fit_actor_max_threshold(train_c3, train_c3_probabilities, "c3")
        if c3_pose_only else fit_binary_threshold(train_c3, train_c3_probabilities, "c3")
    )
    test_c2_probabilities = c2_model.predict(
        xgb.DMatrix(np.asarray(
            [[row[name] for name in c2_feature_names] for row in test_c2],
            dtype=np.float32,
        ))
    )
    test_c3_probabilities = c3_model.predict(
        xgb.DMatrix(np.asarray(
            [[row[name] for name in c3_feature_names] for row in test_c3],
            dtype=np.float32,
        ))
    )
    test_c3_by_key = {
        (row["video"], row["actor_id"], row["source_frame_index"]): float(score)
        for row, score in zip(test_c3, test_c3_probabilities)
    }
    aligned_c2 = []
    aligned_c2_scores = []
    aligned_c3_scores = []
    for row, score in zip(test_c2, test_c2_probabilities):
        key = (row["video"], row["actor_id"], row["source_frame_index"])
        if key not in test_c3_by_key:
            continue
        aligned_c2.append(row)
        aligned_c2_scores.append(float(score))
        aligned_c3_scores.append(test_c3_by_key[key])
    predictions = causal_specialist_replay(
        aligned_c2,
        aligned_c2_scores,
        aligned_c3_scores,
        c3_threshold=c3_threshold,
    )
    with (output_dir / "causal_specialist_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = list(predictions[0]) if predictions else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)

    metrics = {
        "protocol": "actor_only_causal_live_feed_rolling_replay",
        "primary_unit": "(video, actor_id)",
        "metric_labels": ["c2", "c3", "c5"],
        "video_level_metrics": "forbidden",
        "causal": True,
        "future_frames_used_for_decision": False,
        "warmup_frames": CAUSAL_WARMUP_FRAMES,
        "rolling_window_frames": 90,
        "train_prefix_rows": len(train_c2),
        "test_prefix_rows": len(aligned_c2),
        "train_actor_count": len({(row["video"], row["actor_id"]) for row in train_c2}),
        "test_actor_count": len(predictions),
        "c2_train_prefix_counts": c2_counts,
        "c3_train_prefix_counts": c3_counts,
        "c2_threshold": 0.5,
        "c3_threshold_train_only": c3_threshold,
        "c3_feature_family": "pose_only_contract" if c3_pose_only else "historical_pose_face_pnp",
        "c3_threshold_calibration_unit": "train_actor_max_prefix" if c3_pose_only else "train_prefix_row",
        "frame_flag_rule": "first causal prefix frame where specialist priority enters c2 or c3",
        "evidence_rule": "strongest causal prefix score observed up to final frame",
        "actor_flagged_count": sum(bool(row["first_flag_frame_index"] != "") for row in predictions),
        "target_actor_unflagged_count": sum(
            row["truth"] in {"c2", "c3"} and row["first_flag_frame_index"] == ""
            for row in predictions
        ),
        **actor_metrics(predictions),
    }
    (output_dir / "causal_actor_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def _train_quantile(rows, name, truth, quantile, fallback):
    values = [number(row.get(name)) for row in rows if row["truth"] == truth]
    return float(np.quantile(values, quantile)) if values else float(fallback)


def _extended_gate_thresholds(train_rows):
    """Fit all rule limits from train prefix rows only."""
    return {
        # C3: quiet hand/fingers, peer-directed side turn, and no head-down.
        "c3_motion_ceiling": max(
            _train_quantile(train_rows, "hand_motion__q95", "c3", 0.95, 0.0),
            _train_quantile(train_rows, "finger_motion__q95", "c3", 0.95, 0.0),
        ),
        "c3_side_floor": max(0.05, _train_quantile(
            train_rows, "c3_pose_head_peer_delta__max", "c3", 0.25, 0.0
        )),
        # Contract is a hard C3 exclusion: meaningful downward head motion
        # never qualifies as a side-looking C3 cue.
        "c3_down_ceiling": min(
            _train_quantile(train_rows, "strict_head_down_delta__q95", "c3", 0.95, 0.0),
            0.05,
        ),
        # Suspicious: down, active hand, below-hip and own-side evidence.
        "suspicious_down_floor": _train_quantile(
            train_rows, "strict_head_down_delta__q95", "suspicious_activity", 0.25, 0.0
        ),
        "suspicious_motion_floor": max(
            _train_quantile(train_rows, "hand_motion__q95", "suspicious_activity", 0.25, 0.0),
            _train_quantile(train_rows, "finger_motion__q95", "suspicious_activity", 0.25, 0.0),
        ),
        "suspicious_lower_floor": _train_quantile(
            train_rows, "strict_hand_below_hip__max", "suspicious_activity", 0.25, 0.0
        ),
    }


def run_extended_suspicious_causal_replay(train_frame_rows, test_frame_rows, output_dir):
    """Causal C2/C3/suspicious replay on the preserved stage2 path.

    A single prefix stream supplies all three specialists.  This keeps the
    existing causal aggregation, XGBoost training, train-only thresholds and
    actor state machine while avoiding duplicate temporal-prefix materialization.
    """
    shared_bases = tuple(dict.fromkeys((
        *C2_CUE_FEATURES, *C3_POSE_ONLY_FEATURES, *STRICT_C3_SUSPICIOUS_FEATURES,
    )))
    train_prefix, shared_names = causal_aggregate_rows(train_frame_rows, shared_bases)
    test_prefix, _ = causal_aggregate_rows(test_frame_rows, shared_bases)
    train_rows = [row for row in train_prefix if row["warmup_ready"]]
    test_rows = [row for row in test_prefix if row["warmup_ready"]]
    c2_names = [name for name in shared_names if name.split("__", 1)[0] in C2_CUE_FEATURES]
    c3_names = [name for name in shared_names if name.split("__", 1)[0] in (*C3_POSE_ONLY_FEATURES, *STRICT_C3_SUSPICIOUS_FEATURES)]
    suspicious_names = [name for name in shared_names if name.split("__", 1)[0] in STRICT_C3_SUSPICIOUS_FEATURES]
    specs = {
        "c2": c2_names,
        "c3": c3_names,
        "suspicious_activity": suspicious_names,
    }
    models, thresholds, train_scores, test_scores = {}, {}, {}, {}
    for code, names in specs.items():
        model, _ = fit_binary_actor_model(train_rows, names, code, positive_weight=3.0)
        models[code] = model
        model.save_model(str(output_dir / f"causal_{code}_specialist.ubj"))
        (output_dir / f"causal_{code}_feature_names.json").write_text(
            json.dumps(names, indent=2), encoding="utf-8"
        )
        train_scores[code] = model.predict(xgb.DMatrix(np.asarray(
            [[row[name] for name in names] for row in train_rows], dtype=np.float32
        )))
        test_scores[code] = model.predict(xgb.DMatrix(np.asarray(
            [[row[name] for name in names] for row in test_rows], dtype=np.float32
        )))
        thresholds[code] = _fit_actor_max_threshold(train_rows, train_scores[code], code)
    thresholds["c2"] = _fit_current_geometry_actor_threshold(
        train_rows, train_scores["c2"], "c2"
    )
    gates = _extended_gate_thresholds(train_rows)

    def c3_gate(values):
        return (
            number(values.get("strict_hand_quality__mean")) > 0.0
            and number(values.get("hand_motion__q95")) <= gates["c3_motion_ceiling"]
            and number(values.get("finger_motion__q95")) <= gates["c3_motion_ceiling"]
            and number(values.get("c3_pose_head_peer_delta__max")) >= gates["c3_side_floor"]
            and number(values.get("strict_head_down_delta__q95")) <= gates["c3_down_ceiling"]
        )

    def suspicious_gate(values):
        return (
            number(values.get("strict_head_down_delta__q95")) >= gates["suspicious_down_floor"]
            and max(number(values.get("hand_motion__q95")), number(values.get("finger_motion__q95"))) >= gates["suspicious_motion_floor"]
            and number(values.get("strict_hand_below_hip__max")) >= gates["suspicious_lower_floor"]
            and number(values.get("strict_own_side_outside_midpoint__max")) >= 1.0
        )

    predictions = causal_specialist_replay(
        test_rows,
        test_scores["c2"],
        test_scores["c3"],
        c2_threshold=thresholds["c2"],
        c3_threshold=thresholds["c3"],
        suspicious_probabilities=test_scores["suspicious_activity"],
        suspicious_threshold=thresholds["suspicious_activity"],
        c3_gate=c3_gate,
        suspicious_gate=suspicious_gate,
    )
    labels = ["suspicious_activity", "c2", "c3", "c5"]
    truth = [row["truth"] for row in predictions]
    predicted = [row["predicted_class"] for row in predictions]
    report = classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "protocol": "actor_only_causal_live_feed_rolling_replay_extended_suspicious",
        "primary_unit": "(video, actor_id)", "metric_labels": labels,
        "causal": True, "future_frames_used_for_decision": False,
        "official_benchmark_unchanged": True,
        "official_benchmark_labels": ["c2", "c3", "c5"],
        "shared_prefix_stream": True,
        "specialist_thresholds_train_only": thresholds,
        "gate_thresholds_train_only": gates,
        "rules": {
            "c2": "finger-or-hand midpoint evidence through existing explicit-pair gate",
            "c3": "quiet hand/fingers, peer side-turn, no head-down",
            "suspicious_activity": "head down, active hand, below hip, own side outside midpoint",
        },
        "actor_macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "actor_metrics": {label: report[label] for label in labels},
        "actor_confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "test_actor_count": len(predictions),
        "leakage": {
            "raw_actor_id_overlap": sorted({row["actor_id"] for row in train_frame_rows} & {row["actor_id"] for row in test_frame_rows}),
            "clip_id_overlap": sorted({row["clip_id"] for row in train_frame_rows} & {row["clip_id"] for row in test_frame_rows}),
            "split_group_overlap": sorted({row["split_group"] for row in train_frame_rows} & {row["split_group"] for row in test_frame_rows}),
            "truth_used_as_model_feature": False, "identity_used_as_model_feature": False,
            "future_rows_used_for_decision": 0,
        },
    }
    with (output_dir / "causal_specialist_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    (output_dir / "causal_actor_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run(
    input_path: Path,
    manifest_path: Path,
    output_dir: Path,
    mapping_path: Path | None = None,
    feature_mode: str = "absolute",
    json_root: Path | None = None,
    c7_weight: float = 3.0,
    priority_mode: bool = False,
    hand_jitter_aware: bool = False,
    exclude_c7: bool = False,
    causal_replay: bool = False,
    c3_pose_only: bool = False,
    extended_suspicious: bool = False,
):
    global FEATURES, LEGACY_C7_FORMULA, MODEL_CLASSES, TARGET_CLASSES, EXCLUDE_C7, EXTENDED_SUSPICIOUS
    # The default path is the preserved benchmark.  The hand-jitter/Q7 branch
    # is opt-in and must never silently change the deployed c7 formula.
    LEGACY_C7_FORMULA = not hand_jitter_aware
    EXCLUDE_C7 = bool(exclude_c7)
    EXTENDED_SUSPICIOUS = bool(extended_suspicious)
    if EXTENDED_SUSPICIOUS and not (EXCLUDE_C7 and causal_replay and c3_pose_only):
        raise ValueError("--extended-suspicious requires --exclude-c7 --causal-replay --c3-pose-only")
    MODEL_CLASSES = (("suspicious_activity", "c2", "c3", "c5") if EXTENDED_SUSPICIOUS else
                     (("c2", "c3", "c5") if EXCLUDE_C7 else ("c2", "c3", "c5", "c7")))
    TARGET_CLASSES = (("suspicious_activity", "c2", "c3") if EXTENDED_SUSPICIOUS else
                      (("c2", "c3") if EXCLUDE_C7 else ("c2", "c3", "c7")))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    actor_mapping = load_actor_mapping(mapping_path)
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    rows = []
    for source in source_rows:
        if "clip_id" not in source:
            source["clip_id"] = Path(source.get("source_filename", "")).stem
        if "actor_id" not in source:
            key = (source.get("source_filename", ""), source.get("track_id", ""))
            if key not in actor_mapping:
                raise ValueError(f"actor mapping missing for {key}")
            source["actor_id"] = actor_mapping[key]
        if "source_frame_index" not in source:
            source["source_frame_index"] = str(max(0, int(source.get("frame_id", "1")) - 1))
        label = actor_truth(source, manifest)
        row = dict(source)
        row.update(label)
        if EXCLUDE_C7 and row["actor_truth"] == "c7":
            row["excluded_source"] = True
        row["actor_label"] = row["actor_truth"]
        rows.append(row)

    rows = attach_selected_face_points(rows, json_root)
    if c3_pose_only:
        rows = derive_c3_pose_contract(rows)
    rows = _head_pnp_features(rows)
    rows = derive_behavior_motion(rows)
    rows = derive_face_c3_features(rows)
    rows = derive_finger_motion(rows)
    rows = derive_hand_shape_and_pair_cues(rows)
    if EXTENDED_SUSPICIOUS:
        rows = derive_strict_c2_c3_suspicious_cues(rows)
    if hand_jitter_aware:
        rows = derive_hand_jitter_aware_cues(rows)
    train_frame_rows = [row for row in rows if row["split"] == "train" and not row["excluded_source"]]
    test_frame_rows = [row for row in rows if row["split"] == "test" and not row["excluded_source"]]
    # Feature discovery is a train-only operation.  Test rows are never
    # allowed to influence even unsupervised schema selection.
    if feature_mode not in {"absolute", "normalized"}:
        raise ValueError(f"unsupported feature mode: {feature_mode}")
    FEATURES = select_features(
        train_frame_rows,
        allowed_features=NORMALIZED_FEATURES if feature_mode == "normalized" else None,
    )

    if causal_replay:
        return run_causal_replay(
            train_frame_rows,
            test_frame_rows,
            output_dir,
            c7_weight=c7_weight,
            c3_pose_only=c3_pose_only,
        )

    # Aggregate actor model is retained as a comparison artifact only.
    train_rows, actor_feature_names = aggregate_actor_rows(train_frame_rows)
    test_rows, _ = aggregate_actor_rows(test_frame_rows)
    model, train_counts = fit_actor_model(train_rows, actor_feature_names, c7_weight=c7_weight)
    model.save_model(str(output_dir / "actor_only_c2_c3_c5_c7.ubj"))
    (output_dir / "actor_feature_names.json").write_text(json.dumps(actor_feature_names, indent=2), encoding="utf-8")
    importance = model.get_score(importance_type="gain")
    top_importance = sorted(
        ({"feature": actor_feature_names[int(key[1:])], "gain": float(value)} for key, value in importance.items()),
        key=lambda item: item["gain"], reverse=True,
    )
    (output_dir / "aggregate_actor_feature_importance_gain.json").write_text(json.dumps(top_importance, indent=2), encoding="utf-8")

    frame_model, frame_counts = fit_model(train_frame_rows, c7_weight=c7_weight)
    frame_model.save_model(str(output_dir / "actor_one_positive_frame_c2_c3_c5_c7.ubj"))
    frame_importance = frame_model.get_score(importance_type="gain")
    frame_top_importance = sorted(
        ({"feature": FEATURES[int(key[1:])], "gain": float(value)} for key, value in frame_importance.items()),
        key=lambda item: item["gain"], reverse=True,
    )
    (output_dir / "actor_feature_importance_gain.json").write_text(
        json.dumps(frame_top_importance, indent=2), encoding="utf-8"
    )
    (output_dir / "actor_frame_feature_names.json").write_text(json.dumps(FEATURES, indent=2), encoding="utf-8")
    train_frame_probabilities = frame_model.predict(
        xgb.DMatrix(np.asarray([features(row) for row in train_frame_rows], dtype=np.float32))
    )
    if hand_jitter_aware:
        q7_thresholds = fit_c7_q7_thresholds(train_frame_rows, hand_jitter_aware=True)
        q7_train_rows = q7_actor_scores(
            train_frame_rows, train_frame_probabilities, q7_thresholds,
            hand_jitter_aware=True,
        )
        q7_actor_threshold = fit_c7_actor_score_threshold(q7_train_rows)
    else:
        q7_thresholds = None
        q7_actor_threshold = None
    frame_thresholds = fit_actor_thresholds(
        actor_evidence(train_frame_rows, train_frame_probabilities)
    )
    c3_train_rows, c3_feature_names = aggregate_actor_rows(train_frame_rows, C3_FEATURES)
    c3_test_rows, _ = aggregate_actor_rows(test_frame_rows, C3_FEATURES)
    c3_model, _ = fit_actor_model(c3_train_rows, c3_feature_names, c7_weight=1.0)
    c3_model.save_model(str(output_dir / "actor_only_c3_face_shoulder.ubj"))
    c3_frame_model = fit_c3_frame_model(train_frame_rows)
    c3_frame_model.save_model(str(output_dir / "actor_only_c3_face_shoulder_frame.ubj"))

    c2_cue_train_rows, c2_cue_feature_names = aggregate_actor_rows(
        train_frame_rows, C2_CUE_FEATURES
    )
    c2_cue_test_rows, _ = aggregate_actor_rows(test_frame_rows, C2_CUE_FEATURES)
    c2_cue_model, _ = fit_binary_actor_model(
        c2_cue_train_rows, c2_cue_feature_names, "c2", positive_weight=3.0
    )
    c2_cue_model.save_model(str(output_dir / "actor_c2_hand_middle_cue.ubj"))

    c3_cue_train_rows, c3_cue_feature_names = aggregate_actor_rows(
        train_frame_rows, C3_FEATURES
    )
    c3_cue_test_rows, _ = aggregate_actor_rows(test_frame_rows, C3_FEATURES)
    c3_cue_model, _ = fit_binary_actor_model(
        c3_cue_train_rows, c3_cue_feature_names, "c3", positive_weight=3.0
    )
    c3_cue_model.save_model(str(output_dir / "actor_c3_face_tilt_cue.ubj"))

    c7_cue_train_rows = c7_cue_test_rows = c7_cue_feature_names = None
    c7_cue_model = None
    if not EXCLUDE_C7:
        c7_cue_features = (
            (*C7_CUE_FEATURES, *HAND_JITTER_C7_CUE_FEATURES)
            if hand_jitter_aware else C7_CUE_FEATURES_LEGACY
        )
        c7_cue_train_rows, c7_cue_feature_names = aggregate_actor_rows(train_frame_rows, c7_cue_features)
        c7_cue_test_rows, _ = aggregate_actor_rows(test_frame_rows, c7_cue_features)
        c7_cue_model, _ = fit_binary_actor_model(c7_cue_train_rows, c7_cue_feature_names, "c7", positive_weight=4.0)
        c7_cue_model.save_model(str(output_dir / "actor_c7_finger_hand_cue.ubj"))
    c2_cue_train_probabilities = c2_cue_model.predict(
        xgb.DMatrix(np.asarray([[row[name] for name in c2_cue_feature_names] for row in c2_cue_train_rows], dtype=np.float32))
    )
    c3_cue_train_probabilities = c3_cue_model.predict(
        xgb.DMatrix(np.asarray([[row[name] for name in c3_cue_feature_names] for row in c3_cue_train_rows], dtype=np.float32))
    )
    c7_cue_train_probabilities = None
    if not EXCLUDE_C7:
        c7_cue_train_probabilities = c7_cue_model.predict(
            xgb.DMatrix(np.asarray([[row[name] for name in c7_cue_feature_names] for row in c7_cue_train_rows], dtype=np.float32))
        )
    specialist_thresholds = {
        "c2": fit_binary_threshold(c2_cue_train_rows, c2_cue_train_probabilities, "c2"),
        "c3": fit_binary_threshold(c3_cue_train_rows, c3_cue_train_probabilities, "c3"),
    }
    if not EXCLUDE_C7:
        specialist_thresholds["c7"] = fit_binary_threshold(c7_cue_train_rows, c7_cue_train_probabilities, "c7")
    if hand_jitter_aware and not EXCLUDE_C7:
        pair_c7_train_probabilities = pair_propagated_scores(
            c7_cue_train_rows, c7_cue_train_probabilities
        )
        pair_c7_threshold = fit_binary_threshold(
            c7_cue_train_rows, pair_c7_train_probabilities, "c7"
        )
    else:
        pair_c7_threshold = None
    aggregate_probabilities = model.predict(
        xgb.DMatrix(np.asarray([[row[name] for name in actor_feature_names] for row in test_rows], dtype=np.float32))
    )
    aggregate_predictions = []
    mapping = {index: name for index, name in enumerate(MODEL_CLASSES)}
    for row, probability in zip(test_rows, aggregate_probabilities):
        best = int(np.argmax(probability))
        aggregate_predictions.append({
            "video": row["video"], "actor_id": row["actor_id"], "truth": row["truth"],
            "predicted_class": mapping[best], "c2_score": float(probability[0]),
            "c3_score": float(probability[1]), "c5_score": float(probability[2]),
            "c7_score": float(probability[3]) if not EXCLUDE_C7 else 0.0,
            "frames_scanned": row["frames_scanned"],
        })

    c2_cue_probabilities = c2_cue_model.predict(
        xgb.DMatrix(np.asarray([[row[name] for name in c2_cue_feature_names] for row in c2_cue_test_rows], dtype=np.float32))
    )
    c3_cue_probabilities = c3_cue_model.predict(
        xgb.DMatrix(np.asarray([[row[name] for name in c3_cue_feature_names] for row in c3_cue_test_rows], dtype=np.float32))
    )
    c7_cue_probabilities = np.zeros(len(c3_cue_probabilities), dtype=np.float32)
    if not EXCLUDE_C7:
        c7_cue_probabilities = c7_cue_model.predict(
            xgb.DMatrix(np.asarray([[row[name] for name in c7_cue_feature_names] for row in c7_cue_test_rows], dtype=np.float32))
        )
    specialist_predictions = specialist_priority_predictions(
        test_rows, c2_cue_probabilities, c3_cue_probabilities, c7_cue_probabilities,
        thresholds=specialist_thresholds,
    )
    frame_probabilities = frame_model.predict(
        xgb.DMatrix(np.asarray([features(row) for row in test_frame_rows], dtype=np.float32))
    )
    if hand_jitter_aware:
        q7_test_rows = q7_actor_scores(
            test_frame_rows, frame_probabilities, q7_thresholds,
            hand_jitter_aware=True,
        )
        apply_q7_actor_decisions(
            specialist_predictions, q7_test_rows, q7_actor_threshold,
            cue_rows=c7_cue_test_rows,
            cue_probabilities=c7_cue_probabilities,
            cue_threshold=pair_c7_threshold,
        )
    actor_rows = actor_evidence(test_frame_rows, frame_probabilities)
    resolve_actor_predictions(actor_rows, frame_thresholds)
    c3_probabilities = c3_model.predict(
        xgb.DMatrix(np.asarray([[row[name] for name in c3_feature_names] for row in c3_test_rows], dtype=np.float32))
    )
    for row, probability in zip(actor_rows, c3_probabilities):
        row["c3_face_shoulder_score"] = float(probability[1])
    c3_frame_probabilities = c3_frame_model.predict(
        xgb.DMatrix(np.asarray([[number(row.get(name)) for name in C3_FEATURES] for row in test_frame_rows], dtype=np.float32))
    )
    c3_frame_scores = c3_frame_actor_evidence(test_frame_rows, c3_frame_probabilities)
    for row in actor_rows:
        row["c3_frame_face_shoulder_score"] = float(c3_frame_scores[(row["video"], row["actor_id"])])
    attach_specialist_frame_evidence(
        specialist_predictions,
        test_frame_rows,
        frame_probabilities,
        c3_frame_probabilities,
    )
    priority_rows = priority_state_predictions(
        test_frame_rows, frame_probabilities, c3_frame_probabilities
    )
    priority_by_key = {(row["video"], row["actor_id"]): row for row in priority_rows}
    for row in actor_rows:
        row["priority_predicted_class"] = priority_by_key[(row["video"], row["actor_id"])] ["predicted_class"]
        row["priority_best_score"] = priority_by_key[(row["video"], row["actor_id"])] ["priority_best_score"]
    if priority_mode:
        specialist_by_key = {
            (row["video"], row["actor_id"]): row for row in specialist_predictions
        }
        for row in actor_rows:
            row["predicted_class"] = specialist_by_key[(row["video"], row["actor_id"])] ["predicted_class"]
            specialist = specialist_by_key[(row["video"], row["actor_id"])]
            for field in (
                "evidence_class", "evidence_source_actor_id",
                "evidence_frame_index", "evidence_timestamp_ms",
                "evidence_score", "first_flag_frame_index",
                "first_flag_timestamp_ms", "frame_flag_threshold",
                "evidence_status",
            ):
                row[field] = specialist[field]

    with (output_dir / "aggregate_actor_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(aggregate_predictions[0]) if aggregate_predictions else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate_predictions)

    with (output_dir / "specialist_priority_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(specialist_predictions[0]) if specialist_predictions else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(specialist_predictions)

    with (output_dir / "actor_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(actor_rows[0]) if actor_rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(actor_rows)

    with (output_dir / "actor_frame_evidence.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["video", "actor_id", "source_frame_index", "timestamp_ms", "actor_label", "manifest_class_code", "source_actor", "within_action_interval", "c2_score", "c3_score", "c5_score", "c7_score"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, probability in zip(test_frame_rows, frame_probabilities):
            writer.writerow({
                "video": row["clip_id"], "actor_id": row["actor_id"],
                "source_frame_index": row["source_frame_index"], "timestamp_ms": row.get("timestamp_ms", ""),
                "actor_label": row["actor_label"], "manifest_class_code": row["manifest_class_code"],
                "source_actor": row["source_actor"], "within_action_interval": row["within_action_interval"],
                "c2_score": float(probability[0]), "c3_score": float(probability[1]),
                "c5_score": float(probability[2]),
                "c7_score": float(probability[3]) if not EXCLUDE_C7 else 0.0,
            })

    metrics = {
        "protocol": "actor_only_full_video_scan_one_positive_frame",
        "feature_mode": feature_mode,
        "truth": "manifest action_actor_ids + manifest class_code + action_start_s/action_end_s",
        "primary_unit": "(video, actor_id)",
        "video_level_metrics": "forbidden",
        "train_rows": len(train_frame_rows),
        "test_rows": len(test_frame_rows),
        "train_actor_count": len(train_rows),
        "test_actor_count": len(test_rows),
        "train_actor_frame_truth_counts": dict(frame_counts),
        "actor_feature_count": len(actor_feature_names),
        "c3_specialist_feature_count": len(c3_feature_names),
        "c3_frame_specialist_feature_count": len(C3_FEATURES),
        "actor_rows": len(actor_rows),
        **actor_metrics(actor_rows),
        "aggregate_baseline_actor_metrics": actor_metrics(aggregate_predictions),
        "c5_score": "trained model probability; not assumed zero",
        "persistence": "none",
        "positive_frame_rule": "max target-minus-c5 margin over all scanned test frames; class-specific thresholds fit on train actors only",
        "positive_frame_thresholds": frame_thresholds,
        "priority_mode": priority_mode,
        "exclude_c7": EXCLUDE_C7,
        "priority_rule": (
            "pair c2 midpoint event propagates to pair actors; static-hand face cue selects c3"
            if EXCLUDE_C7 else
            "pair c2 midpoint event propagates to pair actors; pair c7 cue event propagates to pair actors; static-hand face cue selects c3"
        ),
        "primary_metric": "actor_macro_f1_c2_c3_c5" if EXCLUDE_C7 else "actor_macro_f1_c2_c3_c5_c7",
        "metric_labels": list((*TARGET_CLASSES, "c5")),
        "specialist_thresholds_train_only": specialist_thresholds,
        "frame_flag_threshold": FRAME_FLAG_THRESHOLD,
        "q7_rule": (
            "V_hand AND own-side AND NOT midpoint AND NOT shared-zone AND "
            "(baseline-relative directional raise OR shape-change)"
            if hand_jitter_aware else None
        ),
        "q7_geometry_thresholds_train_c5_p95": q7_thresholds,
        "q7_actor_score_rule": (
            "max over Q7-qualified frames of p7 - p5; propagated only to explicit interaction_pairs"
            if hand_jitter_aware else None
        ),
        "q7_actor_score_threshold_train_only": q7_actor_threshold,
        "pair_c7_cue_threshold_train_only": pair_c7_threshold,
        "hand_jitter_aware": hand_jitter_aware,
        "c7_weight": c7_weight,
        "head_pose": {
            "method": "SolvePnP_rotation_only",
            "model_points": list(HEAD_PNP_MODEL),
            "torso_relative": True,
            "camera_motion_gate": True,
            "camera_gate_threshold_normalized": CAMERA_MOTION_THRESHOLD,
            "camera_gate_agreement": CAMERA_MOTION_AGREEMENT,
            "reprojection_threshold_normalized": HEAD_PNP_REPROJECTION_THRESHOLD,
            "direction_threshold_degrees": HEAD_PNP_DIRECTION_DEG,
            "ema_alpha": HEAD_PNP_EMA_ALPHA,
            "paper_alignment": "six-point model, approximate intrinsics, iterative solvePnP, Rodrigues, EMA alpha 0.4, yaw alert threshold 25 degrees",
            "frame_resolution_assumed": [int(HEAD_FRAME_WIDTH), int(HEAD_FRAME_HEIGHT)],
            "translation_used_for_behavior": False,
        },
        "priority_actor_metrics": actor_metrics(priority_rows),
        "specialist_priority_actor_metrics": actor_metrics(specialist_predictions),
        "cue_weights": {
            "c2_hand_middle": 3.0,
            "c3_face_tilt": 3.0,
            "c7_finger_hand": 4.0,
        },
        "input_scope": "full video",
        "excluded_source_classes": ["c1", "c4", "c6"],
    }
    (output_dir / "actor_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--feature-mode", choices=("absolute", "normalized"), default="absolute")
    parser.add_argument("--json-root", type=Path)
    parser.add_argument("--c7-weight", type=float, default=3.0)
    parser.add_argument("--priority-mode", action="store_true")
    parser.add_argument("--hand-jitter-aware", action="store_true")
    parser.add_argument(
        "--causal-replay",
        action="store_true",
        help="Run causal live-feed prefix replay; requires --exclude-c7.",
    )
    parser.add_argument(
        "--exclude-c7",
        action="store_true",
        help="Train and evaluate only c2/c3/c5; omit c7 from the model and metrics.",
    )
    parser.add_argument(
        "--c3-pose-only",
        action="store_true",
        help="Opt-in causal C3 A/B branch using contract-faithful pose head/torso features.",
    )
    parser.add_argument(
        "--extended-suspicious",
        action="store_true",
        help="Opt-in C1/C4 -> suspicious_activity causal profile; requires causal C2/C3 pose-only mode.",
    )
    args = parser.parse_args()
    print(json.dumps(run(
        args.input,
        args.manifest,
        args.output_dir,
        args.mapping,
        args.feature_mode,
        args.json_root,
        args.c7_weight,
        args.priority_mode,
        args.hand_jitter_aware,
        args.exclude_c7,
        args.causal_replay,
        args.c3_pose_only,
        args.extended_suspicious,
    ), indent=2))


if __name__ == "__main__":
    main()

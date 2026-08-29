"""Phase 2 canonical actor/frame feature extraction from Holistic JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

POSE_POINTS = (0, 1, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24)
HAND_POINTS = tuple(range(21))
LANDMARK_GROUPS = {
    "pose": "pose_landmarks",
    "left_hand": "left_hand_landmarks",
    "right_hand": "right_hand_landmarks",
    "face": "selected_face_landmarks",
}


def _point(points: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(points, list):
        return None
    for point in points:
        if isinstance(point, dict) and point.get("index") == index:
            return point
    return None


def _value(point: dict[str, Any] | None, key: str) -> str:
    value = point.get(key) if point else None
    if value is None:
        return ""
    try:
        value = float(value)
        return str(value) if math.isfinite(value) else ""
    except (TypeError, ValueError):
        return ""


def _bbox(track: dict[str, Any]) -> tuple[str, str, str, str]:
    values = track.get("bbox_xyxy")
    if not isinstance(values, list) or len(values) != 4:
        return "", "", "", ""
    return tuple(_value({"v": value}, "v") for value in values)  # type: ignore[return-value]


def _manifest_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row["actor_ids"] = json.loads(row.get("actor_ids") or "[]")
            row["action_actor_ids"] = json.loads(row.get("action_actor_ids") or "[]")
            result[row["clip_id"]] = row
    return result


def _mapping_index(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            clip_id = row.get("clip_id") or Path(row.get("source_filename", "")).stem
            actor_id = row.get("actor_id", "").strip()
            track_id = row.get("track_id", "").strip()
            if not clip_id or not actor_id or not track_id:
                raise ValueError("actor mapping requires clip/source filename, actor_id, and track_id")
            normalized = dict(row)
            normalized.update({
                "clip_id": clip_id,
                "actor_id": actor_id,
                "track_id": track_id,
                "spatial_role": row.get("spatial_role") or row.get("track_side", ""),
                "confidence": row.get("confidence") or row.get("mapping_status", ""),
            })
            clip_mapping = result.setdefault(clip_id, {})
            if actor_id in clip_mapping:
                raise ValueError(f"duplicate actor mapping for {clip_id}/{actor_id}")
            if any(value["track_id"] == track_id for value in clip_mapping.values()):
                raise ValueError(f"duplicate track mapping for {clip_id}/{track_id}")
            clip_mapping[actor_id] = normalized
    return result


def _fields() -> list[str]:
    fields = [
        "clip_id", "filename", "split", "split_group", "class_code", "actor_id", "track_id",
        "spatial_role", "mapping_confidence", "is_action_actor", "sample_index", "source_frame_index", "frame_id",
        "timestamp_ms", "dt_ms", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "face_valid",
        "mouth_valid", "face_predicted", "pose_valid_ratio", "left_hand_valid", "right_hand_valid",
        "track_present", "frame_track_count", "continuity_epoch",
    ]
    for group, points in (("pose", POSE_POINTS), ("left_hand", HAND_POINTS), ("right_hand", HAND_POINTS)):
        for index in points:
            fields.extend((f"{group}_{index}_x", f"{group}_{index}_y", f"{group}_{index}_frame_x", f"{group}_{index}_frame_y", f"{group}_{index}_valid"))
    return fields


def _frame_rows(manifest: dict[str, Any], mapping: dict[str, dict[str, str]], payload: dict[str, Any]):
    previous_time: dict[str, float] = {}
    previous_presence: dict[str, bool] = {}
    continuity_epoch: dict[str, int] = {}
    for frame_position, frame in enumerate(payload.get("frames") or []):
        if not isinstance(frame, dict):
            continue
        sample_index = int(frame.get("sample_index", frame_position))
        source_index = int(frame.get("source_frame_index", frame_position))
        frame_id = frame.get("frame_id", source_index + 1)
        timestamp = float(frame.get("timestamp_ms", 0.0) or 0.0)
        tracks = [track for track in frame.get("tracks") or [] if isinstance(track, dict)]
        tracks_by_id = {str(track.get("track_id", "")): track for track in tracks}
        for actor in sorted(mapping.values(), key=lambda value: int(value["track_id"])):
            track_id = actor["track_id"]
            track = tracks_by_id.get(track_id)
            track_present = track is not None
            if previous_presence.get(track_id) is True and not track_present:
                continuity_epoch[track_id] = continuity_epoch.get(track_id, 0) + 1
            previous_presence[track_id] = track_present
            track = track or {"track_id": track_id}
            actor_id = actor["actor_id"]
            dt = timestamp - previous_time.get(track_id, timestamp)
            previous_time[track_id] = timestamp
            row: dict[str, Any] = {
                "clip_id": manifest["clip_id"], "filename": manifest["filename"], "split": manifest["split"],
                "split_group": manifest["split_group"], "class_code": manifest["class_code"],
                "actor_id": actor_id, "track_id": track_id, "spatial_role": actor["spatial_role"],
                "mapping_confidence": actor["confidence"],
                "is_action_actor": int(actor_id in {str(v) for v in manifest["action_actor_ids"]}),
                "sample_index": sample_index, "source_frame_index": source_index,
                "frame_id": frame_id, "timestamp_ms": timestamp,
                "dt_ms": max(0.0, dt), "face_valid": int(bool(track.get("face_valid"))),
                "mouth_valid": int(bool(track.get("mouth_valid"))), "face_predicted": int(bool(track.get("face_predicted"))),
                "pose_valid_ratio": track.get("pose_valid_ratio", ""),
                "left_hand_valid": int(bool(track.get("left_hand_landmarks"))),
                "right_hand_valid": int(bool(track.get("right_hand_landmarks"))),
                "track_present": int(track_present), "frame_track_count": len(tracks),
                "continuity_epoch": continuity_epoch.get(track_id, 0),
            }
            row.update(dict(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), _bbox(track))))
            for group, source in LANDMARK_GROUPS.items():
                points = track.get(source)
                indices = POSE_POINTS if group == "pose" else HAND_POINTS if group != "face" else ()
                for index in indices:
                    point = _point(points, index)
                    x_norm, y_norm = _value(point, "x"), _value(point, "y")
                    x, y = _value(point, "frame_x"), _value(point, "frame_y")
                    row[f"{group}_{index}_x"] = x_norm
                    row[f"{group}_{index}_y"] = y_norm
                    row[f"{group}_{index}_frame_x"] = x
                    row[f"{group}_{index}_frame_y"] = y
                    row[f"{group}_{index}_valid"] = int(x != "" and y != "")
            if row["pose_valid_ratio"] == "":
                row["pose_valid_ratio"] = sum(
                    row[f"pose_{index}_valid"] for index in POSE_POINTS
                ) / len(POSE_POINTS)
            yield row


def extract(manifest_path: Path, mapping_path: Path, json_root: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = _fields()
    manifest = _manifest_index(manifest_path)
    mappings = _mapping_index(mapping_path)
    if set(mappings) != set(manifest):
        missing = sorted(set(manifest) - set(mappings))
        extra = sorted(set(mappings) - set(manifest))
        raise ValueError(f"mapping/manifest clip mismatch: missing={missing}, extra={extra}")
    output_path = output_dir / "canonical_frame_features.csv"
    rows_written = 0
    videos = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for clip_id, row in manifest.items():
            path = json_root / f"{Path(row['filename']).stem}.json"
            if not path.is_file():
                raise FileNotFoundError(f"8 FPS landmark JSON missing: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            observed_track_ids = {
                str(track.get("track_id", ""))
                for frame in payload.get("frames") or [] if isinstance(frame, dict)
                for track in frame.get("tracks") or [] if isinstance(track, dict)
            }
            mapped_track_ids = {value["track_id"] for value in mappings[clip_id].values()}
            if observed_track_ids != mapped_track_ids:
                raise ValueError(
                    f"mapping/JSON track mismatch for {clip_id}: "
                    f"observed={sorted(observed_track_ids)}, mapped={sorted(mapped_track_ids)}"
                )
            count = 0
            for feature_row in _frame_rows(row, mappings.get(clip_id, {}), payload):
                writer.writerow(feature_row)
                rows_written += 1
                count += 1
            if count:
                videos += 1
    mapping_statuses = sorted({
        value.get("mapping_status", "")
        for clip_mapping in mappings.values() for value in clip_mapping.values()
    })
    schema = {
        "fields": fields,
        "coordinate_system": "frame pixels",
        "missing_value": "empty plus *_valid=0",
        "source_manifest": str(manifest_path),
        "landmark_root": str(json_root),
        "mapping": str(mapping_path),
        "mapping_statuses": mapping_statuses,
        "mapping_identity_verified": not any("provisional" in value for value in mapping_statuses),
    }
    (output_dir / "canonical_frame_features.schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    summary = {"videos": videos, "rows": rows_written, "fields": len(fields), "output": str(output_path)}
    (output_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.manifest, args.mapping, args.json_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

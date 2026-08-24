"""Validate streamed landmark JSON against the frame-track contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_QUALITY_FIELDS = {"face_valid", "mouth_valid", "face_predicted"}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_landmarks(
    path: Path,
    *,
    minimum_coverage: float = 0.90,
    maximum_gap: int = 3,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("format_version") != 3:
        errors.append("format_version must be 3")
    if not REQUIRED_QUALITY_FIELDS.issubset(payload.get("quality_fields", [])):
        errors.append("quality_fields missing required face fields")

    seen_keys: set[tuple[int, int]] = set()
    last_timestamp = -1
    track_counts: dict[int, int] = {}
    face_valid = 0
    mouth_valid = 0
    predicted = 0
    per_track: dict[int, list[tuple[int, bool, bool]]] = {}
    frames = payload.get("frames", [])
    for frame in frames:
        frame_id = frame.get("frame_id")
        timestamp = frame.get("timestamp_ms")
        if not isinstance(frame_id, int) or not isinstance(timestamp, int):
            errors.append("frame_id and timestamp_ms must be integers")
            continue
        if timestamp < last_timestamp:
            errors.append(f"timestamp decreased at frame {frame_id}")
        last_timestamp = timestamp
        for track in frame.get("tracks", []):
            track_id = track.get("track_id")
            key = (frame_id, track_id)
            if not isinstance(track_id, int):
                errors.append(f"invalid track_id at frame {frame_id}")
                continue
            if key in seen_keys:
                errors.append(f"duplicate frame-track key {key}")
            seen_keys.add(key)
            track_counts[track_id] = track_counts.get(track_id, 0) + 1
            crop = track.get("crop_bbox_xyxy", [])
            if len(crop) != 4 or not all(_finite(value) for value in crop):
                errors.append(f"invalid crop bbox at {key}")
            elif crop[2] <= crop[0] or crop[3] <= crop[1]:
                errors.append(f"non-positive crop bbox at {key}")
            if track.get("face_valid"):
                face_valid += 1
            if track.get("mouth_valid"):
                mouth_valid += 1
            if track.get("face_predicted"):
                predicted += 1
            for group_name in (
                "pose_landmarks",
                "left_hand_landmarks",
                "right_hand_landmarks",
                "selected_face_landmarks",
            ):
                for point in track.get(group_name, []):
                    for field in ("x", "y", "frame_x", "frame_y"):
                        value = point.get(field)
                        if value is not None and not _finite(value):
                            errors.append(f"non-finite {field} in {group_name} at {key}")

            per_track.setdefault(track_id, []).append(
                (frame_id, bool(track.get("face_valid")), bool(track.get("mouth_valid")))
            )

    gate_tracks: dict[str, Any] = {}
    gate_pass = True
    for track_id, records in per_track.items():
        records.sort()
        total = len(records)
        face_coverage = sum(record[1] for record in records) / total
        mouth_coverage = sum(record[2] for record in records) / total
        longest_face_gap = 0
        current_gap = 0
        for _, has_face, _ in records:
            current_gap = 0 if has_face else current_gap + 1
            longest_face_gap = max(longest_face_gap, current_gap)
        passed = (
            face_coverage >= minimum_coverage
            and mouth_coverage >= minimum_coverage
            and longest_face_gap <= maximum_gap
        )
        gate_tracks[str(track_id)] = {
            "rows": total,
            "face_coverage": round(face_coverage, 4),
            "mouth_coverage": round(mouth_coverage, 4),
            "longest_face_gap": longest_face_gap,
            "passed": passed,
        }
        gate_pass = gate_pass and passed

    return {
        "path": str(path),
        "format_version": payload.get("format_version"),
        "frame_count": len(frames),
        "frame_track_count": len(seen_keys),
        "track_counts": track_counts,
        "face_valid_tracks": face_valid,
        "mouth_valid_tracks": mouth_valid,
        "predicted_face_tracks": predicted,
        "quality_gate": {
            "minimum_coverage": minimum_coverage,
            "maximum_gap": maximum_gap,
            "passed": gate_pass and not errors,
            "tracks": gate_tracks,
        },
        "errors": errors,
        "valid": not errors and gate_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--minimum-coverage", type=float, default=0.90)
    parser.add_argument("--maximum-gap", type=int, default=3)
    args = parser.parse_args()
    report = validate_landmarks(
        args.path,
        minimum_coverage=args.minimum_coverage,
        maximum_gap=args.maximum_gap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()

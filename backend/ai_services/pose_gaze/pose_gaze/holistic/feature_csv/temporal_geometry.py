"""Phase 3 temporal and pair-geometry primitives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


TEMPORAL_POLICY = {
    "target_fps": 8,
    "baseline_valid_frames": 4,
    "head_turn_baseline_valid_frames": 8,
    "window_frames": 24,
    "max_derivative_gap_ms": 450,
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _landmark_root_hash(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return {
        "algorithm": "sha256(sorted relative path, content length, content)",
        "file_count": len(files),
        "sha256": digest.hexdigest().upper(),
    }


def _write_feature_manifest(
    *,
    output_dir: Path,
    landmark_root: Path,
    source_manifest: Path,
    actor_mapping: Path,
    fields: list[str],
    rows: list[dict[str, Any]],
) -> None:
    resolved_landmark_root = landmark_root.resolve()
    if resolved_landmark_root.name != "holistic_output_8fps":
        raise ValueError(
            f"Stage 5 requires holistic_output_8fps, got {resolved_landmark_root}"
        )
    valid_fields = [field for field in fields if field.endswith("_valid")]
    valid_counts = Counter()
    coverage_histogram = Counter()
    row_counts = Counter()
    split_assignments: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (str(row["clip_id"]), str(row["actor_id"]))
        row_counts[key] += 1
        split_assignments.add((
            str(row["clip_id"]), str(row["actor_id"]),
            str(row["split"]), str(row["split_group"]),
        ))
        current_valid = 0
        for field in valid_fields:
            try:
                valid = int(float(row.get(field, 0))) == 1
            except (TypeError, ValueError):
                valid = False
            if valid:
                valid_counts[field] += 1
                current_valid += 1
        coverage = current_valid / len(valid_fields) if valid_fields else 0.0
        bucket = min(10, int(coverage * 10))
        coverage_histogram[f"{bucket / 10:.1f}"] += 1
    row_total = len(rows)
    manifest = {
        "format_version": 1,
        "input": {
            "landmark_root": str(resolved_landmark_root),
            "landmark_root_hash": _landmark_root_hash(resolved_landmark_root),
            "source_manifest": str(source_manifest.resolve()),
            "actor_mapping": str(actor_mapping.resolve()),
            "actor_mapping_sha256": hashlib.sha256(actor_mapping.read_bytes()).hexdigest().upper(),
        },
        "feature_schema": {
            "ordered_feature_names": fields,
            "ordered_feature_name_hash": _canonical_hash(fields),
            "feature_count": len(fields),
        },
        "temporal_policy": TEMPORAL_POLICY,
        "row_counts": {
            "total": row_total,
            "by_clip_actor": [
                {"clip_id": clip_id, "actor_id": actor_id, "rows": count}
                for (clip_id, actor_id), count in sorted(row_counts.items())
            ],
        },
        "validity_coverage_distribution": {
            "valid_fields": {
                field: {
                    "valid_count": valid_counts[field],
                    "invalid_count": row_total - valid_counts[field],
                    "coverage": valid_counts[field] / row_total if row_total else 0.0,
                }
                for field in valid_fields
            },
            "row_coverage_histogram_0_1_step_0_1": dict(sorted(coverage_histogram.items())),
        },
        "split_assignment": {
            "assignments": [
                {
                    "clip_id": clip_id, "actor_id": actor_id,
                    "split": split, "split_group": split_group,
                }
                for clip_id, actor_id, split, split_group in sorted(split_assignments)
            ],
            "sha256": _canonical_hash(sorted(split_assignments)),
        },
        "completion_checks": {
            "landmark_root_is_8fps": True,
            "contains_30fps_input_path": False,
            "all_rows_have_sample_index": all("sample_index" in row for row in rows),
        },
    }
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _point(row: dict[str, str], group: str, index: int) -> tuple[float, float] | None:
    if row.get(f"{group}_{index}_valid") != "1":
        return None
    try:
        return float(row[f"{group}_{index}_frame_x"]), float(row[f"{group}_{index}_frame_y"])
    except (KeyError, TypeError, ValueError):
        return None


def _bbox_center(row: dict[str, str]) -> float | None:
    try:
        return (float(row["bbox_x1"]) + float(row["bbox_x2"])) / 2.0
    except (KeyError, TypeError, ValueError):
        return None


def _dist(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    return math.hypot(a[0] - b[0], a[1] - b[1]) if a and b else None


def _median(values: list[float]) -> float | None:
    return median(values) if values else None


def _sample_index(row: dict[str, str]) -> int:
    if "sample_index" not in row:
        raise ValueError("Stage 3 requires sample_index")
    return int(row["sample_index"])


def _selected_point(row: dict[str, str], midpoint: float | None) -> tuple[tuple[float, float] | None, str]:
    """Select inward hand evidence by fingertip, wrist, then elbow priority."""
    # A fingertip is the strongest exchange cue. Do not let a palm landmark
    # win merely because it happens to be closer to the midpoint.
    tiers = (
        (("left_hand", (4, 8, 12, 16, 20)), ("right_hand", (4, 8, 12, 16, 20))),
        (("left_hand", (0,)), ("right_hand", (0,))),
        (("pose", (16, 15)),),
    )
    for tier in tiers:
        points = []
        for group, indices in tier:
            points.extend((group, index, _point(row, group, index)) for index in indices)
        points = [(group, index, point) for group, index, point in points if point is not None]
        if points:
            selected = min(points, key=lambda item: abs(item[2][0] - midpoint)) if midpoint is not None else points[0]
            return selected[2], f"{selected[0]}_{selected[1]}"
    for group, indices in (("pose", (14, 13)), ("pose", (12, 11))):
        points = [(index, _point(row, group, index)) for index in indices]
        points = [(index, point) for index, point in points if point is not None]
        if points:
            point = min(points, key=lambda item: abs(item[1][0] - midpoint)) if midpoint is not None else points[0]
            return point[1], f"{group}_{point[0]}"
    return None, ""


def _head_depth(row: dict[str, str], baseline: dict[str, float]) -> dict[str, Any]:
    points = {index: _point(row, "pose", index) for index in (0, 1, 4, 9, 10, 11, 12)}
    if any(points[index] is None for index in (0, 11, 12)):
        return {"head_down_valid": 0, "delta_nose_down": "", "delta_mouth_down": "", "delta_eye_down": "", "head_down_candidate": 0}
    left, right = points[11], points[12]
    scale = _dist(left, right)
    if not scale:
        return {"head_down_valid": 0, "delta_nose_down": "", "delta_mouth_down": "", "delta_eye_down": "", "head_down_candidate": 0}
    def depth(point: tuple[float, float] | None) -> float | None:
        if point is None or abs(right[0] - left[0]) < 1e-9:
            return None
        shoulder_y = left[1] + (point[0] - left[0]) * (right[1] - left[1]) / (right[0] - left[0])
        return (point[1] - shoulder_y) / scale
    nose, mouth, eye = depth(points[0]), depth(_mid(points[9], points[10])), depth(_mid(points[1], points[4]))
    if nose is None:
        return {"head_down_valid": 0, "delta_nose_down": "", "delta_mouth_down": "", "delta_eye_down": "", "head_down_candidate": 0}
    dn = nose - baseline.get("nose", nose)
    dm = mouth - baseline["mouth"] if mouth is not None and "mouth" in baseline else None
    de = eye - baseline["eye"] if eye is not None and "eye" in baseline else None
    support = [value for value in (dm, de) if value is not None]
    candidate = int(dn >= baseline.get("nose_threshold", float("inf")) and support and median(support) >= baseline.get("support_threshold", float("inf")))
    return {"head_down_valid": 1, "delta_nose_down": dn, "delta_mouth_down": dm if dm is not None else "", "delta_eye_down": de if de is not None else "", "head_down_candidate": candidate}


def _mid(a: tuple[float, float] | None, b: tuple[float, float] | None) -> tuple[float, float] | None:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0) if a and b else None


def enrich(rows: list[dict[str, str]], baseline_frames: int = 4) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: (_sample_index(row), row["actor_id"]))
    actors = sorted({row["actor_id"] for row in rows}, key=lambda value: int("".join(ch for ch in value if ch.isdigit()) or 0))
    by_frame: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_frame[_sample_index(row)][row["actor_id"]] = row
    paired = None
    midpoint = None
    actor_side: dict[str, int] = {}
    margin = None
    geometry_by_frame: dict[int, tuple[int | None, float | None, dict[str, int], float | None]] = {}
    for frame in sorted(by_frame):
        if paired is None and len(actors) >= 2:
            centers = {actor: _bbox_center(by_frame[frame].get(actor, {})) for actor in actors[:2]}
            if all(value is not None for value in centers.values()):
                paired = frame
                ordered = sorted(centers, key=centers.get)  # type: ignore[arg-type]
                midpoint = (centers[ordered[0]] + centers[ordered[1]]) / 2.0  # type: ignore[operator]
                actor_side = {ordered[0]: -1, ordered[1]: 1}
                margin = 0.10 * abs(centers[ordered[1]] - centers[ordered[0]])  # type: ignore[operator]
        geometry_by_frame[frame] = (paired, midpoint, dict(actor_side), margin)

    baseline_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"nose": [], "mouth": [], "eye": []}
    )
    for row in rows:
        frame = _sample_index(row)
        paired, midpoint, actor_side, margin = geometry_by_frame[frame]
        actor = row["actor_id"]
        point, source = _selected_point(row, midpoint)
        own_distance = actor_side.get(actor, 0) * (point[0] - midpoint) if point is not None and midpoint is not None else ""
        near = int(own_distance != "" and 0 <= own_distance <= margin) if margin is not None else 0
        peer = next((value for value in actors if value != actor), None)
        peer_row = by_frame[frame].get(peer or "")
        pair_distance = _dist(point, _selected_point(peer_row, midpoint)[0]) if peer_row else None
        row.update({
            "baseline_source": f"first_{baseline_frames}_valid_strict_past_observations",
            "pair_mid_x_0": midpoint if midpoint is not None else "", "pair_margin_10pct": margin if margin is not None else "",
            "paired_valid_source_frame": paired if paired is not None else "", "actor_side": actor_side.get(actor, 0),
            "selected_exchange_point": source, "selected_exchange_x": point[0] if point else "", "selected_exchange_y": point[1] if point else "",
            "own_side_distance": own_distance, "near_midpoint_pre_cross": near,
            "pair_hand_distance": pair_distance if pair_distance is not None else "",
        })
        baselines = {
            key: median(values) for key, values in baseline_values[actor].items() if values
        }
        row.update(_head_depth(row, baselines))
        points = {index: _point(row, "pose", index) for index in (0, 1, 4, 9, 10, 11, 12)}
        left, right = points[11], points[12]
        scale = _dist(left, right)
        if left and right and scale and abs(right[0] - left[0]) >= 1e-9:
            def depth(point: tuple[float, float] | None) -> float | None:
                if point is None:
                    return None
                line = left[1] + (point[0] - left[0]) * (right[1] - left[1]) / (right[0] - left[0])
                return (point[1] - line) / scale
            for key, point_value in (("nose", points[0]), ("mouth", _mid(points[9], points[10])), ("eye", _mid(points[1], points[4]))):
                value = depth(point_value)
                if value is not None and len(baseline_values[actor][key]) < baseline_frames:
                    baseline_values[actor][key].append(value)
    return rows


def run(
    input_path: Path,
    output_dir: Path,
    baseline_frames: int,
    *,
    landmark_root: Path | None = None,
    source_manifest: Path | None = None,
    actor_mapping: Path | None = None,
) -> dict[str, int]:
    if baseline_frames != TEMPORAL_POLICY["baseline_valid_frames"]:
        raise ValueError(
            "Stage 5 baseline_frames must match temporal policy: "
            f"{TEMPORAL_POLICY['baseline_valid_frames']}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        base_fields = list(reader.fieldnames or [])
        if "sample_index" not in base_fields:
            raise ValueError("Stage 3 input requires sample_index")
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader: groups[row["clip_id"]].append(row)
    added = ["baseline_source", "pair_mid_x_0", "pair_margin_10pct", "paired_valid_source_frame", "actor_side", "selected_exchange_point", "selected_exchange_x", "selected_exchange_y", "own_side_distance", "near_midpoint_pre_cross", "pair_hand_distance", "head_down_valid", "delta_nose_down", "delta_mouth_down", "delta_eye_down", "head_down_candidate"]
    out = output_dir / "temporal_geometry_features.csv"
    output_rows: list[dict[str, Any]] = []
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields + added)
        writer.writeheader()
        count = 0
        for rows in groups.values():
            for row in enrich(rows, baseline_frames):
                writer.writerow(row)
                output_rows.append(row)
                count += 1
    if any(value is not None for value in (landmark_root, source_manifest, actor_mapping)):
        if not all(value is not None for value in (landmark_root, source_manifest, actor_mapping)):
            raise ValueError(
                "feature manifest requires --landmark-root, --source-manifest, and --actor-mapping"
            )
        _write_feature_manifest(
            output_dir=output_dir,
            landmark_root=landmark_root,
            source_manifest=source_manifest,
            actor_mapping=actor_mapping,
            fields=base_fields + added,
            rows=output_rows,
        )
    summary = {
        "videos": len(groups),
        "rows": count,
        "added_fields": len(added),
        "baseline_frames": baseline_frames,
        "baseline_type": "first_4_valid_strict_past_observations",
    }
    (output_dir / "phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-frames", type=int, default=4)
    parser.add_argument("--landmark-root", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--actor-mapping", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(
        args.input,
        args.output_dir,
        args.baseline_frames,
        landmark_root=args.landmark_root,
        source_manifest=args.source_manifest,
        actor_mapping=args.actor_mapping,
    ), indent=2))


if __name__ == "__main__": main()

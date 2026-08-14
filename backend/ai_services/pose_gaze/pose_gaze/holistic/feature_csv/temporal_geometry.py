"""Phase 3 temporal and pair-geometry primitives."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


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


def enrich(rows: list[dict[str, str]], baseline_frames: int) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: (int(row["source_frame_index"]), row["actor_id"]))
    actors = sorted({row["actor_id"] for row in rows}, key=lambda value: int("".join(ch for ch in value if ch.isdigit()) or 0))
    by_frame: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_frame[int(row["source_frame_index"])][row["actor_id"]] = row
    paired = (
        next(
            (
                frame for frame in sorted(by_frame)
                if all(
                    _bbox_center(by_frame[frame].get(actor, {})) is not None
                    for actor in actors[:2]
                )
            ),
            None,
        )
        if len(actors) >= 2
        else None
    )
    midpoint = None
    actor_side: dict[str, int] = {}
    if paired is not None:
        centers = {actor: _bbox_center(by_frame[paired][actor]) for actor in actors[:2]}
        ordered = sorted(centers, key=centers.get)  # type: ignore[arg-type]
        midpoint = (centers[ordered[0]] + centers[ordered[1]]) / 2.0  # type: ignore[operator]
        actor_side = {ordered[0]: -1, ordered[1]: 1}
    margin = None
    if paired is not None:
        centers = [_bbox_center(by_frame[paired][actor]) for actor in actors[:2]]
        margin = 0.10 * abs(centers[1] - centers[0])  # type: ignore[operator]

    normal_rows = [row for row in rows if int(row["source_frame_index"]) < baseline_frames]
    baselines: dict[str, dict[str, float]] = {}
    for actor in actors:
        actor_rows = [row for row in normal_rows if row["actor_id"] == actor]
        values: dict[str, list[float]] = {"nose": [], "mouth": [], "eye": []}
        for row in actor_rows:
            points = {index: _point(row, "pose", index) for index in (0, 1, 4, 9, 10, 11, 12)}
            left, right = points[11], points[12]
            scale = _dist(left, right)
            if not left or not right or not scale or abs(right[0] - left[0]) < 1e-9:
                continue
            def depth(point: tuple[float, float] | None) -> float | None:
                if not point: return None
                line = left[1] + (point[0] - left[0]) * (right[1] - left[1]) / (right[0] - left[0])
                return (point[1] - line) / scale
            for key, point in (("nose", points[0]), ("mouth", _mid(points[9], points[10])), ("eye", _mid(points[1], points[4]))):
                value = depth(point)
                if value is not None: values[key].append(value)
        baselines[actor] = {key: median(value) for key, value in values.items() if value}
    for row in rows:
        actor = row["actor_id"]
        point, source = _selected_point(row, midpoint)
        own_distance = actor_side.get(actor, 0) * (point[0] - midpoint) if point is not None and midpoint is not None else ""
        near = int(own_distance != "" and 0 <= own_distance <= margin) if margin is not None else 0
        peer = next((value for value in actors if value != actor), None)
        peer_row = by_frame[int(row["source_frame_index"])].get(peer or "")
        pair_distance = _dist(point, _selected_point(peer_row, midpoint)[0]) if peer_row else None
        row.update({
            "baseline_source": f"initial_{baseline_frames}_frames_provisional",
            "pair_mid_x_0": midpoint if midpoint is not None else "", "pair_margin_10pct": margin if margin is not None else "",
            "paired_valid_source_frame": paired if paired is not None else "", "actor_side": actor_side.get(actor, 0),
            "selected_exchange_point": source, "selected_exchange_x": point[0] if point else "", "selected_exchange_y": point[1] if point else "",
            "own_side_distance": own_distance, "near_midpoint_pre_cross": near,
            "pair_hand_distance": pair_distance if pair_distance is not None else "",
        })
        row.update(_head_depth(row, baselines.get(actor, {})))
    return rows


def run(input_path: Path, output_dir: Path, baseline_frames: int) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        base_fields = list(reader.fieldnames or [])
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader: groups[row["clip_id"]].append(row)
    added = ["baseline_source", "pair_mid_x_0", "pair_margin_10pct", "paired_valid_source_frame", "actor_side", "selected_exchange_point", "selected_exchange_x", "selected_exchange_y", "own_side_distance", "near_midpoint_pre_cross", "pair_hand_distance", "head_down_valid", "delta_nose_down", "delta_mouth_down", "delta_eye_down", "head_down_candidate"]
    out = output_dir / "temporal_geometry_features.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields + added)
        writer.writeheader()
        count = 0
        for rows in groups.values():
            for row in enrich(rows, baseline_frames): writer.writerow(row); count += 1
    summary = {"videos": len(groups), "rows": count, "added_fields": len(added), "baseline_frames": baseline_frames, "baseline_type": "provisional_initial_prefix"}
    (output_dir / "phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-frames", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir, args.baseline_frames), indent=2))


if __name__ == "__main__": main()

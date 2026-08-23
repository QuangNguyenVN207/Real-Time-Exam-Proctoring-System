"""Schema and validation helpers for human gaze/temporal annotations."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SUBJECT_EVENTS = (
    "gaze_partner", "gaze_partner_paper", "gaze_self_paper",
    "mouth_active", "hand_retract", "hand_raise_signal",
    "torso_head_lean",
)
PAIR_EVENTS = ("partner_hand_exchange", "mutual_look")
EVENT_TYPES = SUBJECT_EVENTS + PAIR_EVENTS
STATUSES = ("unannotated", "draft", "confirmed", "needs_review")

ANNOTATION_COLUMNS = (
    "source_filename", "clip_id", "split", "split_group", "camera_view_id",
    "video_class_code", "duration_s", "actual_fps", "action_start_s",
    "action_end_s", "actor_id", "track_id", "target_actor_id",
    "event_scope", "event_type", "interval_index", "start_frame",
    "end_frame", "start_time_ms", "end_time_ms", "status", "annotator",
    "confidence", "notes",
)


def _json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed if item not in (None, "")]


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def manifest_index(manifest_path: Path) -> dict[str, dict[str, str]]:
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row.get("filename", "")): row for row in rows if row.get("filename")}


def validate_annotation_rows(
    annotation_path: Path,
    manifest_path: Path,
    *,
    allowed_splits: set[str] | None = None,
) -> list[str]:
    manifest = manifest_index(manifest_path)
    errors: list[str] = []
    intervals: dict[tuple[str, str, str, str, str], list[tuple[float, float, int]]] = defaultdict(list)
    with annotation_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = set(ANNOTATION_COLUMNS).difference(reader.fieldnames or ())
        if missing_columns:
            return [f"missing columns: {sorted(missing_columns)}"]
        for line_number, row in enumerate(reader, start=2):
            filename = row.get("source_filename", "")
            source = manifest.get(filename)
            if source is None:
                errors.append(f"line {line_number}: unknown source_filename {filename!r}")
                continue
            if row.get("camera_view_id") != "front" or source.get("camera_view_id") != "front":
                errors.append(f"line {line_number}: rear/non-front annotation is forbidden")
            actors = set(_json_list(source.get("action_actor_ids") or source.get("actor_ids") or "[]"))
            if row.get("actor_id") not in actors:
                errors.append(f"line {line_number}: actor_id {row.get('actor_id')!r} is not in manifest actors")
            target = row.get("target_actor_id", "")
            if target and target not in actors:
                errors.append(f"line {line_number}: target_actor_id {target!r} is not in manifest actors")
            if allowed_splits and row.get("split") not in allowed_splits:
                errors.append(f"line {line_number}: split {row.get('split')!r} is outside allowed splits")
            if row.get("event_type") not in EVENT_TYPES:
                errors.append(f"line {line_number}: invalid event_type {row.get('event_type')!r}")
            if row.get("status") not in STATUSES:
                errors.append(f"line {line_number}: invalid status {row.get('status')!r}")
            if row.get("event_type") in PAIR_EVENTS:
                if row.get("event_scope") != "pair":
                    errors.append(f"line {line_number}: pair event must have event_scope=pair")
                if not row.get("target_actor_id") or row.get("target_actor_id") == row.get("actor_id"):
                    errors.append(f"line {line_number}: pair event needs a different target_actor_id")
            elif row.get("event_scope") != "subject":
                errors.append(f"line {line_number}: subject event must have event_scope=subject")
            start = _number(row.get("start_time_ms"))
            end = _number(row.get("end_time_ms"))
            duration = (_number(row.get("duration_s")) or 0.0) * 1000.0
            if (start is None) != (end is None):
                errors.append(f"line {line_number}: start/end time must both be set or both blank")
            if start is not None and end is not None:
                if start < 0 or end < start or (duration and end > duration + 1000):
                    errors.append(f"line {line_number}: invalid interval [{start}, {end}] ms")
                key = (filename, row.get("actor_id", ""), row.get("target_actor_id", ""), row.get("event_scope", ""), row.get("event_type", ""))
                intervals[key].append((start, end, line_number))
            if row.get("status") == "confirmed" and not row.get("annotator"):
                errors.append(f"line {line_number}: confirmed annotation needs annotator")
            confidence = _number(row.get("confidence"))
            if confidence is not None and not 0 <= confidence <= 1:
                errors.append(f"line {line_number}: confidence must be within [0, 1]")
    for key, values in intervals.items():
        values.sort()
        for previous, current in zip(values, values[1:]):
            if current[0] < previous[1]:
                errors.append(f"overlap for {key}: lines {previous[2]} and {current[2]}")
    return errors


__all__ = ["ANNOTATION_COLUMNS", "EVENT_TYPES", "PAIR_EVENTS", "STATUSES", "validate_annotation_rows"]

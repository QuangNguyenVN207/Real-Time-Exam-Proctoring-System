"""Build leakage-safe fixed-width temporal windows from frame-track rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TemporalWindowConfig:
    """Deterministic window policy shared by training and realtime inference."""

    size: int = 16
    stride: int = 4
    max_timestamp_gap_ms: int = 250

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("window size must be at least 2")
        if self.stride < 1:
            raise ValueError("window stride must be positive")
        if self.max_timestamp_gap_ms < 1:
            raise ValueError("max timestamp gap must be positive")


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _group_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    split_group = str(row.get("split_group") or row.get("sequence_id") or "")
    source = str(row.get("source_filename") or row.get("source_path") or "")
    track_id = str(row.get("track_id") or "")
    return (str(row.get("split") or ""), split_group, source, track_id)


def _ordered_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(_number(row.get("frame_id")) or 0),
            int(_number(row.get("timestamp_ms")) or 0),
        ),
    )


def _is_contiguous(
    rows: list[Mapping[str, Any]],
    *,
    max_timestamp_gap_ms: int,
) -> bool:
    for previous, current in zip(rows, rows[1:]):
        previous_frame = _number(previous.get("frame_id"))
        current_frame = _number(current.get("frame_id"))
        previous_timestamp = _number(previous.get("timestamp_ms"))
        current_timestamp = _number(current.get("timestamp_ms"))
        if previous_frame is not None and current_frame is not None:
            if current_frame != previous_frame + 1:
                return False
        if previous_timestamp is not None and current_timestamp is not None:
            if current_timestamp < previous_timestamp:
                return False
            if current_timestamp - previous_timestamp > max_timestamp_gap_ms:
                return False
    return True


def build_temporal_windows(
    rows: Iterable[Mapping[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    config: TemporalWindowConfig | None = None,
) -> list[dict[str, Any]]:
    """Flatten contiguous, non-predicted frame rows into XGBoost windows.

    Rows are grouped by split, sequence/video, and track. Predicted face rows
    are rejected before grouping, so held landmarks never enter windows.
    Missing numeric feature values remain finite zero values with a matching
    validity ratio feature supplied by the frame schema.
    """

    policy = config or TemporalWindowConfig()
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("status") or "ok").lower() != "ok":
            continue
        if _truthy(row.get("face_predicted")):
            continue
        groups[_group_key(row)].append(row)

    windows: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        ordered = _ordered_rows(group_rows)
        for start in range(0, len(ordered) - policy.size + 1, policy.stride):
            window_rows = ordered[start : start + policy.size]
            if not _is_contiguous(
                window_rows,
                max_timestamp_gap_ms=policy.max_timestamp_gap_ms,
            ):
                continue
            labels = {str(row.get("label") or row.get("class_code") or "") for row in window_rows}
            target_states = {
                str(row.get("target_state") or ("normal" if str(row.get("class_code")) == "c5" else "event"))
                for row in window_rows
            }
            class_codes = {str(row.get("class_code") or "") for row in window_rows}
            if len(labels) != 1 or not next(iter(labels)):
                continue
            if len(target_states) != 1 or not next(iter(target_states)):
                continue
            if len(class_codes) != 1 or not next(iter(class_codes)):
                continue
            first = window_rows[0]
            last = window_rows[-1]
            output: dict[str, Any] = {
                "split": key[0],
                "split_group": key[1],
                "camera_view_id": str(first.get("camera_view_id") or ""),
                "source_filename": key[2],
                "track_id": key[3],
                "video_class_code": str(first.get("video_class_code") or first.get("class_code") or ""),
                "actor_id": str(first.get("actor_id") or ""),
                "action_actor_id": str(first.get("action_actor_id") or ""),
                "peer_actor_id": str(first.get("peer_actor_id") or ""),
                "actor_role": str(first.get("actor_role") or ""),
                "track_side": str(first.get("track_side") or ""),
                "is_action_actor": str(first.get("is_action_actor") or "0"),
                "window_start_frame": first.get("frame_id"),
                "window_end_frame": last.get("frame_id"),
                "window_start_timestamp_ms": first.get("timestamp_ms"),
                "window_end_timestamp_ms": last.get("timestamp_ms"),
                "window_size": policy.size,
                "class_code": next(iter(class_codes)),
                "label": next(iter(labels)),
                "target_state": next(iter(target_states)),
                "protocol": str(first.get("protocol") or ""),
                "interval_used_for_features": str(first.get("interval_used_for_features") or "False"),
                "label_purity": 1.0,
            }
            for offset, row in enumerate(window_rows):
                for column in feature_columns:
                    value = _number(row.get(column))
                    output[f"t{offset:02d}_{column}"] = 0.0 if value is None else value
            windows.append(output)
    return windows


__all__ = ["TemporalWindowConfig", "build_temporal_windows"]

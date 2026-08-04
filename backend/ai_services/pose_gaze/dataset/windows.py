from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import processed_root
from .split import assign_split


def iter_window_bounds(frame_count: int, window_size: int = 16, overlap: int = 8) -> list[tuple[int, int]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be in [0, window_size)")

    step = window_size - overlap
    bounds: list[tuple[int, int]] = []
    start = 0
    while start + window_size <= frame_count:
        bounds.append((start, start + window_size))
        start += step
    return bounds


def normalize_gaze_direction(*, quality_ok: bool, gaze_direction: str | None) -> str:
    if not quality_ok:
        return "unknown"
    if not gaze_direction:
        return "unknown"
    return gaze_direction


def build_caption(row: dict[str, object]) -> str:
    class_code = str(row.get("class_code", "c?"))
    student_id = str(row.get("student_id", "student"))
    peer_student_id = str(row.get("peer_student_id", "peer"))
    gaze_direction = str(row.get("gaze_direction", "unknown"))
    action = str(row.get("action", class_code))
    return f"{student_id} in {class_code} {action} with gaze {gaze_direction} toward {peer_student_id}"


def _build_scaler(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    numeric_columns = [
        column
        for column in ["yaw", "pitch", "roll", "delta_yaw", "delta_pitch", "delta_roll", "quality_score"]
        if column in frame.columns
    ]
    scaler: dict[str, dict[str, float]] = {}
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        scaler[column] = {"mean": float(values.mean()), "std": float(values.std(ddof=0) or 1.0)}
    return scaler


def build_windows(*, frames_path: Path | None = None, output_path: Path | None = None, scaler_path: Path | None = None, vocab_path: Path | None = None) -> pd.DataFrame:
    frames_path = frames_path or (processed_root() / "frames.parquet")
    output_path = output_path or (processed_root() / "windows.parquet")
    scaler_path = scaler_path or (processed_root() / "scaler.json")
    vocab_path = vocab_path or (processed_root() / "vocab.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = pd.read_parquet(frames_path)
    rows: list[dict[str, object]] = []

    for (video_stem, track_id), group in frames.sort_values(["timestamp_ms", "frame_id"]).groupby(["video_stem", "track_id"]):
        group = group.reset_index(drop=True)
        bounds = iter_window_bounds(len(group), 16, 8)
        split_name = assign_split(tuple(str(value).strip() for value in str(group.loc[0, "subject_ids"]).split(",") if value))
        if split_name is None:
            continue
        for window_index, (start, end) in enumerate(bounds):
            window = group.iloc[start:end].copy()
            quality_ratio = float(pd.to_numeric(window["quality_ok"], errors="coerce").fillna(False).mean())
            if quality_ratio < 0.60:
                continue
            first_row = window.iloc[0].to_dict()
            row = {
                "video_stem": video_stem,
                "track_id": int(track_id),
                "split": split_name,
                "window_index": window_index,
                "window_start_frame": int(first_row["frame_id"]),
                "window_end_frame": int(window.iloc[-1]["frame_id"]),
                "window_start_ms": int(first_row["timestamp_ms"]),
                "window_end_ms": int(window.iloc[-1]["timestamp_ms"]),
                "quality_ratio": quality_ratio,
                "quality_ok": quality_ratio >= 0.60,
                "class_code": str(first_row.get("class_code", "c?")),
                "student_id": str(first_row.get("student_id", "")),
                "peer_student_id": str(first_row.get("peer_student_id", "")),
                "action": str(first_row.get("action", first_row.get("class_code", "c?"))),
                "gaze_direction": normalize_gaze_direction(
                    quality_ok=quality_ratio >= 0.60,
                    gaze_direction=str(window.iloc[-1].get("gaze_direction", "unknown")),
                ),
                "yaw": float(pd.to_numeric(window["yaw"], errors="coerce").mean()),
                "pitch": float(pd.to_numeric(window["pitch"], errors="coerce").mean()),
                "roll": float(pd.to_numeric(window["roll"], errors="coerce").mean()),
                "delta_yaw": float(pd.to_numeric(window["delta_yaw"], errors="coerce").mean()),
                "delta_pitch": float(pd.to_numeric(window["delta_pitch"], errors="coerce").mean()),
                "delta_roll": float(pd.to_numeric(window["delta_roll"], errors="coerce").mean()),
            }
            row["caption"] = build_caption(row)
            rows.append(row)

    columns = [
        "video_stem",
        "track_id",
        "split",
        "window_index",
        "window_start_frame",
        "window_end_frame",
        "window_start_ms",
        "window_end_ms",
        "quality_ratio",
        "quality_ok",
        "class_code",
        "student_id",
        "peer_student_id",
        "action",
        "gaze_direction",
        "yaw",
        "pitch",
        "roll",
        "delta_yaw",
        "delta_pitch",
        "delta_roll",
        "caption",
    ]
    windows = pd.DataFrame(rows, columns=columns)
    windows.to_parquet(output_path, index=False)

    if windows.empty:
        scaler = {}
        vocab = []
    else:
        scaler = _build_scaler(windows.loc[windows["split"] == "train"])
        vocab = sorted({token.lower() for caption in windows.get("caption", pd.Series(dtype=str)).dropna().astype(str) for token in caption.split()})

    scaler_path.write_text(json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")
    vocab_path.write_text(json.dumps({"vocab": vocab}, ensure_ascii=False, indent=2), encoding="utf-8")

    return windows

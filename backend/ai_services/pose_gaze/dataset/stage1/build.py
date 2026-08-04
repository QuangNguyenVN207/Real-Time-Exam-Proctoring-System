from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import (
    ANALYSIS_MAX_HEIGHT,
    ANALYSIS_MAX_WIDTH,
    QUALITY_WEIGHT_BRIGHTNESS,
    QUALITY_WEIGHT_DIVERSITY,
    QUALITY_WEIGHT_MOTION,
    QUALITY_WEIGHT_PERSON,
    QUALITY_WEIGHT_SHARPNESS,
    NON_CHEATING_CLASS_CODES,
    SELECTED_FRAMES_PER_WINDOW,
    TARGET_FPS,
    WINDOW_FRAMES,
    WINDOW_OVERLAP_FRAMES,
    WINDOW_SECONDS,
    stage1_root,
)
from ..manifest import parse_video_filename
from ..split import assign_split


@dataclass(slots=True)
class ClipSpec:
    filename: str
    class_code: str
    action_start_s: float | None
    action_end_s: float | None
    reviewed: bool
    exclude_from_training: bool
    raw_row: dict[str, object]


def _normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _parse_seconds(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"bỏ", "bo", "none", "nan"}:
        return None
    if text.startswith("0p") and text.endswith("s"):
        try:
            return float(text[2:-1].replace("s", ""))
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def _collect_clip_specs(manifest: pd.DataFrame) -> list[ClipSpec]:
    specs: list[ClipSpec] = []
    for _, row in manifest.iterrows():
        filename = str(row.get("filename", "")).strip()
        if not filename:
            continue
        class_code = str(row.get("class_code", "")).strip().lower()
        reviewed = _normalize_bool(row.get("reviewed", False))
        exclude_from_training = _normalize_bool(row.get("exclude_from_training", False)) or class_code == "bỏ"
        if not reviewed and not exclude_from_training:
            raise ValueError(f"Unreviewed clip cannot enter Stage 1: {filename}")
        start_s = _parse_seconds(row.get("action_start_s"))
        end_s = _parse_seconds(row.get("action_end_s"))
        raw_row = row.to_dict()
        # Manifest Stage 0 chính thức dùng actor_ids/recording_session/split_group.
        # Chuẩn hóa alias nội bộ để giữ tương thích với các manifest legacy.
        if not raw_row.get("subject_ids") and raw_row.get("actor_ids"):
            raw_row["subject_ids"] = raw_row["actor_ids"]
        if not raw_row.get("session_id") and raw_row.get("recording_session"):
            raw_row["session_id"] = raw_row["recording_session"]
        if not raw_row.get("group_id") and raw_row.get("split_group"):
            raw_row["group_id"] = raw_row["split_group"]
        # Stage 0 manifests may describe generated clip names and omit explicit
        # subject/session columns.  The canonical v_c*_s* filename still carries
        # stable subject IDs, so reuse the repository parser instead of assigning
        # every clip to an unassigned split.
        if not any(raw_row.get(key) for key in ("subject_id", "student_id", "subject", "session_id", "session", "group_id", "group")):
            try:
                metadata = parse_video_filename(filename)
            except ValueError:
                metadata = None
            if metadata is not None:
                raw_row["subject_ids"] = ",".join(metadata.subject_ids)
                raw_row["subject_id"] = ",".join(metadata.subject_ids)
                raw_row["split_source"] = "filename_subject_ids"
        specs.append(
            ClipSpec(
                filename=filename,
                class_code=class_code,
                action_start_s=start_s,
                action_end_s=end_s,
                reviewed=reviewed,
                exclude_from_training=exclude_from_training,
                raw_row=raw_row,
            )
        )
    return specs


def _resolve_video_path(raw_root: Path, filename: str) -> Path:
    direct = raw_root / filename
    if direct.exists():
        return direct
    candidates = list(raw_root.rglob(filename))
    if candidates:
        return candidates[0]
    stem = Path(filename).stem
    for candidate in raw_root.rglob("*"):
        if candidate.is_file() and candidate.stem == stem:
            return candidate
    raise FileNotFoundError(f"Could not resolve video path for {filename}")


def _probe_video(path: Path) -> dict[str, float | int | bool]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    duration_s = (frame_count / fps) if frame_count and fps else 0.0
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": duration_s,
        "is_landscape": width >= height,
    }


def _video_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_scale(width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 1.0
    return min(1.0, ANALYSIS_MAX_WIDTH / width, ANALYSIS_MAX_HEIGHT / height)


def _normalize_analysis_frame(frame):
    source_height, source_width = frame.shape[:2]
    scale = _analysis_scale(source_width, source_height)
    if scale >= 1.0:
        return frame, source_width, source_height, source_width, source_height, 1.0
    analysis_width = max(1, int(round(source_width * scale)))
    analysis_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
    return resized, source_width, source_height, analysis_width, analysis_height, float(scale)


def _score_sharpness(gray) -> tuple[float, bool]:
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    score = min(1.0, variance / 300.0)
    return score, variance < 60.0


def _score_brightness(gray) -> tuple[float, bool, bool]:
    mean = float(gray.mean())
    dark = mean < 45.0
    overexposed = mean > 220.0
    score = 1.0 - min(1.0, abs(mean - 127.5) / 127.5)
    return max(0.0, score), dark, overexposed


def _score_motion(gray, previous_gray) -> float:
    if previous_gray is None or previous_gray.shape != gray.shape:
        return 0.0
    diff = cv2.absdiff(gray, previous_gray)
    return min(1.0, float(diff.mean()) / 40.0)


def _score_diversity(gray, previous_gray) -> float:
    if previous_gray is None or previous_gray.shape != gray.shape:
        return 0.0
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    previous_hist = cv2.calcHist([previous_gray], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist)
    cv2.normalize(previous_hist, previous_hist)
    similarity = float(cv2.compareHist(hist, previous_hist, cv2.HISTCMP_CORREL))
    return max(0.0, min(1.0, 1.0 - similarity))


def _quality_metadata(frame, previous_gray=None, detector=None) -> tuple[dict[str, object], object]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness_score, blur = _score_sharpness(gray)
    brightness_score, dark, overexposed = _score_brightness(gray)
    motion_score = _score_motion(gray, previous_gray)
    diversity_score = _score_diversity(gray, previous_gray)

    person_count = None
    person_bbox_area_ratio = None
    no_person = False
    occluded = False
    person_score = 0.5
    if detector is not None:
        detections = detector(frame)
        person_count = len(detections)
        if detections:
            areas = []
            height, width = frame.shape[:2]
            frame_area = max(1, width * height)
            for x1, y1, x2, y2 in detections:
                areas.append(max(0, x2 - x1) * max(0, y2 - y1) / frame_area)
            person_bbox_area_ratio = max(areas)
            person_score = min(1.0, person_count / 2.0)
            occluded = person_bbox_area_ratio < 0.05
        else:
            no_person = True
            person_score = 0.0

    frame_quality_score = (
        QUALITY_WEIGHT_SHARPNESS * sharpness_score
        + QUALITY_WEIGHT_BRIGHTNESS * brightness_score
        + QUALITY_WEIGHT_PERSON * person_score
        + QUALITY_WEIGHT_MOTION * motion_score
        + QUALITY_WEIGHT_DIVERSITY * diversity_score
    )
    flags = []
    if blur:
        flags.append("blur")
    if dark:
        flags.append("dark")
    if overexposed:
        flags.append("overexposed")
    if no_person:
        flags.append("no_person")
    if occluded:
        flags.append("occluded")

    return (
        {
            "sharpness_score": round(sharpness_score, 4),
            "brightness_score": round(brightness_score, 4),
            "motion_score": round(motion_score, 4),
            "diversity_score": round(diversity_score, 4),
            "person_count": person_count,
            "person_bbox_area_ratio": None if person_bbox_area_ratio is None else round(float(person_bbox_area_ratio), 4),
            "frame_quality_score": round(float(frame_quality_score), 4),
            "quality_flags": ",".join(flags),
        },
        gray,
    )


def _sample_timestamps(duration_s: float, fps: float = TARGET_FPS) -> list[float]:
    if duration_s <= 0:
        return []
    step = 1.0 / fps
    timestamps = np.arange(0.0, duration_s, step, dtype=np.float64)
    return [float(timestamp) for timestamp in timestamps]


def _decode_frame_at_timestamp(capture: cv2.VideoCapture, timestamp_s: float):
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        return None, None
    reported_timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or (timestamp_s * 1000.0))
    return frame, reported_timestamp_ms


def _write_frame_png(frame, frame_path: Path) -> None:
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(frame_path), frame)
    if not ok:
        raise RuntimeError(f"Failed to write frame: {frame_path}")


def _window_bounds(frame_count: int, window_size: int = WINDOW_FRAMES, overlap: int = WINDOW_OVERLAP_FRAMES) -> list[tuple[int, int]]:
    step = window_size - overlap
    bounds: list[tuple[int, int]] = []
    start = 0
    while start + window_size <= frame_count:
        bounds.append((start, start + window_size))
        start += step
    return bounds


def _interval_overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _action_overlap_ratio(window_start: float, window_end: float, action_start: float | None, action_end: float | None) -> float:
    if action_start is None or action_end is None:
        return 0.0
    action_overlap = _interval_overlap(window_start, window_end, action_start, action_end)
    action_duration = max(0.001, action_end - action_start)
    return action_overlap / action_duration


def _classify_window(
    window_start: float,
    window_end: float,
    action_start: float | None,
    action_end: float | None,
    duration_s: float,
    class_code: str | None = None,
) -> tuple[str, bool]:
    """Gán temporal phase, không gán nhãn binary cho window.

    action_start/end là evidence interval trong clip. Nhãn train binary được
    lấy riêng từ class_code (c5 là non-cheating), không suy diễn từ phase.
    """
    if class_code is not None and _binary_label(class_code) == "non_cheating":
        return "non_action", True
    if action_start is None or action_end is None:
        return "action", True
    overlap = _interval_overlap(window_start, window_end, action_start, action_end)
    return ("action" if overlap > 0.0 else "non_action"), True


def _binary_label(class_code: str) -> str:
    """Taxonomy được xác nhận: chỉ c5 là non-cheating."""
    return "non_cheating" if class_code.strip().lower() in NON_CHEATING_CLASS_CODES else "cheating"


def _frame_label(timestamp_s: float, action_start: float | None, action_end: float | None, class_code: str = "") -> str:
    if _binary_label(class_code) == "non_cheating":
        return "non_action"
    if action_start is None or action_end is None:
        return "action"
    if action_start <= timestamp_s <= action_end:
        return "action"
    return "negative"


def _label_confidence(label: str, overlap_ratio: float) -> float:
    if label == "mixed":
        return 0.5
    if label == "action":
        return max(0.5, min(1.0, overlap_ratio))
    return 1.0


def _selected_frame_indices(frame_index_frame: pd.DataFrame, start_index: int, end_index: int) -> list[tuple[int, str]]:
    local = frame_index_frame.iloc[start_index:end_index].reset_index(drop=False)
    picks: dict[int, str] = {
        0: "start",
        max(0, len(local) // 2): "middle",
        max(0, len(local) - 1): "end",
    }
    metrics = [
        ("sharpness_score", "high_sharpness"),
        ("motion_score", "high_motion"),
        ("diversity_score", "high_diversity"),
    ]
    for column, reason in metrics:
        if column in local:
            order = local[column].astype(float).sort_values(ascending=False).index
            for local_index in order:
                local_index = int(local_index)
                if local_index not in picks:
                    picks[local_index] = reason
                    break
    for local_index in range(len(local)):
        if len(picks) >= SELECTED_FRAMES_PER_WINDOW:
            break
        picks.setdefault(local_index, "coverage")
    return [(start_index + local_index, reason) for local_index, reason in sorted(picks.items())[:SELECTED_FRAMES_PER_WINDOW]]


def _preflight_stage1(*, manifest_path: Path, raw_root: Path, output_root: Path, specs: list[ClipSpec]) -> dict[str, Path]:
    """Validate all cheap prerequisites before decoding or writing frames."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not raw_root.is_dir():
        raise NotADirectoryError(f"Raw video root not found: {raw_root}")
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Stage 1 requires a Parquet engine. Install pyarrow before processing frames."
        ) from error

    resolved: dict[str, Path] = {}
    for spec in specs:
        if spec.exclude_from_training:
            continue
        video_path = _resolve_video_path(raw_root, spec.filename)
        metadata = _probe_video(video_path)
        if not metadata["is_landscape"]:
            raise ValueError(f"Video is not landscape: {video_path}")
        resolved[spec.filename] = video_path

    output_root.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def build_stage1(*, manifest_path: Path, raw_root: Path, output_root: Path | None = None, limit: int | None = None) -> pd.DataFrame:
    output_root = stage1_root(output_root, manifest_path)
    frames_root = output_root / "frames"
    artifact_names = {
        "segments.parquet",
        "sampled_frames.parquet",
        "selected_frames.parquet",
        "windows.parquet",
        "stage1_report.csv",
    }
    existing_artifacts = [output_root / name for name in artifact_names if (output_root / name).exists()]
    frames_exist = frames_root.exists() and any(frames_root.iterdir())
    if existing_artifacts or frames_exist:
        existing_text = ", ".join(str(path) for path in existing_artifacts)
        if frames_exist:
            existing_text = ", ".join(filter(None, [existing_text, str(frames_root)]))
        raise FileExistsError(f"Output root already contains Stage 1 data; choose a new output root to avoid overwrite: {existing_text}")
    manifest = _read_manifest(manifest_path)
    specs = _collect_clip_specs(manifest)
    if limit is not None:
        specs = specs[:limit]
    resolved_video_paths = _preflight_stage1(
        manifest_path=manifest_path,
        raw_root=raw_root,
        output_root=output_root,
        specs=specs,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    segment_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    selected_frame_rows: list[dict[str, object]] = []

    for clip_index, spec in enumerate(specs, start=1):
        if spec.exclude_from_training:
            continue

        video_path = resolved_video_paths[spec.filename]
        metadata = _probe_video(video_path)
        if not metadata["is_landscape"]:
            raise ValueError(f"Video is not landscape: {video_path}")

        clip_id = str(
            spec.raw_row.get("clip_id")
            or f"clip_{clip_index:03d}_{Path(spec.filename).stem}"
        ).strip()
        segment_type = (
            "action"
            if spec.class_code != "c5" and spec.action_start_s is not None and spec.action_end_s is not None
            else "full_clip"
        )
        source_width = int(metadata["width"])
        source_height = int(metadata["height"])
        scale = _analysis_scale(source_width, source_height)
        analysis_width = int(round(source_width * scale))
        analysis_height = int(round(source_height * scale))
        session_id = spec.raw_row.get("session_id") or spec.raw_row.get("session") or None
        subject_id = spec.raw_row.get("subject_id") or spec.raw_row.get("student_id") or spec.raw_row.get("subject") or None
        group_id = spec.raw_row.get("group_id") or spec.raw_row.get("group") or None
        split_group = subject_id or session_id or group_id or None
        subject_ids = tuple(
            token.strip().lower()
            for token in str(spec.raw_row.get("subject_ids", split_group or "")).split(",")
            if token.strip()
        )
        explicit_split = str(spec.raw_row.get("split", "")).strip().lower()
        split_name = explicit_split if explicit_split in {"train", "val", "test"} else (
            assign_split(subject_ids) if subject_ids else None
        )
        split_status = "ready" if split_name is not None else "unassigned"
        split_source = str(spec.raw_row.get("split_source", "manifest_metadata"))
        segment_rows.append(
            {
                "clip_id": clip_id,
                "filename": spec.filename,
                "video_path": str(video_path),
                "video_hash": _video_hash(video_path),
                "class_code": spec.class_code,
                "segment_type": segment_type,
                "action_start_s": spec.action_start_s,
                "action_end_s": spec.action_end_s,
                "source_filename": spec.filename,
                "session_id": session_id,
                "subject_id": subject_id,
                "group_id": group_id,
                "split_group": split_group,
                "subject_ids": ",".join(subject_ids),
                "split": split_name,
                "split_source": split_source,
                "split_status": split_status,
                "source_width": source_width,
                "source_height": source_height,
                "analysis_width": analysis_width,
                "analysis_height": analysis_height,
                "analysis_scale": round(scale, 6),
                **metadata,
            }
        )

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        sampled_frames: list[dict[str, object]] = []
        previous_gray = None
        for frame_number, timestamp_s in enumerate(_sample_timestamps(float(metadata["duration_s"]), TARGET_FPS)):
            frame, timestamp_ms = _decode_frame_at_timestamp(capture, timestamp_s)
            if frame is None:
                sampled_frames.append(
                    {
                        "clip_id": clip_id,
                        "filename": spec.filename,
                        "frame_index": frame_number,
                        "timestamp_s": round(timestamp_s, 3),
                        "timestamp_ms": None,
                        "frame_path": None,
                        "decode_error": True,
                        "frame_label": _frame_label(timestamp_s, spec.action_start_s, spec.action_end_s, spec.class_code),
                        "quality_flags": "decode_error",
                    }
                )
                continue
            analysis_frame, frame_source_width, frame_source_height, frame_analysis_width, frame_analysis_height, frame_scale = _normalize_analysis_frame(frame)
            quality, previous_gray = _quality_metadata(analysis_frame, previous_gray)
            frame_rel_path = Path(clip_id) / f"frame_{frame_number:04d}.png"
            frame_path = frames_root / frame_rel_path
            _write_frame_png(analysis_frame, frame_path)
            sampled_frames.append(
                {
                    "clip_id": clip_id,
                    "filename": spec.filename,
                    "video_hash": segment_rows[-1]["video_hash"],
                    "source_filename": spec.filename,
                    "session_id": session_id,
                    "subject_id": subject_id,
                    "group_id": group_id,
                    "split_group": split_group,
                    "subject_ids": ",".join(subject_ids),
                    "split": split_name,
                    "split_source": split_source,
                    "split_status": split_status,
                    "frame_index": frame_number,
                    "timestamp_s": round(timestamp_s, 3),
                    "timestamp_ms": round(float(timestamp_ms), 1),
                    "actual_timestamp_s": round(float(timestamp_ms) / 1000.0, 3),
                    "frame_path": str(frame_rel_path).replace("\\", "/"),
                    "source_width": frame_source_width,
                    "source_height": frame_source_height,
                    "analysis_width": frame_analysis_width,
                    "analysis_height": frame_analysis_height,
                    "analysis_scale": round(frame_scale, 6),
                    "width": frame_analysis_width,
                    "height": frame_analysis_height,
                    "decode_error": False,
                    "frame_label": _frame_label(timestamp_s, spec.action_start_s, spec.action_end_s, spec.class_code),
                    **quality,
                }
            )
        capture.release()

        if not sampled_frames:
            continue

        frame_index_frame = pd.DataFrame(sampled_frames)
        frame_rows.extend(frame_index_frame.to_dict("records"))
        decoded_frame_index_frame = frame_index_frame[frame_index_frame["decode_error"] == False].reset_index(drop=True)
        if decoded_frame_index_frame.empty:
            continue

        bounds = _window_bounds(len(decoded_frame_index_frame))
        for window_index, (start_index, end_index) in enumerate(bounds):
            window_start_s = float(decoded_frame_index_frame.iloc[start_index]["timestamp_s"])
            window_end_s = float(decoded_frame_index_frame.iloc[end_index - 1]["timestamp_s"])
            window_phase, include_in_training = _classify_window(
                window_start=window_start_s,
                window_end=window_end_s,
                action_start=spec.action_start_s,
                action_end=spec.action_end_s,
                duration_s=float(metadata["duration_s"]),
                class_code=spec.class_code,
            )
            action_overlap_ratio = _action_overlap_ratio(window_start_s, window_end_s, spec.action_start_s, spec.action_end_s)
            binary_label = _binary_label(spec.class_code)
            review_required = False
            low_quality_frames = decoded_frame_index_frame.iloc[start_index:end_index]["quality_flags"].astype(str).str.len() > 0
            low_quality_ratio = float(low_quality_frames.mean())
            rejected_reason = ""
            if low_quality_ratio >= 0.5:
                rejected_reason = ",".join(filter(None, [rejected_reason, "low_quality"]))

            window_id = f"{clip_id}_w{window_index:03d}"
            window_frame_dir = frames_root / clip_id / f"w{window_index:03d}"
            for local_index in range(start_index, end_index):
                rel_path = Path(clip_id) / f"w{window_index:03d}" / f"frame_{local_index - start_index:04d}.png"
                source_rel_path = Path(decoded_frame_index_frame.iloc[local_index]["frame_path"])
                source_path = frames_root / source_rel_path
                target_path = window_frame_dir / f"frame_{local_index - start_index:04d}.png"
                if not target_path.exists():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        raise RuntimeError(f"Failed to read saved frame: {source_path}")
                    ok = cv2.imwrite(str(target_path), image)
                    if not ok:
                        raise RuntimeError(f"Failed to write window frame: {target_path}")

            selected_indices = _selected_frame_indices(decoded_frame_index_frame, start_index, end_index)
            for selected_index, reason in selected_indices:
                source_rel_path = decoded_frame_index_frame.iloc[selected_index]["frame_path"]
                selected_frame_rows.append(
                    {
                        "window_id": window_id,
                        "clip_id": clip_id,
                        "filename": spec.filename,
                        "source_frame_index": int(decoded_frame_index_frame.iloc[selected_index]["frame_index"]),
                        "window_local_index": int(selected_index - start_index),
                        "selection_reason": reason,
                        "frame_path": source_rel_path,
                        "timestamp_s": decoded_frame_index_frame.iloc[selected_index]["timestamp_s"],
                        "frame_quality_score": decoded_frame_index_frame.iloc[selected_index]["frame_quality_score"],
                    }
                )

            window_rows.append(
                {
                    "window_id": window_id,
                    "clip_id": clip_id,
                    "filename": spec.filename,
                    "video_hash": segment_rows[-1]["video_hash"],
                    "source_filename": spec.filename,
                    "session_id": session_id,
                    "subject_id": subject_id,
                    "group_id": group_id,
                    "split_group": split_group,
                    "subject_ids": ",".join(subject_ids),
                    "split": split_name,
                    "split_source": split_source,
                    "split_status": split_status,
                    "class_code": spec.class_code,
                    "label": window_phase,
                    "window_label": window_phase,
                    "window_phase": window_phase,
                    "binary_label": binary_label,
                    "include_in_training": include_in_training,
                    "review_required": review_required,
                    "window_start_s": round(window_start_s, 3),
                    "window_end_s": round(window_end_s, 3),
                    "window_size_frames": WINDOW_FRAMES,
                    "window_overlap_frames": WINDOW_OVERLAP_FRAMES,
                    "window_stride_frames": WINDOW_FRAMES - WINDOW_OVERLAP_FRAMES,
                    "target_fps": TARGET_FPS,
                    "frame_count": end_index - start_index,
                    "frame_dir": str(Path(clip_id) / f"w{window_index:03d}").replace("\\", "/"),
                    "action_start_s": spec.action_start_s,
                    "action_end_s": spec.action_end_s,
                    "action_overlap_ratio": round(action_overlap_ratio, 3),
                    "label_confidence": round(_label_confidence(window_phase, action_overlap_ratio), 3),
                    "annotation_source": "manifest",
                    "selected_frame_count": len(selected_indices),
                    "sampled_frame_count": end_index - start_index,
                    "low_quality_frame_ratio": round(low_quality_ratio, 3),
                    "rejected_reason": rejected_reason,
                }
            )

    segments = pd.DataFrame(segment_rows)
    sampled_frames = pd.DataFrame(frame_rows)
    windows = pd.DataFrame(window_rows)
    selected_frames = pd.DataFrame(selected_frame_rows)

    segments_path = output_root / "segments.parquet"
    sampled_frames_path = output_root / "sampled_frames.parquet"
    windows_path = output_root / "windows.parquet"
    selected_frames_path = output_root / "selected_frames.parquet"
    report_path = output_root / "stage1_report.csv"

    if not segments.empty:
        segments.to_parquet(segments_path, index=False)
    if not sampled_frames.empty:
        sampled_frames.to_parquet(sampled_frames_path, index=False)
    if not windows.empty:
        windows.to_parquet(windows_path, index=False)
    if not selected_frames.empty:
        selected_frames.to_parquet(selected_frames_path, index=False)

    action_windows = int((windows["label"] == "action").sum()) if not windows.empty else 0
    negative_windows = int(windows["label"].isin(["negative_pre", "negative_post"]).sum()) if not windows.empty else 0
    mixed_windows = int((windows["label"] == "mixed").sum()) if not windows.empty else 0
    review_required_windows = int(windows["review_required"].sum()) if not windows.empty else 0
    quality_mean = float(sampled_frames["frame_quality_score"].mean()) if "frame_quality_score" in sampled_frames and not sampled_frames.empty else math.nan
    quality_min = float(sampled_frames["frame_quality_score"].min()) if "frame_quality_score" in sampled_frames and not sampled_frames.empty else math.nan
    low_quality_windows = int((windows["rejected_reason"].astype(str).str.contains("low_quality")).sum()) if not windows.empty else 0
    split_status = "insufficient_metadata" if not segments.empty and (segments["split_status"] != "ready").any() else "ready"
    report = pd.DataFrame(
        [
            {
                "clips": len(segments),
                "sampled_frames": len(sampled_frames),
                "selected_frames": len(selected_frames),
                "windows": len(windows),
                "action_windows": action_windows,
                "negative_pre_windows": int((windows["label"] == "negative_pre").sum()) if not windows.empty else 0,
                "negative_post_windows": int((windows["label"] == "negative_post").sum()) if not windows.empty else 0,
                "negative_windows": negative_windows,
                "negative_windows_cover_action_windows": negative_windows >= action_windows,
                "negative_window_deficit": max(0, action_windows - negative_windows),
                "mixed_windows": mixed_windows,
                "review_required_windows": review_required_windows,
                "low_quality_windows": low_quality_windows,
                "quality_score_min": None if math.isnan(quality_min) else round(quality_min, 4),
                "quality_score_mean": None if math.isnan(quality_mean) else round(quality_mean, 4),
                "split_status": split_status,
                "target_fps": TARGET_FPS,
                "window_frames": WINDOW_FRAMES,
                "window_overlap_frames": WINDOW_OVERLAP_FRAMES,
                "window_stride_frames": WINDOW_FRAMES - WINDOW_OVERLAP_FRAMES,
            }
        ]
    )
    report.to_csv(report_path, index=False)

    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 ingest and normalization for pose/gaze datasets")
    parser.add_argument("--manifest", type=Path, required=True, help="Reviewed manifest CSV/XLSX containing action_start_s/action_end_s")
    parser.add_argument("--raw-root", type=Path, required=True, help="Folder containing the raw video files")
    parser.add_argument("--output-root", type=Path, default=None, help="Where to write stage 1 artifacts")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N clips")
    args = parser.parse_args()

    windows = build_stage1(manifest_path=args.manifest, raw_root=args.raw_root, output_root=args.output_root, limit=args.limit)
    print(windows.to_string(index=False) if not windows.empty else "(no windows)")


if __name__ == "__main__":
    main()

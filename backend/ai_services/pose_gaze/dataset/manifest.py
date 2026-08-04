from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import pandas as pd

from .common import CLASS_COUNTS, default_raw_roots, processed_root, validate_class_counts

VIDEO_EXTENSIONS = {".avi", ".mov", ".mp4", ".mkv", ".wmv", ".webm"}
FILENAME_RE = re.compile(r"^v_(c\d+)_(.+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: Path
    class_code: str
    subject_ids: tuple[str, ...]
    take_code: str | None

    @property
    def stem(self) -> str:
        return self.path.stem


def discover_raw_root() -> Path:
    for candidate in default_raw_roots():
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find the Dataset_cheating raw video folder")


def parse_video_filename(filename: str | Path) -> VideoMetadata:
    path = Path(filename)
    stem = path.stem
    match = FILENAME_RE.match(stem)
    if not match:
        raise ValueError(f"Unrecognized dataset filename: {filename}")

    class_code = match.group(1).lower()
    if class_code not in CLASS_COUNTS:
        raise ValueError(f"Unknown class code: {class_code}")

    suffix_parts = match.group(2).split("_")
    subject_ids: list[str] = []
    take_code: str | None = None
    for part in suffix_parts:
        if re.fullmatch(r"s\d+", part, flags=re.IGNORECASE):
            subject_ids.append(part.lower())
            continue
        if re.fullmatch(r"v\d+", part, flags=re.IGNORECASE):
            take_code = part.lower()
            continue
        raise ValueError(f"Unexpected token in dataset filename: {part}")

    if not subject_ids:
        raise ValueError(f"No subject ids found in filename: {filename}")

    return VideoMetadata(path=path, class_code=class_code, subject_ids=tuple(subject_ids), take_code=take_code)


def iter_video_files(raw_root: Path) -> list[Path]:
    return sorted(
        [path for path in raw_root.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def probe_video(path: Path) -> dict[str, int | float | bool]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("opencv-python is required to probe dataset videos") from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    duration_ms = int(round((frame_count / fps) * 1000)) if frame_count and fps else 0
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_ms": duration_ms,
        "is_readable": True,
    }


def build_manifest(*, raw_root: Path | None = None, output_path: Path | None = None, limit: int | None = None, strict: bool = True) -> pd.DataFrame:
    raw_root = raw_root or discover_raw_root()
    output_path = output_path or (processed_root() / "manifest.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for path in iter_video_files(raw_root):
        metadata = parse_video_filename(path)
        stats = probe_video(path)
        rows.append(
            {
                "video_name": path.name,
                "video_path": str(path),
                "video_stem": metadata.stem,
                "class_code": metadata.class_code,
                "subject_ids": ",".join(metadata.subject_ids),
                "subject_count": len(metadata.subject_ids),
                "take_code": metadata.take_code or "",
                **stats,
            }
        )
        if limit is not None and len(rows) >= limit:
            break

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError(f"No dataset videos found in {raw_root}")

    manifest.to_parquet(output_path, index=False)

    if strict and limit is None and len(manifest) == sum(CLASS_COUNTS.values()):
        class_counts = manifest.groupby("class_code").size().to_dict()
        validate_class_counts(class_counts)

    return manifest

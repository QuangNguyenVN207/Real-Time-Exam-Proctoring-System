"""Batch images through person detection, tracking, Holistic, and split CSVs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re
from time import monotonic
from typing import Any

from backend.ai_services.pose_gaze.holistic.feature_csv import (
    ANNOTATION_COLUMNS,
    CSV_FIELDNAMES,
    build_csv_row,
    model_features_from_result,
)
from backend.ai_services.pose_gaze.holistic.landmark import (
    HolisticLandmarkExtractor,
)
from backend.ai_services.pose_gaze.settings import (
    DEFAULT_HOLISTIC_CONFIDENCE,
    DEFAULT_HOLISTIC_SOFT_CONFIDENCE,
    DEFAULT_PERSON_CONFIDENCE,
    PROJECT_ROOT,
)
from backend.ai_services.pose_gaze.tracking.detectors import (
    PersonDetector,
    UltralyticsPersonDetector,
)
from backend.ai_services.pose_gaze.tracking.schemas import TrackPacket
from backend.ai_services.pose_gaze.tracking.tracker import IoUPersonTracker


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = MODULE_DIR / "data"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)
CLASS_FOLDER_PATTERN = re.compile(r"^(c\d+)(?:(?:__|[-_])(.*))?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """One labeled image scheduled for one output split."""

    path: Path
    split: str
    class_code: str
    label: str
    sequence_id: str | None = None
    annotation: dict[str, str] = field(default_factory=dict)


def _is_true(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_class_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--class-map must contain a JSON object")
    output = {
        str(key).strip().lower(): str(value).strip()
        for key, value in payload.items()
    }
    if not all(output.keys()) or not all(output.values()):
        raise ValueError("--class-map keys and labels must not be empty")
    return output


def parse_class_folder(name: str, class_map: dict[str, str]) -> tuple[str, str]:
    """Parse ``c2`` or ``c2__hand_reach`` without hard-coding the taxonomy."""

    match = CLASS_FOLDER_PATTERN.fullmatch(name.strip())
    if match is None:
        raise ValueError(
            f"Class folder '{name}' must look like c2 or c2__hand_reach"
        )
    class_code = match.group(1).lower()
    folder_label = (match.group(2) or "").strip()
    return class_code, class_map.get(class_code, folder_label or class_code)


def deterministic_split(
    relative_path: str,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: str,
) -> str:
    """Assign an unsplit image reproducibly without depending on input order."""

    digest = hashlib.sha256(f"{seed}\0{relative_path}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + val_ratio:
        return "val"
    return "test"


def _image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        delimiter = "\t" if first_line.count("\t") > first_line.count(",") else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("Manifest has no header")
        return [
            {
                str(key).strip().lower(): str(value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]


def samples_from_manifest(
    input_root: Path,
    manifest_path: Path,
    class_map: dict[str, str],
) -> list[DatasetSample]:
    """Load an image manifest compatible with the supplied annotation columns."""

    resolved_root = input_root.resolve()
    samples: list[DatasetSample] = []
    for row_number, row in enumerate(_manifest_rows(manifest_path), start=2):
        if _is_true(row.get("exclude_from_training")):
            continue
        filename = row.get("filename", "")
        if not filename:
            raise ValueError(f"Manifest row {row_number} has no filename")
        path = (resolved_root / filename).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"Manifest row {row_number} points outside --input-root"
            ) from error
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Manifest row {row_number} is not an image: {filename}. "
                "Extract video frames before using this batch command."
            )
        split = row.get("split", "").lower()
        if split not in SPLITS:
            raise ValueError(
                f"Manifest row {row_number} has split '{split}', expected "
                "train, val, or test"
            )
        class_code = row.get("class_code", "").lower()
        if not class_code:
            raise ValueError(f"Manifest row {row_number} has no class_code")
        label = row.get("label", "") or class_map.get(class_code, class_code)
        annotation = {
            column: row[column]
            for column in ANNOTATION_COLUMNS
            if column in row
        }
        samples.append(
            DatasetSample(
                path=path,
                split=split,
                class_code=class_code,
                label=label,
                sequence_id=row.get("sequence_id") or None,
                annotation=annotation,
            )
        )
    return samples


def samples_from_folders(
    input_root: Path,
    class_map: dict[str, str],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: str,
) -> list[DatasetSample]:
    """Discover pre-split folders or deterministically split class folders."""

    split_roots = {
        split: input_root / split
        for split in SPLITS
        if (input_root / split).is_dir()
    }
    samples: list[DatasetSample] = []
    if split_roots:
        for split, split_root in split_roots.items():
            for path in _image_paths(split_root):
                relative = path.relative_to(split_root)
                if len(relative.parts) < 2:
                    raise ValueError(
                        f"Image must be inside a class folder: {relative}"
                    )
                class_code, label = parse_class_folder(relative.parts[0], class_map)
                sequence_id = relative.parts[1] if len(relative.parts) >= 3 else None
                samples.append(
                    DatasetSample(path, split, class_code, label, sequence_id)
                )
        return samples

    for path in _image_paths(input_root):
        relative = path.relative_to(input_root)
        if len(relative.parts) < 2:
            raise ValueError(f"Image must be inside a class folder: {relative}")
        class_code, label = parse_class_folder(relative.parts[0], class_map)
        sequence_id = relative.parts[1] if len(relative.parts) >= 3 else None
        split = deterministic_split(
            sequence_id or relative.as_posix(),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        samples.append(DatasetSample(path, split, class_code, label, sequence_id))
    return samples


def discover_samples(
    input_root: Path,
    *,
    manifest_path: Path | None,
    class_map: dict[str, str],
    train_ratio: float,
    val_ratio: float,
    seed: str,
) -> list[DatasetSample]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root was not found: {input_root}")
    if manifest_path is not None:
        samples = samples_from_manifest(input_root, manifest_path, class_map)
    else:
        samples = samples_from_folders(
            input_root,
            class_map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    if not samples:
        raise ValueError("No supported images were found")
    return samples


class SplitCsvWriters:
    """Stream three bounded-memory CSVs and publish them only after success."""

    def __init__(self, output_dir: Path, *, overwrite: bool) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, Any] = {}
        self._writers: dict[str, csv.DictWriter] = {}
        self._parts: dict[str, Path] = {}
        self._outputs: dict[str, Path] = {}
        for split in SPLITS:
            output = self.output_dir / f"{split}.csv"
            part = self.output_dir / f"{split}.csv.part"
            if not overwrite and (output.exists() or part.exists()):
                raise FileExistsError(
                    f"Output already exists for {split}; pass --overwrite: {output}"
                )
            if overwrite:
                part.unlink(missing_ok=True)
            handle = part.open("w", encoding="utf-8", newline="", buffering=1)
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            self._handles[split] = handle
            self._writers[split] = writer
            self._parts[split] = part
            self._outputs[split] = output

    def write(self, split: str, row: dict[str, Any]) -> None:
        self._writers[split].writerow(row)

    def commit(self) -> dict[str, Path]:
        for handle in self._handles.values():
            handle.close()
        for split in SPLITS:
            self._parts[split].replace(self._outputs[split])
        return dict(self._outputs)

    def abort(self) -> None:
        for handle in self._handles.values():
            if not handle.closed:
                handle.close()


def _error_row(
    sample: DatasetSample,
    *,
    input_root: Path,
    index: int,
    status: str,
    error: str,
    frame_id: int,
) -> dict[str, Any]:
    return build_csv_row(
        split=sample.split,
        class_code=sample.class_code,
        label=sample.label,
        status=status,
        error=error[:1000],
        source_path=sample.path.relative_to(input_root.resolve()).as_posix(),
        sequence_id=sample.sequence_id,
        source_frame_index=index,
        frame_id=frame_id,
        timestamp_ms=frame_id,
        session_id=f"batch_{sample.split}_{sample.sequence_id or 'independent'}",
        frame_width=None,
        frame_height=None,
        person_count=0,
        annotation=sample.annotation,
    )


def process_sample(
    sample: DatasetSample,
    *,
    input_root: Path,
    index: int,
    detector: PersonDetector,
    holistic: HolisticLandmarkExtractor,
    cv2_module: Any,
    max_people: int,
    tracker: IoUPersonTracker,
    frame_id: int,
) -> list[dict[str, Any]]:
    """Process one image while keeping detector/model instances reusable."""

    source_path = sample.path.relative_to(input_root.resolve()).as_posix()
    frame = cv2_module.imread(str(sample.path))
    if frame is None:
        return [
            _error_row(
                sample,
                input_root=input_root,
                index=index,
                status="read_error",
                error="OpenCV could not decode the image",
                frame_id=frame_id,
            )
        ]

    detections = detector.detect(frame)
    tracks = tuple(
        replace(track, student_id=f"person_{track.track_id:02d}")
        for track in tracker.update(detections)
        if track.is_present
    )
    height, width = frame.shape[:2]
    packet = TrackPacket(
        session_id=f"batch_{sample.split}",
        frame_id=frame_id,
        timestamp_ms=frame_id,
        tracks=tracks,
    )
    if not tracks:
        return [
            build_csv_row(
                split=sample.split,
                class_code=sample.class_code,
                label=sample.label,
                status="no_person",
                source_path=source_path,
                sequence_id=sample.sequence_id,
                source_frame_index=index,
                frame_id=frame_id,
                timestamp_ms=frame_id,
                session_id=packet.session_id,
                frame_width=width,
                frame_height=height,
                person_count=0,
                annotation=sample.annotation,
            )
        ]

    results = {result.track_id: result for result in holistic.process_packet(frame, packet)}
    rows: list[dict[str, Any]] = []
    for track in tracks:
        result = results.get(track.track_id)
        has_landmarks = (
            result is not None
            and (model_features_from_result(result)["all_landmarks_valid_ratio"] or 0.0)
            > 0.0
        )
        rows.append(
            build_csv_row(
                split=sample.split,
                class_code=sample.class_code,
                label=sample.label,
                status="ok" if has_landmarks else "no_landmarks",
                source_path=source_path,
                sequence_id=sample.sequence_id,
                source_frame_index=index,
                frame_id=frame_id,
                timestamp_ms=frame_id,
                session_id=packet.session_id,
                frame_width=width,
                frame_height=height,
                person_count=len(tracks),
                track=track,
                result=result,
                annotation=sample.annotation,
            )
        )
    return rows


def export_dataset(
    samples: list[DatasetSample],
    *,
    input_root: Path,
    output_dir: Path,
    detector: PersonDetector,
    holistic: HolisticLandmarkExtractor,
    cv2_module: Any,
    max_people: int,
    overwrite: bool,
    fail_fast: bool,
    log_every: int,
) -> tuple[dict[str, Path], dict[str, int]]:
    writers = SplitCsvWriters(output_dir, overwrite=overwrite)
    counts: dict[str, int] = {}
    sequence_trackers: dict[str, IoUPersonTracker] = {}
    sequence_frame_ids: dict[str, int] = {}
    started_at = monotonic()
    try:
        for index, sample in enumerate(samples, start=1):
            if sample.sequence_id is None:
                tracker = IoUPersonTracker(
                    max_tracks=max_people,
                    min_iou=0.0,
                    max_missed_frames=0,
                )
                frame_id = 1
            else:
                sequence_key = f"{sample.split}:{sample.sequence_id}"
                tracker = sequence_trackers.setdefault(
                    sequence_key,
                    IoUPersonTracker(max_tracks=max_people),
                )
                frame_id = sequence_frame_ids.get(sequence_key, 0) + 1
                sequence_frame_ids[sequence_key] = frame_id
            try:
                rows = process_sample(
                    sample,
                    input_root=input_root,
                    index=index,
                    detector=detector,
                    holistic=holistic,
                    cv2_module=cv2_module,
                    max_people=max_people,
                    tracker=tracker,
                    frame_id=frame_id,
                )
            except Exception as error:
                rows = [
                    _error_row(
                        sample,
                        input_root=input_root,
                        index=index,
                        status="inference_error",
                        error=f"{type(error).__name__}: {error}",
                        frame_id=frame_id,
                    )
                ]
                if fail_fast:
                    writers.write(sample.split, rows[0])
                    raise
            for row in rows:
                writers.write(sample.split, row)
                key = f"{sample.split}:{row['status']}"
                counts[key] = counts.get(key, 0) + 1
            if index % log_every == 0 or index == len(samples):
                elapsed = max(1e-6, monotonic() - started_at)
                print(
                    f"Processed {index}/{len(samples)} images "
                    f"({index / elapsed:.2f} images/s)"
                )
        return writers.commit(), counts
    except Exception:
        writers.abort()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="Image dataset root")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV/TSV with filename, class_code, label, split, ...",
    )
    parser.add_argument("--class-map", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 processes all images")
    parser.add_argument("--seed", default="pose-gaze-v1")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--model", type=Path, default=None, help="YOLO weights")
    parser.add_argument("--device", default=None, help="Ultralytics device: cpu, 0, ...")
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_PERSON_CONFIDENCE
    )
    parser.add_argument("--max-people", type=int, default=2)
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--holistic-model", type=Path, default=None)
    parser.add_argument("--holistic-input-size", type=int, default=512)
    parser.add_argument(
        "--holistic-confidence",
        type=float,
        default=DEFAULT_HOLISTIC_CONFIDENCE,
    )
    parser.add_argument(
        "--soft-landmark-confidence",
        type=float,
        default=DEFAULT_HOLISTIC_SOFT_CONFIDENCE,
    )
    parser.add_argument("--crop-padding", type=float, default=0.15)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1)")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train and validation ratios must leave room for test")
    if args.limit < 0:
        raise ValueError("--limit must be at least 0")
    if args.max_people < 1:
        raise ValueError("--max-people must be at least 1")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be in [0, 1]")
    if not 0.0 <= args.soft_landmark_confidence <= args.holistic_confidence <= 1.0:
        raise ValueError(
            "Holistic confidence must be in [0, 1] and soft confidence <= it"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_root = args.input_root.resolve()
    class_map = load_class_map(args.class_map)
    samples = discover_samples(
        input_root,
        manifest_path=args.manifest,
        class_map=class_map,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if args.limit:
        samples = samples[: args.limit]
    print(f"Discovered {len(samples)} images")

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Install opencv-python to run batch extraction") from error

    model_path = (args.model or PROJECT_ROOT / "weights" / "yolov8n.pt").resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Person detector weights were not found: {model_path}")
    detector = UltralyticsPersonDetector(
        model_path,
        confidence_threshold=args.confidence,
        device=args.device,
    )
    with HolisticLandmarkExtractor(
        static_image_mode=True,
        model_complexity=args.model_complexity,
        smooth_landmarks=False,
        min_detection_confidence=args.holistic_confidence,
        min_tracking_confidence=args.holistic_confidence,
        soft_landmark_confidence=args.soft_landmark_confidence,
        crop_padding=args.crop_padding,
        task_model_path=args.holistic_model,
        task_input_size=args.holistic_input_size,
    ) as holistic:
        outputs, counts = export_dataset(
            samples,
            input_root=input_root,
            output_dir=args.output_dir.resolve(),
            detector=detector,
            holistic=holistic,
            cv2_module=cv2,
            max_people=args.max_people,
            overwrite=args.overwrite,
            fail_fast=args.fail_fast,
            log_every=args.log_every,
        )
    for split in SPLITS:
        print(f"{split}: {outputs[split]}")
    print("Rows by split/status:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")


if __name__ == "__main__":
    main()

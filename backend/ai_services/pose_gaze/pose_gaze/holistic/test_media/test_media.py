"""Test person tracking and MediaPipe Holistic on one image or video file.

Examples:
    python -m backend.ai_services.pose_gaze.holistic.test_media input.jpg
    python -m backend.ai_services.pose_gaze.holistic.test_media input.mp4 --target-fps 8
    python -m backend.ai_services.pose_gaze.holistic.test_media input.mp4 --no-display
"""

from __future__ import annotations

import argparse
import json
import re
from statistics import median
from pathlib import Path
from time import monotonic
from typing import Any, TextIO
from uuid import uuid4

from ..landmark import (
    HolisticLandmarkExtractor,
    TrackHolisticResult,
)
from ...tracking.schemas import TrackPacket
from ...settings import (
    DEFAULT_HOLISTIC_CONFIDENCE,
    DEFAULT_HOLISTIC_SOFT_CONFIDENCE,
    DEFAULT_MAX_MISSED_FRAMES,
    DEFAULT_MIN_IOU,
    DEFAULT_PERSON_CONFIDENCE,
    PROJECT_ROOT,
)
from ...tracking.webcam import (
    PersonTrackingConfig,
    PersonTrackingModule,
)


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}
WINDOW_NAME = "Exam Proctoring - Image/Video Test"


class LandmarkJsonWriter:
    """Stream landmarks as an indented JSON document without buffering video."""

    def __init__(
        self,
        path: Path,
        *,
        input_path: Path,
        session_id: str,
        media_type: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO = path.open("w", encoding="utf-8")
        self._first_frame = True
        self._closed = False
        metadata = {
            "format_version": 3,
            "input": str(input_path),
            "session_id": session_id,
            "media_type": media_type,
            "student_id_strategy": "student_<track_id>",
            "quality_fields": [
                "face_valid",
                "mouth_valid",
                "face_predicted",
            ],
        }
        metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2)
        # Remove the final brace so that frame records can be streamed into the
        # same document instead of collecting the complete video in memory.
        self._stream.write(metadata_text[:-2])
        self._stream.write(',\n  "frames": [\n')

    def write(self, record: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed landmark JSON file")
        if not self._first_frame:
            self._stream.write(",\n")
        record_text = json.dumps(record, ensure_ascii=False, indent=2)
        indented_record = "\n".join(
            f"    {line}" for line in record_text.splitlines()
        )
        self._stream.write(indented_record)
        self._first_frame = False

    def close(self) -> None:
        if self._closed:
            return
        self._stream.write("\n  ]\n}\n")
        self._stream.close()
        self._closed = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to an image or video")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Annotated output path. Defaults to "
            "test_data_tracking/media_outputs/<name>_annotated.*"
        ),
    )
    parser.add_argument(
        "--landmarks-output",
        type=Path,
        default=None,
        help="Readable JSON landmark path; defaults beside the annotated output",
    )
    parser.add_argument(
        "--landmarks-input", type=Path, default=None,
        help="Render an existing landmark JSON without inference",
    )
    parser.add_argument(
        "--classification-input", type=Path, default=None,
        help="Optional actor classification JSON when rendering existing landmarks",
    )
    parser.add_argument(
        "--no-save-landmarks",
        action="store_true",
        help="Do not write per-frame pose/hand/selected-face landmarks",
    )
    parser.add_argument(
        "--no-save-annotated",
        action="store_true",
        help="Do not write annotated image or video output",
    )
    parser.add_argument("--model", type=Path, default=None, help="YOLO weights")
    parser.add_argument(
        "--object-model",
        type=Path,
        default=None,
        help="Optional three-class object detector used to subtype suspicious_activity",
    )
    parser.add_argument(
        "--object-device",
        default=None,
        help="Object detector device: cpu, 0, ...",
    )
    parser.add_argument("--device", default=None, help="Ultralytics device: cpu, 0, ...")
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_PERSON_CONFIDENCE,
    )
    parser.add_argument("--max-tracks", type=int, default=2)
    parser.add_argument("--min-iou", type=float, default=DEFAULT_MIN_IOU)
    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=DEFAULT_MAX_MISSED_FRAMES,
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=0,
        help=(
            "Video inference samples per source second. Use 0 to process every "
            "source frame. For example, 8 runs inference on exactly 8 samples/s."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many processed frames; use 0 for no limit",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Tracking session ID; a fresh ID is generated when omitted",
    )
    parser.add_argument(
        "--student-prefix",
        default="student_",
        help=(
            "Automatic student ID prefix. Track 1 becomes student_01 by "
            "default."
        ),
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Legacy MediaPipe only; Tasks uses its bundled model",
    )
    parser.add_argument(
        "--holistic-model",
        type=Path,
        default=None,
        help="Path to holistic_landmarker.task",
    )
    parser.add_argument(
        "--holistic-input-size",
        type=int,
        default=512,
        help="Fixed square MediaPipe Tasks input size",
    )
    parser.add_argument(
        "--holistic-confidence",
        type=float,
        default=DEFAULT_HOLISTIC_CONFIDENCE,
        help="Holistic detection/tracking and normal landmark threshold",
    )
    parser.add_argument(
        "--soft-landmark-confidence",
        type=float,
        default=DEFAULT_HOLISTIC_SOFT_CONFIDENCE,
        help="Keep observed x/y above this score but set visibility to null",
    )
    parser.add_argument("--crop-padding", type=float, default=0.15)
    parser.add_argument("--bbox-smoothing-alpha", type=float, default=0.85)
    parser.add_argument("--crop-stabilization-alpha", type=float, default=0.80)
    parser.add_argument(
        "--face-hold-frames",
        type=int,
        default=0,
        help="Held/predicted face frames; 0 disables face prediction (default)",
    )
    parser.add_argument(
        "--face-fallback-model",
        type=Path,
        default=None,
        help="Optional face_landmarker.task used only after repeated face loss",
    )
    parser.add_argument(
        "--xgboost-model-dir",
        type=Path,
        default=None,
        help="Optional actor-level c2/c3/c7 XGBoost model directory",
    )
    parser.add_argument(
        "--c1-model-dir", type=Path, default=None,
        help="Optional causal pose-only C1 specialist artifact directory",
    )
    parser.add_argument(
        "--c4-model-dir", type=Path, default=None,
        help="Optional causal pose-only C4 specialist artifact directory",
    )
    parser.add_argument(
        "--c7-model-dir", type=Path, default=None,
        help="Optional causal per-hand C7 specialist artifact directory",
    )
    parser.add_argument(
        "--causal-live",
        action="store_true",
        help="Run the causal actor state machine during frame ingestion",
    )
    parser.add_argument(
        "--live-pair",
        action="append",
        default=[],
        metavar="ACTOR:ACTOR",
        help="Configured explicit interaction pair for causal c2 propagation",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the annotated preview window",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Play video preview at its sampled FPS instead of as fast as possible",
    )
    return parser.parse_args()


def media_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(
        f"Unsupported input extension '{path.suffix}'. "
        "Use a common image or video extension."
    )


def safe_session_id(input_path: Path, requested: str | None) -> str:
    if requested:
        return requested
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", input_path.stem).strip("_")
    safe_stem = safe_stem or "input"
    return f"media_{safe_stem[:40]}_{uuid4().hex[:8]}"


def default_output_path(input_path: Path, kind: str) -> Path:
    output_root = PROJECT_ROOT / "test_data_tracking" / "media_outputs"
    suffix = input_path.suffix.lower() if kind == "image" else ".mp4"
    return output_root / f"{input_path.stem}_annotated{suffix}"


def validate_args(args: argparse.Namespace, kind: str) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file was not found: {args.input}")
    if args.output == args.input:
        raise ValueError("--output must not overwrite the input file")
    if (
        not args.no_save_landmarks
        and args.landmarks_output in {args.input, args.output}
    ):
        raise ValueError(
            "--landmarks-output must differ from the input and annotated output"
        )
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be in [0, 1]")
    if not 0.0 <= args.min_iou <= 1.0:
        raise ValueError("--min-iou must be in [0, 1]")
    if args.max_tracks < 1:
        raise ValueError("--max-tracks must be at least 1")
    if args.target_fps < 0:
        raise ValueError("--target-fps must be 0 or a positive integer")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be 0 or a positive integer")
    if not 0.0 <= args.holistic_confidence <= 1.0:
        raise ValueError("--holistic-confidence must be in [0, 1]")
    if not 0.0 <= args.soft_landmark_confidence <= args.holistic_confidence:
        raise ValueError(
            "--soft-landmark-confidence must be in [0, --holistic-confidence]"
        )
    if not args.student_prefix:
        raise ValueError("--student-prefix must not be empty")
    if args.causal_live:
        if kind != "video":
            raise ValueError("--causal-live requires a video or camera stream")
        if not any((args.xgboost_model_dir, args.c1_model_dir, args.c4_model_dir, args.c7_model_dir)):
            raise ValueError(
                "--causal-live requires at least one of --xgboost-model-dir, "
                "--c1-model-dir, --c4-model-dir, or --c7-model-dir"
            )
        if args.c7_model_dir is not None and not args.live_pair:
            raise ValueError("--c7-model-dir requires at least one explicit --live-pair")
        if any(":" not in pair or pair.count(":") != 1 for pair in args.live_pair):
            raise ValueError("--live-pair must use ACTOR:ACTOR")
    if args.object_model is not None and not args.causal_live:
        raise ValueError("--object-model requires --causal-live")
    if args.object_model is not None and not args.object_model.is_file():
        raise FileNotFoundError(f"Object detector weights were not found: {args.object_model}")
    if not args.no_save_annotated:
        output_suffix = args.output.suffix.lower()
        allowed = IMAGE_EXTENSIONS if kind == "image" else VIDEO_EXTENSIONS
        if output_suffix not in allowed:
            expected = "image" if kind == "image" else "video"
            raise ValueError(f"--output must have a supported {expected} extension")


def create_object_detector(args: argparse.Namespace) -> Any | None:
    """Load the optional three-class detector for pose-gated subtype evidence."""
    if args.object_model is None:
        return None
    from ultralytics import YOLO
    from backend.ai_services.object_detect.object_detect import ObjectDetectModule
    from backend.ai_services.object_detect.tiled_inference import build_sahi_model
    import torch

    model = YOLO(str(args.object_model.resolve()))
    device = args.object_device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif str(device).isdigit():
        device = f"cuda:{device}"
    # This is the production runtime path: pose gate first, then real SAHI
    # slicing on the current frame. Do not silently fall back to full-frame.
    sahi_model = build_sahi_model(
        args.object_model.resolve(),
        imgsz=1280,
        conf=0.15,
        device=device,
    )
    return ObjectDetectModule(
        model=model,
        device=device,
        enable_smartphone_fallback=False,
        detect_every_n_frames=1,
        # One pose-qualified, actor-owned positive frame is sufficient for
        # this specialist to promote C1/C4; persistence belongs to pose state,
        # not to the object detector's alert decision.
        confirm_frames_by_class={"phone": 1},
        paper_class_names={"baseline_paper", "cheating_paper"},
        flagged_classes={"phone"},
        paper_confidence_threshold=0.15,
        inference_size=1280,
        sahi_model=sahi_model,
    )

def create_tracking(args: argparse.Namespace, session_id: str) -> PersonTrackingModule:
    return PersonTrackingModule(
        PersonTrackingConfig(
            model_path=args.model,
            session_id=session_id,
            restore_session=args.session_id is not None,
            confidence_threshold=args.confidence,
            device=args.device,
            max_tracks=args.max_tracks,
            min_iou=args.min_iou,
            max_missed_frames=args.max_missed_frames,
            bbox_smoothing_alpha=args.bbox_smoothing_alpha,
        )
    )


def auto_assign_students(
    tracking: PersonTrackingModule,
    packet: TrackPacket,
    student_prefix: str,
) -> TrackPacket:
    """Assign deterministic student IDs to visible tracks in track order."""

    for track in sorted(packet.tracks, key=lambda item: item.track_id):
        if not track.is_present or track.student_id:
            continue
        packet = tracking.manager.assign_student(
            tracking.config.session_id,
            track_id=track.track_id,
            student_id=f"{student_prefix}{track.track_id:02d}",
        )
    return packet


def write_landmark_record(
    writer: LandmarkJsonWriter | None,
    *,
    source_frame_index: int,
    packet: TrackPacket,
    results: tuple[TrackHolisticResult, ...],
    classifications: dict[str, dict[str, Any]] | None = None,
) -> None:
    if writer is None:
        return
    record = {
        "source_frame_index": source_frame_index,
        "frame_id": packet.frame_id,
        "timestamp_ms": packet.timestamp_ms,
        "tracks": [result.to_dict() for result in results],
    }
    if classifications is not None:
        record["causal_actor_classifications"] = classifications
    writer.write(record)


def annotate_frame(
    frame: Any,
    *,
    tracking: PersonTrackingModule,
    holistic: HolisticLandmarkExtractor,
    packet: TrackPacket,
    results: tuple[TrackHolisticResult, ...],
    source_frame_index: int,
    inference_ms: float,
    sampled_fps: float | None,
    classifications: dict[str, dict[str, Any]] | None = None,
) -> Any:
    holistic.draw_results(frame, results)
    tracking.draw_tracks(frame, packet)

    import cv2

    if classifications:
        labels = {
            "c1": "Use of Cell", "c2": "Exchange", "c3": "Looking",
            "c4": "Using Cheat", "c5": "Normal",
            "suspicious_activity": "Suspicious Activity",
        }
        for result in results:
            actor_id = result.student_id or f"student_{result.track_id:02d}"
            classification = classifications.get(actor_id)
            if not classification:
                continue
            bbox = result.bbox.to_list()
            if len(bbox) != 4:
                continue
            x1, y1, _, y2 = [round(float(value)) for value in bbox]
            predicted = str(classification.get("predicted_class", "c5"))
            score = classification.get(
                "evidence_score", classification.get(f"{predicted}_score", "")
            )
            score_text = f" {float(score):.2f}" if score not in ("", None) else ""
            text = f"{actor_id}: {labels.get(predicted, predicted)}{score_text}"
            cv2.putText(
                frame, text, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0, 0, 255) if predicted != "c5" else (255, 255, 0), 2,
                cv2.LINE_AA,
            )

    visible_count = sum(track.is_present for track in packet.tracks)
    sample_text = (
        f" | sample {sampled_fps:g} FPS" if sampled_fps is not None else ""
    )
    cv2.putText(
        frame,
        (
            f"Frame {source_frame_index} | tracks {visible_count} | "
            f"poses {sum(result.has_pose for result in results)} | "
            f"inference {inference_ms:.0f} ms{sample_text}"
        ),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def process_one_frame(
    frame: Any,
    *,
    source_frame_index: int,
    timestamp_ms: int,
    tracking: PersonTrackingModule,
    holistic: HolisticLandmarkExtractor,
    student_prefix: str,
    landmark_writer: LandmarkJsonWriter | None,
    sampled_fps: float | None,
    live_classifier: Any | None = None,
    object_rows_by_actor: dict[str, dict[str, Any]] | None = None,
    object_detector: Any | None = None,
    object_session_id: str | None = None,
) -> tuple[Any, TrackPacket]:
    started_at = monotonic()
    packet = tracking.process_frame(frame, timestamp_ms=timestamp_ms)
    packet = auto_assign_students(
        tracking,
        packet,
        student_prefix,
    )
    # Pose is the first-stage gate.  It must be evaluated on this current
    # frame before the expensive object detector is allowed to run.
    results = holistic.process_packet(frame, packet)
    object_rows_by_actor = {}
    classifications = (
        live_classifier.update(
            frame_index=source_frame_index,
            timestamp_ms=timestamp_ms,
            results=results,
        )
        if live_classifier is not None
        else None
    )
    pose_gate = any(
        bool(classification.get("pose_gate", False))
        for classification in (classifications or {}).values()
    )
    if object_detector is not None:
        from ...object_cues import object_row

        people = [
            track
            for track in packet.to_dict().get("tracks", [])
            if track.get("is_present", True)
        ]
        person_rois = [
            {
                "bbox_xyxy": track.get("bbox_xyxy", []),
                "track_id": track.get("track_id"),
                "person_id": track.get("student_id"),
            }
            for track in people
        ]
        detector_result = object_detector.process(
            frame,
            object_session_id or "live",
            source_frame_index,
            person_rois=person_rois,
            pose_suspicious_activity=pose_gate,
        )
        if detector_result is not None:
            object_payload = {
                "object_result": detector_result,
                "raw_objects": detector_result.get("raw_objects", []),
                "people": people,
                "papers": detector_result.get("papers", []),
                "alerts": detector_result.get("alerts", []),
            }
            for track in people:
                actor_id = str(
                    track.get("student_id")
                    or f"{student_prefix}{int(track['track_id']):02d}"
                )
                object_rows_by_actor[actor_id] = object_row(
                    track,
                    object_payload,
                    object_payload["papers"],
                )
    if live_classifier is not None and object_rows_by_actor:
        apply_object_evidence = getattr(
            live_classifier,
            "apply_object_evidence",
            None,
        )
        if apply_object_evidence is not None:
            apply_object_evidence(
                frame_index=source_frame_index,
                timestamp_ms=timestamp_ms,
                object_rows_by_actor=object_rows_by_actor,
            )
            current_decisions = live_classifier.final_decisions()
            classifications = {
                actor_id: {
                    **(classifications or {}).get(actor_id, {}),
                    **decision,
                }
                for actor_id, decision in current_decisions.items()
            }
    inference_ms = (monotonic() - started_at) * 1000.0

    write_landmark_record(
        landmark_writer,
        source_frame_index=source_frame_index,
        packet=packet,
        results=results,
        classifications=classifications,
    )
    annotated = annotate_frame(
        frame,
        tracking=tracking,
        holistic=holistic,
        packet=packet,
        results=results,
        source_frame_index=source_frame_index,
        inference_ms=inference_ms,
        sampled_fps=sampled_fps,
        classifications=classifications,
    )
    return annotated, packet


def create_live_classifier(args: argparse.Namespace, *, clip_id: str):
    """Build one shared causal classifier for media replay or live capture."""
    from .live_actor import (
        CausalLiveActorClassifier,
        CausalPoseActorClassifier,
        CausalC7ActorClassifier,
        CombinedCausalActorClassifier,
    )

    classifiers = []
    if args.xgboost_model_dir is not None:
        classifiers.append(CausalLiveActorClassifier(
            args.xgboost_model_dir.resolve(),
            clip_id=clip_id,
            student_prefix=args.student_prefix,
            explicit_pairs=[tuple(pair.split(":", 1)) for pair in args.live_pair],
            c3_threshold_override=getattr(args, "c3_threshold_override", None),
        ))
    pose_dirs = {
        class_code: path.resolve()
        for class_code, path in (("c1", args.c1_model_dir), ("c4", args.c4_model_dir))
        if path is not None
    }
    if pose_dirs:
        classifiers.append(CausalPoseActorClassifier(
            pose_dirs, student_prefix=args.student_prefix
        ))
    if args.c7_model_dir is not None:
        classifiers.append(CausalC7ActorClassifier(
            args.c7_model_dir.resolve(),
            student_prefix=args.student_prefix,
            explicit_pairs=[tuple(pair.split(":", 1)) for pair in args.live_pair],
        ))
    return classifiers[0] if len(classifiers) == 1 else CombinedCausalActorClassifier(classifiers)

def open_landmark_writer(
    args: argparse.Namespace,
    *,
    session_id: str,
    media_type: str,
) -> LandmarkJsonWriter | None:
    if args.no_save_landmarks:
        return None
    return LandmarkJsonWriter(
        args.landmarks_output,
        input_path=args.input,
        session_id=session_id,
        media_type=media_type,
    )


def process_image(
    args: argparse.Namespace,
    *,
    tracking: PersonTrackingModule,
    landmark_writer: LandmarkJsonWriter | None,
    object_detector: Any | None = None,
    object_session_id: str | None = None,
) -> None:
    import cv2

    frame = cv2.imread(str(args.input))
    if frame is None:
        raise RuntimeError(f"OpenCV could not decode image: {args.input}")

    with HolisticLandmarkExtractor(
        static_image_mode=True,
        model_complexity=args.model_complexity,
        smooth_landmarks=False,
        min_detection_confidence=args.holistic_confidence,
        min_tracking_confidence=args.holistic_confidence,
        soft_landmark_confidence=args.soft_landmark_confidence,
        crop_padding=args.crop_padding,
        crop_stabilization_alpha=args.crop_stabilization_alpha,
        face_hold_frames=args.face_hold_frames,
        face_fallback_model_path=args.face_fallback_model,
        task_model_path=args.holistic_model,
        task_input_size=args.holistic_input_size,
    ) as holistic:
        annotated, _ = process_one_frame(
            frame,
            source_frame_index=0,
            timestamp_ms=0,
            tracking=tracking,
            holistic=holistic,
            student_prefix=args.student_prefix,
            landmark_writer=landmark_writer,
            sampled_fps=None,
            object_detector=object_detector,
            object_session_id=object_session_id,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise RuntimeError(f"Could not write annotated image: {args.output}")

    if args.display:
        cv2.imshow(WINDOW_NAME, annotated)
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def video_writer(path: Path, *, fps: float, width: int, height: int) -> Any:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    codec = "XVID" if path.suffix.lower() == ".avi" else "mp4v"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open annotated video for writing: {path}")
    return writer


def process_video_from_json(args: argparse.Namespace) -> int:
    """Render JSON landmarks while preserving the test_media entry point."""
    import cv2
    from ..landmark.landmarks import (
        HAND_CONNECTIONS,
        POSE_CONNECTIONS,
        SELECTED_FACE_CONNECTIONS,
    )

    payload = json.loads(args.landmarks_input.read_text(encoding="utf-8"))
    records = {int(item["source_frame_index"]): item for item in payload["frames"]}
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {args.input}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    timestamps = [
        float(item.get("timestamp_ms", 0.0))
        for item in payload.get("frames", [])
    ]
    intervals = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
        if current > previous
    ]
    fps = 1000.0 / median(intervals) if intervals else source_fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = video_writer(args.output, fps=fps, width=width, height=height)
    classifications = {}
    live_classifier = (
        create_live_classifier(args, clip_id=args.input.stem)
        if args.causal_live else None
    )
    if getattr(args, "classification_input", None):
        classifications = json.loads(
            args.classification_input.read_text(encoding="utf-8")
        )
    frame_index = 0
    rendered_count = 0
    total_frames = len(records)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            record = records.get(frame_index, {})
            if not record:
                frame_index += 1
                continue
            if live_classifier is not None:
                classifications = live_classifier.update_tracks(
                    frame_index=frame_index,
                    timestamp_ms=int(float(record.get("timestamp_ms", frame_index * 1000.0 / fps))),
                    tracks=record.get("tracks", []),
                )
            frame_classifications = []
            for track in record.get("tracks", []):
                track_id = int(track["track_id"])
                color = (0, 220, 0) if track_id == 1 else (0, 165, 255)
                bbox = track.get("bbox_xyxy", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(round, bbox)
                    classification = (
                        classifications.get(str(track_id))
                        or classifications.get(str(track.get("student_id")))
                    )
                    if classification:
                        activation = classification.get(
                            "first_flag_frame_index",
                            classification.get("activation_frame", 0),
                        )
                        active = activation not in ("", None) and frame_index >= int(activation)
                        predicted_class = classification["predicted_class"] if active else "c5"
                        score = classification.get(
                            "evidence_score",
                            classification.get(f"{predicted_class}_score"),
                        )
                        score_percent = (
                            100 if active and classification.get("pair_c2_event") else
                            round(float(score) * 100.0) if score not in (None, "") else 0
                        )
                        frame_classifications.append((classification, active))
                        label = {
                            "c1": "Use of Cell",
                            "c2": "Exchange Exam P",
                            "c3": "Looking at",
                            "c4": "Using Cheat",
                            "c7": "Using Cheat",
                            "c5": "Not Cheating",
                            "suspicious_activity": "Suspicious Activity",
                        }.get(predicted_class, predicted_class)
                        tag_text = (
                            f"P{track_id}: {label} {score_percent}%"
                        )
                        (text_width, text_height), _ = cv2.getTextSize(
                            tag_text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.72,
                            2,
                        )
                        tag_x = x1 + 18
                        tag_y = max(text_height + 8, y1 + 34)
                        cv2.rectangle(
                            frame,
                            (tag_x - 4, tag_y - text_height - 8),
                            (tag_x + text_width + 6, tag_y + 5),
                            (0, 0, 0),
                            -1,
                        )
                        cv2.putText(
                            frame,
                            tag_text,
                            (tag_x, min(height - 12, tag_y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.72,
                            (0, 165, 255) if active else (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                def draw_set(field, connections, line_color, point_color, thickness, radius):
                    points = {int(p["index"]): p for p in track.get(field, [])}
                    valid = {i: p for i, p in points.items()
                             if p.get("frame_x") is not None and p.get("frame_y") is not None}
                    for a, b in connections:
                        pa, pb = valid.get(a), valid.get(b)
                        if pa and pb:
                            cv2.line(frame, (round(pa["frame_x"]), round(pa["frame_y"])),
                                     (round(pb["frame_x"]), round(pb["frame_y"])),
                                     line_color, thickness, cv2.LINE_AA)
                    for point in valid.values():
                        cv2.circle(frame, (round(point["frame_x"]), round(point["frame_y"])),
                                   radius, point_color, -1, cv2.LINE_AA)
                draw_set("left_hand_landmarks", HAND_CONNECTIONS, (255, 80, 80), (255, 220, 80), 2, 2)
                draw_set("right_hand_landmarks", HAND_CONNECTIONS, (80, 80, 255), (80, 220, 255), 2, 2)
                draw_set("pose_landmarks", POSE_CONNECTIONS, (80, 220, 80), (0, 80, 255), 2, 2)
                draw_set("selected_face_landmarks", SELECTED_FACE_CONNECTIONS, (255, 180, 0), (0, 255, 255), 1, 1)
                pose = {int(p["index"]): p for p in track.get("pose_landmarks", [])}
                for pose_index, hand_field, bridge_color in (
                    (15, "left_hand_landmarks", (255, 80, 80)),
                    (16, "right_hand_landmarks", (80, 80, 255)),
                ):
                    wrist = pose.get(pose_index)
                    hand_wrist = next(
                        (p for p in track.get(hand_field, []) if int(p["index"]) == 0),
                        None,
                    )
                    if (wrist and hand_wrist and wrist.get("frame_x") is not None
                            and hand_wrist.get("frame_x") is not None):
                        cv2.line(
                            frame,
                            (round(wrist["frame_x"]), round(wrist["frame_y"])),
                            (round(hand_wrist["frame_x"]), round(hand_wrist["frame_y"])),
                            bridge_color,
                            2,
                            cv2.LINE_AA,
                        )
            active_cheating = any(active for _, active in frame_classifications)
            if active_cheating:
                cv2.rectangle(frame, (3, 3), (width - 4, height - 4), (0, 0, 255), 5)
                alert = "CHEATING DETECTED"
                (alert_width, alert_height), _ = cv2.getTextSize(
                    alert, cv2.FONT_HERSHEY_SIMPLEX, 1.35, 3
                )
                cv2.putText(
                    frame,
                    alert,
                    ((width - alert_width) // 2, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.35,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                f"YOLOv8 + MediaPipe Holistic | Frame {rendered_count}/{total_frames}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            bar_top = height - 42
            cv2.rectangle(frame, (0, bar_top), (width, height), (25, 25, 25), -1)
            legend = (
                ("Use of Cell", (0, 0, 255)),
                ("Using Cheat", (0, 220, 0)),
                ("Exchange Exam P", (0, 165, 255)),
                ("Looking at", (0, 255, 255)),
                ("Not Cheating", (255, 0, 0)),
            )
            slot = max(1, width // len(legend))
            for index, (legend_text, legend_color) in enumerate(legend):
                cv2.putText(
                    frame,
                    legend_text,
                    (index * slot + 8, height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    legend_color,
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            rendered_count += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    if rendered_count != len(records):
        raise RuntimeError(f"Rendered {rendered_count} frames; JSON has {len(records)}")
    if live_classifier is not None:
        classification_path = args.output.with_name(
            f"{args.output.stem}_live_actor_classification.json"
        )
        classification_path.write_text(
            json.dumps(live_classifier.final_decisions(), indent=2),
            encoding="utf-8",
        )
        print(f"Causal live actor classifications: {classification_path}")
    return rendered_count


def process_video(
    args: argparse.Namespace,
    *,
    tracking: PersonTrackingModule,
    landmark_writer: LandmarkJsonWriter | None,
    live_classifier: Any | None = None,
    object_detector: Any | None = None,
    object_session_id: str | None = None,
) -> int:
    import cv2

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video: {args.input}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0.0:
        source_fps = 30.0
        print("Warning: source FPS is unavailable; using 30 FPS for timestamps.")
    sampled_fps = (
        source_fps
        if args.target_fps == 0
        else min(source_fps, float(args.target_fps))
    )
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = (
        video_writer(args.output, fps=sampled_fps, width=width, height=height)
        if not args.no_save_annotated
        else None
    )

    source_frame_index = -1
    processed_frames = 0
    next_sample_time = 0.0
    stopped_by_user = False

    try:
        with HolisticLandmarkExtractor(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=args.holistic_confidence,
            min_tracking_confidence=args.holistic_confidence,
            soft_landmark_confidence=args.soft_landmark_confidence,
            crop_padding=args.crop_padding,
            crop_stabilization_alpha=args.crop_stabilization_alpha,
            face_hold_frames=args.face_hold_frames,
            face_fallback_model_path=args.face_fallback_model,
            task_model_path=args.holistic_model,
            task_input_size=args.holistic_input_size,
        ) as holistic:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                source_frame_index += 1
                source_time = source_frame_index / source_fps

                if source_time + 1e-9 < next_sample_time:
                    continue
                next_sample_time += 1.0 / sampled_fps

                annotated, _ = process_one_frame(
                    frame,
                    source_frame_index=source_frame_index,
                    timestamp_ms=round(source_time * 1000.0),
                    tracking=tracking,
                    holistic=holistic,
                    student_prefix=args.student_prefix,
                    landmark_writer=landmark_writer,
                    sampled_fps=sampled_fps,
                    live_classifier=live_classifier,
                    object_detector=object_detector,
                    object_session_id=object_session_id,
                )
                if writer is not None:
                    writer.write(annotated)
                processed_frames += 1

                if args.display:
                    cv2.imshow(WINDOW_NAME, annotated)
                    delay_ms = (
                        max(1, round(1000.0 / sampled_fps))
                        if args.realtime
                        else 1
                    )
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key in (ord("q"), 27):
                        stopped_by_user = True
                        break
                    if key == ord(" "):
                        while True:
                            pause_key = cv2.waitKey(0) & 0xFF
                            if pause_key in (ord("q"), 27):
                                stopped_by_user = True
                                break
                            if pause_key == ord(" "):
                                break
                        if stopped_by_user:
                            break

                if args.max_frames and processed_frames >= args.max_frames:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if processed_frames == 0:
        raise RuntimeError(f"No video frames could be decoded from: {args.input}")
    return processed_frames


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    kind = media_kind(args.input)
    args.output = (
        args.output.resolve()
        if args.output is not None
        else default_output_path(args.input, kind).resolve()
    )
    args.landmarks_output = (
        args.landmarks_output.resolve()
        if args.landmarks_output is not None
        else args.output.with_name(f"{args.output.stem}_landmarks.json")
    )
    validate_args(args, kind)

    if args.landmarks_input is not None:
        args.landmarks_input = args.landmarks_input.resolve()
        if not args.landmarks_input.is_file():
            raise FileNotFoundError(f"Landmark JSON was not found: {args.landmarks_input}")
        if kind != "video":
            raise ValueError("--landmarks-input requires a video input")
        processed_frames = process_video_from_json(args)
        print(f"Processed frames: {processed_frames}")
        print(f"Annotated video: {args.output}")
        print(f"Source JSON: {args.landmarks_input}")
        return

    session_id = safe_session_id(args.input, args.session_id)
    tracking = create_tracking(args, session_id)
    object_detector = create_object_detector(args)
    live_classifier = None
    if args.causal_live:
        live_classifier = create_live_classifier(args, clip_id=args.input.stem)
    landmark_writer = open_landmark_writer(
        args,
        session_id=session_id,
        media_type=kind,
    )
    processed_frames = 1

    try:
        if kind == "image":
            process_image(
                args,
                tracking=tracking,
                landmark_writer=landmark_writer,
                object_detector=object_detector,
                object_session_id=session_id,
            )
        else:
            processed_frames = process_video(
                args,
                tracking=tracking,
                landmark_writer=landmark_writer,
                live_classifier=live_classifier,
                object_detector=object_detector,
                object_session_id=session_id,
            )
    finally:
        if landmark_writer is not None:
            landmark_writer.close()
        if object_detector is not None:
            object_detector.cleanup_session(session_id)
        tracking_json = tracking.manager.generate_final_output(session_id)

    if live_classifier is not None:
        classification_path = args.output.with_name(
            f"{args.output.stem}_live_actor_classification.json"
        )
        classification_path.write_text(
            json.dumps(live_classifier.final_decisions(), indent=2),
            encoding="utf-8",
        )
        print(f"Causal live actor classifications: {classification_path}")
    elif args.xgboost_model_dir is not None and kind == "video":
        from .actor_xgboost import classify_landmarks

        classification_path = args.output.with_name(
            f"{args.output.stem}_actor_classification.json"
        )
        classify_landmarks(
            args.landmarks_output,
            args.xgboost_model_dir.resolve(),
            classification_path,
        )
        args.classification_input = classification_path
        # Reuse test_media JSON renderer, preserving exact landmark colors and
        # connections. Only actor classification text is added.
        args.landmarks_input = args.landmarks_output
        args.display = False
        process_video_from_json(args)
        print(f"Actor classifications: {classification_path}")

    print(f"Processed frames: {processed_frames}")
    print(f"Annotated {kind}: {args.output}")
    if not args.no_save_landmarks:
        print(f"Landmarks JSON: {args.landmarks_output}")
    print(f"Tracking JSON: {tracking_json.resolve()}")
    print(f"Session ID: {session_id}")


if __name__ == "__main__":
    main()

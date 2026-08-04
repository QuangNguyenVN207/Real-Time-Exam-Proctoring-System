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
from pathlib import Path
from time import monotonic
from typing import Any, TextIO
from uuid import uuid4

from backend.ai_services.pose_gaze.holistic.landmark import (
    HolisticLandmarkExtractor,
    TrackHolisticResult,
)
from backend.ai_services.pose_gaze.tracking.schemas import TrackPacket
from backend.ai_services.pose_gaze.settings import (
    DEFAULT_HOLISTIC_CONFIDENCE,
    DEFAULT_HOLISTIC_SOFT_CONFIDENCE,
    DEFAULT_MAX_MISSED_FRAMES,
    DEFAULT_MIN_IOU,
    DEFAULT_PERSON_CONFIDENCE,
    PROJECT_ROOT,
)
from backend.ai_services.pose_gaze.tracking.webcam import (
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
            "format_version": 2,
            "input": str(input_path),
            "session_id": session_id,
            "media_type": media_type,
            "student_id_strategy": "student_<track_id>",
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
        "--no-save-landmarks",
        action="store_true",
        help="Do not write per-frame pose/hand/selected-face landmarks",
    )
    parser.add_argument("--model", type=Path, default=None, help="YOLO weights")
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
        help="Keep predicted x/y above this score but set visibility to null",
    )
    parser.add_argument("--crop-padding", type=float, default=0.15)
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

    output_suffix = args.output.suffix.lower()
    allowed = IMAGE_EXTENSIONS if kind == "image" else VIDEO_EXTENSIONS
    if output_suffix not in allowed:
        expected = "image" if kind == "image" else "video"
        raise ValueError(f"--output must have a supported {expected} extension")


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
) -> None:
    if writer is None:
        return
    record = {
        "source_frame_index": source_frame_index,
        "frame_id": packet.frame_id,
        "timestamp_ms": packet.timestamp_ms,
        "tracks": [result.to_dict() for result in results],
    }
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
) -> Any:
    holistic.draw_results(frame, results)
    tracking.draw_tracks(frame, packet)

    import cv2

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
) -> tuple[Any, TrackPacket]:
    started_at = monotonic()
    packet = tracking.process_frame(frame, timestamp_ms=timestamp_ms)
    packet = auto_assign_students(
        tracking,
        packet,
        student_prefix,
    )
    results = holistic.process_packet(frame, packet)
    inference_ms = (monotonic() - started_at) * 1000.0

    write_landmark_record(
        landmark_writer,
        source_frame_index=source_frame_index,
        packet=packet,
        results=results,
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
    )
    return annotated, packet


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


def process_video(
    args: argparse.Namespace,
    *,
    tracking: PersonTrackingModule,
    landmark_writer: LandmarkJsonWriter | None,
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
    writer = video_writer(
        args.output,
        fps=sampled_fps,
        width=width,
        height=height,
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
                )
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

    session_id = safe_session_id(args.input, args.session_id)
    tracking = create_tracking(args, session_id)
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
            )
        else:
            processed_frames = process_video(
                args,
                tracking=tracking,
                landmark_writer=landmark_writer,
            )
    finally:
        if landmark_writer is not None:
            landmark_writer.close()
        tracking_json = tracking.manager.generate_final_output(session_id)

    print(f"Processed frames: {processed_frames}")
    print(f"Annotated {kind}: {args.output}")
    if not args.no_save_landmarks:
        print(f"Landmarks JSON: {args.landmarks_output}")
    print(f"Tracking JSON: {tracking_json.resolve()}")
    print(f"Session ID: {session_id}")


if __name__ == "__main__":
    main()

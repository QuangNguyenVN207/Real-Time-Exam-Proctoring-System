"""One-person webcam demo: actor detection plus live head-turn indicator.

This is a demo/control session. It cannot prove directed C3 because no peer
exists. It fails when no actor is detected in any processed frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from ..landmark import HolisticLandmarkExtractor
from ...settings import (
    DEFAULT_HOLISTIC_CONFIDENCE,
    DEFAULT_HOLISTIC_SOFT_CONFIDENCE,
    DEFAULT_MAX_MISSED_FRAMES,
    DEFAULT_MIN_IOU,
    DEFAULT_PERSON_CONFIDENCE,
    PROJECT_ROOT,
)
from ...tracking.manager import TrackingManager
from ...tracking.webcam import PersonTrackingConfig, PersonTrackingModule, ProcessingRateController
from ...tracking.interactive import WebcamInteractionController, pump_keyboard_until_frame_deadline
from ..debug.session_manifest import SessionManifestRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence", type=float, default=DEFAULT_PERSON_CONFIDENCE)
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--min-iou", type=float, default=DEFAULT_MIN_IOU)
    parser.add_argument("--max-missed-frames", type=int, default=DEFAULT_MAX_MISSED_FRAMES)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--holistic-model", type=Path, default=None)
    parser.add_argument("--holistic-input-size", type=int, default=512)
    parser.add_argument("--holistic-confidence", type=float, default=DEFAULT_HOLISTIC_CONFIDENCE)
    parser.add_argument("--soft-landmark-confidence", type=float, default=DEFAULT_HOLISTIC_SOFT_CONFIDENCE)
    parser.add_argument("--crop-padding", type=float, default=0.15)
    parser.add_argument("--session-id", default=None)
    return parser.parse_args()


def _head_turn_score(result) -> float | None:
    if not result.pose_landmarks or len(result.pose_landmarks) <= 12:
        return None
    nose = result.pose_landmarks[0]
    left = result.pose_landmarks[11]
    right = result.pose_landmarks[12]
    if any(point.frame_x is None for point in (nose, left, right)):
        return None
    shoulder_mid = (float(left.frame_x) + float(right.frame_x)) / 2.0
    shoulder_width = abs(float(left.frame_x) - float(right.frame_x))
    if shoulder_width <= 1.0:
        return None
    return (float(nose.frame_x) - shoulder_mid) / shoulder_width


def main() -> None:
    args = parse_args()
    import cv2

    session_id = args.session_id or TrackingManager.generate_session_id("webcam_one_person")
    output_dir = PROJECT_ROOT / "test_data_tracking" / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "camera.mp4"
    trace_path = output_dir / "one_person_trace.jsonl"
    manifest_path = output_dir / "session_manifest.json"

    tracking = PersonTrackingModule(PersonTrackingConfig(
        model_path=args.model,
        session_id=session_id,
        confidence_threshold=args.confidence,
        device=args.device,
        max_tracks=1,
        min_iou=args.min_iou,
        max_missed_frames=args.max_missed_frames,
    ))
    capture = tracking.open_webcam(args.camera, width=args.width, height=args.height)
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS) or args.target_fps)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
        actual_fps if actual_fps > 0 else float(args.target_fps),
        (actual_width, actual_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video writer: {video_path}")

    recorder = SessionManifestRecorder(
        session_id=session_id,
        working_directory=PROJECT_ROOT,
        model_dir=PROJECT_ROOT / "tmp" / "behavior_actor_extended_suspicious_current_geometry_20260815",
        runtime_arguments={
            "camera": args.camera,
            "target_fps": args.target_fps,
            "max_tracks": 1,
            "confidence": args.confidence,
            "demo": "one_person_head_turn",
        },
        camera_config={
            "index": args.camera,
            "width": actual_width,
            "height": actual_height,
            "target_fps": args.target_fps,
            "reported_camera_fps": actual_fps,
        },
        video_path=video_path,
        trace_path=trace_path,
    )
    rate = ProcessingRateController(args.target_fps)
    interaction = WebcamInteractionController(manager=tracking.manager, session_id=session_id, rate=rate)
    detected_frames = 0
    trace_records = []
    previous_frame_mono = None

    try:
        with HolisticLandmarkExtractor(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=args.holistic_confidence,
            min_tracking_confidence=args.holistic_confidence,
            soft_landmark_confidence=args.soft_landmark_confidence,
            crop_padding=args.crop_padding,
            task_model_path=args.holistic_model,
            task_input_size=args.holistic_input_size,
        ) as holistic:
            while True:
                frame_started = monotonic()
                ok, frame = capture.read()
                if not ok:
                    break
                packet = tracking.process_frame(frame)
                for track in packet.tracks:
                    if track.is_present and not track.student_id:
                        packet = tracking.manager.assign_student(
                            session_id, track_id=track.track_id, student_id="student_01"
                        )
                results = holistic.process_packet(frame, packet)
                inference_ms = (monotonic() - frame_started) * 1000.0
                recorder.record_frame_latency(inference_ms)
                present = next((item for item in results if item.track_id is not None), None)
                if present is not None:
                    detected_frames += 1
                score = _head_turn_score(present) if present is not None else None
                turn = score is not None and abs(score) >= 0.15
                c3_flagged = present is not None
                record = {
                    "timestamp_ms": int(packet.timestamp_ms),
                    "frame_index": int(packet.frame_id),
                    "inter_frame_duration_ms": round(
                        0.0 if previous_frame_mono is None else (frame_started - previous_frame_mono) * 1000.0, 2
                    ),
                    "actor_id": "student_01" if present is not None else None,
                    "actor_detected": present is not None,
                    "actor_bbox": list(present.bbox.to_list()) if present is not None and present.bbox else None,
                    "pose_valid": bool(present and present.pose_landmarks),
                    "head_turn_score": score,
                    "head_turn_detected": bool(turn),
                    "predicted_class": "c3" if c3_flagged else "unknown",
                    "c3_flagged": c3_flagged,
                }
                trace_records.append(record)
                previous_frame_mono = frame_started
                label = "C3 FLAG" if c3_flagged else "NO ACTOR"
                cv2.putText(frame, f"{label} | one-person demo", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                holistic.draw_results(frame, results)
                tracking.draw_tracks(frame, packet)
                writer.write(frame)
                cv2.imshow("Exam Proctoring - One Person Head Turn", frame)
                pump_keyboard_until_frame_deadline(
                    cv2, frame_started_at=frame_started, rate=rate, interaction=interaction
                )
                if interaction.quit_requested:
                    break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()
        trace_path.write_text("".join(json.dumps(item) + "\n" for item in trace_records), encoding="utf-8")
        manifest = recorder.build_manifest()
        manifest.save(manifest_path)
        print(f"Video: {video_path.resolve()}")
        print(f"Trace: {trace_path.resolve()}")
        print(f"Manifest: {manifest_path.resolve()}")
        if detected_frames == 0:
            raise RuntimeError("One-person demo failed: no actor detected in any processed frame")


if __name__ == "__main__":
    main()

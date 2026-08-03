"""Realtime webcam test for person tracking plus MediaPipe Holistic."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import monotonic

# MediaPipe's native library otherwise prints repeated telemetry and model
# diagnostics. Python exceptions are still reported by this application.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from backend.ai_services.pose_gaze.holistic_landmarks import (
    HolisticLandmarkExtractor,
)
from backend.ai_services.pose_gaze.tracking.interactive import (
    WebcamInteractionController,
    pump_keyboard_until_frame_deadline,
)
from backend.ai_services.pose_gaze.tracking.manager import TrackingManager
from backend.ai_services.pose_gaze.tracking.webcam import (
    PersonTrackingConfig,
    PersonTrackingModule,
    ProcessingRateController,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--device", default=None, help="Ultralytics device: cpu, 0, ...")
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--max-tracks", type=int, default=2)
    parser.add_argument(
        "--session-id",
        default=None,
        help="Restore this saved session; omit to start a fresh test session",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Legacy MediaPipe only; current Tasks API uses a fixed model",
    )
    parser.add_argument(
        "--holistic-model",
        type=Path,
        default=None,
        help=(
            "Path to holistic_landmarker.task. Current MediaPipe downloads the "
            "official model to weights/mediapipe on first run when omitted."
        ),
    )
    parser.add_argument(
        "--holistic-input-size",
        type=int,
        default=512,
        help="Fixed square MediaPipe Tasks input; lower to 384 for a weaker CPU",
    )
    parser.add_argument("--crop-padding", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_id = args.session_id or TrackingManager.generate_session_id(
        "webcam_holistic"
    )
    mode = "restoring" if args.session_id is not None else "fresh"
    print(f"Session ID ({mode}): {session_id}")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Install opencv-python to run this webcam test") from error

    tracking = PersonTrackingModule(
        PersonTrackingConfig(
            model_path=args.model,
            session_id=session_id,
            restore_session=args.session_id is not None,
            confidence_threshold=args.confidence,
            device=args.device,
            max_tracks=args.max_tracks,
        )
    )
    capture = tracking.open_webcam(
        args.camera,
        width=args.width,
        height=args.height,
    )
    rate = ProcessingRateController(args.target_fps)
    interaction = WebcamInteractionController(
        manager=tracking.manager,
        session_id=session_id,
        rate=rate,
    )
    latest_packet = tracking.manager.get_packet(session_id)
    latest_holistic_results = ()
    inference_ms = 0.0

    try:
        with HolisticLandmarkExtractor(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            crop_padding=args.crop_padding,
            task_model_path=args.holistic_model,
            task_input_size=args.holistic_input_size,
        ) as holistic:
            while True:
                frame_started_at = monotonic()
                ok, frame = capture.read()
                if not ok:
                    break

                inference_started_at = monotonic()
                latest_packet = tracking.process_frame(frame)
                latest_holistic_results = holistic.process_packet(
                    frame,
                    latest_packet,
                )
                inference_finished_at = monotonic()
                inference_ms = (
                    inference_finished_at - inference_started_at
                ) * 1000.0
                rate.mark_processed(frame_started_at, inference_finished_at)
                interaction.update_tracks(latest_packet)
                packet_update = interaction.consume_packet_update()
                if packet_update is not None:
                    latest_packet = packet_update

                holistic.draw_results(frame, latest_holistic_results)
                tracking.draw_tracks(frame, latest_packet)

                for result in latest_holistic_results:
                    if not result.pose_landmarks:
                        continue
                    nose = result.pose_landmarks[0]
                    x1, _, _, y2 = result.bbox.to_list()
                    if nose.frame_x is not None and nose.frame_y is not None:
                        nose_text = (
                            f"T{result.track_id} nose "
                            f"({nose.frame_x:.0f}, {nose.frame_y:.0f})"
                        )
                        cv2.putText(
                            frame,
                            nose_text,
                            (x1, min(frame.shape[0] - 8, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                cv2.putText(
                    frame,
                    (
                        f"Actual FPS {rate.measured_fps:.0f} | "
                        f"Limit {rate.target_fps} FPS | "
                        f"Inference {inference_ms:.0f} ms | MediaPipe CPU"
                    ),
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    interaction.overlay_lines()[0],
                    (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    interaction.overlay_lines()[1],
                    (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("Exam Proctoring - Tracking + Holistic", frame)

                pump_keyboard_until_frame_deadline(
                    cv2,
                    frame_started_at=frame_started_at,
                    rate=rate,
                    interaction=interaction,
                )
                packet_update = interaction.consume_packet_update()
                if packet_update is not None:
                    latest_packet = packet_update
                if interaction.quit_requested:
                    break
    finally:
        output_path = tracking.manager.generate_final_output(session_id)
        capture.release()
        cv2.destroyAllWindows()
        print(f"\nTracking JSON saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

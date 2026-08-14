"""Realtime webcam test for person tracking plus MediaPipe Holistic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from ..landmark import (
    HolisticLandmarkExtractor,
)
from ...tracking.interactive import (
    WebcamInteractionController,
    pump_keyboard_until_frame_deadline,
)
from ...tracking.manager import TrackingManager
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
    ProcessingRateController,
)


DEFAULT_ACTION_ARTIFACTS = {
    "c2c3": PROJECT_ROOT / "tmp" / "behavior_actor_causal_pose_only_20260812",
}
SUPPORTED_ACTIONS = ("c1", "c2", "c3", "c4", "c7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--device", default=None, help="Ultralytics device: cpu, 0, ...")
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_PERSON_CONFIDENCE,
    )
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--max-tracks", type=int, default=2)
    parser.add_argument("--min-iou", type=float, default=DEFAULT_MIN_IOU)
    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=DEFAULT_MAX_MISSED_FRAMES,
    )
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
        "--actions",
        default="c2,c3",
        help="Comma-separated actions to enable; default c2,c3 uses committed benchmark artifact",
    )
    parser.add_argument(
        "--causal-model-dir",
        type=Path,
        default=None,
        help="Causal c2/c3 specialist artifact directory",
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
    parser.add_argument("--student-prefix", default="student_")
    parser.add_argument(
        "--live-pair",
        action="append",
        default=[],
        metavar="ACTOR:ACTOR",
        help="Configured explicit interaction pair for causal c2 propagation",
    )
    return parser.parse_args()


def configure_action_models(args: argparse.Namespace) -> tuple[str, ...]:
    """Enable requested actions with repo defaults unless explicitly overridden."""
    requested = SUPPORTED_ACTIONS if args.actions.strip().lower() == "all" else tuple(
        item.strip().lower() for item in args.actions.split(",") if item.strip()
    )
    unknown = sorted(set(requested) - set(SUPPORTED_ACTIONS))
    if unknown:
        raise ValueError(f"Unsupported --actions values: {','.join(unknown)}")
    enabled = tuple(action for action in SUPPORTED_ACTIONS if action in requested)
    if not enabled:
        raise ValueError("--actions must enable at least one action")
    if {"c2", "c3"} & set(enabled):
        args.causal_model_dir = args.causal_model_dir or DEFAULT_ACTION_ARTIFACTS["c2c3"]
    # Only the locked C2/C3 benchmark artifact is bundled.  Other specialists
    # require an explicit artifact directory instead of silently referencing
    # local, uncommitted experiment outputs.
    if not ({"c2", "c3"} & set(enabled)):
        args.causal_model_dir = None
    if "c1" not in enabled:
        args.c1_model_dir = None
    if "c4" not in enabled:
        args.c4_model_dir = None
    if "c7" not in enabled:
        args.c7_model_dir = None
    missing_specialists = [
        action for action, model_dir in (
            ("c1", args.c1_model_dir), ("c4", args.c4_model_dir),
            ("c7", args.c7_model_dir),
        )
        if action in enabled and model_dir is None
    ]
    if missing_specialists:
        raise ValueError(
            "requested action requires an explicit model directory: "
            + ", ".join(missing_specialists)
        )
    if not args.live_pair and ({"c2", "c7"} & set(enabled)):
        args.live_pair = ["student_01:student_02"]
    missing = [str(path) for path in (
        args.causal_model_dir, args.c1_model_dir, args.c4_model_dir, args.c7_model_dir
    ) if path is not None and not Path(path).is_dir()]
    if missing:
        raise FileNotFoundError("Action artifact directories were not found: " + ", ".join(missing))
    return enabled


def main() -> None:
    args = parse_args()
    enabled_actions = configure_action_models(args)
    if not 0.0 <= args.holistic_confidence <= 1.0:
        raise ValueError("--holistic-confidence must be in [0, 1]")
    if not 0.0 <= args.soft_landmark_confidence <= args.holistic_confidence:
        raise ValueError(
            "--soft-landmark-confidence must be in [0, --holistic-confidence]"
        )
    if any(":" not in pair or pair.count(":") != 1 for pair in args.live_pair):
        raise ValueError("--live-pair must use ACTOR:ACTOR")
    session_id = args.session_id or TrackingManager.generate_session_id(
        "webcam_holistic"
    )
    mode = "restoring" if args.session_id is not None else "fresh"
    print(f"Session ID ({mode}): {session_id}")
    print(f"Actions: {','.join(enabled_actions)}")
    print(f"Explicit pairs: {','.join(args.live_pair) if args.live_pair else 'none'}")
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
            min_iou=args.min_iou,
            max_missed_frames=args.max_missed_frames,
        )
    )
    capture = tracking.open_webcam(
        args.camera,
        width=args.width,
        height=args.height,
    )
    rate = ProcessingRateController(args.target_fps)
    live_classifier = None
    if args.c7_model_dir is not None and not args.live_pair:
        raise ValueError("--c7-model-dir requires at least one explicit --live-pair")
    if any((args.causal_model_dir, args.c1_model_dir, args.c4_model_dir, args.c7_model_dir)):
        from ..test_media.test_media import create_live_classifier

        args.xgboost_model_dir = args.causal_model_dir
        live_classifier = create_live_classifier(
            args, clip_id=f"webcam_{session_id}"
        )
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
            min_detection_confidence=args.holistic_confidence,
            min_tracking_confidence=args.holistic_confidence,
            soft_landmark_confidence=args.soft_landmark_confidence,
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
                classifications = (
                    live_classifier.update(
                        frame_index=latest_packet.frame_id,
                        timestamp_ms=latest_packet.timestamp_ms,
                        results=latest_holistic_results,
                    )
                    if live_classifier is not None
                    else {}
                )
                rate.mark_processed(frame_started_at, inference_finished_at)
                interaction.update_tracks(latest_packet)
                packet_update = interaction.consume_packet_update()
                if packet_update is not None:
                    latest_packet = packet_update

                holistic.draw_results(frame, latest_holistic_results)
                tracking.draw_tracks(frame, latest_packet)

                if classifications:
                    labels = {
                        "c1": "Use of Cell", "c2": "Exchange", "c3": "Looking",
                        "c4": "Using Cheat", "c5": "Normal",
                        "suspicious_activity": "Suspicious Activity",
                    }
                    for result in latest_holistic_results:
                        actor_id = result.student_id or f"student_{result.track_id:02d}"
                        classification = classifications.get(actor_id)
                        if not classification:
                            continue
                        x1, y1, x2, y2 = [round(float(value)) for value in result.bbox.to_list()]
                        frame_height, frame_width = frame.shape[:2]
                        x1 = max(4, min(x1, frame_width - 8))
                        label_y = y1 - 8
                        if label_y < 100:
                            label_y = min(frame_height - 42, y2 + 22)
                        predicted = str(classification["predicted_class"])
                        score = classification.get(
                            "evidence_score",
                            classification.get(f"{predicted}_score", ""),
                        )
                        if score not in ("", None):
                            score_text = f" {float(score):.2f}"
                        else:
                            seen = int(classification.get("warmup_frames_seen", 0))
                            required = int(classification.get("warmup_frames_required", 30))
                            score_text = f" warming {seen}/{required}" if seen < required else " no pose score"
                        cv2.putText(
                            frame,
                            f"{actor_id}: {labels.get(predicted, predicted)}{score_text}",
                            (x1, max(100, label_y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62,
                            (0, 0, 255) if predicted != "c5" else (255, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                for result in latest_holistic_results:
                    if not result.pose_landmarks:
                        continue
                    nose = result.pose_landmarks[0]
                    x1, _, _, y2 = [round(float(value)) for value in result.bbox.to_list()]
                    if nose.frame_x is not None and nose.frame_y is not None:
                        nose_text = (
                            f"T{result.track_id} nose "
                            f"({nose.frame_x:.0f}, {nose.frame_y:.0f})"
                        )
                        nose_y = min(frame.shape[0] - 10, y2 + 42)
                        cv2.putText(
                            frame,
                            nose_text,
                            (max(4, x1), nose_y),
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
                if interaction.consume_classifier_reset():
                    if live_classifier is None:
                        interaction.status_message = "No classifier is enabled"
                    else:
                        live_classifier.reset()
                        interaction.status_message = (
                            "Classifier reset; collecting a new warmup window"
                        )
                        print(f"\n{interaction.status_message}")
                packet_update = interaction.consume_packet_update()
                if packet_update is not None:
                    latest_packet = packet_update
                if interaction.quit_requested:
                    break
    finally:
        output_path = tracking.manager.generate_final_output(session_id)
        if live_classifier is not None:
            classification_path = PROJECT_ROOT / "test_data_tracking" / (
                f"{session_id}_live_actor_classification.json"
            )
            classification_path.parent.mkdir(parents=True, exist_ok=True)
            classification_path.write_text(
                json.dumps(live_classifier.final_decisions(), indent=2),
                encoding="utf-8",
            )
            print(f"Causal live actor classifications saved to: {classification_path.resolve()}")
        capture.release()
        cv2.destroyAllWindows()
        print(f"\nTracking JSON saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

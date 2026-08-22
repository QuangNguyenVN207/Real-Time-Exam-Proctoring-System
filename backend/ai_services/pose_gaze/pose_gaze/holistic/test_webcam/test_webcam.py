"""Realtime webcam test for person tracking plus MediaPipe Holistic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic, time_ns

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
    "extended": PROJECT_ROOT / "tmp" / "benchmark_face_mesh_restored_cuda_snapshot_verify_final_20260820",
}
SUPPORTED_ACTIONS = ("c2", "c3", "c5", "suspicious_activity")

# Lazy imports so webcam still starts without debug deps installed
try:
    from ..debug.frame_trace import FrameTraceLogger, FrameTraceRecord
    from ..debug.events import C3Trial, OcclusionEvent, SessionEvents, TimeInterval, TrackResetEvent
    from ..debug.session_manifest import SessionManifestRecorder
    _DEBUG_AVAILABLE = True
except ImportError:
    _DEBUG_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--device", default="0", help="Ultralytics CUDA device; default: 0")
    parser.add_argument(
        "--xgboost-device",
        default="cuda:0",
        help="XGBoost inference device; default: cuda:0",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_PERSON_CONFIDENCE,
    )
    parser.add_argument("--target-fps", type=int, default=30)
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
        default="c2,c3,suspicious_activity",
        help="Comma-separated actions to enable; default runs C2/C3/C5 plus suspicious_activity",
    )
    parser.add_argument(
        "--causal-model-dir",
        type=Path,
        default=None,
        help="Causal c2/c3 specialist artifact directory",
    )
    parser.add_argument(
        "--c3-threshold-override",
        type=float,
        default=None,
        help="Experimental live-test C3 threshold; does not modify the artifact or benchmark",
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
    if "suspicious_activity" in enabled:
        args.causal_model_dir = args.causal_model_dir or DEFAULT_ACTION_ARTIFACTS["extended"]
    elif {"c2", "c3"} & set(enabled):
        # The promoted extended artifact contains the deployed C2/C3
        # specialists as well as suspicious_activity.
        args.causal_model_dir = args.causal_model_dir or DEFAULT_ACTION_ARTIFACTS["extended"]
    if not ({"c2", "c3", "suspicious_activity"} & set(enabled)):
        args.causal_model_dir = None
    if not args.live_pair and "c2" in enabled:
        args.live_pair = ["student_01:student_02"]
    missing = [str(args.causal_model_dir)] if (
        args.causal_model_dir is not None and not Path(args.causal_model_dir).is_dir()
    ) else []
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
    if args.c3_threshold_override is not None and not 0.0 <= args.c3_threshold_override <= 1.0:
        raise ValueError("--c3-threshold-override must be in [0, 1]")
    if any(":" not in pair or pair.count(":") != 1 for pair in args.live_pair):
        raise ValueError("--live-pair must use ACTOR:ACTOR")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch CUDA is required for YOLO GPU inference") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; YOLO and XGBoost GPU mode cannot start")
    session_id = args.session_id or TrackingManager.generate_session_id(
        "webcam_holistic"
    )
    mode = "restoring" if args.session_id is not None else "fresh"
    print(f"Session ID ({mode}): {session_id}")
    print(f"Actions: {','.join(enabled_actions)}")
    print(f"Explicit pairs: {','.join(args.live_pair) if args.live_pair else 'none'}")
    print(f"YOLO device: {args.device} | XGBoost device: {args.xgboost_device}")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Install opencv-python to run this webcam test") from error

    # Stage A: per-frame trace + session manifest (only when debug available)
    session_output_dir = PROJECT_ROOT / "test_data_tracking" / session_id
    trace_path = session_output_dir / "frames.jsonl"
    manifest_path = session_output_dir / "session_manifest.json"
    events_path = session_output_dir / "events.json"
    video_path = session_output_dir / "camera.mp4"
    _trace_logger = FrameTraceLogger(trace_path) if _DEBUG_AVAILABLE else None
    _manifest_recorder = None
    if _DEBUG_AVAILABLE:
        try:
            _manifest_recorder = SessionManifestRecorder(
                session_id=session_id,
                working_directory=PROJECT_ROOT,
                model_dir=getattr(args, "causal_model_dir", None),
                runtime_arguments={
                    "camera": args.camera,
                    "target_fps": args.target_fps,
                    "max_tracks": args.max_tracks,
                    "holistic_confidence": args.holistic_confidence,
                    "soft_landmark_confidence": args.soft_landmark_confidence,
                    "c3_threshold_override": args.c3_threshold_override,
                    "xgboost_device": args.xgboost_device,
                    "yolo_device": args.device,
                    "actions": ",".join(enabled_actions),
                    "live_pair": list(args.live_pair),
                    "student_prefix": args.student_prefix,
                },
                camera_config={
                    "index": args.camera,
                    "width": args.width,
                    "height": args.height,
                    "target_fps": args.target_fps,
                },
                video_path=video_path,
                trace_path=trace_path,
            )
        except Exception as _exc:
            print(f"[trace] SessionManifestRecorder init failed: {_exc}")
            _manifest_recorder = None
    _prev_frame_mono: float | None = None
    _capture_start_ms: int | None = None
    _active_annotation: tuple[str, int, str] | None = None
    _neutral_start_ms: int | None = None
    _trials = []
    _neutral_intervals = []
    _occlusions = []
    _track_resets = []
    _stage_a_errors: list[str] = []

    def annotate_key(key: int) -> None:
        """Record A3 timestamps; keys are edge annotations, not model input."""
        nonlocal _active_annotation, _neutral_start_ms
        if _capture_start_ms is None:
            return
        ts = int(latest_packet.timestamp_ms)
        if key in (ord("n"), ord("N")):
            if _neutral_start_ms is None:
                _neutral_start_ms = ts
            else:
                _neutral_intervals.append(TimeInterval(_neutral_start_ms, ts, "neutral"))
                _neutral_start_ms = None
        elif key in (ord("1"), ord("2")):
            source = "student_01" if key == ord("1") else "student_02"
            peer = "student_02" if source == "student_01" else "student_01"
            if _active_annotation is None:
                _active_annotation = ("trial", ts, source + ":" + peer)
            elif _active_annotation[0] == "trial":
                _, start, pair = _active_annotation
                if pair == source + ":" + peer:
                    left, right = pair.split(":")
                    _trials.append(C3Trial(len(_trials) + 1, left, right, start, ts))
                    _active_annotation = None
        elif key in (ord("s"), ord("S"), ord("l"), ord("L")):
            kind = "short_occlusion" if key in (ord("s"), ord("S")) else "long_occlusion"
            if _active_annotation is None:
                _active_annotation = (kind, ts, "student_01")
            elif _active_annotation[0] == kind:
                _, start, actor = _active_annotation
                _occlusions.append(OcclusionEvent(actor, start, ts, kind == "long_occlusion"))
                _active_annotation = None
        elif key in (ord("t"), ord("T")):
            _track_resets.append(TrackResetEvent(ts, "student_01", "student_03"))

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
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width or 0)
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height or 0)
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS) or args.target_fps)
    if _manifest_recorder is not None:
        _manifest_recorder.camera_config.update({
            "measured_width": actual_width,
            "measured_height": actual_height,
            "reported_camera_fps": actual_fps,
        })
    video_writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_fps if actual_fps > 0 else float(args.target_fps),
        (actual_width, actual_height),
    )
    if not video_writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera video writer: {video_path}")
    rate = ProcessingRateController(args.target_fps)
    live_classifier = None
    if args.causal_model_dir is not None:
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
                if _capture_start_ms is None:
                    _capture_start_ms = int(latest_packet.timestamp_ms)
                # Live causal classifiers require stable actor IDs.  The old
                # interactive prompt blocks the frame loop before any model
                # can warm up, so assign deterministic demo IDs immediately.
                for track in latest_packet.tracks:
                    if track.is_present and not track.student_id:
                        latest_packet = tracking.manager.assign_student(
                            session_id,
                            track_id=track.track_id,
                            student_id=f"{args.student_prefix}{track.track_id:02d}",
                        )
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

                # Stage A: emit per-frame trace record for each result
                if _trace_logger is not None and _DEBUG_AVAILABLE:
                    inter_ms = (
                        (frame_started_at - _prev_frame_mono) * 1000.0
                        if _prev_frame_mono is not None
                        else 0.0
                    )
                    # Build raw tracks snapshot for replay
                    _tracks_snapshot = [
                        result.to_dict() for result in latest_holistic_results
                    ]
                    # Index by student_id for O_AB / peer lookups
                    _result_by_id = {
                        (result.student_id or f"{args.student_prefix}{result.track_id:02d}"): result
                        for result in latest_holistic_results
                    }
                    for result in latest_holistic_results:
                        actor_id = result.student_id or f"{args.student_prefix}{result.track_id:02d}"
                        clf_dec = classifications.get(actor_id, {})
                        pred_class = clf_dec.get("predicted_class", "c5")
                        c3_score = float(clf_dec.get("c3_score") or 0.0)
                        warmup_seen = int(clf_dec.get("warmup_frames_seen", 0))
                        warmup_req = int(clf_dec.get("warmup_frames_required", 15))

                        # Determine peer (first other actor in holistic results)
                        peer_result = next(
                            (r for r in latest_holistic_results
                             if (r.student_id or f"{args.student_prefix}{r.track_id:02d}") != actor_id),
                            None,
                        )
                        peer_id = (
                            (peer_result.student_id or f"{args.student_prefix}{peer_result.track_id:02d}")
                            if peer_result is not None else None
                        )

                        # Gate terms from live_classifier gates dict
                        gates = getattr(live_classifier, "gates", {}) if live_classifier else {}
                        latest_scores = getattr(live_classifier, "_latest_scores", {})
                        actor_scores = latest_scores.get(actor_id, {})
                        feature_row = (
                            live_classifier.diagnostic_snapshot(actor_id)
                            if live_classifier is not None
                            and hasattr(live_classifier, "diagnostic_snapshot")
                            else {}
                        )
                        hand_qual = feature_row.get("strict_hand_quality__mean")
                        hand_motion = feature_row.get("hand_motion__q95")
                        finger_motion = feature_row.get("finger_motion__q95")
                        side_delta = feature_row.get("c3_pose_head_peer_delta__max")
                        head_down = feature_row.get("strict_head_down_delta__q95")

                        c3_motion_ceil = gates.get("c3_motion_ceiling")
                        c3_side_fl = gates.get("c3_side_floor")
                        c3_down_ceil = gates.get("c3_down_ceiling")

                        gate_hq = bool(hand_qual) and float(hand_qual or 0) > 0 if hand_qual is not None else None
                        gate_hm = float(hand_motion or 0) <= float(c3_motion_ceil or 1e9) if (hand_motion is not None and c3_motion_ceil is not None) else None
                        gate_fm = float(finger_motion or 0) <= float(c3_motion_ceil or 1e9) if (finger_motion is not None and c3_motion_ceil is not None) else None
                        gate_sf = float(side_delta or 0) >= float(c3_side_fl or 0) if (side_delta is not None and c3_side_fl is not None) else None
                        gate_dc = float(head_down or 0) <= float(c3_down_ceil or 1e9) if (head_down is not None and c3_down_ceil is not None) else None
                        gate_final = (
                            all(g is True for g in [gate_hq, gate_hm, gate_fm, gate_sf, gate_dc]
                                if g is not None)
                            if any(g is not None for g in [gate_hq, gate_hm, gate_fm, gate_sf, gate_dc])
                            else None
                        )

                        actor_c3_thresh = getattr(live_classifier, "c3_threshold", 0.9580807089805603) if live_classifier else 0.9580807089805603
                        actor_c2_thresh = getattr(live_classifier, "c2_threshold", 0.5) if live_classifier else 0.5

                        rec = FrameTraceRecord(
                                timestamp_ms=int(latest_packet.timestamp_ms),
                                frame_index=int(latest_packet.frame_id),
                                inter_frame_duration_ms=round(inter_ms, 2),
                                actor_id=actor_id,
                                peer_id=peer_id,
                                track_present=result.track_id is not None,
                                track_missed_count=0,
                                track_age_frames=warmup_seen,
                                actor_bbox=list(result.bbox.to_list()) if result.bbox else None,
                                peer_bbox=list(peer_result.bbox.to_list()) if peer_result and peer_result.bbox else None,
                                pose_valid=bool(result.pose_landmarks),
                                hand_valid=bool(result.left_hand_landmarks or result.right_hand_landmarks),
                                peer_pose_valid=bool(peer_result.pose_landmarks) if peer_result else None,
                                peer_hand_valid=bool(peer_result.left_hand_landmarks or peer_result.right_hand_landmarks) if peer_result else None,
                                 peer_age_ms=0.0 if peer_result is not None else None,
                                 peer_stale=False,
                                O_A=bool(result.pose_landmarks),
                                O_AB=bool(result.pose_landmarks) and (peer_result is not None and bool(peer_result.pose_landmarks)),
                                R_AB=warmup_seen >= warmup_req,
                                 neutral_baseline_age_ms=float(
                                     feature_row.get("prefix_frames", 0)
                                 ) * 1000.0 / max(float(args.target_fps), 1.0),
                                ready_state="READY" if warmup_seen >= warmup_req else "CALIBRATING",
                                reset_reason=None,
                                p3_A=c3_score,
                                 H_AB=(float(side_delta) if side_delta is not None else None),
                                 Q_A=float(min(
                                     float(feature_row.get("c3_pose_head_valid__mean", 0.0)),
                                     float(feature_row.get("c3_pose_peer_valid__mean", 0.0)),
                                 )),
                                tau_3=actor_c3_thresh,
                                 tau_H=float(gates.get("c3_side_floor", 0.0)),
                                p2_AB=float(clf_dec.get("c2_score") or 0.0),
                                 K_AB=(float(feature_row.get("near_midpoint_pre_cross"))
                                       if feature_row.get("near_midpoint_pre_cross") is not None else None),
                                 Q_hand_AB=(float(feature_row.get("current_hand_quality_mask"))
                                            if feature_row.get("current_hand_quality_mask") is not None else None),
                                tau_2=actor_c2_thresh,
                                tau_K=0.0,
                                legacy_c3_gate_hand_quality_positive=gate_hq,
                                legacy_c3_gate_hand_motion_passed=gate_hm,
                                legacy_c3_gate_finger_motion_passed=gate_fm,
                                legacy_c3_gate_side_floor_passed=gate_sf,
                                legacy_c3_gate_down_ceiling_passed=gate_dc,
                                legacy_c3_gate_final=gate_final,
                                resolver_candidate=pred_class,
                                emitted_class=pred_class,
                                unknown_reason=None,
                                first_flag_timestamp_ms=clf_dec.get("first_flag_timestamp_ms") or None,
                                 latency_ms=None,
                                 tracks_snapshot=_tracks_snapshot,
                                 raw_feature_values=feature_row,
                             )
                        _trace_logger.log(rec)

                if _manifest_recorder is not None:
                    _manifest_recorder.record_frame_latency(inference_ms)
                _prev_frame_mono = frame_started_at

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
                video_writer.write(frame)

                pump_keyboard_until_frame_deadline(
                    cv2,
                    frame_started_at=frame_started_at,
                    rate=rate,
                    interaction=interaction,
                    on_key=annotate_key,
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

        # Stage A: flush trace and write manifest
        if _trace_logger is not None:
            _trace_logger.close()
            print(f"Frame trace saved to: {trace_path.resolve()}")
        video_writer.release()
        print(f"Camera video saved to: {video_path.resolve()}")
        if _manifest_recorder is not None:
            try:
                manifest = _manifest_recorder.build_manifest()
                manifest.save(manifest_path)
                print(f"Session manifest saved to: {manifest_path.resolve()}")
            except Exception as _exc:
                _stage_a_errors.append(f"manifest: {_exc}")
                print(f"[trace] Could not write session manifest: {_exc}")
        if _DEBUG_AVAILABLE and _capture_start_ms is not None:
            if _neutral_start_ms is not None:
                _neutral_intervals.append(
                    TimeInterval(_neutral_start_ms, int(latest_packet.timestamp_ms), "neutral")
                )
            events = SessionEvents(
                session_id=session_id,
                calibration_interval=TimeInterval(_capture_start_ms, _capture_start_ms + 2000, "calibration"),
                neutral_intervals=_neutral_intervals,
                trials=_trials,
                occlusions=_occlusions,
                track_resets=_track_resets,
            )
            events.save(events_path)
            print(f"Events saved to: {events_path.resolve()}")
            try:
                events.validate_protocol()
            except ValueError as _exc:
                _stage_a_errors.append(f"events: {_exc}")
                print(f"[trace] Invalid Stage A events: {_exc}")
        if _trace_logger is not None and not _trace_logger.records:
            _stage_a_errors.append(
                "trace: no processed directed-edge records; camera session has no observable actor pair"
            )

        capture.release()
        cv2.destroyAllWindows()
        print(f"\nTracking JSON saved to: {output_path.resolve()}")
        if _stage_a_errors:
            raise RuntimeError("Stage A capture failed: " + " | ".join(_stage_a_errors))


if __name__ == "__main__":
    main()

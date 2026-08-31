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
    "c2c3": PROJECT_ROOT / "tmp" / "behavior_actor_causal_pose_only_20260812",
    "extended": PROJECT_ROOT / "tmp" / "behavior_actor_extended_suspicious_readiness15_baseline15_20260814",
}
SUPPORTED_ACTIONS = ("c1", "c2", "c3", "c4", "c5", "c7", "suspicious_activity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--device", default="cpu", help="Ultralytics device; default: cpu")
    parser.add_argument(
        "--xgboost-device",
        default="cpu",
        help="XGBoost inference device; default: cpu",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_PERSON_CONFIDENCE,
    )
    parser.add_argument("--target-fps", type=int, default=8)
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
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
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
        "--face-hold-frames",
        type=int,
        default=3,
        help="Processed-observation face hold budget; default: 3",
    )
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
        args.causal_model_dir = args.causal_model_dir or DEFAULT_ACTION_ARTIFACTS["c2c3"]
    # Only the locked C2/C3 benchmark artifact is bundled.  Other specialists
    # require an explicit artifact directory instead of silently referencing
    # local, uncommitted experiment outputs.
    if not ({"c2", "c3", "suspicious_activity"} & set(enabled)):
        args.causal_model_dir = None
    if not args.live_pair and "c2" in enabled:
        args.live_pair = ["student_01:student_02"]
    return enabled


def switch_compute_to_cuda(
    tracking: PersonTrackingModule,
    live_classifier: object | None,
) -> tuple[bool, str | None]:
    """Switch YOLO and every Stage 6 specialist atomically to CUDA."""

    if live_classifier is None:
        return False, "behavior model unavailable"
    detector = tracking.detector
    yolo_model = getattr(detector, "_model", None)
    if yolo_model is None or not hasattr(yolo_model, "to"):
        return False, "YOLO model does not support device switching"
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        live_classifier.set_compute_device("cuda:0")
        yolo_model.to("cuda")
        detector._device = "0"
    except Exception as error:
        rollback_errors = []
        try:
            live_classifier.set_compute_device("cpu")
        except Exception as rollback_error:
            rollback_errors.append(f"XGBoost rollback: {rollback_error}")
        try:
            yolo_model.to("cpu")
            detector._device = "cpu"
        except Exception as rollback_error:
            rollback_errors.append(f"YOLO rollback: {rollback_error}")
        detail = f"{type(error).__name__}: {error}"
        if rollback_errors:
            detail += " | " + " | ".join(rollback_errors)
        return False, detail
    return True, None


def compute_device_snapshot(
    tracking: PersonTrackingModule,
    live_classifier: object | None,
) -> dict[str, str]:
    detector_device = str(getattr(tracking.detector, "_device", "unknown"))
    specialist_device = str(
        getattr(live_classifier, "xgboost_device", "unavailable")
    )
    return {
        "yolo": detector_device,
        "c2": specialist_device,
        "c3": specialist_device,
        "suspicious_activity": specialist_device,
    }


def main() -> None:
    args = parse_args()
    args.device = "cpu"
    args.xgboost_device = "cpu"
    enabled_actions = configure_action_models(args)
    if not 0.0 <= args.holistic_confidence <= 1.0:
        raise ValueError("--holistic-confidence must be in [0, 1]")
    if not 0.0 <= args.soft_landmark_confidence <= args.holistic_confidence:
        raise ValueError(
            "--soft-landmark-confidence must be in [0, --holistic-confidence]"
        )
    if args.face_hold_frames < 0:
        raise ValueError("--face-hold-frames must be non-negative")
    if any(":" not in pair or pair.count(":") != 1 for pair in args.live_pair):
        raise ValueError("--live-pair must use ACTOR:ACTOR")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for inference") from error

    # Kiểm tra CUDA (NVIDIA) và OpenVINO (Intel GPU)
    has_cuda = torch.cuda.is_available()
    has_intel_gpu = False
    try:
        import openvino as ov
        if "GPU" in ov.Core().available_devices:
            has_intel_gpu = True
    except ImportError:
        pass

    if not has_cuda:
        if has_intel_gpu:
            print("[INFO] Đã nhận diện Intel GPU (OpenVINO). Tự động cấu hình thiết bị...")
            if args.device == "0":
                args.device = "GPU"
            if args.xgboost_device.startswith("cuda"):
                args.xgboost_device = "cpu"
        else:
            print("[WARNING] Không tìm thấy CUDA hoặc Intel GPU. Chuyển sang chế độ CPU.")
            if args.device == "0":
                args.device = "cpu"
            if args.xgboost_device.startswith("cuda"):
                args.xgboost_device = "cpu"

    session_id = args.session_id or TrackingManager.generate_session_id(
        "webcam_holistic"
    )
    mode = "restoring" if args.session_id is not None else "fresh"
    print(f"Session ID ({mode}): {session_id}")
    print(f"Actions: {','.join(enabled_actions)}")
    print(f"Explicit pairs: {','.join(args.live_pair) if args.live_pair else 'none'}")
    print("COMPUTE: CPU")
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
                    "face_hold_frames": args.face_hold_frames,
                    "c3_threshold_override": None,
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
                    "face_hold_frames": args.face_hold_frames,
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
    _face_trace_by_track: dict[int, dict[str, object]] = {}
    _face_actor_track: dict[str, int] = {}
    _pending_face_reset_reason: dict[int, str] = {}
    _previous_face_track_ids: set[int] = set()

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
    model_unavailable_error: str | None = None
    model_load_provenance: dict[str, object] = {}
    if args.causal_model_dir is not None:
        from ..test_media.test_media import create_live_classifier

        args.xgboost_model_dir = args.causal_model_dir
        load_result = create_live_classifier(
            args,
            clip_id=f"webcam_{session_id}",
            structured=True,
        )
        live_classifier = load_result.classifier
        model_unavailable_error = load_result.error
        model_load_provenance = dict(load_result.provenance)
        if _manifest_recorder is not None:
            _manifest_recorder.set_bundle_provenance(model_load_provenance)
            _manifest_recorder.runtime_arguments.update({
                "requested_compute_mode": "CPU",
                "active_compute_mode": "CPU",
                "model_available": bool(load_result.available),
                "model_unavailable_error": model_unavailable_error,
                "requested_model_dir": str(args.causal_model_dir),
            })
    interaction = DemoWebcamInteractionController(
        manager=tracking.manager,
        session_id=session_id,
        rate=rate,
    )
    compute_mode = "CPU"
    compute_status = "COMPUTE: CPU"
    cuda_switch_error: str | None = None
    if live_classifier is None:
        interaction.status_message = "MODEL UNAVAILABLE"
        print(
            "MODEL UNAVAILABLE: "
            f"{model_unavailable_error or 'classification disabled'} | "
            f"requested={args.causal_model_dir}"
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
            face_hold_frames=args.face_hold_frames,
        ) as holistic:
            while True:
                frame_started_at = monotonic()
                ok, frame = capture.read()
                if not ok:
                    break

                inference_started_at = monotonic()
                latest_packet = tracking.process_frame(frame)
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
                    _current_face_track_ids = {
                        int(result.track_id) for result in latest_holistic_results
                    }
                    for _missing_track_id in (
                        _previous_face_track_ids - _current_face_track_ids
                    ):
                        if _missing_track_id in _face_trace_by_track:
                            _face_trace_by_track[_missing_track_id]["track_missing"] = True
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
                        face_state = holistic.face_trace_state(result.track_id)
                        previous_face_state = _face_trace_by_track.get(
                            int(result.track_id), {}
                        )
                        previous_actor_track = _face_actor_track.get(actor_id)
                        face_reset_reason = _pending_face_reset_reason.pop(
                            int(result.track_id), None
                        )
                        if (
                            face_reset_reason is None
                            and previous_actor_track is not None
                            and previous_actor_track != int(result.track_id)
                        ):
                            face_reset_reason = "track_change"
                        elif (
                            face_reset_reason is None
                            and previous_face_state.get("track_missing")
                        ):
                            face_reset_reason = "track_reacquisition"
                        elif (
                            face_reset_reason is None
                            and result.face_valid
                            and int(previous_face_state.get("missing_frames", 0)) > 0
                        ):
                            face_reset_reason = "face_reacquired"
                        elif (
                            face_reset_reason is None
                            and face_state["hold_expired"]
                            and not previous_face_state.get("hold_expired", False)
                        ):
                            face_reset_reason = "face_hold_expired"
                        feature_row.update({
                            "face_observed": bool(result.face_valid),
                            "face_predicted": bool(result.face_predicted),
                            "face_missing_frames": face_state["missing_frames"],
                            "face_hold_frames": face_state["hold_frames"],
                            "face_hold_expired": face_state["hold_expired"],
                            "face_track_id": int(result.track_id),
                            "face_reset_reason": face_reset_reason,
                            "compute_mode": compute_mode,
                            "compute_status": compute_status,
                            "model_available": live_classifier is not None,
                            "model_unavailable_error": model_unavailable_error,
                            "requested_model_dir": str(args.causal_model_dir),
                            "model_load_provenance": model_load_provenance,
                            "cuda_switch_error": cuda_switch_error,
                        })
                        _face_trace_by_track[int(result.track_id)] = {
                            **face_state,
                            "track_missing": False,
                        }
                        _face_actor_track[actor_id] = int(result.track_id)
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
                                reset_reason=face_reset_reason,
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
                                  inference_ms=round(inference_ms, 2),
                                  end_to_end_latency_ms=round(
                                      (monotonic() - frame_started_at) * 1000.0,
                                      2,
                                  ),
                                  actor_track_id=int(result.track_id),
                                  peer_track_id=(
                                      int(peer_result.track_id)
                                      if peer_result is not None else None
                                  ),
                                  current_scores=dict(
                                      clf_dec.get("current_scores", {})
                                  ),
                                  current_gates=dict(
                                      clf_dec.get("current_gates", {})
                                  ),
                                  current_class=str(
                                      clf_dec.get("predicted_class", "c5")
                                  ),
                                  alert_history=list(clf_dec.get("history", [])),
                                  evidence_class=(
                                      str(clf_dec.get("evidence_class"))
                                      if clf_dec.get("evidence_class") else None
                                  ),
                                  evidence_score=(
                                      float(clf_dec["evidence_score"])
                                      if clf_dec.get("evidence_score") not in ("", None)
                                      else None
                                  ),
                                  compute_mode=compute_mode,
                                  compute_status=compute_status,
                              )
                        _trace_logger.log(rec)
                    _previous_face_track_ids = _current_face_track_ids

                if _manifest_recorder is not None:
                    _manifest_recorder.record_frame_latency(
                        inference_ms,
                        end_to_end_latency_ms=(
                            monotonic() - frame_started_at
                        ) * 1000.0,
                        observation_timestamp_ms=latest_packet.timestamp_ms,
                    )
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
                        current_score = classification.get("current_scores", {}).get(
                            predicted
                        )
                        if current_score not in ("", None):
                            score_text = f" {float(current_score):.2f}"
                        else:
                            seen = int(classification.get("warmup_frames_seen", 0))
                            required = int(classification.get("warmup_frames_required", 30))
                            score_text = f" warming {seen}/{required}" if seen < required else " no pose score"
                        cv2.putText(
                            frame,
                            f"{actor_id} CURRENT: {labels.get(predicted, predicted)}{score_text}",
                            (x1, max(100, label_y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62,
                            (0, 0, 255) if predicted != "c5" else (255, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        evidence_class = str(
                            classification.get("evidence_class") or ""
                        )
                        evidence_score = classification.get("evidence_score")
                        first_flag = classification.get("first_flag_timestamp_ms")
                        history_text = "ALERT HISTORY: none"
                        if evidence_class:
                            history_text = (
                                f"ALERT HISTORY: {labels.get(evidence_class, evidence_class)} "
                                f"score={float(evidence_score):.2f} first_flag={first_flag}"
                            )
                        cv2.putText(
                            frame,
                            history_text,
                            (x1, min(frame_height - 12, max(122, label_y + 24))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (0, 165, 255),
                            1,
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
                cv2.putText(
                    frame,
                    compute_status,
                    (12, 106),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 0) if compute_mode == "CUDA" else (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                if live_classifier is None:
                    cv2.putText(
                        frame,
                        "MODEL UNAVAILABLE",
                        (12, 132),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 0, 255),
                        2,
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
                if interaction.consume_compute_switch():
                    devices_before = compute_device_snapshot(
                        tracking, live_classifier
                    )
                    if compute_mode != "CUDA":
                        switched, cuda_switch_error = switch_compute_to_cuda(
                            tracking, live_classifier
                        )
                        if switched:
                            compute_mode = "CUDA"
                            compute_status = "COMPUTE: CUDA"
                        else:
                            compute_mode = "CPU"
                            compute_status = "CUDA UNAVAILABLE"
                        interaction.status_message = compute_status
                        print(f"\n{compute_status}")
                    if _manifest_recorder is not None:
                        _manifest_recorder.runtime_arguments[
                            "active_compute_mode"
                        ] = compute_mode
                        _manifest_recorder.record_compute_switch(
                            timestamp_ms=int(latest_packet.timestamp_ms),
                            requested_mode="CUDA",
                            active_mode=compute_mode,
                            result=("success" if compute_mode == "CUDA" else "failed"),
                            devices_before=devices_before,
                            devices_after=compute_device_snapshot(
                                tracking, live_classifier
                            ),
                            error=cuda_switch_error,
                        )
                if interaction.consume_classifier_reset():
                    for result in latest_holistic_results:
                        _pending_face_reset_reason[int(result.track_id)] = "manual_reset"
                    holistic.reset()
                    _face_trace_by_track.clear()
                    _face_actor_track.clear()
                    _previous_face_track_ids.clear()
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

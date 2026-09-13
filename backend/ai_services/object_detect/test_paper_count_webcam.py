"""Manual end-to-end webcam check for count-only paper monitoring.

Run from the project root:
    python -m backend.ai_services.object_detect.test_paper_count_webcam
    python -m backend.ai_services.object_detect.test_paper_count_webcam --source 0

Setup:
1. Place legitimate exam paper(s) on table in front of student(s).
2. Observe status showing SETUP with observed_count, stable_count, baseline_count.
3. Press 'A' to lock/ARM baseline count.
4. Introduce a new sheet/cheat sheet; box turns red and alerts after temporal confirmation.
5. Press 'D' to disarm and return to SETUP mode.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch

from backend.ai_services.object_detect.object_detect import ObjectDetectModule
from backend.ai_services.object_detect.paper_count_pipeline import (
    PaperCountPipeline,
)
from backend.ai_services.pose_gaze.tracking.detectors import (
    UltralyticsPersonDetector,
)
from backend.ai_services.pose_gaze.tracking.manager import AssignmentError
from backend.ai_services.webcam_utils import (
    configure_webcam_capture,
    resize_live_frame,
)
from backend.core.config import settings


PAPER_COLORS = {
    "baseline_paper": (0, 255, 255),        # Cyan/Yellow
    "suspicious_new_paper": (0, 0, 255),    # Red
}


def _draw_label(
    frame: cv2.Mat,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    *,
    scale: float = 0.55,
    bg: bool = True,
) -> None:
    y_pos = max(18, y)
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2
    )
    if bg:
        cv2.rectangle(
            frame,
            (x - 2, y_pos - text_h - 4),
            (x + text_w + 4, y_pos + baseline + 2),
            (20, 20, 20),
            -1,
        )
    cv2.putText(
        frame,
        text,
        (x, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live Webcam test for Count-Only Paper Monitoring."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (e.g., 0) or path to a video file.",
    )
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()

    source: int | str = (
        int(args.source) if args.source.isdigit() else args.source
    )
    is_live_webcam = isinstance(source, int)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        try:
            import openvino as ov
            if "GPU" in ov.Core().available_devices:
                device = "GPU"
        except ImportError:
            pass

    person_model_path = (
        "weights/yolov8n_openvino_model"
        if device == "GPU"
        else "weights/yolov8n.pt"
    )
    person_detector = UltralyticsPersonDetector(
        model_path=Path(person_model_path),
        confidence_threshold=0.55,
        device="cpu" if device == "GPU" else device,
    )
    object_detector = ObjectDetectModule(
        device=device,
        detect_every_n_frames=(
            settings.webcam_object_detect_every_n_frames
            if is_live_webcam
            else None
        ),
        phone_confidence_floor=(
            settings.webcam_phone_confidence_floor
            if is_live_webcam
            else None
        ),
    )
    pipeline = PaperCountPipeline(
        person_detector=person_detector,
        object_detector=object_detector,
        storage_root=Path("test_data_tracking"),
        max_people=2,
        person_detect_every_n_frames=(
            settings.webcam_person_detect_every_n_frames
            if is_live_webcam
            else 1
        ),
    )

    source_stem = "webcam" if is_live_webcam else Path(str(source)).stem
    session_id = (
        args.session_id or f"paper_count_{source_stem}_{int(time.time())}"
    )
    pipeline.create_session(session_id)

    capture = cv2.VideoCapture(source)
    if is_live_webcam:
        configure_webcam_capture(capture)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video/webcam source: {args.source}")

    playback_delay = (
        1
        if is_live_webcam
        else max(1, round(1000 / max(capture.get(cv2.CAP_PROP_FPS), 1.0)))
    )

    print("\n" + "=" * 60)
    print(" PAPER COUNT WEBCAM DEMO RUNNING")
    print(" Controls:")
    print("   [A] Arm / Lock baseline paper count")
    print("   [D] Disarm / Reset setup baseline")
    print("   [Q] / [ESC] Quit")
    print("=" * 60 + "\n")

    frame_id = 0
    previous_alert_count = 0
    status_message = "Mode: SETUP - Place paper and press 'A' to ARM"
    status_until = time.monotonic() + 5.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if is_live_webcam:
                frame = resize_live_frame(frame)

            frame_id += 1
            result = pipeline.process_frame(
                frame,
                session_id=session_id,
                frame_id=frame_id,
                timestamp_ms=int(time.time() * 1000),
            )
            if result is None:
                continue

            state = result["paper_count_state"]
            armed = state["monitoring_armed"]
            observed_count = state["observed_count"]
            stable_count = state["stable_count"]
            baseline_count = state["baseline_count"]
            active_alerts = result["alerts"]

            # Draw Person ROI Bboxes
            for person in result["people"]:
                if not person["is_present"]:
                    continue
                x1, y1, x2, y2 = person["bbox_xyxy"]
                person_id = person["person_id"]
                color = (0, 180, 0) if person_id else (0, 165, 255)
                label = (
                    f"Person: {person_id}"
                    if person_id
                    else f"Person track={person['track_id']}"
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                _draw_label(
                    frame,
                    label,
                    x1,
                    min(frame.shape[0] - 8, y2 + 22),
                    color,
                )

            # Draw Paper Observations Bboxes
            for paper in result["papers"]:
                x1, y1, x2, y2 = paper["bbox_xyxy"]
                status = paper["status"]
                is_suspicious = status == "suspicious_new_paper"
                color = PAPER_COLORS.get(status, (255, 255, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                owner = (
                    paper.get("owner_person_id")
                    or paper.get("owner_track_id")
                    or "None"
                )
                label = (
                    f"CHEAT SHEET (NEW)"
                    if is_suspicious
                    else f"Paper #{paper['observation_index']} (owner={owner})"
                )
                _draw_label(frame, label, x1, y1 - 8, color, scale=0.5)

            # Print terminal alert on new count violation
            if len(active_alerts) > previous_alert_count:
                print(f"[ALERT] Paper count anomaly detected! Active alerts: {len(active_alerts)}")
            previous_alert_count = len(active_alerts)

            # Draw top dashboard panel
            panel_bg_color = (0, 0, 120) if active_alerts else (40, 40, 40)
            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 45),
                panel_bg_color,
                -1,
            )

            mode_str = "ARMED" if armed else "SETUP"
            mode_color = (0, 0, 255) if armed else (0, 255, 255)

            header_text = (
                f"PAPER COUNT: {mode_str} | Observed: {observed_count} | "
                f"Stable: {stable_count} | Baseline: {baseline_count}"
            )
            cv2.putText(
                frame,
                header_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                mode_color,
                2,
            )

            # Draw status message at bottom
            if status_message and time.monotonic() < status_until:
                cv2.rectangle(
                    frame,
                    (0, frame.shape[0] - 35),
                    (frame.shape[1], frame.shape[0]),
                    (20, 20, 20),
                    -1,
                )
                cv2.putText(
                    frame,
                    status_message,
                    (15, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Paper Count Webcam Demo", frame)
            key = cv2.waitKey(playback_delay) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break
            elif key in (ord("a"), ord("A")):
                state_info = pipeline.arm_paper_monitoring(session_id)
                status_message = (
                    f"Paper monitoring ARMED! Baseline count = {state_info['baseline_count']}"
                )
                status_until = time.monotonic() + 4.0
                print(f"[INFO] {status_message}")
            elif key in (ord("d"), ord("D")):
                pipeline.paper_monitor.create_session(session_id)  # reset session
                status_message = "Paper monitoring DISARMED. Re-entered SETUP mode."
                status_until = time.monotonic() + 4.0
                print(f"[INFO] {status_message}")

    finally:
        capture.release()
        cv2.destroyAllWindows()
        pipeline.cleanup_session(session_id)


if __name__ == "__main__":
    main()

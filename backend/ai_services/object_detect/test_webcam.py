"""Visual smoke test for raw object detections.

For identity-aware test-paper/cheat-sheet decisions, run:
    python -m backend.ai_services.object_detect.test_paper_tracking_webcam

The source can also be a video:
    python -m backend.ai_services.object_detect.test_webcam \
        --source data/smartphone.mp4
"""

from __future__ import annotations

import argparse
import cv2
from pathlib import Path

from backend.ai_services.object_detect.object_detect import ObjectDetectModule
from backend.ai_services.webcam_utils import (
    configure_webcam_capture,
    resize_live_frame,
)
from backend.core.config import settings


def draw_objects(
    frame,
    raw_objects: list[dict],
    confirmed_classes: list[str],
) -> None:
    for item in raw_objects:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        display_name = item["display_name"]
        confidence = item["confidence"]
        if item["is_paper_candidate"]:
            color = (255, 255, 0)
            suffix = " [PAPER -> TRACKER]"
        elif display_name in confirmed_classes:
            color = (0, 0, 255)
            suffix = " [CONFIRMED]"
        else:
            color = (0, 255, 255)
            suffix = ""

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{display_name} {confidence:.2f}{suffix}",
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or path to an MP4/video file.",
    )
    parser.add_argument("--session-id", default=None)
    return parser.parse_args()


def _capture_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    args = _arguments()
    source = _capture_source(args.source)
    is_live_webcam = isinstance(source, int)
    module = ObjectDetectModule(
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
    capture = cv2.VideoCapture(source)
    if is_live_webcam:
        configure_webcam_capture(capture)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    source_stem = (
        "webcam"
        if isinstance(source, int)
        else Path(str(source)).stem
    )
    session_id = args.session_id or f"manual_object_test_{source_stem}"
    playback_delay = (
        1
        if isinstance(source, int)
        else max(1, round(1000 / max(capture.get(cv2.CAP_PROP_FPS), 1.0)))
    )
    frame_id = 0
    last_objects: list[dict] = []
    last_confirmed: list[str] = []
    print(
        "Q: quit. Cyan=paper candidate routed to paper tracker; "
        "red=confirmed prohibited non-paper object."
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if is_live_webcam:
                frame = resize_live_frame(frame)
            frame_id += 1
            result = module.process(frame, session_id, frame_id)
            if result is not None:
                last_objects = result.get("raw_objects", last_objects)
                last_confirmed = result.get(
                    "confirmed_classes",
                    last_confirmed,
                )
                if result["label"] != "clear" and frame_id % 15 == 0:
                    print(f"[frame {frame_id}] {result['label']}")

            draw_objects(frame, last_objects, last_confirmed)
            cv2.imshow("Raw object detection", frame)
            if cv2.waitKey(playback_delay) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        module.cleanup_session(session_id)


if __name__ == "__main__":
    main()

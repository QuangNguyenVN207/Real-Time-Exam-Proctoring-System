"""Visual smoke test for the server-facing ObjectDetector contract.

Examples:
    python -m backend.ai_services.object_detect.object_detect_test
    python -m backend.ai_services.object_detect.object_detect_test ^
        --source data/smartphone.mp4 --model "weights/best (1).pt"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from backend.ai_services.object_detect.object_detect import ObjectDetector
from backend.ai_services.webcam_utils import (
    configure_webcam_capture,
    resize_live_frame,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or video path.",
    )
    parser.add_argument(
        "--model",
        default="weights/best (1).pt",
        help="YOLO checkpoint path.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Only confidence strictly above this value is returned.",
    )
    return parser.parse_args()


def _source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _draw_result(frame, result: dict | None) -> None:
    if result is None:
        return
    for detection in result["details"]["detections"]:
        x1, y1, x2, y2 = detection["bbox"]
        label = detection["label"]
        confidence = detection["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )


def main() -> None:
    args = _arguments()
    detector = ObjectDetector(
        Path(args.model),
        confidence_threshold=args.confidence,
    )
    source = _source(args.source)
    is_live_webcam = isinstance(source, int)
    capture = cv2.VideoCapture(source)
    if is_live_webcam:
        configure_webcam_capture(capture)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    playback_delay = (
        1
        if isinstance(source, int)
        else max(
            1,
            round(1000 / max(capture.get(cv2.CAP_PROP_FPS), 1.0)),
        )
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if is_live_webcam:
                frame = resize_live_frame(frame)

            result = detector.process_frame(frame, time.time())
            _draw_result(frame, result)
            cv2.imshow("ObjectDetector contract test", frame)
            if cv2.waitKey(playback_delay) & 0xFF in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

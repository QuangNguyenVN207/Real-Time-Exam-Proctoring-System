"""Manual webcam test for person detection and tracking only."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.ai_services.pose_gaze.tracking.manager import TrackingManager
from backend.ai_services.pose_gaze.tracking.webcam import (
    PersonTrackingConfig,
    PersonTrackingModule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV webcam index")
    parser.add_argument("--model", type=Path, default=None, help="YOLO person weights")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu or 0")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_id = args.session_id or TrackingManager.generate_session_id(
        "webcam_tracking"
    )
    mode = "restoring" if args.session_id is not None else "fresh"
    print(f"Session ID ({mode}): {session_id}")
    module = PersonTrackingModule(
        PersonTrackingConfig(
            model_path=args.model,
            session_id=session_id,
            restore_session=args.session_id is not None,
            confidence_threshold=args.confidence,
            device=args.device,
            max_tracks=args.max_tracks,
        )
    )
    module.run_webcam(
        camera_index=args.camera,
        target_fps=args.target_fps,
        width=args.width,
        height=args.height,
        window_name="Exam Proctoring - Person Tracking",
    )


if __name__ == "__main__":
    main()

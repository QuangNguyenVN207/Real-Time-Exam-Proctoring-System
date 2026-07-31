"""Manual end-to-end webcam check for identity-aware paper monitoring.

Run from the project root:
    python -m backend.ai_services.object_detect.test_paper_tracking_webcam
    python -m backend.ai_services.object_detect.test_paper_tracking_webcam \
        --source data/cheatsheet.mp4

Setup:
1. Put only the legitimate exam paper in front of each student.
2. Wait until its box turns green (authorized_exam_paper).
3. Press A to lock/arm registration.
4. Introduce a second sheet; it should turn red after temporal confirmation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch

from backend.ai_services.object_detect.object_detect import ObjectDetectModule
from backend.ai_services.pose_gaze.paper_pipeline import PoseGazePaperPipeline
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
    "authorized_exam_paper": (0, 180, 0),
    "suspicious": (0, 0, 255),
    "watching": (0, 165, 255),
    "registration_pending": (0, 255, 255),
    "observed": (255, 200, 0),
    "missing": (120, 120, 120),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or path to an MP4/video file.",
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--enable-face-identity",
        action="store_true",
        help=(
            "Use RetinaFace/ArcFace for automatic stable person IDs and "
            "identity-mismatch alerts."
        ),
    )
    parser.add_argument(
        "--face-db",
        default=settings.face_db_path,
        help="Directory of reference images named <person_id>.jpg/png.",
    )
    args = parser.parse_args()
    source: int | str = (
        int(args.source) if args.source.isdigit() else args.source
    )
    is_live_webcam = isinstance(source, int)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    person_detector = UltralyticsPersonDetector(
        model_path=Path("weights/yolov8n.pt"),
        confidence_threshold=0.55,
        device=device,
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
    face_verifier = None
    if args.enable_face_identity:
        from backend.ai_services.face_verify.face_verify import FaceVerifier

        face_verifier = FaceVerifier(db_path=args.face_db)
    pipeline = PoseGazePaperPipeline(
        person_detector=person_detector,
        object_detector=object_detector,
        storage_root=Path("test_data_tracking"),
        max_people=2,
        person_detect_every_n_frames=(
            settings.webcam_person_detect_every_n_frames
            if is_live_webcam
            else 1
        ),
        face_verifier=face_verifier,
    )
    source_stem = (
        "webcam"
        if isinstance(source, int)
        else Path(str(source)).stem
    )
    session_id = args.session_id or f"paper_tracking_{source_stem}"
    pipeline.create_session(session_id, restore_existing=True)

    capture = cv2.VideoCapture(source)
    if is_live_webcam:
        configure_webcam_capture(capture)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    playback_delay = (
        1
        if isinstance(source, int)
        else max(1, round(1000 / max(capture.get(cv2.CAP_PROP_FPS), 1.0)))
    )

    print(
        "Q: quit | A: arm/lock exam-paper registration | "
        "D: reopen registration\n"
        "Type IDs directly in the camera window: ENTER saves, BACKSPACE edits, "
        "ESC skips the current temporary detection."
    )
    frame_id = 0
    previous_alert_ids: set[int] = set()
    ignored_person_track_ids: set[int] = set()
    ignored_temporary_paper_ids: set[int] = set()
    active_prompt: tuple[str, int] | None = None
    input_buffer = ""
    status_message = ""
    status_until = 0.0

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

            # Refresh because assigning a known ID can restore an older numeric
            # person/paper track and its authorization.
            result["people"] = (
                pipeline.manager.get_packet(session_id).to_dict()["tracks"]
            )
            paper_state = pipeline.manager.get_paper_state(session_id)
            result["papers"] = paper_state["papers"]
            result["authorized_papers"] = paper_state["authorized_papers"]

            if active_prompt is None:
                unassigned_people = [
                    track
                    for track in result["people"]
                    if track["is_present"]
                    and track["person_id"] is None
                    and track["track_id"] not in ignored_person_track_ids
                ]
                if unassigned_people:
                    active_prompt = (
                        "person",
                        int(unassigned_people[0]["track_id"]),
                    )
                    input_buffer = ""
                else:
                    unassigned_papers = [
                        paper
                        for paper in result["papers"]
                        if paper["is_present"]
                        and not paper["paper_id_assigned"]
                        and paper["paper_id"]
                        not in ignored_temporary_paper_ids
                    ]
                    if unassigned_papers:
                        active_prompt = (
                            "paper",
                            int(unassigned_papers[0]["paper_id"]),
                        )
                        input_buffer = ""

            for track in result["people"]:
                if not track["is_present"]:
                    continue
                x1, y1, x2, y2 = track["bbox_xyxy"]
                person_id = track["person_id"]
                color = (0, 180, 0) if person_id else (0, 165, 255)
                label = (
                    (
                        f"person_id={person_id} "
                        f"memory={'on' if track['appearance_identity_registered'] else 'off'}"
                    )
                    if person_id
                    else f"UNASSIGNED temp_track={track['track_id']}"
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            for paper in result["papers"]:
                if not paper["is_present"]:
                    continue
                x1, y1, x2, y2 = paper["bbox_xyxy"]
                color = PAPER_COLORS.get(paper["status"], (255, 255, 0))
                owner = (
                    paper["owner_person_id"]
                    if paper.get("owner_person_id")
                    else paper["owner_track_id"]
                )
                paper_identity = (
                    f"paper_id={paper['paper_id']}"
                    if paper["paper_id_assigned"]
                    else f"UNASSIGNED temp_paper={paper['paper_id']}"
                )
                memory_state = (
                    "memory=on"
                    if paper["appearance_identity_registered"]
                    else "memory=off"
                )
                label = (
                    f"{paper_identity} owner={owner} {memory_state} "
                    f"{paper['status']} "
                    f"({paper['stable_label']})"
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    2,
                )

            alert_ids = {
                int(alert["paper_id"])
                for alert in result["alerts"]
                if alert["source"] == "paper_tracking"
            }
            for paper_id in sorted(alert_ids - previous_alert_ids):
                print(f"[ALERT] New suspicious paper_id={paper_id}")
            previous_alert_ids = alert_ids

            mode = "ARMED" if result["paper_monitoring_armed"] else "SETUP"
            cv2.putText(
                frame,
                f"paper monitor: {mode}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255) if mode == "ARMED" else (0, 255, 255),
                2,
            )

            if active_prompt is not None:
                identity_type, temporary_id = active_prompt
                if identity_type == "person":
                    prompt_title = (
                        f"ASSIGN PERSON temp_track={temporary_id}: "
                        f"{input_buffer}_"
                    )
                    prompt_hint = (
                        "Type person ID here | ENTER=save | "
                        "BACKSPACE=edit | ESC=skip"
                    )
                else:
                    prompt_title = (
                        f"ASSIGN PAPER temp_paper={temporary_id}: "
                        f"{input_buffer}_"
                    )
                    prompt_hint = (
                        "Type positive numeric paper ID | ENTER=save | "
                        "BACKSPACE=edit | ESC=skip"
                    )
                cv2.rectangle(
                    frame,
                    (0, 42),
                    (frame.shape[1], 105),
                    (25, 25, 25),
                    -1,
                )
                cv2.putText(
                    frame,
                    prompt_title,
                    (15, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    prompt_hint,
                    (15, 94),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (230, 230, 230),
                    1,
                )

            if status_message and time.monotonic() < status_until:
                cv2.putText(
                    frame,
                    status_message,
                    (15, frame.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Exam paper identity tracking", frame)
            key = cv2.waitKey(playback_delay) & 0xFF
            if active_prompt is not None:
                identity_type, temporary_id = active_prompt
                if key in (10, 13):
                    assigned_id = input_buffer.strip()
                    if not assigned_id:
                        status_message = "ID cannot be empty."
                        status_until = time.monotonic() + 2.5
                    else:
                        try:
                            if identity_type == "person":
                                pipeline.manager.assign_student(
                                    session_id,
                                    track_id=temporary_id,
                                    student_id=assigned_id,
                                )
                                status_message = (
                                    f"Assigned person_id={assigned_id}"
                                )
                                print(status_message)
                            else:
                                stable_paper_id = int(assigned_id)
                                pipeline.assign_paper_id(
                                    session_id,
                                    current_paper_id=temporary_id,
                                    stable_paper_id=stable_paper_id,
                                )
                                status_message = (
                                    f"Assigned paper_id={stable_paper_id}"
                                )
                                print(status_message)
                            active_prompt = None
                            input_buffer = ""
                            status_until = time.monotonic() + 3.0
                        except (AssignmentError, ValueError) as error:
                            status_message = f"Assignment failed: {error}"
                            status_until = time.monotonic() + 3.0
                            active_prompt = None
                            input_buffer = ""
                            print(status_message)
                elif key in (8, 127):
                    input_buffer = input_buffer[:-1]
                elif key == 27:
                    if identity_type == "person":
                        ignored_person_track_ids.add(temporary_id)
                    else:
                        ignored_temporary_paper_ids.add(temporary_id)
                    status_message = (
                        f"Skipped temporary {identity_type} ID {temporary_id}"
                    )
                    status_until = time.monotonic() + 2.0
                    active_prompt = None
                    input_buffer = ""
                elif 32 <= key <= 126:
                    character = chr(key)
                    if identity_type == "person":
                        if (
                            character.isalnum()
                            or character in {"_", "-", "."}
                        ) and len(input_buffer) < 128:
                            input_buffer += character
                    elif character.isdigit() and len(input_buffer) < 9:
                        input_buffer += character
            else:
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("a"), ord("A")):
                    pending_people = [
                        track
                        for track in result["people"]
                        if track["is_present"]
                        and track["person_id"] is None
                        and track["track_id"]
                        not in ignored_person_track_ids
                    ]
                    pending_papers = [
                        paper
                        for paper in result["papers"]
                        if paper["is_present"]
                        and not paper["paper_id_assigned"]
                        and paper["paper_id"]
                        not in ignored_temporary_paper_ids
                    ]
                    unremembered_papers = [
                        paper
                        for paper in result["papers"]
                        if paper["is_present"]
                        and paper["paper_id_assigned"]
                        and not paper["appearance_identity_registered"]
                    ]
                    unremembered_people = [
                        track
                        for track in result["people"]
                        if track["is_present"]
                        and track["person_id"] is not None
                        and not track["appearance_identity_registered"]
                    ]
                    if pending_people or pending_papers:
                        status_message = (
                            "Assign all visible person/paper IDs before ARM."
                        )
                        status_until = time.monotonic() + 3.0
                        print(status_message)
                    elif unremembered_people:
                        status_message = (
                            "Cannot ARM: person fingerprint unavailable. "
                            "Keep face/body visible and improve lighting."
                        )
                        status_until = time.monotonic() + 4.0
                        print(status_message)
                    elif unremembered_papers:
                        status_message = (
                            "Cannot ARM: paper fingerprint unavailable. "
                            "Improve view/lighting."
                        )
                        status_until = time.monotonic() + 4.0
                        print(status_message)
                    else:
                        pipeline.arm_paper_monitoring(session_id)
                        status_message = "Paper monitoring ARMED."
                        status_until = time.monotonic() + 3.0
                        print(
                            "Paper registration locked. "
                            "New paper IDs will be evaluated."
                        )
                if key in (ord("d"), ord("D")):
                    pipeline.manager.disarm_paper_monitoring(session_id)
                    status_message = "Paper registration reopened."
                    status_until = time.monotonic() + 3.0
                    print(status_message)
    finally:
        capture.release()
        cv2.destroyAllWindows()
        pipeline.cleanup_session(session_id)


if __name__ == "__main__":
    main()

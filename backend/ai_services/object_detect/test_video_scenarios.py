"""Repeatable, headless video test for object, person, and paper tracking.

Run from the project root:

    python -m backend.ai_services.object_detect.test_video_scenarios \
        data/smartphone.mp4 data/cheatsheet.mp4

The first ``--setup-seconds`` of each video are treated like the manual webcam
SETUP stage. Foreground people and at most one paper per owner receive stable
test IDs. Registration is then armed and every later paper is evaluated as a
new physical sheet.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import torch

from backend.ai_services.object_detect.object_detect import ObjectDetectModule
from backend.ai_services.pose_gaze.paper_pipeline import PoseGazePaperPipeline
from backend.ai_services.pose_gaze.tracking.detectors import (
    UltralyticsPersonDetector,
)
from backend.ai_services.pose_gaze.tracking.manager import AssignmentError
from backend.core.config import settings


PAPER_COLORS = {
    "authorized_exam_paper": (0, 180, 0),
    "suspicious": (0, 0, 255),
    "watching": (0, 165, 255),
    "registration_pending": (0, 255, 255),
    "observed": (255, 200, 0),
    "missing": (120, 120, 120),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the raw-object and identity-aware paper tests on MP4 files."
        )
    )
    parser.add_argument(
        "videos",
        nargs="*",
        type=Path,
        default=[
            Path("data/smartphone.mp4"),
            Path("data/cheatsheet.mp4"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/video_test_results"),
    )
    parser.add_argument(
        "--setup-seconds",
        type=float,
        default=5.0,
        help="Arm paper monitoring after this source-video timestamp.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=3,
        help="Process every Nth source frame; 3 preserves a 10 FPS timeline.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional early stop used for quick smoke tests.",
    )
    parser.add_argument(
        "--no-annotated-video",
        action="store_true",
        help="Write JSON reports only.",
    )
    return parser.parse_args()


def _new_class_stats() -> dict[str, Any]:
    return {
        "boxes": 0,
        "inference_frames": 0,
        "first_second": None,
        "last_second": None,
        "max_confidence": 0.0,
        "_confidence_sum": 0.0,
    }


def _update_class_stats(
    stats: dict[str, dict[str, Any]],
    raw_objects: list[dict[str, Any]],
    second: float,
) -> None:
    seen_this_frame: set[str] = set()
    for item in raw_objects:
        class_name = str(item["class_name"])
        entry = stats.setdefault(class_name, _new_class_stats())
        confidence = float(item["confidence"])
        entry["boxes"] += 1
        entry["_confidence_sum"] += confidence
        entry["max_confidence"] = max(entry["max_confidence"], confidence)
        entry["first_second"] = (
            second
            if entry["first_second"] is None
            else min(entry["first_second"], second)
        )
        entry["last_second"] = (
            second
            if entry["last_second"] is None
            else max(entry["last_second"], second)
        )
        seen_this_frame.add(class_name)
    for class_name in seen_this_frame:
        stats[class_name]["inference_frames"] += 1


def _finalize_class_stats(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for class_name, raw in sorted(stats.items()):
        boxes = int(raw["boxes"])
        result[class_name] = {
            "boxes": boxes,
            "inference_frames": int(raw["inference_frames"]),
            "first_second": round(float(raw["first_second"]), 3),
            "last_second": round(float(raw["last_second"]), 3),
            "max_confidence": round(float(raw["max_confidence"]), 4),
            "mean_confidence": round(
                float(raw["_confidence_sum"]) / max(boxes, 1),
                4,
            ),
        }
    return result


def _person_slot(track: dict[str, Any], frame_width: int) -> str:
    x1, _, x2, _ = track["bbox_xyxy"]
    return "LEFT" if (x1 + x2) / 2.0 < frame_width / 2.0 else "RIGHT"


def _assign_setup_identities(
    pipeline: PoseGazePaperPipeline,
    *,
    session_id: str,
    frame_width: int,
    second: float,
    assigned_person_slots: set[str],
    assigned_paper_owners: set[int | str],
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    people = pipeline.manager.get_packet(session_id).to_dict()["tracks"]
    for track in sorted(
        (item for item in people if item["is_present"]),
        key=lambda item: item["bbox_xyxy"][0],
    ):
        if track["person_id"] is not None:
            continue
        slot = _person_slot(track, frame_width)
        if slot in assigned_person_slots:
            continue
        person_id = f"STUDENT_{slot}"
        try:
            pipeline.manager.assign_student(
                session_id,
                track_id=int(track["track_id"]),
                student_id=person_id,
            )
        except AssignmentError as error:
            events.append(
                {
                    "second": round(second, 3),
                    "type": "person_assignment_failed",
                    "track_id": int(track["track_id"]),
                    "detail": str(error),
                }
            )
            continue
        assigned_person_slots.add(slot)
        events.append(
            {
                "second": round(second, 3),
                "type": "person_assigned",
                "track_id": int(track["track_id"]),
                "person_id": person_id,
            }
        )

    people = pipeline.manager.get_packet(session_id).to_dict()["tracks"]
    paper_state = pipeline.manager.get_paper_state(session_id)
    papers = paper_state["papers"]
    for paper in sorted(
        (item for item in papers if item["is_present"]),
        key=lambda item: item["bbox_xyxy"][0],
    ):
        if paper["paper_id_assigned"]:
            continue
        owner_key: int | str = (
            int(paper["owner_track_id"])
            if paper["owner_track_id"] is not None
            else f"unowned-{paper['paper_id']}"
        )
        if owner_key in assigned_paper_owners:
            continue
        owner_id = paper.get("owner_person_id")
        stable_paper_id = (
            101
            if owner_id == "STUDENT_LEFT"
            else 102
            if owner_id == "STUDENT_RIGHT"
            else 100 + len(assigned_paper_owners) + 1
        )
        try:
            pipeline.assign_paper_id(
                session_id,
                current_paper_id=int(paper["paper_id"]),
                stable_paper_id=stable_paper_id,
            )
        except (AssignmentError, ValueError) as error:
            events.append(
                {
                    "second": round(second, 3),
                    "type": "paper_assignment_failed",
                    "temporary_paper_id": int(paper["paper_id"]),
                    "detail": str(error),
                }
            )
            continue
        assigned_paper_owners.add(owner_key)
        events.append(
            {
                "second": round(second, 3),
                "type": "paper_assigned",
                "temporary_paper_id": int(paper["paper_id"]),
                "paper_id": stable_paper_id,
                "owner_person_id": owner_id,
            }
        )

    return (
        pipeline.manager.get_packet(session_id).to_dict()["tracks"],
        pipeline.manager.get_paper_state(session_id)["papers"],
    )


def _draw_frame(frame: Any, result: dict[str, Any], second: float) -> None:
    object_result = result.get("object_result") or {}
    confirmed = set(object_result.get("confirmed_classes", []))
    for item in object_result.get("raw_objects", []):
        x1, y1, x2, y2 = item["bbox_xyxy"]
        name = item["display_name"]
        if item["is_paper_candidate"]:
            color = (255, 255, 0)
        elif name in confirmed:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        owner_hint = item.get("owner_person_id_hint")
        if owner_hint is None and item.get("owner_track_id_hint") is not None:
            owner_hint = f"track={item['owner_track_id_hint']}"
        label = f"{name} {item['confidence']:.2f}"
        if owner_hint is not None:
            label += f" owner={owner_hint}"
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    for person in result["people"]:
        if not person["is_present"]:
            continue
        x1, y1, x2, y2 = person["bbox_xyxy"]
        person_id = person["person_id"]
        color = (0, 180, 0) if person_id else (0, 165, 255)
        label = (
            f"{person_id} memory="
            f"{'on' if person['appearance_identity_registered'] else 'off'}"
            if person_id
            else f"temp_person={person['track_id']}"
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, min(frame.shape[0] - 8, y2 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
        )

    for paper in result["papers"]:
        if not paper["is_present"]:
            continue
        x1, y1, x2, y2 = paper["bbox_xyxy"]
        color = PAPER_COLORS.get(paper["status"], (255, 255, 0))
        label = (
            f"paper={paper['paper_id']} {paper['status']} "
            f"owner={paper.get('owner_person_id')}"
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
        )

    mode = "ARMED" if result["paper_monitoring_armed"] else "SETUP"
    cv2.putText(
        frame,
        f"{second:6.2f}s | {mode}",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 255) if mode == "ARMED" else (0, 255, 255),
        2,
    )


def run_video(
    video_path: Path,
    *,
    run_root: Path,
    setup_seconds: float,
    frame_stride: int,
    max_seconds: float | None,
    write_annotated_video: bool,
) -> Path:
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = source_width
    height = source_height

    scenario_dir = run_root / video_path.stem
    scenario_dir.mkdir(parents=True, exist_ok=True)
    session_id = f"video_{video_path.stem}_{run_root.name}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    object_detect_interval = settings.object_detect_every_n_frames
    person_detect_interval = 1
    pipeline = PoseGazePaperPipeline(
        person_detector=UltralyticsPersonDetector(
            model_path=Path("weights/yolov8n.pt"),
            confidence_threshold=0.55,
            device=device,
        ),
        object_detector=ObjectDetectModule(
            device=device,
            detect_every_n_frames=object_detect_interval,
            phone_confidence_floor=None,
            enable_person_roi=True,
            enable_custom_paper_roi=True,
            confirm_frames_by_class=None,
        ),
        storage_root=scenario_dir / "tracking",
        max_people=2,
        capture_evidence=True,
        person_detect_every_n_frames=person_detect_interval,
    )
    pipeline.create_session(session_id)

    writer = None
    # Keep the two rendered scenario videos together at the run root with
    # unambiguous names: smartphone_output.mp4 and cheatsheet_output.mp4.
    annotated_path = run_root / f"{video_path.stem}_output.mp4"
    if write_annotated_video:
        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(source_fps / frame_stride, 1.0),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create video: {annotated_path}")

    started = time.perf_counter()
    processed_frames = 0
    object_inference_frames = 0
    source_frame_id = -1
    armed = False
    assigned_person_slots: set[str] = set()
    assigned_paper_owners: set[int | str] = set()
    raw_stats: dict[str, dict[str, Any]] = {}
    raw_detection_events: list[dict[str, Any]] = []
    confirmed_stats: dict[str, dict[str, Any]] = {}
    people_stats: dict[str, dict[str, Any]] = {}
    paper_stats: dict[int, dict[str, Any]] = {}
    paper_status_counts: dict[str, int] = defaultdict(int)
    events: list[dict[str, Any]] = []
    active_confirmed: set[str] = set()
    active_paper_alerts: set[int] = set()
    unassigned_after_setup: set[int] = set()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source_frame_id += 1
            second = (
                source_frame_id / source_fps
                if source_fps > 0
                else float(source_frame_id)
            )
            if max_seconds is not None and second > max_seconds:
                break
            if source_frame_id % frame_stride != 0:
                continue

            processed_frames += 1
            result = pipeline.process_frame(
                frame,
                session_id=session_id,
                frame_id=source_frame_id,
                timestamp_ms=round(second * 1000),
            )

            if second < setup_seconds:
                people, papers = _assign_setup_identities(
                    pipeline,
                    session_id=session_id,
                    frame_width=width,
                    second=second,
                    assigned_person_slots=assigned_person_slots,
                    assigned_paper_owners=assigned_paper_owners,
                    events=events,
                )
                result["people"] = people
                result["papers"] = papers

            if not armed and second >= setup_seconds:
                pipeline.arm_paper_monitoring(session_id)
                armed = True
                result["paper_monitoring_armed"] = True
                events.append(
                    {
                        "second": round(second, 3),
                        "type": "paper_monitoring_armed",
                    }
                )

            object_result = result.get("object_result")
            if object_result and object_result.get("inference_ran"):
                object_inference_frames += 1
                _update_class_stats(
                    raw_stats,
                    object_result.get("raw_objects", []),
                    second,
                )
                for item in object_result.get("raw_objects", []):
                    raw_detection_events.append(
                        {
                            "source_frame_id": source_frame_id,
                            "second": round(second, 3),
                            "class_name": item["class_name"],
                            "source": item.get("source", "unknown"),
                            "confidence": round(
                                float(item["confidence"]),
                                4,
                            ),
                            "bbox_xyxy": list(item["bbox_xyxy"]),
                            "owner_track_id_hint": item.get(
                                "owner_track_id_hint"
                            ),
                            "owner_person_id_hint": item.get(
                                "owner_person_id_hint"
                            ),
                        }
                    )
                current_confirmed = set(
                    object_result.get("confirmed_classes", [])
                )
                for class_name in sorted(current_confirmed):
                    entry = confirmed_stats.setdefault(
                        class_name,
                        {
                            "inference_frames": 0,
                            "first_second": second,
                            "last_second": second,
                        },
                    )
                    entry["inference_frames"] += 1
                    entry["last_second"] = second
                for class_name in sorted(current_confirmed - active_confirmed):
                    events.append(
                        {
                            "second": round(second, 3),
                            "type": "object_alert_started",
                            "class_name": class_name,
                        }
                    )
                for class_name in sorted(active_confirmed - current_confirmed):
                    events.append(
                        {
                            "second": round(second, 3),
                            "type": "object_alert_ended",
                            "class_name": class_name,
                        }
                    )
                active_confirmed = current_confirmed

                for paper in result["papers"]:
                    paper_status_counts[paper["status"]] += 1

            for person in result["people"]:
                if not person["is_present"]:
                    continue
                key = person["person_id"] or f"temp:{person['track_id']}"
                entry = people_stats.setdefault(
                    str(key),
                    {
                        "track_ids": set(),
                        "first_second": second,
                        "last_second": second,
                        "present_processed_frames": 0,
                        "memory_on_frames": 0,
                    },
                )
                entry["track_ids"].add(int(person["track_id"]))
                entry["last_second"] = second
                entry["present_processed_frames"] += 1
                if person["appearance_identity_registered"]:
                    entry["memory_on_frames"] += 1
                if second >= setup_seconds and person["person_id"] is None:
                    track_id = int(person["track_id"])
                    if track_id not in unassigned_after_setup:
                        unassigned_after_setup.add(track_id)
                        events.append(
                            {
                                "second": round(second, 3),
                                "type": "unassigned_person_after_setup",
                                "track_id": track_id,
                            }
                        )

            for paper in result["papers"]:
                paper_id = int(paper["paper_id"])
                entry = paper_stats.setdefault(
                    paper_id,
                    {
                        "first_second": second,
                        "last_second": second,
                        "present_processed_frames": 0,
                        "max_risk_score": 0.0,
                        "statuses": set(),
                        "owners": set(),
                        "paper_id_assigned": False,
                        "memory_registered": False,
                    },
                )
                entry["last_second"] = second
                entry["statuses"].add(str(paper["status"]))
                entry["max_risk_score"] = max(
                    entry["max_risk_score"],
                    float(paper["risk_score"]),
                )
                entry["paper_id_assigned"] = (
                    entry["paper_id_assigned"]
                    or bool(paper["paper_id_assigned"])
                )
                entry["memory_registered"] = (
                    entry["memory_registered"]
                    or bool(paper["appearance_identity_registered"])
                )
                owner = paper.get("owner_person_id")
                if owner is not None:
                    entry["owners"].add(str(owner))
                if paper["is_present"]:
                    entry["present_processed_frames"] += 1

            current_paper_alerts = {
                int(alert["paper_id"])
                for alert in result["alerts"]
                if alert.get("source") == "paper_tracking"
            }
            for paper_id in sorted(current_paper_alerts - active_paper_alerts):
                paper = next(
                    (
                        item
                        for item in result["papers"]
                        if int(item["paper_id"]) == paper_id
                    ),
                    None,
                )
                events.append(
                    {
                        "second": round(second, 3),
                        "type": "paper_alert_started",
                        "paper_id": paper_id,
                        "owner_person_id": (
                            paper.get("owner_person_id") if paper else None
                        ),
                        "reasons": paper.get("reasons", []) if paper else [],
                    }
                )
            for paper_id in sorted(active_paper_alerts - current_paper_alerts):
                events.append(
                    {
                        "second": round(second, 3),
                        "type": "paper_alert_ended",
                        "paper_id": paper_id,
                    }
                )
            active_paper_alerts = current_paper_alerts

            if writer is not None:
                annotated = frame.copy()
                _draw_frame(annotated, result, second)
                writer.write(annotated)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        pipeline.cleanup_session(session_id)

    for entry in confirmed_stats.values():
        entry["first_second"] = round(float(entry["first_second"]), 3)
        entry["last_second"] = round(float(entry["last_second"]), 3)
    serializable_people = {}
    for key, entry in people_stats.items():
        serializable_people[key] = {
            **entry,
            "track_ids": sorted(entry["track_ids"]),
            "first_second": round(float(entry["first_second"]), 3),
            "last_second": round(float(entry["last_second"]), 3),
        }
    serializable_papers = {}
    for paper_id, entry in paper_stats.items():
        serializable_papers[str(paper_id)] = {
            **entry,
            "statuses": sorted(entry["statuses"]),
            "owners": sorted(entry["owners"]),
            "first_second": round(float(entry["first_second"]), 3),
            "last_second": round(float(entry["last_second"]), 3),
            "max_risk_score": round(float(entry["max_risk_score"]), 3),
        }

    report = {
        "video": str(video_path.resolve()),
        "scenario": video_path.stem,
        "session_id": session_id,
        "device": device,
        "source": {
            "fps": source_fps,
            "frames": source_frames,
            "duration_seconds": (
                source_frames / source_fps if source_fps > 0 else None
            ),
            "width": source_width,
            "height": source_height,
        },
        "test_config": {
            "setup_seconds": setup_seconds,
            "frame_stride": frame_stride,
            "processed_fps": source_fps / frame_stride,
            "object_detect_every_n_processed_frames": (
                object_detect_interval
            ),
            "person_detect_every_n_processed_frames": (
                person_detect_interval
            ),
            "profile": "full_output_reference",
            "processing_width": width,
            "processing_height": height,
            "person_roi_enabled": True,
            "custom_paper_roi_enabled": True,
            "smartphone_confirm_frames": 3,
        },
        "processed_frames": processed_frames,
        "object_inference_frames": object_inference_frames,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "raw_object_detections": _finalize_class_stats(raw_stats),
        "raw_detection_events": raw_detection_events,
        "confirmed_object_alerts": confirmed_stats,
        "people": serializable_people,
        "unassigned_person_tracks_after_setup": sorted(unassigned_after_setup),
        "paper_status_inference_counts": dict(
            sorted(paper_status_counts.items())
        ),
        "authorized_papers_at_end": (
            pipeline.manager.get_paper_state(session_id)["authorized_papers"]
        ),
        "papers": serializable_papers,
        "events": events,
        "evidence_files": [
            str(path.resolve())
            for path in sorted(
                (settings.session_log_dir / session_id).glob("**/*")
            )
            if path.is_file()
        ],
        "annotated_video": (
            str(annotated_path.resolve()) if writer is not None else None
        ),
    }
    report_path = scenario_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def _sync_canonical_outputs(
    report_paths: list[Path],
    *,
    output_root: Path,
) -> list[Path]:
    """Keep the latest four user-facing files together like the reference."""

    canonical_dir = output_root / "outputs"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        scenario = str(report["scenario"])
        canonical_report = canonical_dir / f"{scenario}_report.json"
        shutil.copy2(report_path, canonical_report)
        copied.append(canonical_report)

        annotated_value = report.get("annotated_video")
        if not annotated_value:
            continue
        annotated_path = Path(str(annotated_value))
        if not annotated_path.is_file():
            continue
        canonical_video = canonical_dir / f"{scenario}_output.mp4"
        shutil.copy2(annotated_path, canonical_video)
        copied.append(canonical_video)
    return copied


def main() -> None:
    args = _arguments()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = args.output_dir / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    report_paths = []
    for video in args.videos:
        print(f"[video-test] Processing {video}...")
        report_path = run_video(
            video,
            run_root=run_root,
            setup_seconds=args.setup_seconds,
            frame_stride=args.frame_stride,
            max_seconds=args.max_seconds,
            write_annotated_video=not args.no_annotated_video,
        )
        report_paths.append(report_path)
        print(f"[video-test] Report: {report_path}")

    index_path = run_root / "reports.json"
    index_path.write_text(
        json.dumps(
            {"reports": [str(path.resolve()) for path in report_paths]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    canonical_outputs = []
    if not args.no_annotated_video:
        canonical_outputs = _sync_canonical_outputs(
            report_paths,
            output_root=args.output_dir,
        )
    print(f"[video-test] Complete: {index_path}")
    if canonical_outputs:
        print(
            "[video-test] Latest outputs: "
            f"{(args.output_dir / 'outputs').resolve()}"
        )


if __name__ == "__main__":
    main()

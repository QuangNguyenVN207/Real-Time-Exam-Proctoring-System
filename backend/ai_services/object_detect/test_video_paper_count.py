"""Run the count-only paper experiment on recorded videos.

Example from the project root::

    python -m backend.ai_services.object_detect.test_video_paper_count \
        "C:/path/to/smartphone.mp4" "C:/path/to/cheatsheet.mp4"

Unlike ``test_video_scenarios``, this test never assigns or tracks paper IDs.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
from backend.core.config import settings


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test person-aware paper-count changes on video files."
    )
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/paper_count_results"),
    )
    parser.add_argument("--setup-seconds", type=float, default=3.0)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--no-annotated-video", action="store_true")
    return parser.parse_args()


def _person_slot(person: dict[str, Any], frame_width: int) -> str:
    x1, _, x2, _ = person["bbox_xyxy"]
    return "LEFT" if (x1 + x2) / 2.0 < frame_width / 2.0 else "RIGHT"


def _assign_people_during_setup(
    pipeline: PaperCountPipeline,
    *,
    session_id: str,
    frame_width: int,
    second: float,
    assigned_slots: set[str],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    people = pipeline.manager.get_packet(session_id).to_dict()["tracks"]
    present = sorted(
        (person for person in people if person["is_present"]),
        key=lambda person: person["bbox_xyxy"][0],
    )
    for person in present:
        if person["person_id"] is not None:
            continue
        slot = _person_slot(person, frame_width)
        if slot in assigned_slots:
            continue
        person_id = f"STUDENT_{slot}"
        try:
            pipeline.manager.assign_student(
                session_id,
                track_id=int(person["track_id"]),
                student_id=person_id,
            )
        except AssignmentError as error:
            events.append(
                {
                    "second": round(second, 3),
                    "type": "person_assignment_failed",
                    "track_id": int(person["track_id"]),
                    "detail": str(error),
                }
            )
            continue
        assigned_slots.add(slot)
        events.append(
            {
                "second": round(second, 3),
                "type": "person_assigned",
                "track_id": int(person["track_id"]),
                "person_id": person_id,
            }
        )
    return pipeline.manager.get_packet(session_id).to_dict()["tracks"]


def _new_detection_stats() -> dict[str, Any]:
    return {
        "boxes": 0,
        "inference_frames": 0,
        "first_second": None,
        "last_second": None,
        "max_confidence": 0.0,
        "confidence_sum": 0.0,
    }


def _update_detection_stats(
    stats: dict[str, dict[str, Any]],
    detections: list[dict[str, Any]],
    second: float,
) -> None:
    seen: set[str] = set()
    for detection in detections:
        name = str(detection["class_name"])
        entry = stats.setdefault(name, _new_detection_stats())
        confidence = float(detection["confidence"])
        entry["boxes"] += 1
        entry["confidence_sum"] += confidence
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
        seen.add(name)
    for name in seen:
        stats[name]["inference_frames"] += 1


def _serialize_detection_stats(
    stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output = {}
    for name, entry in sorted(stats.items()):
        boxes = int(entry["boxes"])
        output[name] = {
            "boxes": boxes,
            "inference_frames": int(entry["inference_frames"]),
            "first_second": round(float(entry["first_second"]), 3),
            "last_second": round(float(entry["last_second"]), 3),
            "max_confidence": round(float(entry["max_confidence"]), 4),
            "mean_confidence": round(
                float(entry["confidence_sum"]) / max(boxes, 1),
                4,
            ),
        }
    return output


def _draw_label(
    frame: Any,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    *,
    scale: float = 0.52,
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


def _draw_frame(frame: Any, result: dict[str, Any], second: float) -> None:
    object_result = result.get("object_result") or {}
    confirmed = set(object_result.get("confirmed_classes", []))

    # Paper candidates are rendered only after clustering below.  This avoids
    # making one sheet look like several sheets in the output video.
    for item in object_result.get("raw_objects", []):
        if item.get("is_paper_candidate"):
            continue
        x1, y1, x2, y2 = item["bbox_xyxy"]
        color = (0, 0, 255) if item["display_name"] in confirmed else (0, 210, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        owner = item.get("owner_person_id_hint")
        label = f"{item['display_name']} {item['confidence']:.2f}"
        if owner:
            label += f" owner={owner}"
        _draw_label(frame, label, x1, y1 - 8, color)

    for person in result["people"]:
        if not person["is_present"]:
            continue
        x1, y1, x2, y2 = person["bbox_xyxy"]
        person_id = person["person_id"]
        color = (0, 180, 0) if person_id else (0, 165, 255)
        label = (
            f"{person_id} memory=on"
            if person_id and person.get("appearance_identity_registered")
            else person_id or f"temp_person={person['track_id']}"
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _draw_label(frame, label, x1, min(frame.shape[0] - 8, y2 + 22), color)

    # Sort papers by x1 coordinate and stagger label height if adjacent
    papers = result["papers"]
    for idx, paper in enumerate(papers):
        x1, y1, x2, y2 = paper["bbox_xyxy"]
        suspicious = paper["status"] == "suspicious_new_paper"
        color = (0, 0, 255) if suspicious else (255, 220, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = (
            "cheat_sheet NEW"
            if suspicious
            else f"paper observation={paper['observation_index']}"
        )
        if paper.get("owner_person_id"):
            label += f" owner={paper['owner_person_id']}"
        # Stagger height for alternating adjacent paper labels
        offset_y = 8 if (idx % 2 == 0) else 28
        _draw_label(frame, label, x1, y1 - offset_y, color, scale=0.48)

    state = result["paper_count_state"]
    mode = "ARMED" if state["monitoring_armed"] else "SETUP"
    baseline = state["baseline_count"]
    stable = state["stable_count"]
    observed = state["observed_count"]
    color = (0, 0, 255) if state["active_alerts"] else (0, 255, 255)
    _draw_label(
        frame,
        (
            f"{second:6.2f}s | PAPER COUNT {mode} | "
            f"observed={observed} stable={stable} baseline={baseline}"
        ),
        15,
        32,
        color,
        scale=0.66,
    )


def run_video(
    video_path: Path,
    *,
    run_root: Path,
    output_root: Path,
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
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scenario_dir = run_root / video_path.stem
    scenario_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = output_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    session_id = f"paper_count_{video_path.stem}_{run_root.name}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    object_interval = settings.object_detect_every_n_frames
    person_interval = 2
    pipeline = PaperCountPipeline(
        person_detector=UltralyticsPersonDetector(
            Path("weights/yolov8n.pt"),
            confidence_threshold=0.55,
            device=device,
        ),
        object_detector=ObjectDetectModule(
            device=device,
            detect_every_n_frames=object_interval,
            phone_confidence_floor=None,
            enable_person_roi=True,
            enable_custom_paper_roi=True,
            confirm_frames_by_class=None,
        ),
        storage_root=scenario_dir / "tracking",
        max_people=2,
        person_detect_every_n_frames=person_interval,
    )
    pipeline.create_session(session_id)

    annotated_path = outputs_dir / f"{video_path.stem}_paper_count_output.mp4"
    writer = None
    if write_annotated_video:
        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(source_fps / frame_stride, 1.0),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create video: {annotated_path}")

    source_frame_id = -1
    processed_frames = 0
    failed_frames = 0
    inference_frames = 0
    armed = False
    assigned_slots: set[str] = set()
    events: list[dict[str, Any]] = []
    count_events: list[dict[str, Any]] = []
    count_timeline: list[dict[str, Any]] = []
    observed_histogram: Counter[int] = Counter()
    stable_histogram: Counter[int] = Counter()
    paper_status_counts: Counter[str] = Counter()
    object_stats: dict[str, dict[str, Any]] = {}
    object_alert_frames: Counter[str] = Counter()
    people_stats: dict[str, dict[str, Any]] = {}
    final_state: dict[str, Any] = {}
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source_frame_id += 1
            second = source_frame_id / source_fps if source_fps > 0 else 0.0
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
            if result is None:
                failed_frames += 1
                if writer is not None:
                    writer.write(frame)
                continue

            if second < setup_seconds:
                result["people"] = _assign_people_during_setup(
                    pipeline,
                    session_id=session_id,
                    frame_width=width,
                    second=second,
                    assigned_slots=assigned_slots,
                    events=events,
                )

            if not armed and second >= setup_seconds:
                final_state = pipeline.arm_paper_monitoring(session_id)
                armed = True
                result["paper_count_state"] = final_state
                result["papers"] = final_state["papers"]
                result["paper_monitoring_armed"] = True
                events.append(
                    {
                        "second": round(second, 3),
                        "type": "paper_count_monitoring_armed",
                        "baseline_count": final_state["baseline_count"],
                    }
                )

            object_result = result.get("object_result") or {}
            if object_result.get("inference_ran"):
                inference_frames += 1
                _update_detection_stats(
                    object_stats,
                    object_result.get("raw_objects", []),
                    second,
                )
                for name in object_result.get("confirmed_classes", []):
                    object_alert_frames[str(name)] += 1
                state = result["paper_count_state"]
                final_state = state
                observed_histogram[int(state["observed_count"])] += 1
                if state["stable_count"] is not None:
                    stable_histogram[int(state["stable_count"])] += 1
                for paper in state["papers"]:
                    paper_status_counts[str(paper["status"])] += 1
                count_timeline.append(
                    {
                        "source_frame_id": source_frame_id,
                        "second": round(second, 3),
                        "observed_count": state["observed_count"],
                        "stable_count": state["stable_count"],
                        "baseline_count": state["baseline_count"],
                        "candidate_count": state["candidate_count"],
                        "candidate_streak": state["candidate_streak"],
                    }
                )
                for event in state["new_events"]:
                    serialized = {**event, "second": round(second, 3)}
                    count_events.append(serialized)
                    events.append(serialized)

            for person in result["people"]:
                if not person["is_present"]:
                    continue
                key = str(person["person_id"] or f"temp:{person['track_id']}")
                entry = people_stats.setdefault(
                    key,
                    {
                        "track_ids": set(),
                        "first_second": second,
                        "last_second": second,
                        "present_frames": 0,
                        "memory_on_frames": 0,
                    },
                )
                entry["track_ids"].add(int(person["track_id"]))
                entry["last_second"] = second
                entry["present_frames"] += 1
                if person.get("appearance_identity_registered"):
                    entry["memory_on_frames"] += 1

            if writer is not None:
                annotated = frame.copy()
                _draw_frame(annotated, result, second)
                writer.write(annotated)

            if processed_frames % 100 == 0:
                elapsed = max(time.perf_counter() - started, 0.001)
                print(
                    f"[paper-count-test] {video_path.name}: "
                    f"{second:.1f}s, {processed_frames / elapsed:.2f} processed FPS"
                )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        pipeline.cleanup_session(session_id)

    serializable_people = {
        key: {
            **entry,
            "track_ids": sorted(entry["track_ids"]),
            "first_second": round(float(entry["first_second"]), 3),
            "last_second": round(float(entry["last_second"]), 3),
        }
        for key, entry in people_stats.items()
    }
    report = {
        "video": str(video_path.resolve()),
        "scenario": video_path.stem,
        "mode": "paper_count_only_no_paper_id",
        "device": device,
        "source": {
            "fps": source_fps,
            "frames": source_frames,
            "duration_seconds": (
                source_frames / source_fps if source_fps > 0 else None
            ),
            "width": width,
            "height": height,
        },
        "test_config": {
            "setup_seconds": setup_seconds,
            "frame_stride": frame_stride,
            "object_detect_every_n_processed_frames": object_interval,
            "person_detect_every_n_processed_frames": person_interval,
            "paper_count_confirm_inferences": (
                settings.paper_count_confirm_inferences
            ),
            "book_as_cheat_sheet": settings.book_as_cheatsheet_enabled,
        },
        "processed_frames": processed_frames,
        "failed_frames": failed_frames,
        "object_inference_frames": inference_frames,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "baseline_count": final_state.get("baseline_count"),
        "final_stable_count": final_state.get("stable_count"),
        "active_paper_alerts_at_end": final_state.get("active_alerts", []),
        "observed_count_inference_histogram": {
            str(key): value for key, value in sorted(observed_histogram.items())
        },
        "stable_count_inference_histogram": {
            str(key): value for key, value in sorted(stable_histogram.items())
        },
        "paper_observation_status_counts": dict(paper_status_counts),
        "paper_count_events": count_events,
        "count_timeline": count_timeline,
        "raw_object_detections": _serialize_detection_stats(object_stats),
        "confirmed_object_alert_inference_frames": dict(object_alert_frames),
        "people": serializable_people,
        "events": events,
        "annotated_video": (
            str(annotated_path.resolve()) if write_annotated_video else None
        ),
    }
    report_path = outputs_dir / f"{video_path.stem}_paper_count_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    args = _arguments()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = args.output_dir / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for video in args.videos:
        print(f"[paper-count-test] Processing {video}...")
        report = run_video(
            video,
            run_root=run_root,
            output_root=args.output_dir,
            setup_seconds=args.setup_seconds,
            frame_stride=args.frame_stride,
            max_seconds=args.max_seconds,
            write_annotated_video=not args.no_annotated_video,
        )
        reports.append(report)
        print(f"[paper-count-test] Report: {report}")
    index_path = run_root / "reports.json"
    index_path.write_text(
        json.dumps(
            {"reports": [str(report.resolve()) for report in reports]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[paper-count-test] Complete: {index_path}")
    print(f"[paper-count-test] Outputs: {(args.output_dir / 'outputs').resolve()}")


if __name__ == "__main__":
    main()

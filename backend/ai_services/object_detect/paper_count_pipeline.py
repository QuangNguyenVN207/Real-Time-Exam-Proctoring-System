"""Person-aware, count-only paper monitoring pipeline."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from backend.ai_services.pose_gaze.tracking.manager import TrackingManager
from backend.ai_services.pose_gaze.tracking.tracker import (
    person_fingerprint_from_frame,
)
from backend.core.config import settings

from .paper_count_monitor import CountBasedPaperMonitor


class PaperCountPipeline:
    """Keep stable people, but decide paper violations from count changes.

    This pipeline intentionally never calls ``process_paper_detections`` and
    never calculates a paper appearance fingerprint.  Consequently no
    ``paper_id`` or paper tracker is involved in the decision.
    """

    def __init__(
        self,
        *,
        person_detector: Any,
        object_detector: Any,
        tracking_manager: TrackingManager | None = None,
        paper_monitor: CountBasedPaperMonitor | None = None,
        storage_root: Path | None = None,
        max_people: int = 2,
        person_detect_every_n_frames: int = 1,
    ) -> None:
        if person_detect_every_n_frames < 1:
            raise ValueError("person_detect_every_n_frames must be at least 1")
        self.person_detector = person_detector
        self.object_detector = object_detector
        self.person_detect_every_n_frames = person_detect_every_n_frames
        self.manager = tracking_manager or TrackingManager(
            storage_root or settings.session_log_dir,
            max_tracks=max_people,
            person_appearance_match_threshold=(
                settings.person_appearance_match_threshold
            ),
            paper_registration_frames=settings.paper_registration_frames,
            paper_alert_confirm_frames=settings.paper_alert_confirm_frames,
            paper_max_missed_frames=settings.paper_max_missed_frames,
            paper_auto_register_first=settings.paper_auto_register_first,
            paper_appearance_match_threshold=(
                settings.paper_appearance_match_threshold
            ),
        )
        self.paper_monitor = paper_monitor or CountBasedPaperMonitor(
            confirmation_frames=settings.paper_count_confirm_inferences,
            duplicate_overlap_threshold=(
                settings.paper_count_duplicate_overlap_threshold
            ),
            duplicate_center_distance_ratio=(
                settings.paper_count_duplicate_center_distance_ratio
            ),
        )
        self._person_frames_seen: dict[str, int] = {}
        self._cached_person_detections: dict[str, list[Any]] = {}

    def create_session(
        self,
        session_id: str,
        *,
        restore_existing: bool = False,
    ) -> None:
        self.manager.create_session(
            session_id,
            restore_existing=restore_existing,
        )
        self.paper_monitor.create_session(session_id)

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        session_id: str,
        frame_id: int,
        timestamp_ms: int | None = None,
    ) -> dict[str, Any] | None:
        """Process safely so a damaged OpenCV frame cannot stop the stream."""

        try:
            return self._process_frame(
                frame,
                session_id=session_id,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
            )
        except Exception as error:  # noqa: BLE001 - realtime safety boundary
            print(
                "[paper_count][ERROR] Frame skipped: "
                f"{type(error).__name__}: {error}"
            )
            return None

    def _process_frame(
        self,
        frame: np.ndarray,
        *,
        session_id: str,
        frame_id: int,
        timestamp_ms: int | None,
    ) -> dict[str, Any]:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy array")
        self.manager.ensure_session(session_id)
        self.paper_monitor.ensure_session(session_id)
        resolved_timestamp_ms = (
            int(timestamp_ms)
            if timestamp_ms is not None
            else int(time.time() * 1000)
        )

        person_seen = self._person_frames_seen.get(session_id, 0) + 1
        self._person_frames_seen[session_id] = person_seen
        run_person_inference = (
            session_id not in self._cached_person_detections
            or (person_seen - 1) % self.person_detect_every_n_frames == 0
        )
        if run_person_inference:
            person_detections = []
            for detection in self.person_detector.detect(frame):
                fingerprint = person_fingerprint_from_frame(
                    frame,
                    detection.bbox,
                )
                if fingerprint is not None:
                    detection = replace(
                        detection,
                        appearance_fingerprint=fingerprint,
                    )
                person_detections.append(detection)
            self._cached_person_detections[session_id] = person_detections
        else:
            person_detections = list(
                self._cached_person_detections[session_id]
            )

        people_packet = self.manager.process_detections(
            session_id,
            frame_id=frame_id,
            timestamp_ms=resolved_timestamp_ms,
            detections=person_detections,
        )
        people = people_packet.to_dict()["tracks"]
        person_rois = [
            {
                "bbox_xyxy": person["bbox_xyxy"],
                "track_id": person["track_id"],
                "person_id": person["person_id"],
            }
            for person in people
            if person["is_present"]
        ]
        object_result = self.object_detector.process(
            frame,
            session_id,
            frame_id,
            person_rois=person_rois,
        )
        if object_result is not None:
            object_result.setdefault("model_capabilities", {})[
                "paper_decision_mode"
            ] = "count_only_no_paper_id"
        if object_result is not None and object_result.get("inference_ran", False):
            paper_state = self.paper_monitor.update(
                session_id,
                paper_detections=object_result.get("paper_detections", []),
                people=people,
                frame_id=frame_id,
                timestamp_ms=resolved_timestamp_ms,
            )
        else:
            paper_state = self.paper_monitor.get_state(session_id)

        alerts = list(paper_state["active_alerts"])
        if object_result is not None and object_result.get("label") != "clear":
            alerts.append(
                {
                    "source": "object_detect",
                    "label": object_result["label"],
                    "risk_score": float(object_result["risk_score"]),
                    "confirmed_classes": object_result.get(
                        "confirmed_classes",
                        [],
                    ),
                }
            )

        return {
            "session_id": session_id,
            "frame_id": frame_id,
            "timestamp_ms": resolved_timestamp_ms,
            "people": people,
            "papers": paper_state["papers"],
            "paper_count_state": paper_state,
            "paper_monitoring_armed": paper_state["monitoring_armed"],
            "object_result": object_result,
            "alerts": alerts,
            "risk_score": max(
                (float(alert.get("risk_score", 0.0)) for alert in alerts),
                default=0.0,
            ),
        }

    def arm_paper_monitoring(self, session_id: str) -> dict[str, Any]:
        return self.paper_monitor.arm(session_id)

    def cleanup_session(self, session_id: str) -> None:
        self.object_detector.cleanup_session(session_id)
        self.paper_monitor.cleanup_session(session_id)
        self._person_frames_seen.pop(session_id, None)
        self._cached_person_detections.pop(session_id, None)
        try:
            self.manager.close_session(session_id)
        except KeyError:
            pass

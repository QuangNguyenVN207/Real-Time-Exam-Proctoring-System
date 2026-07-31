"""Combined person + paper pipeline built on the pose/gaze tracking session."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from backend.core.config import settings
from backend.ai_services.face_verify.identity_guard import IdentityGuard

from .tracking.manager import TrackingManager
from .tracking.paper_tracking import (
    paper_detection_from_dict,
    paper_fingerprint_from_frame,
)
from .tracking.tracker import person_fingerprint_from_frame


class PoseGazePaperPipeline:
    """Run both detectors and make identity-aware paper decisions per frame."""

    def __init__(
        self,
        *,
        person_detector: Any,
        object_detector: Any,
        tracking_manager: TrackingManager | None = None,
        storage_root: Path | None = None,
        max_people: int = 2,
        capture_evidence: bool = True,
        person_detect_every_n_frames: int = 1,
        face_verifier: Any | None = None,
        identity_guard: IdentityGuard | None = None,
        identity_scan_every_n_frames: int = 5,
        identity_assignment_confirmations: int = 3,
        identity_mismatch_confirmations: int = 3,
    ) -> None:
        if person_detect_every_n_frames < 1:
            raise ValueError(
                "person_detect_every_n_frames must be at least 1"
            )
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
        self.capture_evidence = capture_evidence
        if face_verifier is not None and identity_guard is not None:
            raise ValueError("Pass face_verifier or identity_guard, not both")
        self.identity_guard = identity_guard
        if face_verifier is not None:
            self.identity_guard = IdentityGuard(
                face_verifier,
                self.manager,
                scan_every_n_frames=identity_scan_every_n_frames,
                assignment_confirmations=identity_assignment_confirmations,
                mismatch_confirmations=identity_mismatch_confirmations,
            )
        self._active_paper_alerts: dict[str, set[int]] = {}
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

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        session_id: str,
        frame_id: int,
        timestamp_ms: int | None = None,
    ) -> dict[str, Any]:
        self.manager.ensure_session(session_id)
        resolved_timestamp_ms = (
            timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        )

        person_seen = self._person_frames_seen.get(session_id, 0) + 1
        self._person_frames_seen[session_id] = person_seen
        run_person_inference = (
            session_id not in self._cached_person_detections
            or (
                (person_seen - 1)
                % self.person_detect_every_n_frames
                == 0
            )
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
        identity_alerts: list[dict[str, Any]] = []
        if self.identity_guard is not None:
            identity_alerts = self.identity_guard.sync(
                session_id,
                frame,
                resolved_timestamp_ms / 1000.0,
            )
            # Auto-assignment may have remapped a temporary track onto an old,
            # stable ID. Always build downstream ROIs from the refreshed packet.
            people_packet = self.manager.get_packet(session_id)

        person_rois = [
            {
                "bbox_xyxy": track.bbox.to_list(),
                "track_id": track.track_id,
                "person_id": track.student_id,
            }
            for track in people_packet.tracks
            if track.is_present
        ]
        object_result = self.object_detector.process(
            frame,
            session_id,
            frame_id,
            person_rois=person_rois,
        )
        if object_result is not None and object_result.get("inference_ran", False):
            paper_detections = []
            for payload in object_result.get("paper_detections", []):
                detection = paper_detection_from_dict(payload)
                fingerprint = paper_fingerprint_from_frame(
                    frame,
                    detection.bbox,
                )
                if fingerprint is not None:
                    detection = replace(
                        detection,
                        appearance_fingerprint=fingerprint,
                    )
                paper_detections.append(detection)
            paper_state = self.manager.process_paper_detections(
                session_id,
                detections=paper_detections,
                supports_test_paper=bool(
                    object_result.get("model_capabilities", {}).get(
                        "supports_test_paper",
                        False,
                    )
                ),
            )
        else:
            paper_state = self.manager.get_paper_state(session_id)

        paper_alerts = paper_state["alerts"]
        direct_object_alert = (
            object_result
            if object_result is not None and object_result.get("label") != "clear"
            else None
        )
        alerts: list[dict[str, Any]] = []
        for identity_alert in identity_alerts:
            identity_alert.setdefault("source", "face_verify")
            identity_alert.setdefault("risk_score", 1.0)
            alerts.append(identity_alert)
        if direct_object_alert is not None:
            alerts.append(
                {
                    "source": "object_detect",
                    "label": direct_object_alert["label"],
                    "risk_score": direct_object_alert["risk_score"],
                    "confirmed_classes": direct_object_alert.get(
                        "confirmed_classes",
                        [],
                    ),
                }
            )
        alerts.extend(
            {
                "source": "paper_tracking",
                "label": "possible_cheat_sheet",
                **alert,
            }
            for alert in paper_alerts
        )

        self._capture_new_paper_alerts(
            frame,
            session_id=session_id,
            frame_id=frame_id,
            paper_alerts=paper_alerts,
        )

        return {
            "session_id": session_id,
            "frame_id": frame_id,
            "timestamp_ms": resolved_timestamp_ms,
            "people": people_packet.to_dict()["tracks"],
            "papers": paper_state["papers"],
            "paper_monitoring_armed": paper_state["monitoring_armed"],
            "authorized_papers": paper_state["authorized_papers"],
            "object_result": object_result,
            "alerts": alerts,
            "risk_score": max(
                [float(alert["risk_score"]) for alert in alerts],
                default=0.0,
            ),
        }

    def register_exam_paper(
        self,
        session_id: str,
        *,
        owner_track_id: int,
        paper_id: int,
        replace: bool = False,
    ) -> dict[str, Any]:
        return self.manager.register_exam_paper(
            session_id,
            owner_track_id=owner_track_id,
            paper_id=paper_id,
            replace=replace,
        )

    def assign_paper_id(
        self,
        session_id: str,
        *,
        current_paper_id: int,
        stable_paper_id: int,
    ) -> dict[str, Any]:
        return self.manager.assign_paper_id(
            session_id,
            current_paper_id=current_paper_id,
            stable_paper_id=stable_paper_id,
        )

    def arm_paper_monitoring(self, session_id: str) -> dict[str, Any]:
        return self.manager.arm_paper_monitoring(session_id)

    def cleanup_session(self, session_id: str) -> None:
        self.object_detector.cleanup_session(session_id)
        if self.identity_guard is not None:
            self.identity_guard.cleanup_session(session_id)
        self._active_paper_alerts.pop(session_id, None)
        self._person_frames_seen.pop(session_id, None)
        self._cached_person_detections.pop(session_id, None)

    def _capture_new_paper_alerts(
        self,
        frame: np.ndarray,
        *,
        session_id: str,
        frame_id: int,
        paper_alerts: list[dict[str, Any]],
    ) -> None:
        current_alert_ids = {int(alert["paper_id"]) for alert in paper_alerts}
        previous_alert_ids = self._active_paper_alerts.get(session_id, set())
        new_alert_ids = current_alert_ids - previous_alert_ids
        self._active_paper_alerts[session_id] = current_alert_ids

        if not self.capture_evidence or not new_alert_ids:
            return
        try:
            import cv2
        except ImportError:  # pragma: no cover - production dependency
            return

        new_alerts = [
            alert for alert in paper_alerts if int(alert["paper_id"]) in new_alert_ids
        ]
        session_dir = settings.session_log_dir / session_id
        snapshot_dir = session_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.time()
        paper_tag = "-".join(str(paper_id) for paper_id in sorted(new_alert_ids))
        snapshot_filename = (
            f"paper_tracking_ids-{paper_tag}_{frame_id}_{int(timestamp)}.jpg"
        )
        snapshot_path = snapshot_dir / snapshot_filename
        annotated = frame.copy()

        for alert in new_alerts:
            x1, y1, x2, y2 = alert["bbox_xyxy"]
            owner = alert.get("owner_track_id")
            label = f"possible cheat paper #{alert['paper_id']} owner={owner}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        cv2.imwrite(str(snapshot_path), annotated)
        log_entry = {
            "module": "paper_tracking",
            "status": "alert",
            "timestamp": timestamp,
            "session_id": session_id,
            "frame_id": frame_id,
            "snapshot_file": snapshot_filename,
            "papers": new_alerts,
        }
        try:
            with (session_dir / "paper_tracking_log.jsonl").open(
                "a",
                encoding="utf-8",
            ) as output:
                output.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except OSError as error:
            print(f"[paper_tracking][ERROR] Could not write evidence log: {error}")

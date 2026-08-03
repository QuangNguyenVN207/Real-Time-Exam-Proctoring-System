from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.ai_services.pose_gaze.paper_pipeline import PoseGazePaperPipeline
from backend.ai_services.pose_gaze.tracking.manager import TrackingManager
from backend.ai_services.pose_gaze.tracking.schemas import (
    BoundingBox,
    PersonDetection,
)


class _FakePersonDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, _frame):
        self.calls += 1
        return [PersonDetection(BoundingBox(0, 0, 400, 650), 0.95)]


class _FakeObjectDetector:
    def __init__(self) -> None:
        self.paper_boxes: list[list[int]] = [[60, 430, 180, 570]]
        self.person_rois = None

    def process(
        self,
        _frame,
        _session_id: str,
        frame_id: int,
        *,
        person_rois=None,
    ):
        self.person_rois = person_rois
        return {
            "label": "clear",
            "risk_score": 0.0,
            "inference_ran": True,
            "frame_id": frame_id,
            "paper_detections": [
                {
                    "bbox_xyxy": bbox,
                    "confidence": 0.9,
                    "class_name": "cheat_sheet",
                }
                for bbox in self.paper_boxes
            ],
            "model_capabilities": {"supports_test_paper": False},
        }

    def cleanup_session(self, _session_id: str) -> None:
        return None


class PoseGazePaperPipelineTests(unittest.TestCase):
    def test_live_mode_reuses_person_detection_between_inferences(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            people = _FakePersonDetector()
            objects = _FakeObjectDetector()
            pipeline = PoseGazePaperPipeline(
                person_detector=people,
                object_detector=objects,
                tracking_manager=TrackingManager(
                    Path(directory),
                    max_tracks=1,
                ),
                capture_evidence=False,
                person_detect_every_n_frames=2,
            )
            frame = np.zeros((720, 480, 3), dtype=np.uint8)

            for frame_id in range(1, 6):
                result = pipeline.process_frame(
                    frame,
                    session_id="live_cache",
                    frame_id=frame_id,
                )

            self.assertEqual(people.calls, 3)
            self.assertEqual(len(result["people"]), 1)
            self.assertTrue(result["people"][0]["is_present"])

    def test_end_to_end_second_paper_alert(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(
                Path(directory),
                max_tracks=1,
                paper_registration_frames=2,
                paper_alert_confirm_frames=2,
            )
            objects = _FakeObjectDetector()
            pipeline = PoseGazePaperPipeline(
                person_detector=_FakePersonDetector(),
                object_detector=objects,
                tracking_manager=manager,
                capture_evidence=False,
            )
            frame = np.zeros((720, 480, 3), dtype=np.uint8)

            for frame_id in range(1, 3):
                result = pipeline.process_frame(
                    frame,
                    session_id="integration_room",
                    frame_id=frame_id,
                )
            self.assertEqual(
                result["papers"][0]["status"],
                "authorized_exam_paper",
            )
            self.assertEqual(len(objects.person_rois), 1)
            self.assertEqual(
                objects.person_rois[0]["bbox_xyxy"],
                [0, 0, 400, 650],
            )

            objects.paper_boxes = [
                [60, 430, 180, 570],
                [235, 440, 365, 580],
            ]
            for frame_id in range(3, 6):
                result = pipeline.process_frame(
                    frame,
                    session_id="integration_room",
                    frame_id=frame_id,
                )

            paper_alerts = [
                alert
                for alert in result["alerts"]
                if alert["source"] == "paper_tracking"
            ]
            self.assertEqual(len(paper_alerts), 1)
            self.assertEqual(paper_alerts[0]["label"], "possible_cheat_sheet")
            self.assertIn("additional_paper", paper_alerts[0]["reasons"])


if __name__ == "__main__":
    unittest.main()

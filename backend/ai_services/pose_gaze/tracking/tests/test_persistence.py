from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.ai_services.pose_gaze.tracking.manager import TrackingManager
from backend.ai_services.pose_gaze.tracking.paper_tracking import PaperDetection
from backend.ai_services.pose_gaze.tracking.schemas import (
    BoundingBox,
    PersonDetection,
)


PERSON = (0.8, 0.2, 0.1, 0.05, 0.3, 0.12, 0.4, 0.6)
PAPER = (0.1, 0.7, 0.2, 0.4, 0.05, 0.2, 0.6, 0.3)


class TrackingPersistenceTests(unittest.TestCase):
    def test_person_and_paper_identity_survive_manager_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory)
            first = TrackingManager(
                storage,
                max_tracks=1,
                paper_registration_frames=1,
            )
            first.create_session("exam")
            people = first.process_detections(
                "exam",
                frame_id=1,
                timestamp_ms=1000,
                detections=[
                    PersonDetection(
                        BoundingBox(10, 10, 210, 310),
                        0.95,
                        appearance_fingerprint=PERSON,
                    )
                ],
            )
            first.assign_student(
                "exam",
                track_id=people.tracks[0].track_id,
                student_id="SV_A",
            )
            first.process_paper_detections(
                "exam",
                detections=[
                    PaperDetection(
                        BoundingBox(40, 220, 180, 300),
                        0.9,
                        "test_paper",
                        appearance_fingerprint=PAPER,
                    )
                ],
                supports_test_paper=True,
            )
            first.assign_paper_id(
                "exam",
                current_paper_id=1,
                stable_paper_id=42,
            )
            first.close_session("exam")

            restarted = TrackingManager(
                storage,
                max_tracks=1,
                paper_registration_frames=1,
            )
            restarted.restore_session("exam")
            restored_before_frame = restarted.get_packet("exam")
            self.assertFalse(restored_before_frame.tracks[0].is_present)
            self.assertEqual(restored_before_frame.tracks[0].student_id, "SV_A")

            restored_people = restarted.process_detections(
                "exam",
                frame_id=2,
                timestamp_ms=2000,
                detections=[
                    PersonDetection(
                        BoundingBox(500, 10, 700, 310),
                        0.94,
                        appearance_fingerprint=PERSON,
                    )
                ],
            )
            self.assertEqual(restored_people.tracks[0].track_id, 1)
            self.assertEqual(restored_people.tracks[0].student_id, "SV_A")

            restored_papers = restarted.process_paper_detections(
                "exam",
                detections=[
                    PaperDetection(
                        BoundingBox(530, 220, 670, 300),
                        0.88,
                        "test_paper",
                        appearance_fingerprint=PAPER,
                    )
                ],
                supports_test_paper=True,
            )
            self.assertEqual(restored_papers["papers"][0]["paper_id"], 42)
            self.assertTrue(restored_papers["papers"][0]["authorized"])


if __name__ == "__main__":
    unittest.main()

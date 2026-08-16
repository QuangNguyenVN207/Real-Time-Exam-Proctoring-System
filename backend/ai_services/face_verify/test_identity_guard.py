from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.ai_services.face_verify.identity_guard import IdentityGuard
from backend.ai_services.pose_gaze.tracking.manager import TrackingManager
from backend.ai_services.pose_gaze.tracking.schemas import (
    BoundingBox,
    PersonDetection,
)


def detection(x1: int, y1: int, x2: int, y2: int) -> PersonDetection:
    return PersonDetection(BoundingBox(x1, y1, x2, y2), 0.95)


class FakeFaceVerifier:
    def __init__(self) -> None:
        self.identity = "SV_A"
        self.mismatch = False

    def identify(self, _frame, _bbox):
        return (self.identity, 0.91)

    def verify_assigned_identity(
        self,
        _frame,
        bbox,
        expected_student_id,
        timestamp,
    ):
        if not self.mismatch:
            return None
        return {
            "module": "face_verify",
            "status": "alert",
            "timestamp": timestamp,
            "message": "identity_mismatch",
            "details": {
                "expected_student_id": expected_student_id,
                "matched_student_id": "SV_B",
                "bbox": bbox,
            },
        }


class IdentityGuardTests(unittest.TestCase):
    def test_face_confirmation_restores_original_track_id_after_reentry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(
                Path(directory),
                max_tracks=2,
                max_missed_frames=0,
            )
            manager.create_session("exam")
            face = FakeFaceVerifier()
            guard = IdentityGuard(
                face,
                manager,
                scan_every_n_frames=1,
                assignment_confirmations=2,
            )

            first = manager.process_detections(
                "exam",
                frame_id=1,
                timestamp_ms=1000,
                detections=[detection(10, 10, 110, 210)],
            )
            original_track_id = first.tracks[0].track_id
            guard.sync("exam", object(), 1.0)
            manager.process_detections(
                "exam",
                frame_id=2,
                timestamp_ms=2000,
                detections=[detection(12, 10, 112, 210)],
            )
            guard.sync("exam", object(), 2.0)
            self.assertEqual(
                manager.get_packet("exam").tracks[0].student_id,
                "SV_A",
            )

            manager.process_detections(
                "exam",
                frame_id=3,
                timestamp_ms=3000,
                detections=[],
            )
            returned = manager.process_detections(
                "exam",
                frame_id=4,
                timestamp_ms=4000,
                detections=[detection(400, 10, 500, 210)],
            )
            temporary_track_id = next(
                track.track_id for track in returned.tracks if track.is_present
            )
            self.assertNotEqual(temporary_track_id, original_track_id)

            guard.sync("exam", object(), 4.0)
            manager.process_detections(
                "exam",
                frame_id=5,
                timestamp_ms=5000,
                detections=[detection(402, 10, 502, 210)],
            )
            guard.sync("exam", object(), 5.0)

            restored = manager.get_packet("exam")
            visible = [track for track in restored.tracks if track.is_present]
            self.assertEqual(len(visible), 1)
            self.assertEqual(visible[0].track_id, original_track_id)
            self.assertEqual(visible[0].student_id, "SV_A")

    def test_mismatch_requires_multiple_face_scans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(Path(directory), max_tracks=1)
            manager.create_session("exam")
            packet = manager.process_detections(
                "exam",
                frame_id=1,
                timestamp_ms=1000,
                detections=[detection(10, 10, 110, 210)],
            )
            manager.assign_student(
                "exam",
                track_id=packet.tracks[0].track_id,
                student_id="SV_A",
            )
            face = FakeFaceVerifier()
            face.mismatch = True
            guard = IdentityGuard(
                face,
                manager,
                scan_every_n_frames=1,
                mismatch_confirmations=2,
            )

            self.assertEqual(guard.sync("exam", object(), 1.0), [])
            alerts = guard.sync("exam", object(), 2.0)

            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]["message"], "identity_mismatch")
            self.assertEqual(alerts[0]["details"]["confirmation_count"], 2)


if __name__ == "__main__":
    unittest.main()

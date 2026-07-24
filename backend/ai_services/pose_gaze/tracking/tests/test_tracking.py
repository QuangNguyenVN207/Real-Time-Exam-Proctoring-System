from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.ai_services.pose_gaze.tracking.manager import AssignmentError, TrackingManager
from backend.ai_services.pose_gaze.tracking.schemas import BoundingBox, PersonDetection
from backend.ai_services.pose_gaze.tracking.tracker import IoUPersonTracker


def detection(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> PersonDetection:
    return PersonDetection(BoundingBox(x1, y1, x2, y2), confidence)


class IoUPersonTrackerTests(unittest.TestCase):
    def test_track_ids_remain_stable_when_students_move_slightly(self) -> None:
        tracker = IoUPersonTracker(max_tracks=2, min_iou=0.2)
        first = tracker.update([detection(10, 20, 110, 220), detection(300, 20, 400, 220)])
        second = tracker.update([detection(14, 22, 114, 222), detection(295, 20, 395, 220)])

        self.assertEqual([track.track_id for track in first], [1, 2])
        self.assertEqual([track.track_id for track in second], [1, 2])
        self.assertTrue(all(track.is_present for track in second))

    def test_missing_track_is_not_reported_as_present(self) -> None:
        tracker = IoUPersonTracker(max_tracks=2, max_missed_frames=2)
        tracker.update([detection(10, 20, 110, 220)])
        result = tracker.update([])

        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].is_present)
        self.assertEqual(result[0].missed_frames, 1)


class TrackingManagerTests(unittest.TestCase):
    def test_assignment_is_included_in_pose_gaze_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(Path(directory), max_tracks=2)
            manager.create_session("room_01")
            packet = manager.process_detections(
                "room_01",
                frame_id=10,
                timestamp_ms=1000,
                detections=[detection(10, 20, 110, 220), detection(300, 20, 400, 220)],
            )
            manager.assign_student("room_01", track_id=packet.tracks[0].track_id, student_id="SV_A")
            manager.assign_student("room_01", track_id=packet.tracks[1].track_id, student_id="SV_B")

            handoff = manager.get_pose_gaze_input("room_01")
            self.assertTrue(handoff["ready"])
            self.assertEqual({track["student_id"] for track in handoff["tracks"]}, {"SV_A", "SV_B"})

    def test_student_cannot_be_assigned_to_two_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(Path(directory), max_tracks=2)
            manager.create_session("room_02")
            packet = manager.process_detections(
                "room_02",
                frame_id=1,
                timestamp_ms=1,
                detections=[detection(10, 20, 110, 220), detection(300, 20, 400, 220)],
            )
            manager.assign_student("room_02", track_id=packet.tracks[0].track_id, student_id="SV_A")
            with self.assertRaises(AssignmentError):
                manager.assign_student("room_02", track_id=packet.tracks[1].track_id, student_id="SV_A")


if __name__ == "__main__":
    unittest.main()

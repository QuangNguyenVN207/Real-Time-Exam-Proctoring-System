"""Regression tests for person tracking and session persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pose_gaze.tracking.manager import AssignmentError, TrackingManager
from pose_gaze.tracking.schemas import BoundingBox, PersonDetection
from pose_gaze.tracking.tracker import IoUPersonTracker
from pose_gaze.tracking.webcam import (
    PersonTrackingConfig,
    PersonTrackingModule,
    ProcessingRateController,
)


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

    def test_high_confidence_outsider_does_not_hide_existing_students(self) -> None:
        tracker = IoUPersonTracker(max_tracks=2, min_iou=0.2)
        tracker.update(
            [
                detection(10, 20, 110, 220, confidence=0.70),
                detection(300, 20, 400, 220, confidence=0.65),
            ]
        )

        result = tracker.update(
            [
                detection(500, 20, 600, 220, confidence=0.99),
                detection(14, 22, 114, 222, confidence=0.75),
                detection(296, 20, 396, 220, confidence=0.72),
            ]
        )

        self.assertEqual([track.track_id for track in result], [1, 2])
        self.assertTrue(all(track.is_present for track in result))
        self.assertEqual(result[0].bbox, BoundingBox(14, 22, 114, 222))
        self.assertEqual(result[1].bbox, BoundingBox(296, 20, 396, 220))

    def test_new_track_slots_prefer_confident_foreground_detection(self) -> None:
        tracker = IoUPersonTracker(max_tracks=1)

        result = tracker.update(
            [
                detection(10, 20, 60, 120, confidence=0.95),
                detection(100, 20, 260, 340, confidence=0.80),
            ]
        )

        self.assertEqual(result[0].bbox, BoundingBox(100, 20, 260, 340))

    def test_tracking_telemetry_counts_duplicates_matches_and_misses(self) -> None:
        tracker = IoUPersonTracker(max_tracks=1, min_iou=0.2)
        tracker.update([detection(10, 20, 110, 220), detection(12, 22, 108, 218)])
        tracker.update([])

        self.assertEqual(
            tracker.telemetry(),
            {
                "frames": 2,
                "detections_seen": 2,
                "duplicates_suppressed": 1,
                "matches": 0,
                "new_tracks": 1,
                "misses": 1,
            },
        )


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

    def test_session_and_assignments_are_restored_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage_root = Path(directory)
            first_manager = TrackingManager(storage_root, max_tracks=2)
            first_manager.create_session("room_restart")
            packet = first_manager.process_detections(
                "room_restart",
                frame_id=25,
                timestamp_ms=2500,
                detections=[detection(10, 20, 110, 220), detection(300, 20, 400, 220)],
            )
            first_manager.assign_student(
                "room_restart",
                track_id=packet.tracks[0].track_id,
                student_id="SV_A",
            )

            restarted_manager = TrackingManager(storage_root, max_tracks=2)
            restarted_manager.restore_session("room_restart")
            restored = restarted_manager.get_packet("room_restart")

            self.assertEqual(restored.frame_id, 25)
            self.assertEqual(restored.timestamp_ms, 2500)
            self.assertEqual([track.track_id for track in restored.tracks], [1, 2])
            self.assertEqual(restored.tracks[0].student_id, "SV_A")
            self.assertFalse(restored.tracks[0].is_present)

            refreshed = restarted_manager.process_detections(
                "room_restart",
                frame_id=26,
                timestamp_ms=2600,
                detections=[
                    detection(10, 20, 110, 220),
                    detection(300, 20, 400, 220),
                ],
            )
            self.assertEqual([track.track_id for track in refreshed.tracks], [1, 2])
            self.assertEqual(refreshed.tracks[0].student_id, "SV_A")
            self.assertTrue(refreshed.tracks[0].is_present)

    def test_missing_track_can_be_reassigned_during_grace_period(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(
                Path(directory),
                max_tracks=3,
                max_missed_frames=5,
            )
            manager.create_session("room_retrack")
            first = manager.process_detections(
                "room_retrack",
                frame_id=1,
                timestamp_ms=100,
                detections=[detection(10, 20, 110, 220)],
            )
            manager.assign_student(
                "room_retrack",
                track_id=first.tracks[0].track_id,
                student_id="SV_A",
            )

            manager.process_detections(
                "room_retrack",
                frame_id=2,
                timestamp_ms=200,
                detections=[],
            )
            reappeared = manager.process_detections(
                "room_retrack",
                frame_id=3,
                timestamp_ms=300,
                detections=[detection(300, 20, 400, 220)],
            )
            new_track = next(track for track in reappeared.tracks if track.is_present)

            reassigned = manager.assign_student(
                "room_retrack",
                track_id=new_track.track_id,
                student_id="SV_A",
            )

            self.assertEqual(len(reassigned.tracks), 1)
            self.assertEqual(reassigned.tracks[0].track_id, first.tracks[0].track_id)
            self.assertEqual(reassigned.tracks[0].student_id, "SV_A")
            self.assertTrue(reassigned.tracks[0].is_present)

    def test_legacy_assignment_only_state_is_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage_root = Path(directory)
            state_path = storage_root / "legacy_room" / "tracking_state.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "session_id": "legacy_room",
                        "assignments": [
                            {"track_id": 1, "student_id": "SV_LEGACY"}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manager = TrackingManager(storage_root, max_tracks=2)
            manager.restore_session("legacy_room")
            packet = manager.process_detections(
                "legacy_room",
                frame_id=1,
                timestamp_ms=100,
                detections=[detection(10, 20, 110, 220)],
            )

            self.assertEqual(packet.tracks[0].student_id, "SV_LEGACY")


class PersonTrackingModuleTests(unittest.TestCase):
    def test_process_frame_reuses_detector_and_increments_frame_id(self) -> None:
        class FakeDetector:
            def detect(self, frame) -> list[PersonDetection]:
                return [detection(10, 20, 110, 220)]

        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(Path(directory), max_tracks=2)
            module = PersonTrackingModule(
                PersonTrackingConfig(session_id="webcam_test"),
                detector=FakeDetector(),
                manager=manager,
            )

            first = module.process_frame(object(), timestamp_ms=100)
            second = module.process_frame(object(), timestamp_ms=200)

            self.assertEqual(first.frame_id, 1)
            self.assertEqual(second.frame_id, 2)
            self.assertEqual(second.tracks[0].track_id, 1)
            self.assertTrue(second.tracks[0].is_present)

    def test_processing_rate_can_be_adjusted(self) -> None:
        rate = ProcessingRateController(target_fps=10)
        self.assertTrue(rate.should_process(now=1.0))
        rate.mark_processed(started_at=1.0, finished_at=1.01)
        self.assertFalse(rate.should_process(now=1.05))
        self.assertTrue(rate.should_process(now=1.10))

        rate.decrease(step=5)
        self.assertEqual(rate.target_fps, 5)
        rate.increase(step=2)
        self.assertEqual(rate.target_fps, 7)


if __name__ == "__main__":
    unittest.main()

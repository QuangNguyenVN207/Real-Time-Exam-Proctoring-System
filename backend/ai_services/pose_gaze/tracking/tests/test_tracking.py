from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.ai_services.pose_gaze.tracking.manager import AssignmentError, TrackingManager
from backend.ai_services.pose_gaze.tracking.schemas import BoundingBox, PersonDetection
from backend.ai_services.pose_gaze.tracking.tracker import (
    IoUPersonTracker,
    person_fingerprint_from_frame,
)


def detection(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    confidence: float = 0.9,
    appearance: tuple[float, ...] | None = None,
) -> PersonDetection:
    return PersonDetection(
        BoundingBox(x1, y1, x2, y2),
        confidence,
        appearance_fingerprint=appearance,
    )


PERSON_A = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
PERSON_A_CHANGED = (0.98, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
PERSON_B = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class IoUPersonTrackerTests(unittest.TestCase):
    def test_track_ids_remain_stable_when_students_move_slightly(self) -> None:
        tracker = IoUPersonTracker(max_tracks=2, min_iou=0.2)
        first = tracker.update([detection(10, 20, 110, 220), detection(300, 20, 400, 220)])
        second = tracker.update([detection(14, 22, 114, 222), detection(295, 20, 395, 220)])

        self.assertEqual([track.track_id for track in first], [1, 2])
        self.assertEqual([track.track_id for track in second], [1, 2])
        self.assertTrue(all(track.is_present for track in second))

    def test_small_background_person_does_not_displace_foreground_students(
        self,
    ) -> None:
        tracker = IoUPersonTracker(max_tracks=2)
        result = tracker.update(
            [
                detection(10, 10, 60, 110, confidence=0.99),
                detection(100, 100, 400, 700, confidence=0.80),
                detection(500, 100, 800, 700, confidence=0.78),
            ]
        )

        present_boxes = {
            tuple(track.bbox.to_list())
            for track in result
            if track.is_present
        }
        self.assertEqual(
            present_boxes,
            {
                (100, 100, 400, 700),
                (500, 100, 800, 700),
            },
        )

    def test_missing_track_is_not_reported_as_present(self) -> None:
        tracker = IoUPersonTracker(max_tracks=2, max_missed_frames=2)
        tracker.update([detection(10, 20, 110, 220)])
        result = tracker.update([])

        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].is_present)
        self.assertEqual(result[0].missed_frames, 1)

    def test_track_id_survives_motion_without_box_overlap(self) -> None:
        tracker = IoUPersonTracker(max_tracks=1, min_iou=0.2)
        first = tracker.update([detection(10, 20, 110, 220)])
        moved = tracker.update([detection(125, 20, 225, 220)])

        self.assertEqual(first[0].track_id, moved[0].track_id)
        self.assertTrue(moved[0].is_present)

    def test_registered_person_is_automatically_restored_after_track_expiry(self) -> None:
        tracker = IoUPersonTracker(max_tracks=2, max_missed_frames=0)
        first = tracker.update(
            [detection(10, 20, 110, 220, appearance=PERSON_A)]
        )
        original_track_id = first[0].track_id
        self.assertTrue(tracker.register_identity(original_track_id))

        tracker.update([])
        returned = tracker.update(
            [
                detection(
                    600,
                    40,
                    700,
                    240,
                    appearance=PERSON_A_CHANGED,
                )
            ]
        )
        present = [track for track in returned if track.is_present]

        self.assertEqual(len(present), 1)
        self.assertEqual(present[0].track_id, original_track_id)
        self.assertTrue(present[0].appearance_identity_registered)

    def test_different_person_at_same_position_does_not_inherit_registered_id(self) -> None:
        tracker = IoUPersonTracker(max_tracks=2, max_missed_frames=0)
        original = tracker.update(
            [detection(10, 20, 110, 220, appearance=PERSON_A)]
        )[0]
        tracker.register_identity(original.track_id)
        tracker.update([])

        replacement = tracker.update(
            [detection(10, 20, 110, 220, appearance=PERSON_B)]
        )
        present = [track for track in replacement if track.is_present]

        self.assertEqual(len(present), 1)
        self.assertNotEqual(present[0].track_id, original.track_id)
        self.assertFalse(present[0].appearance_identity_registered)

    def test_registered_appearance_wins_over_nearby_temporary_track(self) -> None:
        tracker = IoUPersonTracker(max_tracks=2, max_missed_frames=0)
        original = tracker.update(
            [detection(10, 20, 110, 220, appearance=PERSON_A)]
        )[0]
        tracker.register_identity(original.track_id)
        tracker.update([])

        temporary = tracker.update(
            [detection(500, 20, 600, 220, appearance=PERSON_B)]
        )[0]
        returned = tracker.update(
            [
                detection(
                    500,
                    20,
                    600,
                    220,
                    appearance=PERSON_A_CHANGED,
                )
            ]
        )
        present = [track for track in returned if track.is_present]

        self.assertNotEqual(temporary.track_id, original.track_id)
        self.assertEqual(len(present), 1)
        self.assertEqual(present[0].track_id, original.track_id)

    def test_person_fingerprint_is_robust_to_brightness_but_separates_appearance(
        self,
    ) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/numpy are required for image fingerprint test")

        first = np.full((250, 180, 3), 35, dtype=np.uint8)
        cv2.rectangle(first, (20, 10), (140, 230), (70, 70, 70), -1)
        cv2.circle(first, (80, 70), 37, (135, 175, 210), -1)
        cv2.circle(first, (67, 62), 4, (20, 20, 20), -1)
        cv2.circle(first, (93, 62), 4, (20, 20, 20), -1)
        cv2.line(first, (68, 85), (92, 85), (30, 30, 30), 3)
        cv2.rectangle(first, (38, 112), (122, 225), (180, 70, 35), -1)
        brighter = cv2.convertScaleAbs(first, alpha=1.08, beta=12)

        different = np.full((250, 180, 3), 190, dtype=np.uint8)
        cv2.rectangle(different, (20, 10), (140, 230), (120, 120, 120), -1)
        cv2.circle(different, (80, 70), 37, (80, 110, 145), -1)
        cv2.line(different, (58, 60), (102, 78), (255, 255, 255), 8)
        cv2.rectangle(different, (38, 112), (122, 225), (25, 210, 45), -1)

        box = BoundingBox(20, 10, 140, 230)
        first_fp = person_fingerprint_from_frame(first, box)
        brighter_fp = person_fingerprint_from_frame(brighter, box)
        different_fp = person_fingerprint_from_frame(different, box)

        self.assertIsNotNone(first_fp)
        self.assertIsNotNone(brighter_fp)
        self.assertIsNotNone(different_fp)
        same_similarity = sum(
            left * right
            for left, right in zip(first_fp, brighter_fp, strict=True)
        )
        different_similarity = sum(
            left * right
            for left, right in zip(first_fp, different_fp, strict=True)
        )
        self.assertGreater(same_similarity, 0.95)
        self.assertLess(different_similarity, 0.50)


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
            self.assertEqual({track["person_id"] for track in handoff["tracks"]}, {"SV_A", "SV_B"})

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

    def test_known_person_id_restores_original_track_after_retracking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(
                Path(directory),
                max_tracks=2,
                max_missed_frames=5,
            )
            manager.create_session("room_retrack")
            first = manager.process_detections(
                "room_retrack",
                frame_id=1,
                timestamp_ms=1,
                detections=[detection(10, 20, 110, 220)],
            )
            original_track_id = first.tracks[0].track_id
            manager.assign_student(
                "room_retrack",
                track_id=original_track_id,
                student_id="SV_STABLE",
            )

            # This box is too far away for geometric association, so the
            # low-level tracker creates a temporary second ID while the old
            # track remains in its missed-frame grace period.
            retracked = manager.process_detections(
                "room_retrack",
                frame_id=2,
                timestamp_ms=2,
                detections=[detection(1000, 20, 1100, 220)],
            )
            new_track = next(track for track in retracked.tracks if track.is_present)
            self.assertNotEqual(new_track.track_id, original_track_id)

            restored = manager.assign_student(
                "room_retrack",
                track_id=new_track.track_id,
                student_id="SV_STABLE",
            )
            present = [track for track in restored.tracks if track.is_present]

            self.assertEqual(len(present), 1)
            self.assertEqual(present[0].track_id, original_track_id)
            self.assertEqual(present[0].student_id, "SV_STABLE")
            self.assertEqual(present[0].to_dict()["person_id"], "SV_STABLE")

    def test_assigned_person_is_automatically_reidentified_on_return(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(
                Path(directory),
                max_tracks=2,
                max_missed_frames=0,
            )
            manager.create_session("room_auto_reid")
            first = manager.process_detections(
                "room_auto_reid",
                frame_id=1,
                timestamp_ms=1,
                detections=[
                    detection(10, 20, 110, 220, appearance=PERSON_A)
                ],
            )
            original_track_id = first.tracks[0].track_id
            assigned = manager.assign_student(
                "room_auto_reid",
                track_id=original_track_id,
                student_id="SV_REMEMBERED",
            )
            self.assertTrue(
                assigned.tracks[0].appearance_identity_registered
            )

            manager.process_detections(
                "room_auto_reid",
                frame_id=2,
                timestamp_ms=2,
                detections=[],
            )
            returned = manager.process_detections(
                "room_auto_reid",
                frame_id=3,
                timestamp_ms=3,
                detections=[
                    detection(
                        600,
                        40,
                        700,
                        240,
                        appearance=PERSON_A_CHANGED,
                    )
                ],
            )
            present = [track for track in returned.tracks if track.is_present]

            self.assertEqual(len(present), 1)
            self.assertEqual(present[0].track_id, original_track_id)
            self.assertEqual(present[0].student_id, "SV_REMEMBERED")
            self.assertTrue(present[0].appearance_identity_registered)

    def test_other_person_cannot_inherit_assigned_person_id_from_same_seat(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(
                Path(directory),
                max_tracks=2,
                max_missed_frames=0,
            )
            manager.create_session("room_no_identity_leak")
            first = manager.process_detections(
                "room_no_identity_leak",
                frame_id=1,
                timestamp_ms=1,
                detections=[
                    detection(10, 20, 110, 220, appearance=PERSON_A)
                ],
            )
            original_track_id = first.tracks[0].track_id
            manager.assign_student(
                "room_no_identity_leak",
                track_id=original_track_id,
                student_id="SV_A",
            )
            manager.process_detections(
                "room_no_identity_leak",
                frame_id=2,
                timestamp_ms=2,
                detections=[],
            )

            replacement = manager.process_detections(
                "room_no_identity_leak",
                frame_id=3,
                timestamp_ms=3,
                detections=[
                    detection(10, 20, 110, 220, appearance=PERSON_B)
                ],
            )
            present = [track for track in replacement.tracks if track.is_present]

            self.assertEqual(len(present), 1)
            self.assertNotEqual(present[0].track_id, original_track_id)
            self.assertIsNone(present[0].student_id)


if __name__ == "__main__":
    unittest.main()

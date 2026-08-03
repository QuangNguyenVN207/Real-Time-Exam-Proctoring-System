from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.ai_services.pose_gaze.tracking.manager import TrackingManager
from backend.ai_services.pose_gaze.tracking.paper_tracking import (
    IoUPaperTracker,
    PaperAuthorizationPolicy,
    PaperDetection,
    fingerprint_similarity,
    paper_fingerprint_from_frame,
)
from backend.ai_services.pose_gaze.tracking.schemas import (
    BoundingBox,
    PersonDetection,
    TrackedPerson,
)


def paper(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    class_name: str = "cheat_sheet",
    confidence: float = 0.9,
    appearance_fingerprint: tuple[float, ...] | None = None,
) -> PaperDetection:
    return PaperDetection(
        bbox=BoundingBox(x1, y1, x2, y2),
        confidence=confidence,
        class_name=class_name,
        appearance_fingerprint=appearance_fingerprint,
    )


def person(track_id: int = 1) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        bbox=BoundingBox(0, 0, 400, 650),
        confidence=0.95,
        age_frames=10,
        missed_frames=0,
        is_present=True,
    )


class IoUPaperTrackerTests(unittest.TestCase):
    def test_visual_fingerprint_distinguishes_page_layouts(self) -> None:
        horizontal_page = np.full((200, 200, 3), 255, dtype=np.uint8)
        horizontal_page[60:66, 40:160] = 0
        horizontal_page[95:101, 40:145] = 0
        horizontal_page[130:136, 40:155] = 0
        vertical_page = np.full((200, 200, 3), 255, dtype=np.uint8)
        vertical_page[40:160, 60:66] = 0
        vertical_page[40:145, 95:101] = 0
        vertical_page[40:155, 130:136] = 0
        bbox = BoundingBox(20, 20, 180, 180)

        horizontal = paper_fingerprint_from_frame(horizontal_page, bbox)
        same_horizontal = paper_fingerprint_from_frame(horizontal_page, bbox)
        vertical = paper_fingerprint_from_frame(vertical_page, bbox)

        self.assertIsNotNone(horizontal)
        self.assertIsNotNone(vertical)
        self.assertGreater(
            fingerprint_similarity(horizontal, same_horizontal),
            0.99,
        )
        self.assertLess(
            fingerprint_similarity(horizontal, vertical),
            0.86,
        )

    def test_paper_id_survives_label_flip(self) -> None:
        tracker = IoUPaperTracker(supports_test_paper=True)
        for offset in range(3):
            result = tracker.update(
                [paper(80 + offset, 420, 250 + offset, 570, "test_paper")]
            )
        result = tracker.update([paper(85, 423, 255, 573, "cheat_sheet")])

        self.assertEqual(result[0].paper_id, 1)
        self.assertEqual(result[0].stable_label, "test_paper")
        self.assertEqual(result[0].visible_frames, 4)

    def test_four_class_checkpoint_exposes_uncertain_paper_label(self) -> None:
        tracker = IoUPaperTracker(supports_test_paper=False)
        result = tracker.update([paper(80, 420, 250, 570, "cheat_sheet")])

        self.assertEqual(result[0].raw_class_name, "cheat_sheet")
        self.assertEqual(result[0].stable_label, "paper_unknown")

    def test_short_occlusion_keeps_paper_id(self) -> None:
        tracker = IoUPaperTracker(max_missed_frames=2)
        first = tracker.update([paper(80, 420, 250, 570)])[0]
        tracker.update([])
        returned = tracker.update([paper(86, 423, 256, 573)])[0]

        self.assertEqual(first.paper_id, returned.paper_id)
        self.assertTrue(returned.is_present)

    def test_paper_is_linked_to_nearest_person_track(self) -> None:
        tracker = IoUPaperTracker()
        tracker.update([paper(80, 420, 250, 570)])
        result = tracker.associate_owners([person(7)])

        self.assertEqual(result[0].owner_track_id, 7)

    def test_owner_can_be_remapped_after_manual_person_reidentification(self) -> None:
        tracker = IoUPaperTracker()
        tracker.update([paper(80, 420, 250, 570)])
        tracker.associate_owners([person(7)])

        tracker.remap_owner(current_track_id=7, target_track_id=3)

        self.assertEqual(tracker.snapshot()[0].owner_track_id, 3)

    def test_temporary_paper_track_can_receive_manual_id(self) -> None:
        tracker = IoUPaperTracker()
        temporary = tracker.update([paper(80, 420, 250, 570)])[0]

        tracker.remap_track(
            current_paper_id=temporary.paper_id,
            target_paper_id=42,
        )
        moved = tracker.update([paper(84, 422, 254, 572)])[0]

        self.assertEqual(moved.paper_id, 42)
        self.assertTrue(moved.is_present)

    def test_manual_paper_id_cannot_overwrite_visible_paper(self) -> None:
        tracker = IoUPaperTracker()
        tracked = tracker.update(
            [
                paper(50, 430, 180, 570),
                paper(235, 440, 365, 580),
            ]
        )
        tracker.remap_track(
            current_paper_id=tracked[0].paper_id,
            target_paper_id=42,
        )

        with self.assertRaises(ValueError):
            tracker.remap_track(
                current_paper_id=tracked[1].paper_id,
                target_paper_id=42,
            )

    def test_registered_paper_is_automatically_reidentified_by_appearance(
        self,
    ) -> None:
        fingerprint = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        tracker = IoUPaperTracker(max_missed_frames=0)
        temporary = tracker.update(
            [
                paper(
                    80,
                    420,
                    250,
                    570,
                    appearance_fingerprint=fingerprint,
                )
            ],
            people=[person()],
        )[0]
        tracker.remap_track(
            current_paper_id=temporary.paper_id,
            target_paper_id=42,
        )
        self.assertTrue(
            tracker.register_identity(paper_id=42, owner_track_id=1)
        )
        tracker.update([], people=[person()])

        returned = tracker.update(
            [
                paper(
                    82,
                    421,
                    252,
                    571,
                    appearance_fingerprint=fingerprint,
                )
            ],
            people=[person()],
        )

        self.assertEqual(returned[0].paper_id, 42)

    def test_different_paper_in_same_position_does_not_inherit_registered_id(
        self,
    ) -> None:
        first_fingerprint = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        other_fingerprint = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        tracker = IoUPaperTracker(max_missed_frames=0)
        temporary = tracker.update(
            [
                paper(
                    80,
                    420,
                    250,
                    570,
                    appearance_fingerprint=first_fingerprint,
                )
            ],
            people=[person()],
        )[0]
        tracker.remap_track(
            current_paper_id=temporary.paper_id,
            target_paper_id=42,
        )
        tracker.register_identity(paper_id=42, owner_track_id=1)

        replacement = tracker.update(
            [
                paper(
                    80,
                    420,
                    250,
                    570,
                    appearance_fingerprint=other_fingerprint,
                )
            ],
            people=[person()],
        )

        self.assertNotEqual(replacement[0].paper_id, 42)

    def test_identical_exam_layouts_are_reidentified_by_owner(self) -> None:
        fingerprint = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        left_student = TrackedPerson(
            track_id=1,
            bbox=BoundingBox(0, 0, 400, 650),
            confidence=0.95,
            age_frames=10,
            missed_frames=0,
            is_present=True,
        )
        right_student = TrackedPerson(
            track_id=2,
            bbox=BoundingBox(500, 0, 900, 650),
            confidence=0.95,
            age_frames=10,
            missed_frames=0,
            is_present=True,
        )
        people = [left_student, right_student]
        tracker = IoUPaperTracker(max_missed_frames=0)
        initial = tracker.update(
            [
                paper(
                    80,
                    420,
                    250,
                    570,
                    appearance_fingerprint=fingerprint,
                ),
                paper(
                    580,
                    420,
                    750,
                    570,
                    appearance_fingerprint=fingerprint,
                ),
            ],
            people=people,
        )
        tracker.associate_owners(people)
        tracker.remap_track(
            current_paper_id=initial[0].paper_id,
            target_paper_id=101,
        )
        tracker.remap_track(
            current_paper_id=initial[1].paper_id,
            target_paper_id=202,
        )
        tracker.register_identity(paper_id=101, owner_track_id=1)
        tracker.register_identity(paper_id=202, owner_track_id=2)
        tracker.update([], people=people)

        returned = tracker.update(
            [
                paper(
                    580,
                    420,
                    750,
                    570,
                    appearance_fingerprint=fingerprint,
                ),
                paper(
                    80,
                    420,
                    250,
                    570,
                    appearance_fingerprint=fingerprint,
                ),
            ],
            people=people,
        )
        ids_by_side = {
            "left" if item.bbox.center[0] < 450 else "right": item.paper_id
            for item in returned
        }

        self.assertEqual(ids_by_side, {"left": 101, "right": 202})


class PaperAuthorizationPolicyTests(unittest.TestCase):
    def test_single_paper_from_four_class_model_is_auto_authorized(self) -> None:
        tracker = IoUPaperTracker(supports_test_paper=False)
        policy = PaperAuthorizationPolicy(
            registration_frames=2,
            alert_confirm_frames=2,
            supports_test_paper=False,
        )

        for _ in range(2):
            tracker.update([paper(80, 420, 250, 570, "cheat_sheet")])
            tracked = tracker.associate_owners([person()])
            assessments = policy.evaluate(tracked)

        self.assertEqual(assessments[0].status, "authorized_exam_paper")
        self.assertFalse(
            any(assessment.status == "suspicious" for assessment in assessments)
        )

    def test_second_physical_paper_becomes_suspicious(self) -> None:
        tracker = IoUPaperTracker(supports_test_paper=False)
        policy = PaperAuthorizationPolicy(
            registration_frames=2,
            alert_confirm_frames=2,
            supports_test_paper=False,
        )

        for _ in range(2):
            tracker.update([paper(50, 430, 180, 570)])
            policy.evaluate(tracker.associate_owners([person()]))

        for offset in range(3):
            tracker.update(
                [
                    paper(50 + offset, 430, 180 + offset, 570),
                    paper(235 + offset, 440, 365 + offset, 580),
                ]
            )
            assessments = policy.evaluate(tracker.associate_owners([person()]))

        suspicious = [
            assessment
            for assessment in assessments
            if assessment.status == "suspicious"
        ]
        self.assertEqual(len(suspicious), 1)
        self.assertIn("additional_paper", suspicious[0].reasons)
        self.assertNotEqual(
            suspicious[0].paper.paper_id,
            policy.authorized_mapping()[1],
        )

    def test_six_class_cheat_sheet_semantics_can_raise_alert(self) -> None:
        tracker = IoUPaperTracker(supports_test_paper=True)
        policy = PaperAuthorizationPolicy(
            registration_frames=2,
            alert_confirm_frames=2,
            supports_test_paper=True,
        )

        for _ in range(3):
            tracker.update([paper(80, 420, 250, 570, "cheat_sheet")])
            assessments = policy.evaluate(tracker.associate_owners([person()]))

        self.assertEqual(assessments[0].status, "suspicious")
        self.assertIn("classifier_cheat_sheet", assessments[0].reasons)
        self.assertEqual(policy.authorized_mapping(), {})


class PaperTrackingManagerTests(unittest.TestCase):
    def test_manager_returns_authorized_paper_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(
                Path(directory),
                max_tracks=1,
                paper_registration_frames=2,
                paper_alert_confirm_frames=2,
            )
            manager.create_session("paper_room")
            person_detection = PersonDetection(
                BoundingBox(0, 0, 400, 650),
                0.95,
            )

            for frame_id in range(1, 3):
                manager.process_detections(
                    "paper_room",
                    frame_id=frame_id,
                    timestamp_ms=frame_id * 100,
                    detections=[person_detection],
                )
                state = manager.process_paper_detections(
                    "paper_room",
                    detections=[paper(80, 420, 250, 570)],
                    supports_test_paper=False,
                )

            self.assertEqual(
                state["papers"][0]["status"],
                "authorized_exam_paper",
            )
            self.assertEqual(
                state["authorized_papers"],
                [
                    {
                        "owner_track_id": 1,
                        "owner_person_id": None,
                        "paper_id": 1,
                        "paper_id_assigned": False,
                    }
                ],
            )

    def test_manual_reidentification_moves_paper_owner_to_stable_person(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(
                Path(directory),
                max_tracks=2,
                max_missed_frames=5,
                paper_registration_frames=1,
            )
            manager.create_session("paper_retrack")
            first_person = manager.process_detections(
                "paper_retrack",
                frame_id=1,
                timestamp_ms=100,
                detections=[
                    PersonDetection(BoundingBox(0, 0, 400, 650), 0.95)
                ],
            )
            original_track_id = first_person.tracks[0].track_id
            manager.assign_student(
                "paper_retrack",
                track_id=original_track_id,
                student_id="SV_STABLE",
            )
            manager.process_paper_detections(
                "paper_retrack",
                detections=[paper(80, 420, 250, 570)],
                supports_test_paper=False,
            )

            retracked = manager.process_detections(
                "paper_retrack",
                frame_id=2,
                timestamp_ms=200,
                detections=[
                    PersonDetection(BoundingBox(1000, 0, 1400, 650), 0.95)
                ],
            )
            temporary_track = next(
                track for track in retracked.tracks if track.is_present
            )
            manager.process_paper_detections(
                "paper_retrack",
                detections=[paper(1080, 420, 1250, 570)],
                supports_test_paper=False,
            )

            manager.assign_student(
                "paper_retrack",
                track_id=temporary_track.track_id,
                student_id="SV_STABLE",
            )
            state = manager.get_paper_state("paper_retrack")

            self.assertTrue(state["papers"])
            self.assertTrue(
                all(
                    item["owner_track_id"] == original_track_id
                    and item["owner_person_id"] == "SV_STABLE"
                    for item in state["papers"]
                    if item["is_present"]
                )
            )

    def test_known_manual_paper_id_is_restored_after_retracking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(
                Path(directory),
                max_tracks=1,
                paper_registration_frames=1,
                paper_max_missed_frames=0,
            )
            manager.create_session("stable_paper")
            person_detection = PersonDetection(
                BoundingBox(0, 0, 400, 650),
                0.95,
            )
            manager.process_detections(
                "stable_paper",
                frame_id=1,
                timestamp_ms=100,
                detections=[person_detection],
            )
            initial = manager.process_paper_detections(
                "stable_paper",
                detections=[paper(80, 420, 250, 570)],
                supports_test_paper=False,
            )
            temporary_id = initial["papers"][0]["paper_id"]
            assigned = manager.assign_paper_id(
                "stable_paper",
                current_paper_id=temporary_id,
                stable_paper_id=77,
            )

            self.assertEqual(assigned["papers"][0]["paper_id"], 77)
            self.assertTrue(assigned["papers"][0]["paper_id_assigned"])
            self.assertEqual(
                assigned["papers"][0]["status"],
                "authorized_exam_paper",
            )

            manager.process_paper_detections(
                "stable_paper",
                detections=[],
                supports_test_paper=False,
            )
            returned = manager.process_paper_detections(
                "stable_paper",
                detections=[paper(82, 421, 252, 571)],
                supports_test_paper=False,
            )
            new_temporary_id = returned["papers"][0]["paper_id"]
            self.assertNotEqual(new_temporary_id, 77)

            restored = manager.assign_paper_id(
                "stable_paper",
                current_paper_id=new_temporary_id,
                stable_paper_id=77,
            )

            self.assertEqual(restored["papers"][0]["paper_id"], 77)
            self.assertTrue(restored["papers"][0]["paper_id_assigned"])
            self.assertEqual(
                restored["papers"][0]["status"],
                "authorized_exam_paper",
            )

    def test_registered_exam_paper_auto_reidentifies_and_other_paper_alerts_owner(
        self,
    ) -> None:
        exam_fingerprint = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        other_fingerprint = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        with tempfile.TemporaryDirectory() as directory:
            manager = TrackingManager(
                Path(directory),
                max_tracks=1,
                paper_registration_frames=1,
                paper_alert_confirm_frames=1,
                paper_max_missed_frames=0,
            )
            manager.create_session("appearance_reid")
            person_packet = manager.process_detections(
                "appearance_reid",
                frame_id=1,
                timestamp_ms=100,
                detections=[
                    PersonDetection(BoundingBox(0, 0, 400, 650), 0.95)
                ],
            )
            manager.assign_student(
                "appearance_reid",
                track_id=person_packet.tracks[0].track_id,
                student_id="SV001",
            )
            initial = manager.process_paper_detections(
                "appearance_reid",
                detections=[
                    paper(
                        80,
                        420,
                        250,
                        570,
                        appearance_fingerprint=exam_fingerprint,
                    )
                ],
                supports_test_paper=False,
            )
            assigned = manager.assign_paper_id(
                "appearance_reid",
                current_paper_id=initial["papers"][0]["paper_id"],
                stable_paper_id=101,
            )
            self.assertTrue(
                assigned["papers"][0]["appearance_identity_registered"]
            )

            manager.process_paper_detections(
                "appearance_reid",
                detections=[],
                supports_test_paper=False,
            )
            returned_exam = manager.process_paper_detections(
                "appearance_reid",
                detections=[
                    paper(
                        82,
                        421,
                        252,
                        571,
                        appearance_fingerprint=exam_fingerprint,
                    )
                ],
                supports_test_paper=False,
            )

            self.assertEqual(returned_exam["papers"][0]["paper_id"], 101)
            self.assertEqual(
                returned_exam["papers"][0]["status"],
                "authorized_exam_paper",
            )
            self.assertEqual(
                returned_exam["papers"][0]["owner_person_id"],
                "SV001",
            )

            different_paper = manager.process_paper_detections(
                "appearance_reid",
                detections=[
                    paper(
                        82,
                        421,
                        252,
                        571,
                        appearance_fingerprint=other_fingerprint,
                    )
                ],
                supports_test_paper=False,
            )

            self.assertNotEqual(different_paper["papers"][0]["paper_id"], 101)
            self.assertEqual(
                different_paper["papers"][0]["status"],
                "suspicious",
            )
            self.assertEqual(
                different_paper["alerts"][0]["owner_person_id"],
                "SV001",
            )
            self.assertIn(
                "paper_replacement",
                different_paper["alerts"][0]["reasons"],
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from backend.ai_services.pose_gaze.dataset.manifest import parse_video_filename
from backend.ai_services.pose_gaze.dataset.split import assign_split
from backend.ai_services.pose_gaze.dataset.windows import build_caption, iter_window_bounds, normalize_gaze_direction


class DatasetParserTests(unittest.TestCase):
    def test_parse_two_subject_filename(self) -> None:
        metadata = parse_video_filename("v_c1_s1_s2.MOV")
        self.assertEqual(metadata.class_code, "c1")
        self.assertEqual(metadata.subject_ids, ("s1", "s2"))
        self.assertIsNone(metadata.take_code)

    def test_parse_single_subject_with_take(self) -> None:
        metadata = parse_video_filename("v_c3_s7.MOV")
        self.assertEqual(metadata.class_code, "c3")
        self.assertEqual(metadata.subject_ids, ("s7",))
        self.assertIsNone(metadata.take_code)


class DatasetSplitTests(unittest.TestCase):
    def test_split_assignment_respects_subject_bands(self) -> None:
        self.assertEqual(assign_split(("s1", "s4")), "train")
        self.assertEqual(assign_split(("s5", "s6")), "val")
        self.assertEqual(assign_split(("s7",)), "test")
        self.assertIsNone(assign_split(("s4", "s5")))


class DatasetWindowTests(unittest.TestCase):
    def test_window_bounds_are_16_frame_strides_of_8(self) -> None:
        self.assertEqual(iter_window_bounds(16), [(0, 16)])
        self.assertEqual(iter_window_bounds(24), [(0, 16), (8, 24)])
        self.assertEqual(iter_window_bounds(31), [(0, 16), (8, 24)])

    def test_quality_gate_forces_unknown_gaze(self) -> None:
        self.assertEqual(normalize_gaze_direction(quality_ok=False, gaze_direction="left"), "unknown")
        self.assertEqual(normalize_gaze_direction(quality_ok=True, gaze_direction="left"), "left")

    def test_caption_template_includes_class_and_students(self) -> None:
        caption = build_caption(
            {
                "class_code": "c2",
                "student_id": "s1",
                "peer_student_id": "s2",
                "gaze_direction": "right",
                "action": "looking_toward_peer",
            }
        )
        self.assertIn("c2", caption)
        self.assertIn("s1", caption)
        self.assertIn("s2", caption)
        self.assertIn("looking_toward_peer", caption)


if __name__ == "__main__":
    unittest.main()
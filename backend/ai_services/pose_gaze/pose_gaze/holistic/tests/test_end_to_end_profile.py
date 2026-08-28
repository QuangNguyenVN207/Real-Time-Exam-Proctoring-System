from __future__ import annotations

import unittest

from pose_gaze.holistic.feature_csv.benchmark_end_to_end_causal import (
    EXTENDED_BENCHMARK_LABELS,
    OFFICIAL_BENCHMARK_LABELS,
    _truth,
)


class EndToEndProfileTests(unittest.TestCase):
    def test_official_and_extended_profiles_are_distinct(self):
        self.assertEqual(OFFICIAL_BENCHMARK_LABELS, ("c2", "c3", "c5"))
        self.assertEqual(
            EXTENDED_BENCHMARK_LABELS,
            ("suspicious_activity", "c2", "c3", "c5", "c7"),
        )
        self.assertNotEqual(OFFICIAL_BENCHMARK_LABELS, EXTENDED_BENCHMARK_LABELS)

    def test_pose_c1_c4_truth_maps_to_shared_public_label(self):
        for class_code in ("c1", "c4"):
            source = {
                "class_code": class_code,
                "action_actor_ids_parsed": {"s1"},
            }
            self.assertEqual(_truth(source, "s1"), "suspicious_activity")
            self.assertEqual(_truth(source, "s2"), "c5")


if __name__ == "__main__":
    unittest.main()

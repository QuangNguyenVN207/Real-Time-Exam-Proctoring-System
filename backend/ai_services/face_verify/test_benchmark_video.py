from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.ai_services.face_verify.benchmark_video import (
    ManifestRow,
    UNKNOWN,
    actors_requested,
    apply_policy,
    natural_actor_key,
    wilson_interval,
    wilson_interval,
)


class BenchmarkVideoTests(unittest.TestCase):
    def row(self, note: str = "") -> ManifestRow:
        return ManifestRow(
            data_path=Path("sample.png"),
            pair_id="pair02",
            split="enroll",
            left_actor_id="P03",
            right_actor_id="P04",
            note=note,
        )

    def test_actor_specific_note_selects_only_requested_seat(self) -> None:
        self.assertEqual(
            actors_requested(self.row("just_use_for_enroll_P04")),
            {"right": "P04"},
        )

    def test_general_note_keeps_both_seats(self) -> None:
        self.assertEqual(
            actors_requested(self.row("good")),
            {"left": "P03", "right": "P04"},
        )

    def test_policy_requires_similarity_and_margin(self) -> None:
        frame = pd.DataFrame(
            {
                "top1_actor_id": ["P01", "P02", "P03"],
                "top1_score": [0.60, 0.39, 0.60],
                "margin": [0.10, 0.20, 0.02],
            }
        )
        self.assertEqual(
            apply_policy(frame, 0.40, 0.08).tolist(),
            ["P01", UNKNOWN, UNKNOWN],
        )

    def test_actor_sort_is_numeric(self) -> None:
        actors = ["P010", "P02", "P1"]
        self.assertEqual(sorted(actors, key=natural_actor_key), ["P1", "P02", "P010"])

    def test_wilson_interval_is_honest_for_small_perfect_sample(self) -> None:
        lower, upper = wilson_interval(4, 4)
        self.assertAlmostEqual(upper, 1.0)
        self.assertGreater(lower, 0.50)
        self.assertLess(lower, 0.52)

    def test_wilson_interval_is_honest_for_small_perfect_sample(self) -> None:
        lower, upper = wilson_interval(4, 4)
        self.assertAlmostEqual(upper, 1.0)
        self.assertGreater(lower, 0.50)
        self.assertLess(lower, 0.52)


if __name__ == "__main__":
    unittest.main()

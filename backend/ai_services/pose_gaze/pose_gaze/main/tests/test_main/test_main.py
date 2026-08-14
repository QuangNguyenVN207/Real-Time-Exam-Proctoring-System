"""Tests for realtime prediction decoding and smoothing."""

from __future__ import annotations

import unittest

from backend.ai_services.pose_gaze.main import (
    PredictionSmoother,
    decode_prediction,
)


class RealtimeMainTests(unittest.TestCase):
    def test_binary_probability_is_expanded(self) -> None:
        probabilities = decode_prediction([0.8], 2)
        self.assertAlmostEqual(probabilities[0], 0.2)
        self.assertAlmostEqual(probabilities[1], 0.8)

    def test_multiclass_probabilities_are_normalized(self) -> None:
        self.assertEqual(decode_prediction([[2.0, 1.0, 1.0]], 3), (0.5, 0.25, 0.25))

    def test_softmax_class_index_becomes_one_hot(self) -> None:
        self.assertEqual(decode_prediction([2.0], 3), (0.0, 0.0, 1.0))

    def test_smoothing_is_isolated_per_track(self) -> None:
        smoother = PredictionSmoother(alpha=0.5)
        self.assertEqual(smoother.update(1, (1.0, 0.0)), (1.0, 0.0))
        self.assertEqual(smoother.update(1, (0.0, 1.0)), (0.5, 0.5))
        self.assertEqual(smoother.update(2, (0.0, 1.0)), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()

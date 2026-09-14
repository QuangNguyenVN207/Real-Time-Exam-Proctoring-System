from __future__ import annotations

import unittest

import numpy as np

from backend.ai_services.whisper.phobert.decision_fusion import DecisionFusionService
from backend.ai_services.whisper.phobert.evaluate_p1 import (
    choose_threshold,
    classification_metrics,
    threshold_table,
)


class DecisionFusionTests(unittest.TestCase):
    def setUp(self):
        self.fusion = DecisionFusionService(ai_cheating_threshold=0.60)

    @staticmethod
    def result(probability: float) -> dict:
        return {
            "label": "Cheating" if probability >= 0.5 else "Normal",
            "all_probs": {"Normal": 1 - probability, "Cheating": probability},
        }

    def test_medium_keyword_risk_triggers(self):
        result = self.fusion.fuse("sample", "medium", self.result(0.01))
        self.assertEqual(result["final_label"], "Cheating")

    def test_safe_rule_uses_configured_probability_threshold(self):
        self.assertEqual(
            self.fusion.fuse("sample", "safe", self.result(0.59))["final_label"],
            "Normal",
        )
        self.assertEqual(
            self.fusion.fuse("sample", "safe", self.result(0.60))["final_label"],
            "Cheating",
        )

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            DecisionFusionService(ai_cheating_threshold=1.01)


class MetricTests(unittest.TestCase):
    def test_confusion_metrics(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.8, 0.9, 0.2])
        metrics = classification_metrics(labels, scores, 0.5)
        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_threshold_choice_prefers_value_nearest_half_on_f1_tie(self):
        labels = np.array([0, 1])
        scores = np.array([0.1, 0.9])
        selected = choose_threshold(threshold_table(labels, scores))
        self.assertEqual(selected, 0.5)


if __name__ == "__main__":
    unittest.main()

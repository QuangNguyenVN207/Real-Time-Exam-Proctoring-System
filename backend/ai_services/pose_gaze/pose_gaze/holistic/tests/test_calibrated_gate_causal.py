from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose_gaze.holistic.feature_csv.benchmark_calibrated_gate_causal import (
    _checkpoint_scores, _meta_model, _meta_rows, _pair_max, _validate_feature_names, _validate_oof_folds,
    _validate_score_vector,
)


class CalibratedGatePreflightTests(unittest.TestCase):
    def test_missing_specialist_prefix_becomes_no_evidence(self):
        key = ("video", "actor")
        row = _meta_rows({key}, {key: "c5"}, {key: "group"}, {}, {}, {}, {})[0]
        self.assertEqual(row["x"], [0.0, 0.0, 0.0, 0.0])

    def test_forbidden_feature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "forbidden leakage"):
            _validate_feature_names("bad", ["actor_id__max"])

    def test_oof_fold_without_positive_training_class_is_rejected(self):
        rows = [
            {"group": "a", "truth": "c2"},
            {"group": "b", "truth": "c5"},
        ]
        with self.assertRaisesRegex(ValueError, "lacks a binary class"):
            _validate_oof_folds(rows, "c2")

    def test_non_finite_oof_score_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _validate_score_vector("c2", [{}], np.asarray([np.nan]))

    def test_c2_pair_propagates_only_eligible_midpoint_evidence(self):
        records = [
            {"key": ("v", "a"), "frame": 1, "mid": 0},
            {"key": ("v", "b"), "frame": 1, "mid": 1},
        ]
        manifest = {"v": {"interaction_pairs": '[{"source":"a","peer":"b"}]'}}
        result = _pair_max(records, [0.99, 0.7], manifest, lambda row: row["mid"] == 1)
        self.assertEqual(result[("v", "a")], 0.7)
        self.assertEqual(result[("v", "b")], 0.7)

    def test_production_meta_model_fits_all_five_classes(self):
        x = np.asarray([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
            [0, 0, 0, 0], [0, 0, 0, 1],
        ], dtype=np.float64)
        model = _meta_model().fit(x, [
            "suspicious_activity", "c2", "c3", "c5", "c7",
        ])
        self.assertEqual(set(model.classes_), {
            "suspicious_activity", "c2", "c3", "c5", "c7",
        })

    def test_checkpoint_round_trip_and_provenance_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            first = _checkpoint_scores(
                path, {"stage": "c2"},
                lambda: (np.asarray([0.2]), np.asarray([0.3])),
            )
            self.assertFalse(first[2])
            second = _checkpoint_scores(
                path, {"stage": "c2"},
                lambda: self.fail("valid checkpoint should resume"),
            )
            self.assertTrue(second[2])
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                _checkpoint_scores(path, {"stage": "c3"}, lambda: None)


if __name__ == "__main__":
    unittest.main()

import unittest

from pose_gaze.holistic.feature_csv import (
    MODEL_FEATURE_COLUMNS,
    VARIANT_COLUMNS,
    generate_feature_variants,
)


class FeatureVariantTests(unittest.TestCase):
    def test_generates_ordered_variants(self) -> None:
        rows = [{"frame_id": 1, "pose_000_x": 0.25, "pose_000_visibility": None}]
        variants = generate_feature_variants(rows)

        self.assertEqual(tuple(variants), ("2d", "2d_quality", "2d_world", "full", "2d_geometry", "2d_world_upper", "2d_world_behavior"))
        self.assertEqual(tuple(variants["full"][0])[:1], ("frame_id",))
        self.assertEqual(variants["full"][0]["pose_000_x"], 0.25)
        self.assertEqual(len(VARIANT_COLUMNS["full"]), len(MODEL_FEATURE_COLUMNS))
        self.assertFalse(any(column.startswith("pose_world_") for column in VARIANT_COLUMNS["2d"]))
        self.assertFalse(any(column.startswith("pose_world_") for column in VARIANT_COLUMNS["2d_quality"]))
        self.assertTrue(any(column.startswith("pose_world_") for column in VARIANT_COLUMNS["2d_world"]))
        self.assertTrue(any(column.startswith("pose_world_011_") for column in VARIANT_COLUMNS["2d_world_upper"]))
        self.assertFalse(any(column.startswith("pose_023_") for column in VARIANT_COLUMNS["2d_world_upper"]))
        self.assertFalse(any(column.startswith("pose_world_023_") for column in VARIANT_COLUMNS["2d_world_upper"]))

    def test_predicted_rows_are_excluded_from_all_variants(self) -> None:
        rows = [
            {"frame_id": 1, "face_predicted": True, "pose_000_x": 0.1},
            {"frame_id": 2, "face_predicted": False, "pose_000_x": 0.2},
        ]
        variants = generate_feature_variants(rows)

        for output in variants.values():
            self.assertEqual([row["frame_id"] for row in output], [2])


if __name__ == "__main__":
    unittest.main()

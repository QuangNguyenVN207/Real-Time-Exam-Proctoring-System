"""Tests for the shared fixed-width landmark CSV schema."""

from __future__ import annotations

import unittest

from backend.ai_services.pose_gaze.holistic.feature_csv import (
    CSV_FIELDNAMES,
    MODEL_FEATURE_COLUMNS,
    build_csv_row,
    model_features_from_result,
)
from backend.ai_services.pose_gaze.holistic.landmark import (
    LandmarkPoint,
    TrackHolisticResult,
)
from backend.ai_services.pose_gaze.tracking.schemas import (
    BoundingBox,
    TrackedPerson,
)


class FeatureCsvTests(unittest.TestCase):
    def setUp(self) -> None:
        bbox = BoundingBox(10, 20, 110, 220)
        self.track = TrackedPerson(1, bbox, 0.9, 1, 0, True, "student_01")
        self.result = TrackHolisticResult(
            track_id=1,
            student_id="student_01",
            bbox=bbox,
            crop_bbox=bbox,
            pose_landmarks=(
                LandmarkPoint(
                    index=0,
                    x=0.25,
                    y=0.50,
                    frame_x=35.0,
                    frame_y=120.0,
                    visibility=0.8,
                    presence=0.9,
                ),
            ),
        )

    def test_schema_contains_no_depth_columns(self) -> None:
        self.assertFalse(any(column.endswith("_z") for column in CSV_FIELDNAMES))
        self.assertFalse(any("frame_x" in column for column in MODEL_FEATURE_COLUMNS))

    def test_result_is_flattened_in_fixed_model_schema(self) -> None:
        features = model_features_from_result(self.result)

        self.assertEqual(tuple(features), MODEL_FEATURE_COLUMNS)
        self.assertEqual(features["pose_000_x"], 0.25)
        self.assertEqual(features["pose_000_y"], 0.50)
        self.assertIsNone(features["pose_001_x"])
        self.assertAlmostEqual(features["pose_valid_ratio"], 1 / 25)

    def test_csv_row_keeps_annotation_separate_from_features(self) -> None:
        row = build_csv_row(
            split="train",
            class_code="c2",
            label="hand_reach_toward_friend",
            status="ok",
            source_path="train/c2/frame.jpg",
            sequence_id=None,
            source_frame_index=1,
            frame_id=1,
            timestamp_ms=1,
            session_id="batch_train",
            frame_width=640,
            frame_height=480,
            person_count=1,
            track=self.track,
            result=self.result,
            annotation={"action_actor_ids": '["s10"]'},
        )

        self.assertEqual(tuple(row), CSV_FIELDNAMES)
        self.assertEqual(row["class_code"], "c2")
        self.assertEqual(row["action_actor_ids"], '["s10"]')
        self.assertEqual(row["pose_000_x"], 0.25)


if __name__ == "__main__":
    unittest.main()

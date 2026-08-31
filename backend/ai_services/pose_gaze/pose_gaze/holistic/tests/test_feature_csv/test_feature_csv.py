"""Tests for the shared fixed-width landmark CSV schema."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from pose_gaze.holistic.feature_csv import (
    CSV_FIELDNAMES,
    MODEL_FEATURE_COLUMNS,
    build_csv_row,
    is_training_row,
    model_features_from_result,
)
from pose_gaze.holistic.landmark import (
    LandmarkPoint,
    TrackHolisticResult,
)
from pose_gaze.tracking.schemas import (
    BoundingBox,
    TrackedPerson,
)
from pose_gaze.holistic.feature_csv.export_json_features import export


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

    def test_world_result_uses_world_points_x_and_y(self) -> None:
        result = TrackHolisticResult(
            track_id=1,
            student_id="student_01",
            bbox=self.track.bbox,
            crop_bbox=self.track.bbox,
            pose_world_landmarks=(
                LandmarkPoint(index=0, x=0.12, y=-0.34),
            ),
        )
        features = model_features_from_result(result)

        self.assertEqual(features["pose_world_000_x"], 0.12)
        self.assertEqual(features["pose_world_000_y"], -0.34)
        self.assertAlmostEqual(features["pose_world_valid_ratio"], 1 / 25)

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

    def test_predicted_face_row_is_marked_and_excluded(self) -> None:
        predicted = TrackHolisticResult(
            track_id=1,
            student_id="student_01",
            bbox=self.track.bbox,
            crop_bbox=self.track.bbox,
            face_valid=False,
            face_predicted=True,
        )
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
            result=predicted,
        )

        self.assertFalse(row["face_observed_mask"])
        self.assertTrue(row["face_predicted"])
        self.assertFalse(is_training_row(row))

    def test_action_only_keeps_interval_starting_at_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("filename", "class_code", "action_start_s", "action_end_s", "split", "split_group", "actor_ids", "action_actor_ids", "interaction_pairs"))
                writer.writerow(("clip.mp4", "c1", "0", "0.5", "train", "group_01", '["s1"]', '["s1"]', "[]"))
            json_dir = root / "json"
            json_dir.mkdir()
            (json_dir / "clip.json").write_text(
                json.dumps(
                    {"frames": [
                        {"frame_id": 0, "timestamp_ms": 0, "tracks": [{"track_id": "1", "bbox_xyxy": [10, 10, 100, 200]}]},
                        {"frame_id": 1, "timestamp_ms": 500, "tracks": [{"track_id": "1", "bbox_xyxy": [10, 10, 100, 200]}]},
                        {"frame_id": 2, "timestamp_ms": 501, "tracks": [{"track_id": "1", "bbox_xyxy": [10, 10, 100, 200]}]},
                    ]}
                ),
                encoding="utf-8",
            )
            output = root / "output"
            export(manifest, json_dir, output, selected_variant="2d", action_scope="action_only")
            with (output / "features_2d.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["timestamp_ms"] for row in rows], ["0", "500"])

    def test_action_only_rejects_partial_non_c5_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "filename,class_code,action_start_s,action_end_s\nclip.mp4,c4,1,\n",
                encoding="utf-8",
            )
            json_dir = root / "json"
            json_dir.mkdir()
            (json_dir / "clip.json").write_text('{"frames": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "action_only requires finite"):
                export(manifest, json_dir, root / "output", selected_variant="2d", action_scope="action_only")

    def test_export_uses_manifest_subset_of_json_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("filename", "class_code", "split", "split_group", "actor_ids", "action_actor_ids", "interaction_pairs"))
                writer.writerow(("front.mp4", "c5", "train", "group_01", '["s1"]', "[]", "[]"))
            json_dir = root / "json"
            json_dir.mkdir()
            for stem in ("front", "rear"):
                (json_dir / f"{stem}.json").write_text(
                    json.dumps({"frames": [{"frame_id": 0, "timestamp_ms": 0, "tracks": [{"track_id": "1", "bbox_xyxy": [10, 10, 100, 200]}]}]}),
                    encoding="utf-8",
                )
            output = root / "output"
            export(manifest, json_dir, output, selected_variant="2d", action_scope="action_only")
            with (output / "features_2d.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual([row["source_filename"] for row in rows], ["front.mp4"])


if __name__ == "__main__":
    unittest.main()

import csv
import json
import tempfile
import unittest
from pathlib import Path

from pose_gaze.holistic.annotation_schema import ANNOTATION_COLUMNS, validate_annotation_rows
from pose_gaze.holistic.build_annotation_template import build


class AnnotationSchemaTests(unittest.TestCase):
    def test_template_is_front_only_and_pair_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "clip_id", "camera_view_id", "action_actor_ids", "class_code", "duration_s", "actual_fps", "action_start_s", "action_end_s", "split", "split_group"])
                writer.writerow(["front.mp4", "clip1", "front", json.dumps(["a", "b"]), "c1", "10", "30", "0", "10", "train", "group_10"])
                writer.writerow(["rear.mp4", "clip2", "rear", json.dumps(["a"]), "c1", "10", "30", "0", "10", "train", "group_10"])
            output = root / "annotations.csv"
            count = build(manifest, output)
            self.assertEqual(count, 18)  # 2 actors * 7 subject + 2 pair events
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue(all(row["camera_view_id"] == "front" for row in rows))
            self.assertTrue(all(row["event_scope"] == "pair" or not row["target_actor_id"] for row in rows))

    def test_validator_rejects_overlap_and_confirmed_without_annotator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "clip_id", "camera_view_id", "action_actor_ids", "class_code", "duration_s", "split", "split_group"])
                writer.writerow(["front.mp4", "clip1", "front", json.dumps(["a", "b"]), "c1", "10", "train", "group_10"])
            annotation = root / "annotations.csv"
            row = {column: "" for column in ANNOTATION_COLUMNS}
            row.update({"source_filename": "front.mp4", "camera_view_id": "front", "duration_s": "10", "split": "train", "split_group": "group_10", "actor_id": "a", "event_scope": "subject", "event_type": "mouth_active", "status": "confirmed", "start_time_ms": "100", "end_time_ms": "500"})
            row2 = dict(row)
            row2.update({"interval_index": "2", "status": "draft", "start_time_ms": "400", "end_time_ms": "600"})
            with annotation.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
                writer.writeheader()
                writer.writerows([row, row2])
            errors = validate_annotation_rows(annotation, manifest)
            self.assertTrue(any("confirmed annotation needs annotator" in error for error in errors))
            self.assertTrue(any("overlap" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from pose_gaze.holistic.feature_csv.prepare_front_v3_manifest import build_front_v3_manifest


class FrontV3ManifestTests(unittest.TestCase):
    def test_filters_rear_and_assigns_primary_challenge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            source.write_text(
                "filename,camera_view_id,split_group,split\n"
                "a.mp4,front,group_06,train\n"
                "b.mp4,front,group_01,test\n"
                "c.mp4,front,group_10,test\n"
                "d.mp4,rear,group_10,test\n",
                encoding="utf-8",
            )
            output = root / "front_v3.csv"
            self.assertEqual(build_front_v3_manifest(source, output), {"test": 1, "train": 1, "challenge": 1})
            with output.open(encoding="utf-8", newline="") as handle:
                rows = {row["filename"]: row for row in csv.DictReader(handle)}

        self.assertEqual(set(rows), {"a.mp4", "b.mp4", "c.mp4"})
        self.assertEqual((rows["a.mp4"]["split"], rows["a.mp4"]["evaluation_role"]), ("test", "primary"))
        self.assertEqual((rows["b.mp4"]["split"], rows["b.mp4"]["evaluation_role"]), ("train", "train"))
        self.assertEqual((rows["c.mp4"]["split"], rows["c.mp4"]["evaluation_role"]), ("challenge", "stress"))

    def test_moves_challenge_group_into_train_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            source.write_text(
                "filename,camera_view_id,split_group,split\n"
                "a.mp4,front,group_06,train\n"
                "b.mp4,front,group_10,test\n",
                encoding="utf-8",
            )
            output = root / "front_v4.csv"
            self.assertEqual(build_front_v3_manifest(source, output, challenge_as_train=True), {"test": 1, "train": 1})
            with output.open(encoding="utf-8", newline="") as handle:
                rows = {row["filename"]: row for row in csv.DictReader(handle)}

        self.assertEqual((rows["b.mp4"]["split"], rows["b.mp4"]["evaluation_role"]), ("train", "train"))


if __name__ == "__main__":
    unittest.main()

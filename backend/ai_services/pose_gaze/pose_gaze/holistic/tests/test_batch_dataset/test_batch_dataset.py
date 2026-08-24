"""Tests for dataset discovery and deterministic split assignment."""

from __future__ import annotations

import csv
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from pose_gaze.holistic.batch_dataset import (
    DatasetSample,
    deterministic_split,
    discover_samples,
    export_dataset,
    parse_class_folder,
)
from pose_gaze.holistic.landmark import LandmarkPoint, TrackHolisticResult
from pose_gaze.tracking.schemas import BoundingBox, PersonDetection


class _Frame:
    shape = (480, 640, 3)


class _Cv2:
    @staticmethod
    def imread(path: str) -> _Frame:
        return _Frame()


class _Detector:
    @staticmethod
    def detect(frame: _Frame) -> list[PersonDetection]:
        return [PersonDetection(BoundingBox(10, 20, 110, 220), 0.9)]


class _Holistic:
    @staticmethod
    def process_packet(frame: _Frame, packet):
        return tuple(
            TrackHolisticResult(
                track_id=track.track_id,
                student_id=track.student_id,
                bbox=track.bbox,
                crop_bbox=track.bbox,
                pose_landmarks=(LandmarkPoint(0, x=0.5, y=0.5),),
            )
            for track in packet.tracks
        )


class BatchDatasetTests(unittest.TestCase):
    def test_class_folder_can_keep_code_and_label(self) -> None:
        self.assertEqual(
            parse_class_folder("c2__hand_reach_toward_friend", {}),
            ("c2", "hand_reach_toward_friend"),
        )
        self.assertEqual(parse_class_folder("c5", {"c5": "normal"}), ("c5", "normal"))

    def test_deterministic_split_is_stable(self) -> None:
        first = deterministic_split(
            "c5/frame_001.jpg",
            train_ratio=0.8,
            val_ratio=0.1,
            seed="same",
        )
        second = deterministic_split(
            "c5/frame_001.jpg",
            train_ratio=0.8,
            val_ratio=0.1,
            seed="same",
        )
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "val", "test"})

    def test_pre_split_folders_are_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "train" / "c5__normal" / "frame.jpg"
            image.parent.mkdir(parents=True)
            image.touch()

            samples = discover_samples(
                root,
                manifest_path=None,
                class_map={},
                train_ratio=0.8,
                val_ratio=0.1,
                seed="test",
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].split, "train")
        self.assertEqual(samples[0].class_code, "c5")
        self.assertEqual(samples[0].label, "normal")

    def test_sequence_keeps_track_id_and_increments_frame_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "train" / "c5" / "video_01" / "frame_001.jpg"
            second = root / "train" / "c5" / "video_01" / "frame_002.jpg"
            first.parent.mkdir(parents=True)
            first.touch()
            second.touch()
            samples = [
                DatasetSample(first, "train", "c5", "normal", "video_01"),
                DatasetSample(second, "train", "c5", "normal", "video_01"),
            ]

            with redirect_stdout(io.StringIO()):
                outputs, counts = export_dataset(
                    samples,
                    input_root=root.resolve(),
                    output_dir=root / "output",
                    detector=_Detector(),
                    holistic=_Holistic(),
                    cv2_module=_Cv2(),
                    max_people=2,
                    overwrite=False,
                    fail_fast=True,
                    log_every=10,
                )
            with outputs["train"].open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(counts["train:ok"], 2)
        self.assertEqual([row["track_id"] for row in rows], ["1", "1"])
        self.assertEqual([row["frame_id"] for row in rows], ["1", "2"])

    def test_unsplit_sequence_is_kept_in_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = root / "c5" / "video_01"
            sequence.mkdir(parents=True)
            (sequence / "frame_001.jpg").touch()
            (sequence / "frame_002.jpg").touch()

            samples = discover_samples(
                root,
                manifest_path=None,
                class_map={"c5": "normal"},
                train_ratio=0.8,
                val_ratio=0.1,
                seed="group-safe",
            )

        self.assertEqual({sample.sequence_id for sample in samples}, {"video_01"})
        self.assertEqual(len({sample.split for sample in samples}), 1)

    def test_manifest_headers_from_dataset_note_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frame.jpg").touch()
            manifest = root / "annotations.tsv"
            manifest.write_text(
                "filename\tclass_code\tlabel\tsplit\tObserved_sign_code\tQuality\n"
                "frame.jpg\tc7\tsign_code\ttrain\tTRUE\tnormal\n",
                encoding="utf-8",
            )

            samples = discover_samples(
                root,
                manifest_path=manifest,
                class_map={},
                train_ratio=0.8,
                val_ratio=0.1,
                seed="unused",
            )

        self.assertEqual(samples[0].annotation["observed_sign_code"], "TRUE")
        self.assertEqual(samples[0].annotation["quality"], "normal")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import dataclass
import json
import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_gaze.holistic.landmark import (
    HAND_LANDMARK_INDICES,
    HolisticLandmarkExtractor,
    LandmarkPoint,
    POSE_LANDMARK_INDICES,
    SELECTED_FACE_CONNECTIONS,
    SELECTED_FACE_LANDMARK_INDICES,
    _LetterboxTransform,
)
from pose_gaze.holistic.test_media import LandmarkJsonWriter
from pose_gaze.tracking.schemas import BoundingBox


@dataclass
class _Landmark:
    x: float
    y: float
    visibility: float | None = None
    presence: float | None = None


@dataclass
class _LegacyLandmarkList:
    landmark: list[_Landmark]


class _LegacyOptionalScores:
    x = 0.50
    y = 0.25
    visibility = 0.0
    presence = 0.0

    @staticmethod
    def HasField(name: str) -> bool:
        return False


class _FakeCV2:
    INTER_AREA = 1
    INTER_LINEAR = 2
    BORDER_CONSTANT = 0

    @staticmethod
    def resize(image, size, interpolation):
        del interpolation
        width, height = size
        return np.zeros((height, width, image.shape[2]), dtype=image.dtype)

    @staticmethod
    def copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        border_type,
        value,
    ):
        del border_type, value
        return np.pad(
            image,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
        )


class HolisticLandmarkExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        # These helpers do not need OpenCV or MediaPipe to be installed.
        self.extractor = object.__new__(HolisticLandmarkExtractor)
        self.extractor._last_timestamps = {}
        self.extractor._min_tracking_confidence = 0.50
        self.extractor._soft_landmark_confidence = 0.20

    def test_face_prediction_is_disabled_by_default(self) -> None:
        default = inspect.signature(HolisticLandmarkExtractor.__init__).parameters[
            "face_hold_frames"
        ].default
        self.assertEqual(default, 0)

    def test_hand_support_rejects_points_far_from_pose_wrist(self) -> None:
        pose = [
            LandmarkPoint(index=11, frame_x=100, frame_y=100),
            LandmarkPoint(index=12, frame_x=200, frame_y=100),
            LandmarkPoint(index=15, frame_x=110, frame_y=200),
        ]
        hand = tuple(
            LandmarkPoint(index=index, frame_x=500 + index, frame_y=500)
            for index in range(4)
        )

        accepted, accepted_world = self.extractor._validate_hand_support(
            pose, hand, hand, pose_wrist_index=15
        )

        self.assertEqual(accepted, ())
        self.assertEqual(accepted_world, ())

    def test_hand_support_accepts_points_anchored_to_pose_wrist(self) -> None:
        pose = [
            LandmarkPoint(index=11, frame_x=100, frame_y=100),
            LandmarkPoint(index=12, frame_x=200, frame_y=100),
            LandmarkPoint(index=15, frame_x=110, frame_y=200),
        ]
        hand = tuple(
            LandmarkPoint(index=index, frame_x=110 + index, frame_y=200 + index)
            for index in range(4)
        )

        accepted, accepted_world = self.extractor._validate_hand_support(
            pose, hand, hand, pose_wrist_index=15
        )

        self.assertEqual(accepted, hand)
        self.assertEqual(accepted_world, hand)

    def test_normalized_points_accept_tasks_list(self) -> None:
        landmarks = [_Landmark(0.25, 0.50, 0.90, 0.80)]

        points = self.extractor._normalized_points(
            landmarks,
            BoundingBox(100, 50, 300, 250),
        )

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].frame_x, 150)
        self.assertEqual(points[0].frame_y, 150)
        self.assertEqual(points[0].visibility, 0.90)
        self.assertEqual(points[0].presence, 0.80)

    def test_normalized_points_accept_legacy_protobuf_shape(self) -> None:
        landmarks = _LegacyLandmarkList([_Landmark(0.50, 0.25)])

        points = self.extractor._normalized_points(
            landmarks,
            BoundingBox(20, 40, 220, 140),
        )

        self.assertEqual(points[0].frame_x, 120)
        self.assertEqual(points[0].frame_y, 65)

    def test_unset_legacy_scores_do_not_become_false_zero_confidence(self) -> None:
        points = self.extractor._normalized_points(
            _LegacyLandmarkList([_LegacyOptionalScores()]),
            BoundingBox(20, 40, 220, 140),
        )

        self.assertEqual(points[0].x, 0.50)
        self.assertEqual(points[0].y, 0.25)
        self.assertIsNone(points[0].visibility)
        self.assertIsNone(points[0].presence)

    def test_letterboxed_points_map_back_to_original_crop(self) -> None:
        # A 200x400 crop becomes 256x512 with 128px horizontal padding.
        transform = _LetterboxTransform(
            input_width=512,
            input_height=512,
            content_width=256,
            content_height=512,
            pad_left=128,
            pad_top=0,
        )
        landmarks = [_Landmark(0.50, 0.50)]

        points = self.extractor._normalized_points(
            landmarks,
            BoundingBox(100, 50, 300, 450),
            transform,
        )

        self.assertAlmostEqual(points[0].x, 0.50)
        self.assertAlmostEqual(points[0].y, 0.50)
        self.assertAlmostEqual(points[0].frame_x, 200)
        self.assertAlmostEqual(points[0].frame_y, 250)
        self.assertNotIn("z", points[0].to_dict())

    def test_only_requested_landmark_indices_are_materialized(self) -> None:
        landmarks = [
            _Landmark(0.25, 0.50, visibility=0.90, presence=0.90)
            for _ in range(33)
        ]

        points = self.extractor._normalized_points(
            landmarks,
            BoundingBox(0, 0, 100, 100),
            included_indices=POSE_LANDMARK_INDICES,
        )

        self.assertEqual(
            tuple(point.index for point in points),
            tuple(range(25)),
        )
        self.assertNotIn(25, {point.index for point in points})

    def test_world_points_use_same_filter_and_have_no_depth_field(self) -> None:
        landmarks = [
            _Landmark(0.10, 0.20, visibility=0.90, presence=0.90)
            for _ in range(33)
        ]

        points = self.extractor._world_points(
            landmarks,
            included_indices=POSE_LANDMARK_INDICES,
        )

        self.assertEqual(len(points), 25)
        self.assertNotIn("z", points[0].to_dict())

    def test_landmark_writer_emits_version_two_xy_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "landmarks.json"
            writer = LandmarkJsonWriter(
                output_path,
                input_path=Path("input.jpg"),
                session_id="test_session",
                media_type="image",
            )
            writer.write(
                {
                    "tracks": [
                        {
                            "pose_landmarks": [
                                self.extractor._normalized_points(
                                    [
                                        _Landmark(
                                            0.25,
                                            0.50,
                                            visibility=0.90,
                                            presence=0.80,
                                        )
                                    ],
                                    BoundingBox(0, 0, 100, 100),
                                )[0].to_dict()
                            ]
                        }
                    ]
                }
            )
            writer.close()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["format_version"], 3)
        point = payload["frames"][0]["tracks"][0]["pose_landmarks"][0]
        self.assertNotIn("z", point)

    def test_soft_confidence_keeps_xy_and_nulls_visibility(self) -> None:
        points = self.extractor._normalized_points(
            [_Landmark(0.40, 0.60, visibility=0.30, presence=0.80)],
            BoundingBox(0, 0, 100, 100),
        )

        self.assertEqual(points[0].x, 0.40)
        self.assertEqual(points[0].y, 0.60)
        self.assertEqual(points[0].frame_x, 40)
        self.assertEqual(points[0].frame_y, 60)
        self.assertIsNone(points[0].visibility)
        self.assertEqual(points[0].presence, 0.80)

    def test_below_soft_confidence_nulls_all_measurements(self) -> None:
        points = self.extractor._normalized_points(
            [_Landmark(0.40, 0.60, visibility=0.10, presence=0.90)],
            BoundingBox(0, 0, 100, 100),
        )

        payload = points[0].to_dict()
        self.assertEqual(payload["index"], 0)
        self.assertTrue(
            all(value is None for key, value in payload.items() if key != "index")
        )

    def test_connection_index_sets_match_the_export_contract(self) -> None:
        self.assertEqual(POSE_LANDMARK_INDICES, frozenset(range(25)))
        self.assertEqual(HAND_LANDMARK_INDICES, frozenset(range(21)))
        self.assertNotIn(389, SELECTED_FACE_LANDMARK_INDICES)

    def test_changing_rois_produce_the_same_tasks_input_shape(self) -> None:
        self.extractor._cv2 = _FakeCV2()
        self.extractor._task_input_size = 512

        first, _ = self.extractor._letterbox_task_input(
            np.zeros((630, 430, 3), dtype=np.uint8)
        )
        second, _ = self.extractor._letterbox_task_input(
            np.zeros((626, 427, 3), dtype=np.uint8)
        )

        self.assertEqual(first.shape, (512, 512, 3))
        self.assertEqual(second.shape, (512, 512, 3))

    def test_video_timestamp_is_strictly_monotonic_per_track(self) -> None:
        self.assertEqual(self.extractor._monotonic_timestamp(7, 1000), 1000)
        self.assertEqual(self.extractor._monotonic_timestamp(7, 1000), 1001)
        self.assertEqual(self.extractor._monotonic_timestamp(7, 999), 1002)
        self.assertEqual(self.extractor._monotonic_timestamp(8, 500), 500)

    def test_selected_face_contains_mouth_forehead_and_head_axis(self) -> None:
        self.assertIn((61, 146), SELECTED_FACE_CONNECTIONS)
        self.assertIn((10, 109), SELECTED_FACE_CONNECTIONS)
        self.assertIn((10, 1), SELECTED_FACE_CONNECTIONS)


if __name__ == "__main__":
    unittest.main()

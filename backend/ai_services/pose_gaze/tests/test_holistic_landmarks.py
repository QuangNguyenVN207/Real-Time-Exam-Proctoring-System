from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from backend.ai_services.pose_gaze.holistic_landmarks import (
    HolisticLandmarkExtractor,
    SELECTED_FACE_CONNECTIONS,
    _LetterboxTransform,
)
from backend.ai_services.pose_gaze.tracking.schemas import BoundingBox


@dataclass
class _Landmark:
    x: float
    y: float
    z: float
    visibility: float | None = None
    presence: float | None = None


@dataclass
class _LegacyLandmarkList:
    landmark: list[_Landmark]


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

    def test_normalized_points_accept_tasks_list(self) -> None:
        landmarks = [_Landmark(0.25, 0.50, -0.10, 0.90, 0.80)]

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
        landmarks = _LegacyLandmarkList([_Landmark(0.50, 0.25, 0.20)])

        points = self.extractor._normalized_points(
            landmarks,
            BoundingBox(20, 40, 220, 140),
        )

        self.assertEqual(points[0].frame_x, 120)
        self.assertEqual(points[0].frame_y, 65)

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
        landmarks = [_Landmark(0.50, 0.50, 0.10)]

        points = self.extractor._normalized_points(
            landmarks,
            BoundingBox(100, 50, 300, 450),
            transform,
        )

        self.assertAlmostEqual(points[0].x, 0.50)
        self.assertAlmostEqual(points[0].y, 0.50)
        self.assertAlmostEqual(points[0].z, 0.20)
        self.assertAlmostEqual(points[0].frame_x, 200)
        self.assertAlmostEqual(points[0].frame_y, 250)

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

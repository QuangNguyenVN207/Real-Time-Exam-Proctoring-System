from __future__ import annotations

import math
import unittest

from pose_gaze.holistic.landmark.face_mapping import (
    OneEuroFilter2D,
    fit_affine_2d,
    fit_kabsch_2d,
    fit_pose_anchor_transform,
)
from pose_gaze.holistic.landmark.landmarks import LandmarkPoint
from pose_gaze.holistic.landmark.landmarks import HolisticLandmarkExtractor


class FaceMappingTests(unittest.TestCase):
    def test_world_fields_are_optional_in_json(self) -> None:
        legacy = LandmarkPoint(index=1, x=0.2, y=0.3).to_dict()
        mapped = LandmarkPoint(index=1, x=0.2, y=0.3, world_x=4.0, world_y=5.0).to_dict()

        self.assertNotIn("world_x", legacy)
        self.assertEqual((mapped["world_x"], mapped["world_y"]), (4.0, 5.0))

    def test_face_mapper_writes_finite_pseudo_world_coordinates(self) -> None:
        face = (LandmarkPoint(index=1, frame_x=50.0, frame_y=20.0),)
        pose = (
            LandmarkPoint(index=11, frame_x=25.0, frame_y=50.0),
            LandmarkPoint(index=12, frame_x=75.0, frame_y=50.0),
        )

        mapped, valid, source = HolisticLandmarkExtractor._map_face_to_pseudo_world(
            face,
            pose,
            frame_width=100,
            frame_height=100,
        )

        self.assertTrue(valid)
        self.assertEqual(source, "shoulders")
        self.assertTrue(math.isfinite(mapped[0].world_x))
        self.assertTrue(math.isfinite(mapped[0].world_y))

    def test_kabsch_recovers_known_transform(self) -> None:
        source = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        target = [(2.0, 3.0), (2.0, 5.0), (0.0, 3.0), (0.0, 5.0)]

        transform = fit_kabsch_2d(source, target)

        self.assertIsNotNone(transform)
        self.assertAlmostEqual(transform.scale, 2.0)
        self.assertLess(transform.residual, 1e-9)
        self.assertEqual(transform.apply((0.25, 0.5)), (1.0, 3.5))

    def test_affine_benchmark_recovers_known_mapping(self) -> None:
        source = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        target = [(2.0, 3.0), (4.0, 3.0), (2.0, 6.0), (4.0, 6.0)]

        transform = fit_affine_2d(source, target)

        self.assertIsNotNone(transform)
        self.assertLess(transform.residual, 1e-9)
        self.assertEqual(transform.apply((0.25, 0.5)), (2.5, 4.5))

    def test_affine_rejects_wrong_point_count(self) -> None:
        self.assertIsNone(fit_affine_2d([(0.0, 0.0)] * 3, [(0.0, 0.0)] * 3))

    def test_kabsch_rejects_degenerate_or_non_finite_anchors(self) -> None:
        target = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]

        self.assertIsNone(fit_kabsch_2d([(1.0, 1.0)] * 4, target))
        self.assertIsNone(fit_kabsch_2d([(math.nan, 0.0)] * 4, target))

    def test_pose_anchor_selection_prefers_shoulders_then_uses_hips(self) -> None:
        fitted = fit_pose_anchor_transform({11: (0.2, 0.2), 12: (0.8, 0.2)})
        self.assertIsNotNone(fitted)
        self.assertEqual(fitted[1], "shoulders")

        fitted = fit_pose_anchor_transform({23: (0.2, 0.8), 24: (0.8, 0.8)})
        self.assertIsNotNone(fitted)
        self.assertEqual(fitted[1], "hips")

    def test_one_euro_reduces_stationary_jitter(self) -> None:
        filter_ = OneEuroFilter2D(min_cutoff=0.5, beta=0.0)
        samples = [(0.5 + (-1) ** index * 0.1, 0.5) for index in range(20)]

        filtered = [filter_.update(point, index * 33.0) for index, point in enumerate(samples)]

        self.assertLess(abs(filtered[-1][0] - 0.5), abs(samples[-1][0] - 0.5))


if __name__ == "__main__":
    unittest.main()
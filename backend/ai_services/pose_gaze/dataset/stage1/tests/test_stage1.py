from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from backend.ai_services.pose_gaze.dataset.stage1.build import (
    _classify_window,
    _binary_label,
    _normalize_analysis_frame,
    _sample_timestamps,
    _selected_frame_indices,
    _window_bounds,
)
from backend.ai_services.pose_gaze.dataset.stage1.common import WINDOW_FRAMES, WINDOW_OVERLAP_FRAMES


class Stage1WindowTests(unittest.TestCase):
    def test_sample_timestamps_at_10_fps(self) -> None:
        timestamps = _sample_timestamps(1.05)
        self.assertGreaterEqual(len(timestamps), 10)
        self.assertAlmostEqual(timestamps[1] - timestamps[0], 0.1, places=3)

    def test_window_classification(self) -> None:
        label, include = _classify_window(0.0, 3.0, 1.0, 2.0, 10.0)
        self.assertEqual(label, "action")
        self.assertTrue(include)

        label, include = _classify_window(2.0, 4.9, 5.0, 7.0, 12.0)
        self.assertEqual(label, "non_action")
        self.assertTrue(include)

    def test_window_overlap_is_10_and_stride_is_20(self) -> None:
        self.assertEqual(WINDOW_OVERLAP_FRAMES, 10)
        bounds = _window_bounds(70)
        self.assertEqual(bounds[:3], [(0, 30), (20, 50), (40, 70)])
        self.assertEqual(WINDOW_FRAMES - WINDOW_OVERLAP_FRAMES, 20)

    def test_normalize_does_not_upscale_or_crop(self) -> None:
        small = np.zeros((360, 640, 3), dtype=np.uint8)
        normalized, source_width, source_height, analysis_width, analysis_height, scale = _normalize_analysis_frame(small)
        self.assertEqual(normalized.shape[:2], (360, 640))
        self.assertEqual((source_width, source_height), (640, 360))
        self.assertEqual((analysis_width, analysis_height), (640, 360))
        self.assertEqual(scale, 1.0)

        large = np.zeros((1080, 1920, 3), dtype=np.uint8)
        normalized, source_width, source_height, analysis_width, analysis_height, scale = _normalize_analysis_frame(large)
        self.assertEqual((source_width, source_height), (1920, 1080))
        self.assertLessEqual(analysis_width, 1280)
        self.assertLessEqual(analysis_height, 720)
        self.assertAlmostEqual(analysis_width / analysis_height, 1920 / 1080, places=2)
        self.assertLess(scale, 1.0)

    def test_window_overlapping_cheating_is_positive(self) -> None:
        label, include = _classify_window(0.0, 3.0, 4.0, 5.0, 10.0)
        self.assertEqual(label, "non_action")
        self.assertTrue(include)

        label, include = _classify_window(3.0, 5.0, 4.0, 5.0, 10.0)
        self.assertEqual(label, "action")
        self.assertTrue(include)

    def test_c5_is_the_only_non_cheating_binary_class(self) -> None:
        self.assertEqual(_binary_label("c5"), "non_cheating")
        self.assertEqual(_binary_label("c1"), "cheating")
        self.assertEqual(_binary_label("c7"), "cheating")
        phase, include = _classify_window(1.0, 4.0, 1.0, 4.0, 10.0, "c5")
        self.assertEqual(phase, "non_action")
        self.assertTrue(include)

    def test_selected_frames_do_not_replace_continuous_sampled_frames(self) -> None:
        rows = []
        for index in range(30):
            rows.append(
                {
                    "frame_index": index,
                    "sharpness_score": float(index),
                    "motion_score": float(30 - index),
                    "diversity_score": float(index % 7),
                }
            )
        frame_index_frame = pd.DataFrame(rows)
        selected = _selected_frame_indices(frame_index_frame, 0, 30)
        self.assertLess(len(selected), 30)
        self.assertEqual(30, WINDOW_FRAMES)
        self.assertIn((0, "start"), selected)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from pose_gaze.holistic.temporal_windows import (
    TemporalWindowConfig,
    build_temporal_windows,
)


class TemporalWindowTests(unittest.TestCase):
    def _rows(self, count: int = 5) -> list[dict[str, object]]:
        return [
            {
                "status": "ok",
                "split": "train",
                "split_group": "session_a",
                "camera_view_id": "front",
                "source_filename": "video.mp4",
                "track_id": "1",
                "frame_id": index,
                "timestamp_ms": index * 40,
                "class_code": "c3",
                "label": "looking_friend",
                "face_predicted": False,
                "pose_000_x": index / 10,
            }
            for index in range(count)
        ]

    def test_builds_fixed_width_contiguous_windows(self) -> None:
        windows = build_temporal_windows(
            self._rows(),
            feature_columns=("pose_000_x",),
            config=TemporalWindowConfig(size=3, stride=1),
        )

        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0]["t00_pose_000_x"], 0.0)
        self.assertEqual(windows[0]["t02_pose_000_x"], 0.2)
        self.assertEqual(windows[0]["window_size"], 3)
        self.assertEqual(windows[0]["camera_view_id"], "front")
        self.assertEqual(windows[0]["label_purity"], 1.0)

    def test_mixed_class_window_is_rejected(self) -> None:
        rows = self._rows()
        rows[2]["class_code"] = "c5"
        windows = build_temporal_windows(
            rows,
            feature_columns=("pose_000_x",),
            config=TemporalWindowConfig(size=3, stride=1),
        )

        self.assertEqual(windows, [])

    def test_event_background_boundary_is_rejected(self) -> None:
        rows = self._rows()
        for row in rows[:2]:
            row["target_state"] = "background"
            row["label"] = "background"
        for row in rows[2:]:
            row["target_state"] = "event"
            row["label"] = "c3"
        windows = build_temporal_windows(
            rows,
            feature_columns=("pose_000_x",),
            config=TemporalWindowConfig(size=3, stride=1),
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["target_state"], "event")
        self.assertEqual(windows[0]["window_start_frame"], 2)

    def test_background_window_keeps_background_label(self) -> None:
        rows = self._rows()
        for row in rows:
            row["target_state"] = "background"
            row["label"] = "background"
        windows = build_temporal_windows(
            rows,
            feature_columns=("pose_000_x",),
            config=TemporalWindowConfig(size=3, stride=1),
        )

        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0]["label"], "background")
        self.assertEqual(windows[0]["target_state"], "background")

    def test_predicted_face_rows_never_enter_window(self) -> None:
        rows = self._rows()
        rows[2]["face_predicted"] = True
        windows = build_temporal_windows(
            rows,
            feature_columns=("pose_000_x",),
            config=TemporalWindowConfig(size=3, stride=1),
        )

        self.assertEqual(windows, [])

    def test_timestamp_gap_breaks_window(self) -> None:
        rows = self._rows()
        rows[2]["timestamp_ms"] = 1000
        windows = build_temporal_windows(
            rows,
            feature_columns=("pose_000_x",),
            config=TemporalWindowConfig(size=3, stride=1),
        )

        self.assertEqual(windows, [])


if __name__ == "__main__":
    unittest.main()

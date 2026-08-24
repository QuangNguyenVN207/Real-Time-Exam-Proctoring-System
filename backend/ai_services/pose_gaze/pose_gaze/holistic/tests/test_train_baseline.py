from __future__ import annotations

import unittest

import numpy as np

from pose_gaze.holistic.feature_csv.train_baseline import _actor_track_metrics


class ActorTrackMetricTests(unittest.TestCase):
    def test_state_aggregation_keeps_background_and_event_separate(self) -> None:
        labels = np.asarray([0, 1, 1], dtype=np.int32)
        probabilities = np.eye(2, dtype=np.float32)[labels]
        sources = np.asarray(["video.mp4", "video.mp4", "video.mp4"])
        tracks = np.asarray(["1", "1", "1"])
        states = np.asarray(["background", "event", "event"])

        metrics = _actor_track_metrics(labels, probabilities, sources, tracks, states)

        self.assertEqual(metrics["count"], 2.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

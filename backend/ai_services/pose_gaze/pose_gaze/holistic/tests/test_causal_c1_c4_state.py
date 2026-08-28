from __future__ import annotations

import unittest
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.ai_services.pose_gaze.pose_gaze.holistic.feature_csv.causal_c1_c4_state import (
    C1C4Thresholds,
    CausalC1C4ActorState,
)
from backend.ai_services.pose_gaze.pose_gaze.holistic.feature_csv.c1_c4_pose_features import (
    apply_causal_episode_runs,
)
from backend.ai_services.pose_gaze.pose_gaze.holistic.feature_csv.benchmark_c1_causal import (
    FEATURES,
    PoseStream,
)
from backend.ai_services.pose_gaze.pose_gaze.holistic.test_media.live_actor import (
    CausalPoseActorClassifier,
)


def frame(head_y: float, wrist_y: float) -> dict[int, tuple[float, float]]:
    return {
        0: (0.5, head_y),
        11: (0.4, 0.4),
        12: (0.6, 0.4),
        13: (0.43, 0.55),
        14: (0.57, 0.55),
        15: (0.44, wrist_y),
        16: (0.56, wrist_y),
        23: (0.44, 0.7),
        24: (0.56, 0.7),
    }


class CausalC1C4StateTest(unittest.TestCase):
    @staticmethod
    def _artifact(root: Path, class_code: str) -> None:
        names = [f"{name}__{stat}" for name in FEATURES for stat in ("last", "mean", "max", "min", "std")]
        x = np.asarray([[0.0] * len(names), [1.0] * len(names)], dtype=np.float64)
        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42)).fit(x, [0, 1])
        (root / f"{class_code}_actor_metrics.json").write_text(json.dumps({
            "causal": True, "future_frames_used_for_decision": False,
            "pose_threshold_train_only": 0.0,
        }), encoding="utf-8")
        (root / f"{class_code}_feature_names.json").write_text(json.dumps(names), encoding="utf-8")
        with (root / f"{class_code}_pose_specialist.pkl").open("wb") as handle:
            pickle.dump(model, handle)

    @staticmethod
    def _track(points: dict[int, tuple[float, float]]) -> dict[str, object]:
        return {
            "track_id": 1, "student_id": "student_01",
            "pose_landmarks": [
                {"index": index, "frame_x": x, "frame_y": y}
                for index, (x, y) in points.items()
            ],
        }

    def test_shared_live_pose_classifier_is_prefix_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifact(root, "c1")

            def replay(rows):
                classifier = CausalPoseActorClassifier({"c1": root}, warmup_frames=2)
                return [classifier.update_tracks(
                    frame_index=index, timestamp_ms=index * 33,
                    tracks=[self._track(points)],
                ) for index, points in enumerate(rows)]

            rows = [frame(0.30, 0.50), frame(0.32, 0.55), frame(0.55, 0.70)]
            self.assertEqual(replay(rows[:2]), replay(rows)[:2])

    def test_pose_only_c1_c4_are_exposed_as_suspicious_activity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifact(root, "c1")
            self._artifact(root, "c4")
            classifier = CausalPoseActorClassifier(
                {"c1": root, "c4": root}, warmup_frames=1
            )
            decision = classifier.update_tracks(
                frame_index=0, timestamp_ms=0,
                tracks=[self._track(frame(0.55, 0.70))],
            )["student_01"]
            self.assertEqual(decision["predicted_class"], "suspicious_activity")
            self.assertIn(decision["source_specialist"], {"c1", "c4"})

    def test_actor_owned_object_subtypes_pose_superclass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._artifact(root, "c1")
            classifier = CausalPoseActorClassifier({"c1": root}, warmup_frames=1)
            track = self._track(frame(0.55, 0.70))

            pose_only = classifier.update_tracks(
                frame_index=0, timestamp_ms=0, tracks=[track]
            )["student_01"]
            self.assertEqual(pose_only["predicted_class"], "suspicious_activity")

            refined = classifier.update_tracks(
                frame_index=1,
                timestamp_ms=33,
                tracks=[track],
                object_rows_by_actor={
                    "student_01": {
                        "direct_object_class": "c1",
                        "direct_object_score": 0.91,
                    }
                },
            )["student_01"]
            self.assertEqual(refined["predicted_class"], "c1")
            self.assertEqual(refined["object_class"], "c1")
            self.assertAlmostEqual(refined["object_score"], 0.91)

    def test_actor_owned_object_can_create_c4_without_pose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            classifier = CausalPoseActorClassifier({}, warmup_frames=1)
            decision = classifier.update_tracks(
                frame_index=0,
                timestamp_ms=0,
                tracks=[self._track(frame(0.55, 0.70))],
                object_rows_by_actor={
                    "student_01": {
                        "direct_object_class": "c4",
                        "direct_object_score": 0.95,
                    }
                },
            )["student_01"]
            self.assertEqual(decision["predicted_class"], "c4")
            self.assertTrue(decision["object_priority"])
            self.assertEqual(decision["first_flag_frame_index"], 0)

    def test_c1_pose_prefix_features_are_future_invariant(self) -> None:
        def row(index: int, head_y: float, wrist_y: float):
            values = {
                "source_frame_index": str(index),
                "timestamp_ms": str(index * 33),
            }
            points = frame(head_y, wrist_y)
            for point_index, (x, y) in points.items():
                values[f"pose_{point_index}_valid"] = "1"
                values[f"pose_{point_index}_frame_x"] = str(x)
                values[f"pose_{point_index}_frame_y"] = str(y)
            return values

        rows = [
            row(0, 0.30, 0.50),
            row(1, 0.32, 0.55),
            row(2, 0.55, 0.70),
            row(3, 0.56, 0.72),
        ]
        prefix_stream = PoseStream(baseline_frames=2)
        prefix = [prefix_stream.update(item) for item in rows[:3]]
        full_stream = PoseStream(baseline_frames=2)
        full = [full_stream.update(item) for item in rows]
        self.assertEqual(prefix, full[:3])

    def test_legacy_feature_exporter_no_longer_backfills_future_run(self) -> None:
        prefix = [
            {"head_down_pose": 0, "arm_departure_pose": 0},
            {"head_down_pose": 1, "arm_departure_pose": 1},
        ]
        full = [dict(row) for row in prefix] + [
            {"head_down_pose": 1, "arm_departure_pose": 1}
        ]
        thresholds = {"enter": 2, "arm": 2, "overlap": 2}
        apply_causal_episode_runs(prefix, thresholds)
        apply_causal_episode_runs(full, thresholds)
        self.assertEqual(prefix, full[:2])
        self.assertEqual(full[0]["shared_c1_c4_episode_pose"], 0)
        self.assertEqual(full[2]["shared_c1_c4_episode_pose"], 1)

    def test_prefix_outputs_do_not_change_when_future_is_appended(self) -> None:
        thresholds = C1C4Thresholds(enter_frames=3, deep_head_absolute=0.1)
        stream = [frame(0.30, 0.50), frame(0.30, 0.50), frame(0.55, 0.70), frame(0.55, 0.70)]

        prefix_state = CausalC1C4ActorState(thresholds, baseline_frames=2)
        prefix = [
            prefix_state.update(frame_index=index, timestamp_ms=index * 33.0, points=points)
            for index, points in enumerate(stream[:3])
        ]

        full_state = CausalC1C4ActorState(thresholds, baseline_frames=2)
        full = [
            full_state.update(frame_index=index, timestamp_ms=index * 33.0, points=points)
            for index, points in enumerate(stream)
        ]
        self.assertEqual(prefix, full[:3])
        self.assertTrue(all(not row["future_frames_used"] for row in full))

    def test_no_retroactive_flag(self) -> None:
        thresholds = C1C4Thresholds(enter_frames=2, deep_head_absolute=0.1)
        state = CausalC1C4ActorState(thresholds, baseline_frames=1)
        outputs = [
            state.update(frame_index=index, timestamp_ms=index * 33.0, points=points)
            for index, points in enumerate(
                [frame(0.30, 0.50), frame(0.55, 0.70), frame(0.55, 0.70)]
            )
        ]
        self.assertFalse(outputs[0]["shared_c1_c4_qualified"])
        self.assertFalse(outputs[1]["shared_c1_c4_qualified"])
        self.assertTrue(outputs[2]["shared_c1_c4_qualified"])
        self.assertEqual(outputs[2]["first_flag_frame"], 2)


if __name__ == "__main__":
    unittest.main()

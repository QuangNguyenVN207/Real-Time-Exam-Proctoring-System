import unittest
import json
import math

from pose_gaze.holistic.feature_csv.behavior_features import BEHAVIOR_COLUMNS, derive_behavior_features
from pose_gaze.holistic.feature_csv.train_baseline import _event_summary_features, _selective_event_summary_features
from pose_gaze.holistic.feature_csv import behavior_subset_stage2 as stage2


def _row(frame: int, wrist_y: float = 0.60) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "frame_id": frame, "timestamp_ms": (frame - 1) * 33, "track_id": "1",
        "source_filename": "synthetic.json", "split": "train", "split_group": "g1",
    }
    pose = {0: (0.50, 0.30), 7: (0.45, 0.27), 8: (0.55, 0.27), 11: (0.40, 0.40), 12: (0.60, 0.40), 15: (0.40, wrist_y), 16: (0.60, wrist_y), 23: (0.45, 0.80), 24: (0.55, 0.80)}
    for index, (x, y) in pose.items():
        row[f"pose_{index:03d}_x"], row[f"pose_{index:03d}_y"] = x, y
    face = {1: (0.50, 0.30), 13: (0.49, 0.34), 14: (0.49, 0.38), 33: (0.45, 0.29), 263: (0.55, 0.29), 61: (0.46, 0.35), 291: (0.54, 0.35)}
    for index, (x, y) in face.items():
        row[f"face_{index:03d}_x"], row[f"face_{index:03d}_y"] = x, y
    for side, x in (("left", 0.40), ("right", 0.60)):
        for index in range(21):
            row[f"{side}_hand_{index:03d}_x"] = x + (index % 3 - 1) * 0.01
            row[f"{side}_hand_{index:03d}_y"] = wrist_y + (index % 4) * 0.01
        row[f"{side}_hand_000_x"] = x
        row[f"{side}_hand_000_y"] = wrist_y
    return row


class BehaviorFeatureTests(unittest.TestCase):
    def test_causal_c2_gate_requires_current_frame_midpoint(self) -> None:
        rows = []
        for frame in range(31):
            for actor, peer in (("s1", "s2"), ("s2", "s1")):
                rows.append({
                    "video": "clip",
                    "actor_id": actor,
                    "truth": "suspicious_activity",
                    "actor_label": "suspicious_activity",
                    "interaction_peer_ids": f'["{peer}"]',
                    "source_frame_index": frame,
                    "timestamp_ms": frame * 33,
                    "warmup_ready": int(frame >= 29),
                    # Historical rolling max would retain this evidence at
                    # frame 30. Same-frame gate must reject it.
                    "near_midpoint_pre_cross__max": 1.0 if frame == 29 else 0.0,
                    "near_midpoint_pre_cross": 1.0 if frame == 29 else 0.0,
                })

        predictions = stage2.causal_specialist_replay(
            rows,
            [0.1] * 60 + [0.9] * 2,
            [0.0] * 62,
            c3_threshold=0.9,
        )

        self.assertEqual({row["predicted_class"] for row in predictions}, {"c5"})

    def test_solvepnp_head_pose_returns_orientation_for_valid_face(self) -> None:
        import cv2
        import numpy as np

        object_points = np.asarray([stage2.HEAD_PNP_MODEL[index] for index in stage2.HEAD_PNP_MODEL], dtype=np.float64)
        camera = np.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
        projected, _ = cv2.projectPoints(
            object_points, np.zeros((3, 1), dtype=np.float64),
            np.asarray([[0.0], [0.0], [500.0]], dtype=np.float64), camera,
            np.zeros((4, 1), dtype=np.float64),
        )
        points = {
            index: tuple(float(value) for value in point)
            for index, point in zip(stage2.HEAD_PNP_MODEL, projected.reshape(-1, 2))
        }
        pose = stage2._head_pnp_pose(points)
        self.assertEqual(pose["valid"], 1.0)
        self.assertLess(pose["reprojection_error"], stage2.HEAD_PNP_REPROJECTION_THRESHOLD)
        self.assertTrue(all(math.isfinite(pose[name]) for name in ("yaw", "pitch", "roll")))

    def test_solvepnp_rejects_missing_face_points(self) -> None:
        pose = stage2._head_pnp_pose({1: (0.5, 0.5)})
        self.assertEqual(pose["valid"], 0.0)

    def test_stage2_solvepnp_uses_pixel_intrinsics(self) -> None:
        import cv2
        import numpy as np

        object_points = np.asarray([stage2.HEAD_PNP_MODEL[index] for index in stage2.HEAD_PNP_MODEL], dtype=np.float64)
        camera = np.asarray([[1920.0, 0.0, 960.0], [0.0, 1920.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        projected, _ = cv2.projectPoints(
            object_points, np.zeros((3, 1), dtype=np.float64),
            np.asarray([[0.0], [0.0], [500.0]], dtype=np.float64), camera,
            np.zeros((4, 1), dtype=np.float64),
        )
        points = {
            index: tuple(float(value) for value in point)
            for index, point in zip(stage2.HEAD_PNP_MODEL, projected.reshape(-1, 2))
        }
        pose = stage2._head_pnp_pose(points)
        self.assertEqual(pose["valid"], 1.0)
        self.assertLess(pose["reprojection_error"], stage2.HEAD_PNP_REPROJECTION_THRESHOLD)

    @staticmethod
    def _hand_row(frame: int, valid: bool = True, wrist_y: float = 50.0) -> dict[str, str]:
        row = {
            "clip_id": "synthetic", "actor_id": "s1", "source_frame_index": str(frame),
            "actor_side": "-1", "pair_mid_x_0": "100", "pair_margin_10pct": "10",
            "interaction_peer_ids": json.dumps([]),
            "pose_11_valid": "1", "pose_11_frame_x": "80", "pose_11_frame_y": "40",
            "pose_12_valid": "1", "pose_12_frame_x": "120", "pose_12_frame_y": "40",
        }
        for side, x in (("left_hand", 70.0), ("right_hand", 130.0)):
            for index in range(21):
                row[f"{side}_{index}_valid"] = "1" if valid else "0"
                row[f"{side}_{index}_frame_x"] = str(x + (index % 3) * 0.5)
                row[f"{side}_{index}_frame_y"] = str(wrist_y + (index % 4) * 0.5)
            row[f"{side}_0_frame_x"] = str(x)
            row[f"{side}_0_frame_y"] = str(wrist_y)
        return row

    def test_hand_gap_is_unknown_and_does_not_become_zero_motion(self) -> None:
        rows = [self._hand_row(0), self._hand_row(1, valid=False), self._hand_row(2, wrist_y=40.0)]
        stage2.derive_hand_jitter_aware_cues(rows)
        self.assertEqual(rows[1]["c7_selected_hand_valid"], 0.0)
        self.assertEqual(rows[2]["c7_selected_hand_gap_frames"], 1.0)
        self.assertGreater(rows[2]["c7_selected_hand_raise_displacement"], 0.0)

    def test_q7_requires_same_hand_coherence(self) -> None:
        row = self._hand_row(0)
        row.update({
            "c7_selected_hand_valid": 1.0,
            "c7_selected_hand_finger_valid": 1.0,
            "c7_selected_hand_coherence": 0.0,
            "c7_selected_hand_own_side_distance": 20.0,
            "c7_selected_hand_near_midpoint": 0.0,
            "c7_selected_hand_shared_zone": 0.0,
            "c7_selected_hand_raise_displacement": 1.0,
            "c7_selected_hand_raise_speed": 1.0,
            "c7_selected_hand_jitter_speed": 1.0,
            "c7_selected_hand_shape_change": 0.0,
            "c7_selected_hand_gap_frames": 0.0,
        })
        thresholds = {"raise_displacement": .2, "raise_speed": .2, "jitter_speed": .1, "shape_change": .2, "coherence": .5}
        self.assertFalse(stage2.q7_frame(row, thresholds, hand_jitter_aware=True))
        row["c7_selected_hand_shape_change"] = 1.0
        row["c7_selected_hand_jitter_speed"] = 0.01
        self.assertTrue(stage2.q7_frame(row, thresholds, hand_jitter_aware=True))

    def test_specialist_evidence_adds_first_flag_without_changing_class(self) -> None:
        import numpy as np

        rows = [
            {
                "clip_id": "clip", "actor_id": "s1", "source_frame_index": str(frame),
                "timestamp_ms": str(frame * 33), "near_midpoint_pre_cross": "0",
                "interaction_peer_ids": "[]", "actor_label": "c3",
            }
            for frame in (1, 2, 3)
        ]
        predictions = [{"video": "clip", "actor_id": "s1", "predicted_class": "c3"}]
        stage2.attach_specialist_frame_evidence(
            predictions,
            rows,
            np.asarray([[0.1, 0.2, 0.7], [0.1, 0.3, 0.6], [0.1, 0.4, 0.5]], dtype=np.float32),
            np.asarray([0.2, 0.55, 0.85], dtype=np.float32),
        )

        self.assertEqual(predictions[0]["predicted_class"], "c3")
        self.assertEqual(predictions[0]["evidence_frame_index"], "3")
        self.assertEqual(predictions[0]["first_flag_frame_index"], "2")

    def test_causal_aggregate_prefix_never_uses_future_values(self) -> None:
        rows = []
        for frame, value in ((1, 1.0), (2, 9.0)):
            rows.append({
                "clip_id": "clip", "actor_id": "s1", "actor_truth": "c3",
                "actor_label": "c3", "interaction_peer_ids": "[]",
                "source_actor": 1, "manifest_class_code": "c3",
                "source_frame_index": str(frame), "timestamp_ms": str(frame * 33),
                "near_midpoint_pre_cross": "0", "causal_value": value,
            })
        output, names = stage2.causal_aggregate_rows(
            rows, ("causal_value",), warmup_frames=2
        )

        self.assertEqual(tuple(names), (
            "causal_value__mean", "causal_value__std", "causal_value__max",
            "causal_value__q95", "causal_value__min",
        ))
        self.assertEqual(output[0]["causal_value__max"], 1.0)
        self.assertEqual(output[0]["prefix_frames"], 1)
        self.assertEqual(output[0]["warmup_ready"], 0)
        self.assertEqual(output[1]["causal_value__max"], 9.0)
        self.assertEqual(output[1]["prefix_frames"], 2)

    def test_event_summary_preserves_onset_and_hold(self) -> None:
        import numpy as np

        values = np.zeros((1, 10, 2), dtype=np.float32)
        values[0, 2:5, 0] = 1.0
        values[0, 7:9, 0] = 1.0
        values[0, 5:, 1] = 0.4
        output = _event_summary_features(values, np.asarray(["mouth_open", "mouth_open_duration"]))
        # Base summary has mean/std/velocity/acceleration for two columns.
        event_values = output[0, 8:13]
        self.assertAlmostEqual(float(event_values[0]), 0.5)  # active fraction
        self.assertAlmostEqual(float(event_values[1]), 0.3)  # longest run / 10
        self.assertAlmostEqual(float(event_values[2]), 0.2)  # two onsets / 10
        self.assertAlmostEqual(float(event_values[3]), 1.0)  # peak
        self.assertAlmostEqual(float(event_values[4]), 0.0)  # event ended before window end

    def test_selective_event_summary_keeps_retract_raise_direction(self) -> None:
        import numpy as np

        values = np.zeros((2, 10, 2), dtype=np.float32)
        values[0, 2, 0] = 1.0  # retract
        values[1, 2, 1] = 1.0  # raise
        output = _selective_event_summary_features(
            values,
            np.asarray(["hand_rest_to_retract", "hand_rest_to_raise"]),
        )
        # The retract/raise interaction is the first five values after the
        # eight selected events (base summary is eight values).
        self.assertGreater(float(output[0, 48]), float(output[0, 49]))
        self.assertLess(float(output[1, 48]), float(output[1, 49]))

    def test_rest_to_raise_is_detected_and_valid(self) -> None:
        rows = [_row(frame) for frame in range(1, 22)]
        rows.append(_row(22, 0.20))
        derive_behavior_features(rows)
        raised = rows[-1]
        self.assertEqual(raised["hand_rest_to_raise"], 1.0)
        self.assertEqual(raised["hand_rest_to_retract"], 0.0)
        self.assertEqual(raised["hand_rest_to_raise"], 1.0)
        self.assertTrue(all(column in raised for column in BEHAVIOR_COLUMNS))

    def test_invalid_hand_does_not_create_raise_event(self) -> None:
        rows = [_row(frame) for frame in range(1, 22)]
        row = _row(22, 0.20)
        for side in ("left", "right"):
            for index in range(21):
                row.pop(f"{side}_hand_{index:03d}_x")
                row.pop(f"{side}_hand_{index:03d}_y")
        rows.append(row)
        derive_behavior_features(rows)
        self.assertEqual(rows[-1]["hand_rest_to_raise"], 0.0)
        self.assertEqual(rows[-1]["hand_rest_valid"], 0.0)

    def test_pose_fallback_is_separate_from_face_angles(self) -> None:
        rows = [_row(frame) for frame in range(1, 32)]
        for baseline_row in rows:
            baseline_row["pose_valid_ratio"] = 0.8
        shifted = _row(32)
        for index in (0, 7, 8, 11, 12):
            shifted[f"pose_{index:03d}_visibility"] = 0.8
            shifted[f"pose_{index:03d}_presence"] = 0.8
        shifted["pose_007_x"] = 0.65
        shifted["pose_008_x"] = 0.75
        for key in list(shifted):
            if key.startswith("face_"):
                shifted.pop(key)
        rows.append(shifted)

        derive_behavior_features(rows)
        result = rows[-1]
        self.assertEqual(result["yaw_valid"], 0.0)
        self.assertEqual(result["pitch_valid"], 0.0)
        self.assertEqual(result["pose_fallback_active"], 1.0)
        self.assertGreater(float(result["pose_fallback_dx"]), 0.08)
        self.assertEqual(result["pose_fallback_valid"], 1.0)

    def test_pose_fallback_quality_threshold_does_not_change_face_angles(self) -> None:
        row = _row(1)
        for index in (0, 7, 8, 11, 12):
            row[f"pose_{index:03d}_visibility"] = 0.29
            row[f"pose_{index:03d}_presence"] = 0.29
        derive_behavior_features([row])
        self.assertEqual(row["yaw_valid"], 1.0)
        self.assertEqual(row["pose_fallback_active"], 0.0)
        self.assertEqual(row["pose_fallback_valid"], 0.0)


if __name__ == "__main__":
    unittest.main()

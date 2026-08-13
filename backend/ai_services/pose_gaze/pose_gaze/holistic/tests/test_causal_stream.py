from __future__ import annotations

import unittest

from pose_gaze.holistic.feature_csv.behavior_subset_stage2 import (
    derive_c3_pose_contract,
)
from pose_gaze.holistic.feature_csv.causal_stream import (
    CausalActorWindow,
    CausalOrderError,
    CausalSpecialistState,
)


class CausalStreamTests(unittest.TestCase):
    @staticmethod
    def _pose_row(actor_id: str, frame: int, *, peer: str, nose_x: float) -> dict[str, object]:
        row: dict[str, object] = {
            "clip_id": "clip",
            "actor_id": actor_id,
            "source_frame_index": frame,
            "interaction_peer_ids": f'["{peer}"]',
            "bbox_x1": 100 if actor_id == "s1" else 500,
            "bbox_x2": 200 if actor_id == "s1" else 600,
        }
        points = {
            0: (nose_x, 100), 1: (nose_x - 5, 90), 4: (nose_x + 5, 90),
            11: (100, 120), 12: (140, 120), 23: (105, 180), 24: (135, 180),
        }
        if actor_id == "s2":
            points = {index: (x + 400, y) for index, (x, y) in points.items()}
        for index, (x, y) in points.items():
            row.update({
                f"pose_{index}_valid": "1",
                f"pose_{index}_frame_x": x,
                f"pose_{index}_frame_y": y,
            })
        return row

    def test_c3_pose_contract_uses_hip_torso_and_explicit_peer(self) -> None:
        rows = []
        for frame, nose_x in ((0, 120), (1, 120), (2, 120), (3, 140)):
            rows.extend((
                self._pose_row("s1", frame, peer="s2", nose_x=nose_x),
                self._pose_row("s2", frame, peer="s1", nose_x=520),
            ))
        derived = derive_c3_pose_contract(rows, baseline_frames=3)
        current = next(row for row in derived if row["actor_id"] == "s1" and row["source_frame_index"] == 3)
        self.assertEqual(current["c3_pose_required_valid"], 1.0)
        self.assertEqual(current["c3_pose_peer_valid"], 1.0)
        self.assertGreater(current["c3_pose_head_peer_delta"], 0.0)
        self.assertEqual(current["c3_pose_torso_valid"], 1.0)

    def test_c3_pose_contract_temporarily_allows_a_solo_live_actor(self) -> None:
        rows = []
        for frame, nose_x in ((0, 120), (1, 120), (2, 120), (3, 140)):
            row = self._pose_row("s1", frame, peer="unused", nose_x=nose_x)
            row["interaction_peer_ids"] = "[]"
            rows.append(row)

        derived = derive_c3_pose_contract(rows, baseline_frames=3)
        current = derived[-1]
        self.assertEqual(current["c3_pose_peer_valid"], 1.0)
        self.assertGreater(current["c3_pose_head_peer_delta"], 0.0)

    def test_rolling_window_drops_old_frames(self) -> None:
        window = CausalActorWindow("s1", ("score",), max_frames=2)
        first = window.update(frame_index=1, timestamp_ms=10, features={"score": 1})
        second = window.update(frame_index=2, timestamp_ms=20, features={"score": 2})
        third = window.update(frame_index=3, timestamp_ms=30, features={"score": 9})

        self.assertEqual(first.features["score__max"], 1.0)
        self.assertEqual(second.features["score__max"], 2.0)
        self.assertEqual(third.features["score__max"], 9.0)
        self.assertEqual(third.window_start_frame, 2)
        self.assertEqual(third.window_size, 2)

    def test_future_frame_cannot_be_inserted(self) -> None:
        window = CausalActorWindow("s1", ("score",))
        window.update(frame_index=2, timestamp_ms=20, features={"score": 2})

        with self.assertRaises(CausalOrderError):
            window.update(frame_index=1, timestamp_ms=10, features={"score": 1})

    def test_pair_c2_is_propagated_only_to_explicit_pair(self) -> None:
        state = CausalSpecialistState(("s1", "s2", "s3"), c3_threshold=0.7)
        decisions = state.update(
            frame_index=30,
            timestamp_ms=1000,
            scores_by_actor={
                "s1": {"c2": 0.8, "c3": 0.1},
                "s2": {"c2": 0.4, "c3": 0.1},
                "s3": {"c2": 0.9, "c3": 0.1},
            },
            explicit_pairs=(("s1", "s2"),),
            near_midpoint_by_actor={"s1": 1, "s2": 0, "s3": 1},
        )

        self.assertEqual(decisions["s1"].class_code, "c2")
        self.assertEqual(decisions["s2"].class_code, "c2")
        self.assertEqual(decisions["s3"].class_code, "c5")
        self.assertEqual(decisions["s1"].first_flag_frame_index, 30)
        self.assertEqual(decisions["s2"].source_actor_id, "s1")

    def test_c3_flag_is_first_and_evidence_is_strongest(self) -> None:
        state = CausalSpecialistState(("s1",), c3_threshold=0.5)
        state.update(
            frame_index=30,
            timestamp_ms=1000,
            scores_by_actor={"s1": {"c2": 0.0, "c3": 0.6}},
        )
        decisions = state.update(
            frame_index=31,
            timestamp_ms=1033,
            scores_by_actor={"s1": {"c2": 0.0, "c3": 0.9}},
        )

        self.assertEqual(decisions["s1"].class_code, "c3")
        self.assertEqual(decisions["s1"].first_flag_frame_index, 30)
        self.assertEqual(decisions["s1"].evidence_frame_index, 31)
        self.assertEqual(decisions["s1"].evidence_score, 0.9)

    def test_c3_gate_blocks_head_down_but_suspicious_can_flag(self) -> None:
        c3_gate = lambda values: values["head_down"] <= 0.05
        suspicious_gate = lambda values: values["own_side"] >= 1.0
        state = CausalSpecialistState(
            ("s1",), c3_threshold=0.5, suspicious_threshold=0.5,
            c3_gate=c3_gate, suspicious_gate=suspicious_gate,
        )
        decision = state.update(
            frame_index=30, timestamp_ms=1000,
            scores_by_actor={"s1": {
                "c2": 0.0, "c3": 0.99, "suspicious_activity": 0.8,
                "head_down": 0.2, "own_side": 1.0,
            }},
        )["s1"]
        self.assertEqual(decision.class_code, "suspicious_activity")

    def test_midpoint_c2_stays_out_of_suspicious_gate(self) -> None:
        suspicious_gate = lambda values: values["own_side_outside_midpoint"] >= 1.0
        state = CausalSpecialistState(
            ("s1",), c3_threshold=0.5, suspicious_threshold=0.5,
            suspicious_gate=suspicious_gate,
        )
        decision = state.update(
            frame_index=30, timestamp_ms=1000,
            scores_by_actor={"s1": {
                "c2": 0.0, "c3": 0.0, "suspicious_activity": 0.95,
                "own_side_outside_midpoint": 0.0,
            }},
        )["s1"]
        self.assertEqual(decision.class_code, "c5")

    def test_higher_later_suspicious_evidence_replaces_c3(self) -> None:
        state = CausalSpecialistState(
            ("s1",), c3_threshold=0.5, suspicious_threshold=0.5,
        )
        state.update(
            frame_index=30, timestamp_ms=1000,
            scores_by_actor={"s1": {"c2": 0.0, "c3": 0.6, "suspicious_activity": 0.0}},
        )
        decision = state.update(
            frame_index=31, timestamp_ms=1033,
            scores_by_actor={"s1": {"c2": 0.0, "c3": 0.0, "suspicious_activity": 0.8}},
        )["s1"]
        self.assertEqual(decision.class_code, "suspicious_activity")
        self.assertEqual(decision.evidence_frame_index, 31)


if __name__ == "__main__":
    unittest.main()

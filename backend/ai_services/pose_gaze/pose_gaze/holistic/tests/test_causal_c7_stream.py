from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from backend.ai_services.pose_gaze.pose_gaze.holistic.feature_csv.benchmark_c7_causal import C7Stream
from backend.ai_services.pose_gaze.pose_gaze.holistic.test_media.live_actor import CombinedCausalActorClassifier
from backend.ai_services.pose_gaze.pose_gaze.holistic.test_media.test_media import parse_args as parse_media_args
from backend.ai_services.pose_gaze.pose_gaze.holistic.test_webcam.test_webcam import (
    SUPPORTED_ACTIONS,
    configure_action_models,
)


def row(frame: int, *, hand: bool = True) -> dict[str, str]:
    output = {
        "timestamp_ms": str(frame * 33), "pair_mid_x_0": "250",
        "pair_margin_10pct": "20", "actor_side": "-1",
    }
    for index, point in {11: (100, 100), 12: (200, 100)}.items():
        output[f"pose_{index}_valid"] = "1"
        output[f"pose_{index}_frame_x"] = str(point[0])
        output[f"pose_{index}_frame_y"] = str(point[1])
    if hand:
        for index, point in {0: (120, 180-frame), 4: (110, 140-frame), 5: (120, 165-frame), 8: (115, 135-frame), 9: (125, 160-frame), 12: (120, 132-frame), 16: (125, 135-frame), 20: (130, 140-frame)}.items():
            output[f"left_hand_{index}_valid"] = "1"
            output[f"left_hand_{index}_frame_x"] = str(point[0])
            output[f"left_hand_{index}_frame_y"] = str(point[1])
    return output


class CausalC7StreamTest(unittest.TestCase):
    @staticmethod
    def _args(actions: str) -> Namespace:
        return Namespace(
            actions=actions, causal_model_dir=None, c1_model_dir=None,
            c4_model_dir=None, c7_model_dir=None, live_pair=[],
        )

    def test_webcam_committed_default_profile_enables_c2_c3(self) -> None:
        args = self._args("c2,c3")
        enabled = configure_action_models(args)
        self.assertEqual(enabled, ("c2", "c3"))
        self.assertEqual(args.live_pair, ["student_01:student_02"])
        self.assertTrue(Path(args.causal_model_dir).is_dir())
        self.assertIsNone(args.c1_model_dir)
        self.assertIsNone(args.c4_model_dir)
        self.assertIsNone(args.c7_model_dir)

    def test_webcam_runtime_exposes_only_current_causal_classes(self) -> None:
        self.assertEqual(
            SUPPORTED_ACTIONS,
            ("c2", "c3", "c5", "suspicious_activity"),
        )

    def test_webcam_rejects_removed_specialist_classes(self) -> None:
        for action in ("c1", "c4", "c7"):
            with self.subTest(action=action):
                with self.assertRaisesRegex(ValueError, "Unsupported --actions"):
                    configure_action_models(self._args(action))

    def test_media_rejects_removed_object_subtype_option(self) -> None:
        with patch(
            "sys.argv",
            ["test_media", "input.mp4", "--object-model", "best.pt"],
        ):
            with self.assertRaises(SystemExit) as raised:
                parse_media_args()
        self.assertEqual(raised.exception.code, 2)

    def test_combined_classifier_keeps_only_adapter_methods(self) -> None:
        self.assertEqual(
            {name for name in CombinedCausalActorClassifier.__dict__ if name in {"update", "update_tracks", "final_decisions", "reset"}},
            {"update", "update_tracks", "final_decisions", "reset"},
        )

    def test_combined_classifier_resets_every_specialist(self) -> None:
        class Specialist:
            def __init__(self) -> None:
                self.reset_count = 0

            def reset(self) -> None:
                self.reset_count += 1

        first, second = Specialist(), Specialist()
        CombinedCausalActorClassifier((first, second)).reset()
        self.assertEqual((first.reset_count, second.reset_count), (1, 1))

    def test_prefix_is_future_invariant(self) -> None:
        rows = [row(0), row(1), row(2), row(3)]
        prefix_stream = C7Stream(baseline_frames=2)
        prefix = [prefix_stream.update(item) for item in rows[:3]]
        full_stream = C7Stream(baseline_frames=2)
        full = [full_stream.update(item) for item in rows]
        self.assertEqual(prefix, full[:3])

    def test_missing_hand_is_uncertainty_not_zero_motion_evidence(self) -> None:
        stream = C7Stream(baseline_frames=2)
        stream.update(row(0))
        observed = stream.update(row(1))
        missing = stream.update(row(2, hand=False))
        self.assertEqual(missing["selected_hand_valid__last"], 0.0)
        self.assertEqual(missing["selected_hand_gap_frames__last"], 1.0)
        self.assertEqual(
            missing["selected_hand_raise__mean"],
            observed["selected_hand_raise__mean"],
        )


if __name__ == "__main__":
    unittest.main()

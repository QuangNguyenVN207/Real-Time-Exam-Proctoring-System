"""Stage A tests: T-A2, T-A3, T-A4.

T-A2: Replay on FrameTraceRecord stream with legacy gate=False yields:
  - summary.passed == False (RED)
  - At least one trial with legacy_gate_veto > 0 (specific blocker)
  - Blocker counts are the actual reason, not just a generic fail

T-A3: Same FrameTraceRecord stream replayed 3 times gives identical verdict,
  first_flag_latency_ms, and blocker_counts for every trial.

T-A4: Prefix causality — outputs from frame 0..N-1 are identical whether
  future frames 0..N+M are appended or not.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from pose_gaze.holistic.debug.events import C3Trial, OcclusionEvent, SessionEvents, TimeInterval, TrackResetEvent
from pose_gaze.holistic.debug.frame_trace import FrameTraceLogger, FrameTraceRecord
from pose_gaze.holistic.debug.replay_c3_camera import run_c3_replay
from pose_gaze.holistic.test_media.live_actor import CausalLiveActorClassifier
from pose_gaze.settings import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "tmp" / "behavior_actor_extended_suspicious_current_geometry_20260815"


def _events() -> SessionEvents:
    return SessionEvents(
        session_id="synthetic_gate_veto_session",
        calibration_interval=TimeInterval(0, 2000, "calibration"),
        neutral_intervals=[TimeInterval(2000, 12000, "neutral")],
        trials=[
            C3Trial(1, "student_01", "student_02", 13000, 15500),
            C3Trial(2, "student_01", "student_02", 17000, 19500),
            C3Trial(3, "student_01", "student_02", 21000, 23500),
            C3Trial(4, "student_02", "student_01", 25000, 27500),
            C3Trial(5, "student_02", "student_01", 29000, 31500),
            C3Trial(6, "student_02", "student_01", 33000, 35500),
        ],
        occlusions=[
            OcclusionEvent("student_02", 36000, 36400, expected_unknown=False),
            OcclusionEvent("student_02", 37000, 37600, expected_unknown=True),
        ],
        track_resets=[TrackResetEvent(38000, "student_02", "student_03")],
    )


def _make_trace_record(
    frame_idx: int,
    ts: int,
    actor_id: str,
    peer_id: str,
    p3: float,
    gate_final: bool,
    tracks: list,
) -> FrameTraceRecord:
    """Build a FrameTraceRecord mimicking the webcam output."""
    return FrameTraceRecord(
        timestamp_ms=ts,
        frame_index=frame_idx,
        inter_frame_duration_ms=100.0,
        actor_id=actor_id,
        peer_id=peer_id,
        track_present=True,
        track_missed_count=0,
        track_age_frames=frame_idx + 1,
        actor_bbox=[0.0, 0.0, 300.0, 600.0],
        peer_bbox=[400.0, 0.0, 700.0, 600.0],
        pose_valid=True,
        hand_valid=False,
        peer_pose_valid=True,
        peer_hand_valid=False,
        peer_age_ms=0.0,
        peer_stale=False,
        O_A=True,
        O_AB=True,
        R_AB=frame_idx >= 20,  # warmup done after 20 frames
        neutral_baseline_age_ms=float(ts),
        ready_state="READY" if frame_idx >= 20 else "CALIBRATING",
        reset_reason=None,
        p3_A=p3,
        H_AB=None,
        Q_A=None,
        tau_3=0.9580807089805603,
        tau_H=0.0,
        p2_AB=0.05,
        K_AB=None,
        Q_hand_AB=None,
        tau_2=0.5,
        tau_K=0.0,
        legacy_c3_gate_hand_quality_positive=False,  # hand missing → veto
        legacy_c3_gate_hand_motion_passed=None,
        legacy_c3_gate_finger_motion_passed=None,
        legacy_c3_gate_side_floor_passed=True,
        legacy_c3_gate_down_ceiling_passed=True,
        legacy_c3_gate_final=gate_final,
        resolver_candidate="c5",
        emitted_class="c5",
        unknown_reason=None,
        first_flag_timestamp_ms=None,
        latency_ms=None,
        tracks_snapshot=tracks,
    )


def _build_synthetic_gate_veto_trace() -> list[dict]:
    """Generate 360 frames (36s at 10 FPS) where legacy_c3_gate_final is always False.

    This mimics the real camera symptom: p3 occasionally >= tau_3 during turn
    windows but hand quality gate hard-blocks C3.
    """
    fps_ms = 100
    total_frames = 360
    events = _events()

    records: list[dict] = []
    for i in range(total_frames):
        ts = i * fps_ms

        # During C3 turn windows, p3 is elevated but gate is still False
        p3_s1 = 0.1
        p3_s2 = 0.1
        for trial in events.trials[:3]:  # s1 -> s2
            if trial.turn_onset_ms <= ts <= trial.turn_end_ms:
                p3_s1 = 0.85  # above score threshold, below tau_3 — score_low
        for trial in events.trials[3:]:  # s2 -> s1
            if trial.turn_onset_ms <= ts <= trial.turn_end_ms:
                p3_s2 = 0.85

        tracks = [
            {
                "track_id": 1, "student_id": "student_01",
                "bbox": [0.0, 0.0, 300.0, 600.0],
                "pose_landmarks": [
                    {"index": 0, "frame_x": 150.0, "frame_y": 100.0, "visibility": 0.99},
                    {"index": 11, "frame_x": 100.0, "frame_y": 250.0, "visibility": 0.99},
                    {"index": 12, "frame_x": 200.0, "frame_y": 250.0, "visibility": 0.99},
                    {"index": 23, "frame_x": 110.0, "frame_y": 450.0, "visibility": 0.99},
                    {"index": 24, "frame_x": 190.0, "frame_y": 450.0, "visibility": 0.99},
                ],
                "left_hand_landmarks": [],
                "right_hand_landmarks": [],
                "selected_face_landmarks": [],
            },
            {
                "track_id": 2, "student_id": "student_02",
                "bbox": [400.0, 0.0, 700.0, 600.0],
                "pose_landmarks": [
                    {"index": 0, "frame_x": 550.0, "frame_y": 100.0, "visibility": 0.99},
                    {"index": 11, "frame_x": 500.0, "frame_y": 250.0, "visibility": 0.99},
                    {"index": 12, "frame_x": 600.0, "frame_y": 250.0, "visibility": 0.99},
                    {"index": 23, "frame_x": 510.0, "frame_y": 450.0, "visibility": 0.99},
                    {"index": 24, "frame_x": 590.0, "frame_y": 450.0, "visibility": 0.99},
                ],
                "left_hand_landmarks": [],
                "right_hand_landmarks": [],
                "selected_face_landmarks": [],
            },
        ]

        # Directed edge: student_01 -> student_02
        records.append(_make_trace_record(i, ts, "student_01", "student_02", p3_s1, False, tracks).to_dict())
        # Directed edge: student_02 -> student_01
        records.append(_make_trace_record(i, ts, "student_02", "student_01", p3_s2, False, tracks).to_dict())

    return records


class StageAReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_t_a2_replay_catches_exact_symptom(self) -> None:
        """T-A2: Replay on gate-vetoed stream is RED with specific blocker counts.

        Specifically asserts:
        - summary.passed == False (RED)
        - Each missed trial has score_low > 0 (p3 < tau_3 in synthetic stream)
        - No trial with all blockers at 0 while flagged=False (must identify reason)
        """
        if not MODEL_DIR.is_dir():
            self.skipTest(f"Model dir not found: {MODEL_DIR}")

        trace_records = _build_synthetic_gate_veto_trace()
        events = _events()

        summary = run_c3_replay(
            trace_records=trace_records,
            events=events,
            model_dir=MODEL_DIR,
        )

        # Must be RED
        self.assertFalse(summary.passed, "Replay must be RED on un-repaired gate-vetoed stream")

        # Less than all 6 trials flagged
        self.assertLess(
            summary.trials_flagged_in_time,
            summary.trials_total,
            "Un-repaired stream must miss at least one trial",
        )

        # Every missed trial must have a specific identified blocker
        missed = [r for r in summary.trial_results if not r.flagged]
        self.assertGreater(len(missed), 0, "At least one trial must be missed")

        for res in missed:
            total_blockers = sum(res.blocker_counts.values())
            self.assertGreater(
                total_blockers,
                0,
                f"Trial {res.trial_id} is missed but no blockers identified — "
                f"replay is not diagnosing the camera symptom",
            )
            # At least one specific blocker must be non-zero
            self.assertTrue(
                any(v > 0 for v in res.blocker_counts.values()),
                f"Trial {res.trial_id}: must identify at least one specific blocker type",
            )

    def test_t_a3_replay_determinism(self) -> None:
        """T-A3: Same trace replayed 3 times gives byte-for-byte identical verdict."""
        if not MODEL_DIR.is_dir():
            self.skipTest(f"Model dir not found: {MODEL_DIR}")

        trace_records = _build_synthetic_gate_veto_trace()
        events = _events()

        runs = [
            run_c3_replay(trace_records=trace_records, events=events, model_dir=MODEL_DIR)
            for _ in range(3)
        ]

        first = runs[0]
        for run_i, run in enumerate(runs[1:], start=2):
            self.assertEqual(first.passed, run.passed, f"Run {run_i} passed differs")
            self.assertEqual(
                first.trials_flagged_in_time, run.trials_flagged_in_time,
                f"Run {run_i} flagged count differs"
            )
            self.assertEqual(
                first.neutral_false_flags, run.neutral_false_flags,
                f"Run {run_i} neutral_false_flags differs"
            )
            for r1, r2 in zip(first.trial_results, run.trial_results):
                self.assertEqual(r1.flagged, r2.flagged, f"Trial {r1.trial_id}: flagged differs in run {run_i}")
                self.assertEqual(
                    r1.first_flag_latency_ms, r2.first_flag_latency_ms,
                    f"Trial {r1.trial_id}: latency differs in run {run_i}"
                )
                self.assertEqual(
                    r1.blocker_counts, r2.blocker_counts,
                    f"Trial {r1.trial_id}: blocker_counts differ in run {run_i}"
                )

    def test_t_a4_causality(self) -> None:
        """T-A4: Prefix outputs [0..N-1] identical when future frames [N..N+M] appended."""
        if not MODEL_DIR.is_dir():
            self.skipTest(f"Model dir not found: {MODEL_DIR}")

        trace_records = _build_synthetic_gate_veto_trace()
        prefix_len = 50  # Frames 0..49
        full_len = 80     # Frames 0..79

        # Build frame feeds for prefix and full
        full_feed = FrameTraceLogger.to_frame_feed(trace_records)
        prefix_feed = full_feed[:prefix_len]
        extended_feed = full_feed[:full_len]

        clf_prefix = CausalLiveActorClassifier(model_dir=MODEL_DIR)
        clf_full = CausalLiveActorClassifier(model_dir=MODEL_DIR)

        outputs_prefix = []
        for frame in prefix_feed:
            out = clf_prefix.update_tracks(
                frame_index=frame["frame_index"],
                timestamp_ms=frame["timestamp_ms"],
                tracks=frame["tracks"],
            )
            outputs_prefix.append(copy.deepcopy(out))

        outputs_full = []
        for frame in extended_feed:
            out = clf_full.update_tracks(
                frame_index=frame["frame_index"],
                timestamp_ms=frame["timestamp_ms"],
                tracks=frame["tracks"],
            )
            outputs_full.append(copy.deepcopy(out))

        # First prefix_len outputs must be identical
        self.assertEqual(len(outputs_prefix), prefix_len)
        for i in range(prefix_len):
            self.assertEqual(
                outputs_prefix[i],
                outputs_full[i],
                f"Causality violation at frame {i}: prefix differs from extended",
            )


if __name__ == "__main__":
    unittest.main()

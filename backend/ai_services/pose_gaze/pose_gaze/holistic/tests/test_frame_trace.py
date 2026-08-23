"""Unit tests for Stage A2 / T-A1: Trace Completeness."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pose_gaze.holistic.debug.frame_trace import FrameTraceLogger, FrameTraceRecord


def _make_record(
    frame_index: int,
    timestamp_ms: int,
    actor_id: str,
    peer_id: str = "student_02",
    tracks_snapshot: list | None = None,
) -> FrameTraceRecord:
    return FrameTraceRecord(
        timestamp_ms=timestamp_ms,
        frame_index=frame_index,
        inter_frame_duration_ms=100.0,
        actor_id=actor_id,
        peer_id=peer_id,
        track_present=True,
        track_missed_count=0,
        track_age_frames=frame_index + 1,
        actor_bbox=[0.0, 0.0, 200.0, 400.0],
        peer_bbox=[300.0, 0.0, 500.0, 400.0],
        pose_valid=True,
        hand_valid=False,
        peer_pose_valid=True,
        peer_hand_valid=False,
        peer_age_ms=0.0,
        peer_stale=False,
        O_A=True,
        O_AB=True,
        R_AB=True,
        neutral_baseline_age_ms=float(frame_index * 100),
        ready_state="READY",
        reset_reason=None,
        p3_A=0.85,
        H_AB=None,
        Q_A=None,
        tau_3=0.9580807089805603,
        tau_H=0.0,
        p2_AB=0.1,
        K_AB=None,
        Q_hand_AB=None,
        tau_2=0.5,
        tau_K=0.0,
        legacy_c3_gate_hand_quality_positive=False,   # hand missing → veto
        legacy_c3_gate_hand_motion_passed=None,
        legacy_c3_gate_finger_motion_passed=None,
        legacy_c3_gate_side_floor_passed=True,
        legacy_c3_gate_down_ceiling_passed=True,
        legacy_c3_gate_final=False,                   # gate vetoes this frame
        resolver_candidate="c5",
        emitted_class="c5",
        unknown_reason=None,
        first_flag_timestamp_ms=None,
        latency_ms=None,
        tracks_snapshot=tracks_snapshot or [
            {"track_id": int(actor_id[-2:]), "student_id": actor_id},
            {"track_id": int(peer_id[-2:]), "student_id": peer_id},
        ],
    )


EXPECTED_FIELDS = {
    "timestamp_ms", "frame_index", "inter_frame_duration_ms",
    "actor_id", "peer_id", "track_present", "track_missed_count",
    "track_age_frames", "actor_bbox", "peer_bbox", "pose_valid",
    "hand_valid", "peer_pose_valid", "peer_hand_valid", "peer_age_ms",
    "peer_stale", "O_A", "O_AB", "R_AB", "neutral_baseline_age_ms",
    "ready_state", "reset_reason", "p3_A", "H_AB", "Q_A", "tau_3",
    "tau_H", "p2_AB", "K_AB", "Q_hand_AB", "tau_2", "tau_K",
    "legacy_c3_gate_hand_quality_positive", "legacy_c3_gate_hand_motion_passed",
    "legacy_c3_gate_finger_motion_passed", "legacy_c3_gate_side_floor_passed",
    "legacy_c3_gate_down_ceiling_passed", "legacy_c3_gate_final",
    "resolver_candidate", "emitted_class", "unknown_reason",
    "first_flag_timestamp_ms", "latency_ms",
    "tracks_snapshot",  # Required for replay
}


class FrameTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_t_a1_one_record_per_edge_all_fields_present_timestamps_monotonic(self) -> None:
        """T-A1: 20 frames × 2 directed edges = 40 records, all A2 fields present, monotonic."""
        trace_file = self.tmp_path / "frames.jsonl"
        logger = FrameTraceLogger(trace_file)

        for frame_idx in range(20):
            ts = 1000 + frame_idx * 100
            logger.log(_make_record(frame_idx, ts, "student_01", "student_02"))
            logger.log(_make_record(frame_idx, ts, "student_02", "student_01"))

        logger.close()

        loaded = FrameTraceLogger.load_jsonl(trace_file)
        self.assertEqual(len(loaded), 40, "Exactly one record per directed edge per frame")

        prev_ts = -1
        for entry in loaded:
            # All fields present
            missing = EXPECTED_FIELDS - set(entry.keys())
            self.assertEqual(missing, set(), f"Missing fields: {missing}")
            # Monotonic
            self.assertGreaterEqual(entry["timestamp_ms"], prev_ts)
            prev_ts = entry["timestamp_ms"]
            # tracks_snapshot is a list (non-empty since each record carries it)
            self.assertIsInstance(entry["tracks_snapshot"], list)
            self.assertGreater(len(entry["tracks_snapshot"]), 0)

    def test_gate_terms_are_raw_scalars_not_booleans_only(self) -> None:
        """Gate terms must record raw bool/None, not be absent or always True."""
        logger = FrameTraceLogger()
        rec = _make_record(0, 0, "student_01")
        logger.log(rec)
        loaded = logger.records[0].to_dict()
        self.assertFalse(loaded["legacy_c3_gate_hand_quality_positive"])
        self.assertIsNone(loaded["legacy_c3_gate_hand_motion_passed"])
        self.assertFalse(loaded["legacy_c3_gate_final"])

    def test_non_monotonic_timestamp_rejected(self) -> None:
        logger = FrameTraceLogger()
        logger.log(_make_record(0, 200, "student_01"))
        with self.assertRaisesRegex(ValueError, "Non-monotonic timestamp"):
            logger.log(_make_record(1, 100, "student_01"))

    def test_to_frame_feed_deduplicates_and_preserves_tracks(self) -> None:
        """to_frame_feed must produce one entry per frame_index with tracks_snapshot."""
        tracks = [{"track_id": 1, "student_id": "student_01"}]
        records = [
            _make_record(0, 0, "student_01", tracks_snapshot=tracks).to_dict(),
            _make_record(0, 0, "student_02", tracks_snapshot=tracks).to_dict(),  # same frame
            _make_record(1, 100, "student_01", tracks_snapshot=tracks).to_dict(),
        ]
        feed = FrameTraceLogger.to_frame_feed(records)
        self.assertEqual(len(feed), 2)
        self.assertEqual(feed[0]["frame_index"], 0)
        self.assertEqual(feed[1]["frame_index"], 1)
        self.assertIsInstance(feed[0]["tracks"], list)

    def test_context_manager_closes_file(self) -> None:
        trace_file = self.tmp_path / "ctx.jsonl"
        with FrameTraceLogger(trace_file) as logger:
            logger.log(_make_record(0, 0, "student_01"))
        self.assertTrue(trace_file.exists())
        loaded = FrameTraceLogger.load_jsonl(trace_file)
        self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Stage A3: events protocol validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pose_gaze.holistic.debug.events import (
    C3Trial,
    OcclusionEvent,
    SessionEvents,
    TimeInterval,
    TrackResetEvent,
)


def _valid_events() -> SessionEvents:
    return SessionEvents(
        session_id="capture_01",
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
            OcclusionEvent("student_02", 37000, 37400, expected_unknown=False),  # 400 ms
            OcclusionEvent("student_02", 39000, 39600, expected_unknown=True),   # 600 ms
        ],
        track_resets=[
            TrackResetEvent(41000, "student_02", "student_03"),
        ],
    )


class EventsProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_protocol_passes_and_round_trips(self) -> None:
        ev = _valid_events()
        ev.validate_protocol()
        saved = ev.save(self.tmp_path / "events.json")
        loaded = SessionEvents.load(saved)
        self.assertEqual(len(loaded.trials), 6)
        self.assertEqual(len(loaded.occlusions), 2)
        self.assertEqual(len(loaded.track_resets), 1)

    def test_short_calibration_rejected(self) -> None:
        ev = _valid_events()
        ev.calibration_interval = TimeInterval(0, 1000, "cal")
        with self.assertRaisesRegex(ValueError, "Calibration duration"):
            ev.validate_protocol()

    def test_short_neutral_rejected(self) -> None:
        ev = _valid_events()
        ev.neutral_intervals = [TimeInterval(2000, 5000, "neutral")]
        with self.assertRaisesRegex(ValueError, "Total neutral"):
            ev.validate_protocol()

    def test_too_few_trials_rejected(self) -> None:
        ev = _valid_events()
        ev.trials = ev.trials[:4]
        with self.assertRaisesRegex(ValueError, "at least 6 trials"):
            ev.validate_protocol()

    def test_direction_imbalance_rejected(self) -> None:
        ev = _valid_events()
        # All 6 in same direction: student_01 -> student_02
        ev.trials = [
            C3Trial(i + 1, "student_01", "student_02", 13000 + i * 4000, 15500 + i * 4000)
            for i in range(6)
        ]
        with self.assertRaisesRegex(ValueError, "source=student_02"):
            ev.validate_protocol()

    def test_missing_short_occlusion_rejected(self) -> None:
        ev = _valid_events()
        # Only long occlusion
        ev.occlusions = [OcclusionEvent("student_02", 39000, 39600, expected_unknown=True)]
        with self.assertRaisesRegex(ValueError, "occlusion < 500 ms"):
            ev.validate_protocol()

    def test_missing_long_occlusion_rejected(self) -> None:
        ev = _valid_events()
        ev.occlusions = [OcclusionEvent("student_02", 37000, 37400, expected_unknown=False)]
        with self.assertRaisesRegex(ValueError, "occlusion >= 500 ms"):
            ev.validate_protocol()

    def test_missing_track_reset_rejected(self) -> None:
        ev = _valid_events()
        ev.track_resets = []
        with self.assertRaisesRegex(ValueError, "TrackResetEvent"):
            ev.validate_protocol()

    def test_overlapping_trials_rejected(self) -> None:
        ev = _valid_events()
        # Trial 1 and 2 overlap
        ev.trials[1] = C3Trial(2, "student_01", "student_02", 14000, 16500)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            ev.validate_protocol()

    def test_duplicate_trial_ids_rejected(self) -> None:
        ev = _valid_events()
        ev.trials[2] = C3Trial(1, "student_01", "student_02", 22000, 24500)  # id=1 again
        with self.assertRaisesRegex(ValueError, "Duplicate trial IDs"):
            ev.validate_protocol()

    def test_short_trial_rejected(self) -> None:
        ev = _valid_events()
        ev.trials[0] = C3Trial(1, "student_01", "student_02", 13000, 14000)  # 1 s
        with self.assertRaisesRegex(ValueError, "duration < 2000"):
            ev.validate_protocol()

    def test_inconsistent_occlusion_expected_unknown_rejected(self) -> None:
        # expected_unknown=True but duration < 500ms
        with self.assertRaisesRegex(ValueError, "expected_unknown=True but duration"):
            OcclusionEvent("student_02", 37000, 37400, expected_unknown=True).validate()


if __name__ == "__main__":
    unittest.main()

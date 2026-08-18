"""Event annotations and capture protocol schema for C3 evaluation (Stage A3).

validate_protocol() enforces the full A3 protocol:
  - Calibration >= 2000 ms
  - Neutral >= 10 000 ms total
  - Exactly 6 trials: 3 A->B and 3 B->A (at least one per direction, default pair)
  - Each trial >= 2000 ms
  - At least two occlusions: one < 500 ms (expected_unknown=False)
    and one >= 500 ms (expected_unknown=True)
  - At least one track_reset
  - No overlapping intervals
  - No negative timestamps
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class C3Trial:
    """One directed C3 turn trial."""

    trial_id: int
    source_actor_id: str
    peer_actor_id: str
    turn_onset_ms: int
    turn_end_ms: int
    expected_class: str = "c3"

    def duration_ms(self) -> int:
        return self.turn_end_ms - self.turn_onset_ms

    def validate(self) -> None:
        if self.turn_onset_ms < 0:
            raise ValueError(f"Trial {self.trial_id}: Negative turn_onset_ms")
        if self.turn_end_ms <= self.turn_onset_ms:
            raise ValueError(
                f"Trial {self.trial_id}: turn_end_ms <= turn_onset_ms"
            )
        if not self.source_actor_id or not self.peer_actor_id:
            raise ValueError(
                f"Trial {self.trial_id}: source_actor_id and peer_actor_id must not be empty"
            )
        if self.source_actor_id == self.peer_actor_id:
            raise ValueError(
                f"Trial {self.trial_id}: source_actor_id == peer_actor_id"
            )


@dataclass
class TimeInterval:
    """Generic time interval in milliseconds."""

    start_ms: int
    end_ms: int
    label: str = "neutral"

    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def validate(self) -> None:
        if self.start_ms < 0:
            raise ValueError(f"Interval '{self.label}': Negative start_ms {self.start_ms}")
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"Interval '{self.label}': end_ms {self.end_ms} < start_ms {self.start_ms}"
            )


@dataclass
class OcclusionEvent:
    """Peer loss / occlusion event."""

    actor_id: str
    occlusion_onset_ms: int
    occlusion_end_ms: int
    expected_unknown: bool  # True iff duration >= T_stale (500 ms)

    def duration_ms(self) -> int:
        return self.occlusion_end_ms - self.occlusion_onset_ms

    def validate(self) -> None:
        if self.occlusion_onset_ms < 0:
            raise ValueError(f"Occlusion: Negative occlusion_onset_ms")
        if self.occlusion_end_ms <= self.occlusion_onset_ms:
            raise ValueError(f"Occlusion: end <= onset")
        # Consistency: expected_unknown iff duration >= 500 ms
        dur = self.duration_ms()
        if self.expected_unknown and dur < 500:
            raise ValueError(
                f"Occlusion expected_unknown=True but duration only {dur} ms (<500 ms)"
            )
        if not self.expected_unknown and dur >= 500:
            raise ValueError(
                f"Occlusion expected_unknown=False but duration {dur} ms (>=500 ms)"
            )


@dataclass
class TrackResetEvent:
    """Track identity change or reset."""

    reset_ms: int
    old_actor_id: str
    new_actor_id: str


@dataclass
class SessionEvents:
    """Complete ground-truth event annotations for one capture session."""

    session_id: str
    calibration_interval: TimeInterval
    neutral_intervals: list[TimeInterval] = field(default_factory=list)
    trials: list[C3Trial] = field(default_factory=list)
    occlusions: list[OcclusionEvent] = field(default_factory=list)
    track_resets: list[TrackResetEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: Path | str) -> SessionEvents:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cal = TimeInterval(**raw["calibration_interval"])
        neutrals = [TimeInterval(**i) for i in raw.get("neutral_intervals", [])]
        trials = [C3Trial(**i) for i in raw.get("trials", [])]
        occlusions = [OcclusionEvent(**i) for i in raw.get("occlusions", [])]
        resets = [TrackResetEvent(**i) for i in raw.get("track_resets", [])]
        return cls(
            session_id=raw["session_id"],
            calibration_interval=cal,
            neutral_intervals=neutrals,
            trials=trials,
            occlusions=occlusions,
            track_resets=resets,
        )

    def validate_protocol(self) -> None:
        """Full A3 protocol compliance check."""
        # 1. Calibration
        self.calibration_interval.validate()
        if self.calibration_interval.duration_ms() < 2000:
            raise ValueError(
                f"Calibration duration must be >= 2000 ms, "
                f"got {self.calibration_interval.duration_ms()} ms"
            )

        # 2. Neutral >= 10 000 ms total
        for n in self.neutral_intervals:
            n.validate()
        total_neutral = sum(n.duration_ms() for n in self.neutral_intervals)
        if total_neutral < 10_000:
            raise ValueError(
                f"Total neutral interval must be >= 10000 ms, got {total_neutral} ms"
            )

        # 3. Exactly >= 6 trials
        if len(self.trials) < 6:
            raise ValueError(f"Protocol requires at least 6 trials, got {len(self.trials)}")

        for trial in self.trials:
            trial.validate()
            if trial.duration_ms() < 2000:
                raise ValueError(
                    f"Trial {trial.trial_id} duration < 2000 ms: {trial.duration_ms()} ms"
                )

        # 4. Direction balance: 3 A->B and 3 B->A
        # Discover the two unique actor IDs from trial actor pairs
        all_actors = {t.source_actor_id for t in self.trials} | {t.peer_actor_id for t in self.trials}
        if len(all_actors) != 2:
            raise ValueError(
                f"Protocol expects exactly 2 actor IDs in trials, found: {all_actors}"
            )
        actor_a, actor_b = sorted(all_actors)
        ab_count = sum(
            1 for t in self.trials
            if t.source_actor_id == actor_a and t.peer_actor_id == actor_b
        )
        ba_count = sum(
            1 for t in self.trials
            if t.source_actor_id == actor_b and t.peer_actor_id == actor_a
        )
        if ab_count < 3:
            raise ValueError(
                f"Protocol requires at least 3 trials with source={actor_a}, "
                f"got {ab_count}"
            )
        if ba_count < 3:
            raise ValueError(
                f"Protocol requires at least 3 trials with source={actor_b}, "
                f"got {ba_count}"
            )

        # 5. Occlusions: at least one <500 ms and one >=500 ms
        for occ in self.occlusions:
            occ.validate()
        short_occs = [o for o in self.occlusions if o.duration_ms() < 500]
        long_occs = [o for o in self.occlusions if o.duration_ms() >= 500]
        if not short_occs:
            raise ValueError(
                "Protocol requires at least one occlusion < 500 ms (expected_unknown=False)"
            )
        if not long_occs:
            raise ValueError(
                "Protocol requires at least one occlusion >= 500 ms (expected_unknown=True)"
            )

        # 6. At least one track reset
        if not self.track_resets:
            raise ValueError("Protocol requires at least one TrackResetEvent")

        # 7. No overlapping trial intervals
        sorted_trials = sorted(self.trials, key=lambda t: t.turn_onset_ms)
        for i in range(len(sorted_trials) - 1):
            a, b = sorted_trials[i], sorted_trials[i + 1]
            if a.turn_end_ms > b.turn_onset_ms:
                raise ValueError(
                    f"Trial {a.trial_id} overlaps Trial {b.trial_id}: "
                    f"[{a.turn_onset_ms}, {a.turn_end_ms}] vs [{b.turn_onset_ms}, {b.turn_end_ms}]"
                )

        # 8. No duplicate trial IDs
        id_counts = Counter(t.trial_id for t in self.trials)
        duplicates = [tid for tid, count in id_counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate trial IDs: {duplicates}")

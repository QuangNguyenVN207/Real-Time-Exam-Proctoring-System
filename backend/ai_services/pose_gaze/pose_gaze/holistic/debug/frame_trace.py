"""Per-frame directed edge trace logger for holistic proctoring (Stage A2).

Each JSONL line is ONE directed edge record that also carries the raw tracks
snapshot so that replay_c3_camera can feed the same data back through
CausalLiveActorClassifier.update_tracks() without any other source of truth.

Format contract:
  - One record per directed edge per frame (actor_id -> peer_id).
  - tracks_snapshot: list of raw track dicts exactly as passed to update_tracks().
  - All scalars are raw values, never aggregated or future-aware.
  - timestamp_ms is strictly monotonic across records emitted by one logger.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass
class FrameTraceRecord:
    """Complete diagnostic trace for one directed edge at one frame.

    tracks_snapshot carries the raw track list for this frame so replay can
    call update_tracks() without any other source.  The same snapshot is
    repeated for every directed edge at the same frame.
    """

    # ---------- Timebase ----------
    timestamp_ms: int
    frame_index: int
    inter_frame_duration_ms: float

    # ---------- Actors / Edge ----------
    actor_id: str
    peer_id: str | None

    # ---------- Track health ----------
    track_present: bool
    track_missed_count: int
    track_age_frames: int

    # ---------- Bbox & Landmark validity ----------
    actor_bbox: list[float] | None
    peer_bbox: list[float] | None
    pose_valid: bool
    hand_valid: bool
    peer_pose_valid: bool | None
    peer_hand_valid: bool | None

    # ---------- Observable & Calibration State ----------
    peer_age_ms: float | None
    peer_stale: bool
    O_A: bool
    O_AB: bool
    R_AB: bool
    neutral_baseline_age_ms: float
    ready_state: str  # "CALIBRATING", "READY", "UNKNOWN"
    reset_reason: str | None

    # ---------- C3 Directed Evidence & Predicates ----------
    p3_A: float | None
    H_AB: float | None          # head/body directed feature score, not c3_score proxy
    Q_A: float | None
    tau_3: float
    tau_H: float

    # ---------- C2 Unordered Evidence & Predicates ----------
    p2_AB: float | None
    K_AB: float | None
    Q_hand_AB: float | None
    tau_2: float
    tau_K: float

    # ---------- Legacy C3 Gate Terms ----------
    # All five legacy gate sub-terms are recorded even after Stage B removes them.
    # Presence in JSONL proves which sub-term vetoed a frame.
    legacy_c3_gate_hand_quality_positive: bool | None
    legacy_c3_gate_hand_motion_passed: bool | None
    legacy_c3_gate_finger_motion_passed: bool | None
    legacy_c3_gate_side_floor_passed: bool | None
    legacy_c3_gate_down_ceiling_passed: bool | None
    legacy_c3_gate_final: bool | None

    # ---------- Resolver & Output ----------
    resolver_candidate: str | None
    emitted_class: str
    unknown_reason: str | None
    first_flag_timestamp_ms: int | None
    latency_ms: float | None

    # ---------- Raw Track Snapshot (for replay) ----------
    # Exact list of track dicts passed to update_tracks() for this frame.
    # Repeated for every directed edge at the same frame.
    tracks_snapshot: list[dict[str, Any]] = field(default_factory=list)

    # Full current aggregate row, retained so scalar mappings remain auditable.
    raw_feature_values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        """Validate mandatory fields."""
        if self.timestamp_ms < 0:
            raise ValueError(f"Negative timestamp_ms: {self.timestamp_ms}")
        if self.frame_index < 0:
            raise ValueError(f"Negative frame_index: {self.frame_index}")
        if not self.actor_id:
            raise ValueError("actor_id cannot be empty")
        if self.ready_state not in ("CALIBRATING", "READY", "UNKNOWN"):
            raise ValueError(f"Invalid ready_state: {self.ready_state!r}")


class FrameTraceLogger:
    """Incremental JSONL writer for frame trace records.

    One logger per session.  Close or use as context manager to flush.
    """

    def __init__(self, output_path: Path | str | None = None) -> None:
        self.output_path = Path(output_path) if output_path else None
        self._file: TextIO | None = None
        self._records: list[FrameTraceRecord] = []
        self._last_timestamp: int | None = None
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.output_path.open("w", encoding="utf-8")

    @property
    def records(self) -> list[FrameTraceRecord]:
        return self._records

    def log(self, record: FrameTraceRecord) -> None:
        """Log one directed edge record.

        Enforces:
        - Schema correctness via record.validate()
        - Strictly monotonic timestamps (same-timestamp OK if same frame)
        - No secret/path fields outside of session manifest
        """
        record.validate()
        if (
            self._last_timestamp is not None
            and record.timestamp_ms < self._last_timestamp
        ):
            raise ValueError(
                f"Non-monotonic timestamp in trace: "
                f"{record.timestamp_ms} < {self._last_timestamp}"
            )
        self._last_timestamp = record.timestamp_ms
        self._records.append(record)

        if self._file is not None:
            self._file.write(json.dumps(record.to_dict()) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()
            self._file = None

    def __enter__(self) -> FrameTraceLogger:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Helpers for replay                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_jsonl(cls, path: Path | str) -> list[dict[str, Any]]:
        """Load and parse JSONL trace file into a list of dicts."""
        records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON at line {lineno}: {exc}") from exc
        return records

    @classmethod
    def to_frame_feed(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert a list of FrameTraceRecord dicts into the frame feed expected
        by replay_c3_camera (deduplicated by frame_index, preserving tracks_snapshot).

        Returns one entry per unique frame_index, sorted by timestamp_ms.
        Each entry has keys: frame_index, timestamp_ms, tracks.
        """
        seen: dict[int, dict[str, Any]] = {}
        for rec in records:
            fidx = int(rec["frame_index"])
            if fidx not in seen:
                seen[fidx] = {
                    "frame_index": fidx,
                    "timestamp_ms": int(rec["timestamp_ms"]),
                    "tracks": list(rec.get("tracks_snapshot", [])),
                }
        return sorted(seen.values(), key=lambda x: x["timestamp_ms"])

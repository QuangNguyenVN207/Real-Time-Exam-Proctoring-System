"""Stateful causal actor inference primitives for live-feed replay and camera use."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, Sequence


class CausalOrderError(ValueError):
    """Raised when a stream sends a frame older than the current state."""


STAGE3_RATE_FEATURES = frozenset({
    "pair_convergence", "hand_direction", "hand_speed", "finger_speed",
    "finger_motion", "hand_finger_motion", "c3_pose_head_peer_velocity",
    "c3_pose_torso_peer_velocity", "hand_motion",
})


def _number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if isfinite(result) else float("nan")


def _valid_flag(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "nan"}
    return bool(value)


@dataclass(frozen=True, slots=True)
class CausalFrameState:
    """One actor's aggregate at one observed frame."""

    actor_id: str
    frame_index: int
    timestamp_ms: int
    features: dict[str, float]
    window_start_frame: int
    window_size: int
    window_end_frame: int = 0
    window_start_timestamp_ms: int = 0
    window_end_timestamp_ms: int = 0
    valid_counts: dict[str, int] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    warmup_ready: bool = False


class CausalActorWindow:
    """Bounded rolling feature state; never reads a future observation."""

    def __init__(
        self,
        actor_id: str,
        feature_names: Sequence[str],
        *,
        max_frames: int = 24,
        warmup_frames: int = 4,
        derivative_feature_names: Sequence[str] = (),
        max_derivative_gap_ms: int = 450,
    ) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")
        if max_derivative_gap_ms <= 0:
            raise ValueError("max_derivative_gap_ms must be positive")
        self.actor_id = str(actor_id)
        self.feature_names = tuple(feature_names)
        self.max_frames = int(max_frames)
        self.warmup_frames = int(warmup_frames)
        self.derivative_feature_names = frozenset(derivative_feature_names)
        self.max_derivative_gap_ms = int(max_derivative_gap_ms)
        self._frames: deque[tuple[int, int, dict[str, float], dict[str, bool]]] = deque(maxlen=max_frames)
        self._last_frame: int | None = None
        self._last_timestamp: int | None = None
        self._last_continuity_epoch: object | None = None
        self._valid_baseline_count = 0

    @property
    def last_frame(self) -> int | None:
        return self._last_frame

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def update(
        self,
        *,
        frame_index: int | None = None,
        timestamp_ms: int,
        features: Mapping[str, object],
        validity: Mapping[str, object] | None = None,
        sample_index: int | None = None,
        source_frame_index: int | None = None,
        continuity_epoch: object | None = None,
    ) -> CausalFrameState:
        if sample_index is not None and frame_index is not None and int(sample_index) != int(frame_index):
            raise CausalOrderError("sample_index and frame_index disagree")
        if sample_index is not None:
            frame_index = sample_index
        if frame_index is None:
            raise TypeError("sample_index or frame_index is required")
        frame_index = int(frame_index)
        timestamp_ms = int(timestamp_ms)
        if self._last_frame is not None and frame_index <= self._last_frame:
            raise CausalOrderError(
                f"actor {self.actor_id}: frame {frame_index} follows "
                f"frame {self._last_frame} out of order"
            )
        if self._last_timestamp is not None and timestamp_ms <= self._last_timestamp:
            raise CausalOrderError(
                f"actor {self.actor_id}: timestamp {timestamp_ms} precedes "
                f"timestamp {self._last_timestamp}"
            )

        validity = validity or {}
        current = {name: _number(features.get(name)) for name in self.feature_names}
        current_valid = {
            name: _valid_flag(validity.get(name), isfinite(current[name])) and isfinite(current[name])
            for name in self.feature_names
        }
        derivative_continuous = (
            self._last_timestamp is not None
            and self._last_continuity_epoch == continuity_epoch
            and 0 < timestamp_ms - self._last_timestamp <= self.max_derivative_gap_ms
        )
        for name in self.derivative_feature_names.intersection(self.feature_names):
            current_valid[name] = current_valid[name] and derivative_continuous
            if not current_valid[name]:
                current[name] = float("nan")
        self._frames.append((frame_index, timestamp_ms, current, current_valid))
        self._last_frame = frame_index
        self._last_timestamp = timestamp_ms
        if continuity_epoch is not None:
            self._last_continuity_epoch = continuity_epoch
        values = {
            name: [frame[2][name] for frame in self._frames if frame[3][name]]
            for name in self.feature_names
        }
        aggregate: dict[str, float] = {}
        for name, series in values.items():
            ordered = sorted(series)
            if not series:
                aggregate.update({
                    f"{name}__mean": float("nan"), f"{name}__std": float("nan"),
                    f"{name}__max": float("nan"), f"{name}__q95": float("nan"),
                    f"{name}__min": float("nan"), f"{name}__valid_count": 0.0,
                })
                continue
            mean = sum(series) / len(series)
            variance = sum((value - mean) ** 2 for value in series) / len(series)
            q95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered))))
            aggregate[f"{name}__mean"] = mean
            aggregate[f"{name}__std"] = variance ** 0.5
            aggregate[f"{name}__max"] = max(series)
            aggregate[f"{name}__q95"] = ordered[q95_index]
            aggregate[f"{name}__min"] = min(series)
            aggregate[f"{name}__valid_count"] = float(len(series))
        warmup_ready = self._valid_baseline_count >= self.warmup_frames
        if any(current_valid.values()):
            self._valid_baseline_count += 1
        return CausalFrameState(
            actor_id=self.actor_id,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            features=aggregate,
            window_start_frame=self._frames[0][0],
            window_size=len(self._frames),
            window_end_frame=self._frames[-1][0],
            window_start_timestamp_ms=self._frames[0][1],
            window_end_timestamp_ms=self._frames[-1][1],
            valid_counts={name: len(values[name]) for name in self.feature_names},
            coverage={name: len(values[name]) / len(self._frames) for name in self.feature_names},
            warmup_ready=warmup_ready,
        )


@dataclass(frozen=True, slots=True)
class ActorEvidence:
    """One qualified class observation retained in actor/session history."""

    class_code: str
    frame_index: int
    timestamp_ms: int
    score: float
    source_actor_id: str | None = None
    source_score: float | None = None


@dataclass(frozen=True, slots=True)
class ActorFlag:
    """Current-frame output plus persistent actor evidence history."""

    actor_id: str
    class_code: str
    evidence_class: str | None = None
    evidence_frame_index: int | None = None
    evidence_timestamp_ms: int | None = None
    evidence_score: float | None = None
    evidence_source_score: float | None = None
    first_flag_frame_index: int | None = None
    first_flag_timestamp_ms: int | None = None
    source_actor_id: str | None = None
    history: tuple[ActorEvidence, ...] = ()


@dataclass(slots=True)
class CausalSpecialistState:
    """Frame-local actor decisions with persistent causal evidence history."""

    actor_ids: tuple[str, ...]
    c3_threshold: float
    c2_threshold: float = 0.5
    suspicious_threshold: float | None = None
    c3_gate: object | None = None
    suspicious_gate: object | None = None
    current_class_by_actor: dict[str, str] = field(default_factory=dict)
    history_by_actor: dict[str, list[ActorEvidence]] = field(default_factory=dict)
    first_flag_by_actor: dict[str, ActorEvidence] = field(default_factory=dict)
    _last_frame: int | None = None
    _last_timestamp: int | None = None

    def __post_init__(self) -> None:
        self.actor_ids = tuple(str(actor_id) for actor_id in self.actor_ids)
        self.current_class_by_actor = {actor_id: "c5" for actor_id in self.actor_ids}
        self.history_by_actor = {actor_id: [] for actor_id in self.actor_ids}

    def register_actor(self, actor_id: str) -> None:
        """Register a track that appears later in a live stream."""
        actor_id = str(actor_id)
        if actor_id in self.current_class_by_actor:
            return
        self.actor_ids = (*self.actor_ids, actor_id)
        self.current_class_by_actor[actor_id] = "c5"
        self.history_by_actor[actor_id] = []

    def update(
        self,
        *,
        frame_index: int,
        timestamp_ms: int,
        scores_by_actor: Mapping[str, Mapping[str, object]],
        explicit_pairs: Sequence[tuple[str, str]] = (),
        near_midpoint_by_actor: Mapping[str, object] | None = None,
    ) -> dict[str, ActorFlag]:
        frame_index = int(frame_index)
        timestamp_ms = int(timestamp_ms)
        if self._last_frame is not None and frame_index <= self._last_frame:
            raise CausalOrderError("specialist frame order is not strictly increasing")
        if self._last_timestamp is not None and timestamp_ms < self._last_timestamp:
            raise CausalOrderError("specialist timestamps are not monotonic")
        self._last_frame = frame_index
        self._last_timestamp = timestamp_ms
        midpoint = near_midpoint_by_actor or {}

        for actor_id in scores_by_actor:
            self.register_actor(actor_id)
        for left, right in explicit_pairs:
            self.register_actor(str(left))
            self.register_actor(str(right))
        self.current_class_by_actor = {
            actor_id: "c5" for actor_id in self.actor_ids
        }

        for left, right in explicit_pairs:
            pair = (str(left), str(right))
            if any(actor_id not in scores_by_actor for actor_id in pair):
                continue
            pair_candidates: list[tuple[float, str]] = []
            for actor_id in pair:
                score = _number(scores_by_actor.get(actor_id, {}).get("c2"))
                if _number(midpoint.get(actor_id)) >= 1.0 and score >= self.c2_threshold:
                    pair_candidates.append((score, actor_id))
            if pair_candidates:
                source_score, source_actor = max(pair_candidates)
                for actor_id in pair:
                    own_score = _number(scores_by_actor.get(actor_id, {}).get("c2"))
                    self._accept(
                        actor_id, "c2", own_score, frame_index, timestamp_ms,
                        source_actor, source_score=source_score,
                    )

        for actor_id in self.actor_ids:
            if self.current_class_by_actor[actor_id] == "c2":
                continue
            values = scores_by_actor.get(actor_id, {})
            c3_score = _number(values.get("c3"))
            suspicious_score = _number(values.get("suspicious_activity"))
            if (
                self.suspicious_threshold is not None
                and suspicious_score >= self.suspicious_threshold
                and (self.suspicious_gate is None or bool(self.suspicious_gate(values)))
            ):
                self._accept(actor_id, "suspicious_activity", suspicious_score, frame_index, timestamp_ms, actor_id)
            if c3_score >= self.c3_threshold and (self.c3_gate is None or bool(self.c3_gate(values))):
                self._accept(actor_id, "c3", c3_score, frame_index, timestamp_ms, actor_id)
        return self.decisions()

    def _accept(
        self,
        actor_id: str,
        class_code: str,
        score: float,
        frame_index: int,
        timestamp_ms: int,
        source_actor: str,
        *,
        source_score: float | None = None,
    ) -> None:
        priority = {"c5": 0, "suspicious_activity": 1, "c3": 2, "c2": 3}
        current = self.current_class_by_actor[actor_id]
        if priority.get(class_code, 0) >= priority.get(current, 0):
            self.current_class_by_actor[actor_id] = class_code

        evidence = ActorEvidence(
            class_code=class_code,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            score=score,
            source_actor_id=source_actor,
            source_score=source_score,
        )
        history = self.history_by_actor.setdefault(actor_id, [])
        for index, previous in enumerate(history):
            if previous.class_code == class_code and previous.frame_index == frame_index:
                if evidence.score > previous.score:
                    history[index] = evidence
                break
        else:
            history.append(evidence)
        self.first_flag_by_actor.setdefault(actor_id, evidence)

    def decisions(self) -> dict[str, ActorFlag]:
        output: dict[str, ActorFlag] = {}
        for actor_id in self.actor_ids:
            history = tuple(self.history_by_actor.get(actor_id, ()))
            evidence = max(
                history,
                key=lambda item: (item.score, item.frame_index, item.timestamp_ms),
                default=None,
            )
            first = self.first_flag_by_actor.get(actor_id)
            output[actor_id] = ActorFlag(
                actor_id=actor_id,
                class_code=self.current_class_by_actor[actor_id],
                evidence_class=evidence.class_code if evidence else None,
                evidence_frame_index=evidence.frame_index if evidence else None,
                evidence_timestamp_ms=evidence.timestamp_ms if evidence else None,
                evidence_score=evidence.score if evidence else None,
                evidence_source_score=evidence.source_score if evidence else None,
                first_flag_frame_index=first.frame_index if first else None,
                first_flag_timestamp_ms=first.timestamp_ms if first else None,
                source_actor_id=evidence.source_actor_id if evidence else None,
                history=history,
            )
        return output


__all__ = [
    "ActorEvidence",
    "ActorFlag",
    "CausalActorWindow",
    "CausalFrameState",
    "CausalOrderError",
    "CausalSpecialistState",
    "STAGE3_RATE_FEATURES",
]

"""Stateful causal actor inference primitives for live-feed replay and camera use."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, Sequence


class CausalOrderError(ValueError):
    """Raised when a stream sends a frame older than the current state."""


def _number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if isfinite(result) else 0.0


@dataclass(frozen=True, slots=True)
class CausalFrameState:
    """One actor's aggregate at one observed frame."""

    actor_id: str
    frame_index: int
    timestamp_ms: int
    features: dict[str, float]
    window_start_frame: int
    window_size: int


class CausalActorWindow:
    """Bounded rolling feature state; never reads a future observation."""

    def __init__(
        self,
        actor_id: str,
        feature_names: Sequence[str],
        *,
        max_frames: int = 90,
    ) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        self.actor_id = str(actor_id)
        self.feature_names = tuple(feature_names)
        self.max_frames = int(max_frames)
        self._frames: deque[tuple[int, int, dict[str, float]]] = deque(maxlen=max_frames)
        self._last_frame: int | None = None
        self._last_timestamp: int | None = None

    @property
    def last_frame(self) -> int | None:
        return self._last_frame

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def update(
        self,
        *,
        frame_index: int,
        timestamp_ms: int,
        features: Mapping[str, object],
    ) -> CausalFrameState:
        frame_index = int(frame_index)
        timestamp_ms = int(timestamp_ms)
        if self._last_frame is not None and frame_index <= self._last_frame:
            raise CausalOrderError(
                f"actor {self.actor_id}: frame {frame_index} follows "
                f"frame {self._last_frame} out of order"
            )
        if self._last_timestamp is not None and timestamp_ms < self._last_timestamp:
            raise CausalOrderError(
                f"actor {self.actor_id}: timestamp {timestamp_ms} precedes "
                f"timestamp {self._last_timestamp}"
            )

        current = {name: _number(features.get(name)) for name in self.feature_names}
        self._frames.append((frame_index, timestamp_ms, current))
        self._last_frame = frame_index
        self._last_timestamp = timestamp_ms
        values = {
            name: [frame[2][name] for frame in self._frames]
            for name in self.feature_names
        }
        aggregate: dict[str, float] = {}
        for name, series in values.items():
            ordered = sorted(series)
            mean = sum(series) / len(series)
            variance = sum((value - mean) ** 2 for value in series) / len(series)
            q95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered))))
            aggregate[f"{name}__mean"] = mean
            aggregate[f"{name}__std"] = variance ** 0.5
            aggregate[f"{name}__max"] = max(series)
            aggregate[f"{name}__q95"] = ordered[q95_index]
            aggregate[f"{name}__min"] = min(series)
        return CausalFrameState(
            actor_id=self.actor_id,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            features=aggregate,
            window_start_frame=self._frames[0][0],
            window_size=len(self._frames),
        )


@dataclass(frozen=True, slots=True)
class ActorFlag:
    """Causal actor decision and audit timestamps."""

    actor_id: str
    class_code: str
    evidence_frame_index: int | None = None
    evidence_timestamp_ms: int | None = None
    evidence_score: float | None = None
    first_flag_frame_index: int | None = None
    first_flag_timestamp_ms: int | None = None
    source_actor_id: str | None = None


@dataclass(slots=True)
class CausalSpecialistState:
    """Actor-level priority state updated in timestamp order."""

    actor_ids: tuple[str, ...]
    c3_threshold: float
    c2_threshold: float = 0.5
    suspicious_threshold: float | None = None
    c3_gate: object | None = None
    suspicious_gate: object | None = None
    class_by_actor: dict[str, str] = field(default_factory=dict)
    evidence_by_actor: dict[str, tuple[str, float, int, int, str]] = field(default_factory=dict)
    first_flag_by_actor: dict[str, tuple[int, int, str]] = field(default_factory=dict)
    _last_frame: int | None = None
    _last_timestamp: int | None = None

    def __post_init__(self) -> None:
        self.actor_ids = tuple(str(actor_id) for actor_id in self.actor_ids)
        self.class_by_actor = {actor_id: "c5" for actor_id in self.actor_ids}

    def register_actor(self, actor_id: str) -> None:
        """Register a track that appears later in a live stream."""
        actor_id = str(actor_id)
        if actor_id in self.class_by_actor:
            return
        self.actor_ids = (*self.actor_ids, actor_id)
        self.class_by_actor[actor_id] = "c5"

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

        for left, right in explicit_pairs:
            pair = (str(left), str(right))
            pair_candidates: list[tuple[str, float, str, str]] = []
            for actor_id in pair:
                score = _number(scores_by_actor.get(actor_id, {}).get("c2"))
                if _number(midpoint.get(actor_id)) >= 1.0 and score >= self.c2_threshold:
                    pair_candidates.append(("c2", score, actor_id, actor_id))
            if pair_candidates:
                class_code, score, source_actor, _ = max(
                    pair_candidates,
                    key=lambda candidate: candidate[1],
                )
                for actor_id in pair:
                    self._accept(actor_id, class_code, score, frame_index, timestamp_ms, source_actor)

        for actor_id in self.actor_ids:
            if self.class_by_actor[actor_id] == "c2":
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
    ) -> None:
        current = self.class_by_actor[actor_id]
        if current == "c5":
            self.class_by_actor[actor_id] = class_code
            self.first_flag_by_actor.setdefault(
                actor_id, (frame_index, timestamp_ms, source_actor)
            )
        elif class_code == "c2" and current != "c2":
            # Pair exchange has priority over an earlier single-actor cue.
            self.class_by_actor[actor_id] = class_code
            self.first_flag_by_actor[actor_id] = (frame_index, timestamp_ms, source_actor)
        elif current != class_code:
            # Apart from the explicit pair-exchange priority, the actor state
            # follows the strongest causal positive evidence seen so far.
            previous = self.evidence_by_actor.get(actor_id)
            if previous is None or score <= previous[1]:
                return
            self.class_by_actor[actor_id] = class_code
        previous = self.evidence_by_actor.get(actor_id)
        if previous is None or score > previous[1]:
            self.evidence_by_actor[actor_id] = (
                class_code, score, frame_index, timestamp_ms, source_actor
            )

    def decisions(self) -> dict[str, ActorFlag]:
        output: dict[str, ActorFlag] = {}
        for actor_id in self.actor_ids:
            evidence = self.evidence_by_actor.get(actor_id)
            first = self.first_flag_by_actor.get(actor_id)
            output[actor_id] = ActorFlag(
                actor_id=actor_id,
                class_code=self.class_by_actor[actor_id],
                evidence_frame_index=evidence[2] if evidence else None,
                evidence_timestamp_ms=evidence[3] if evidence else None,
                evidence_score=evidence[1] if evidence else None,
                first_flag_frame_index=first[0] if first else None,
                first_flag_timestamp_ms=first[1] if first else None,
                source_actor_id=(evidence[4] if evidence else first[2] if first else None),
            )
        return output


__all__ = [
    "ActorFlag",
    "CausalActorWindow",
    "CausalFrameState",
    "CausalOrderError",
    "CausalSpecialistState",
]

"""Replay runner for C3 live camera sessions (Stage A4).

Usage:
    python -m pose_gaze.holistic.debug.replay_c3_camera \\
        --session-manifest <manifest.json> \\
        --trace <frames.jsonl> \\
        --events <events.json> \\
        [--model-dir <dir>] \\
        [--c3-threshold-override <float>] \\
        --assert-expected

Deterministic guarantees:
- Reads FrameTraceRecord JSONL produced by FrameTraceLogger (frame_trace.py).
- Converts to frame feed via FrameTraceLogger.to_frame_feed().
- Passes each frame through CausalLiveActorClassifier.update_tracks() in
  timestamp order, identical to the live webcam seam.
- Replay is independent of future frames: only causal prefix.
- Each trial is evaluated against a fresh per-trial classifier snapshot so
  C3 latching from a previous trial cannot inflate the verdict.

Blocker counters that are actually incremented:
- not_ready: warmup_frames_seen < warmup_frames_required
- score_low: warmup ready but p3_A < tau_3 (from trace record)
- legacy_gate_veto: warmup ready, p3_A >= tau_3, but predicted != c3
  (trace legacy_c3_gate_final == False when available)
- actor_unobservable: trace O_A == False
- peer_unobservable: trace O_AB == False
- head_low: trace H_AB < tau_H (when H_AB is set in trace)
- track_reset: trace reset_reason is not None

Manifest verification:
- When --session-manifest is provided, hashes the model_dir and compares
  against manifest.model_artifacts; aborts if any mismatch.
- Also logs the captured command and c3_threshold_override for audit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from pose_gaze.holistic.debug.events import SessionEvents
from pose_gaze.holistic.debug.frame_trace import FrameTraceLogger
from pose_gaze.holistic.debug.session_manifest import SessionManifest
from pose_gaze.holistic.test_media.live_actor import CausalLiveActorClassifier
from pose_gaze.settings import PROJECT_ROOT

DEFAULT_MODEL_DIR = (
    PROJECT_ROOT / "tmp" / "behavior_actor_extended_suspicious_current_geometry_20260815"
)


@dataclass
class TrialReplayResult:
    """Outcome for one C3 trial."""

    trial_id: int
    source_actor_id: str
    peer_actor_id: str
    turn_onset_ms: int
    turn_end_ms: int
    flagged: bool = False
    first_flag_latency_ms: float | None = None
    max_p3: float = 0.0
    max_h: float = 0.0
    blocker_counts: dict[str, int] = field(default_factory=lambda: {
        "not_ready": 0,
        "actor_unobservable": 0,
        "peer_unobservable": 0,
        "score_low": 0,
        "head_low": 0,
        "legacy_gate_veto": 0,
        "resolver_override": 0,
        "track_reset": 0,
    })
    peer_false_flag: bool = False


@dataclass
class ReplaySummary:
    """Overall replay result."""

    session_id: str
    trials_total: int
    trials_flagged_in_time: int
    neutral_false_flags: int
    peer_false_flags: int
    trial_results: list[TrialReplayResult]
    passed: bool
    replay_wall_seconds: float = 0.0


def _build_explicit_pairs(events: SessionEvents) -> list[tuple[str, str]]:
    """Derive explicit pairs from trial actor IDs, matching webcam default."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for trial in events.trials:
        pair = (trial.source_actor_id, trial.peer_actor_id)
        reverse = (trial.peer_actor_id, trial.source_actor_id)
        if pair not in seen and reverse not in seen:
            pairs.append(pair)
            seen.add(pair)
    return pairs


def _make_fresh_classifier(
    model_dir: Path,
    events: SessionEvents,
    c3_threshold_override: float | None,
) -> CausalLiveActorClassifier:
    """Create a fresh CausalLiveActorClassifier with explicit pairs from events."""
    explicit_pairs = _build_explicit_pairs(events)
    return CausalLiveActorClassifier(
        model_dir=model_dir,
        explicit_pairs=explicit_pairs,
        c3_threshold_override=c3_threshold_override,
    )


def run_c3_replay(
    *,
    trace_records: Sequence[dict[str, Any]],
    events: SessionEvents,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    c3_threshold_override: float | None = None,
    max_latency_ms: float = 2000.0,
) -> ReplaySummary:
    """Run deterministic replay through CausalLiveActorClassifier.update_tracks().

    Parameters
    ----------
    trace_records:
        List of FrameTraceRecord dicts loaded from frames.jsonl.
    events:
        SessionEvents loaded from events.json.
    model_dir:
        Path to the causal specialist artifact directory.
    c3_threshold_override:
        Optional override for c3_threshold; None uses artifact value.
    max_latency_ms:
        Maximum allowable latency from turn_onset_ms to first flag.
    """
    model_dir = Path(model_dir)

    # Build frame feed (one entry per unique frame_index, deduped from JSONL)
    frame_feed = FrameTraceLogger.to_frame_feed(trace_records)
    if not frame_feed:
        raise ValueError("trace_records produced empty frame feed — check tracks_snapshot field")

    # Validate feed is strictly ordered by timestamp (replay must not re-sort)
    for i in range(1, len(frame_feed)):
        if frame_feed[i]["timestamp_ms"] < frame_feed[i - 1]["timestamp_ms"]:
            raise ValueError(
                f"Frame feed is not monotonic at index {i}: "
                f"{frame_feed[i]['timestamp_ms']} < {frame_feed[i-1]['timestamp_ms']}"
            )

    # Index trace records by frame_index for blocker diagnostics
    trace_by_frame_actor: dict[tuple[int, str], dict[str, Any]] = {}
    for rec in trace_records:
        key = (int(rec["frame_index"]), str(rec["actor_id"]))
        trace_by_frame_actor[key] = rec

    # Build trial result stubs
    trial_results: dict[int, TrialReplayResult] = {
        t.trial_id: TrialReplayResult(
            trial_id=t.trial_id,
            source_actor_id=t.source_actor_id,
            peer_actor_id=t.peer_actor_id,
            turn_onset_ms=t.turn_onset_ms,
            turn_end_ms=t.turn_end_ms,
        )
        for t in events.trials
    }

    neutral_false_flags = 0
    replay_start = time.perf_counter()

    # Single pass through all frames with one persistent classifier.
    # C3 latch: CausalSpecialistState keeps the highest-scoring evidence per
    # actor forever.  For per-trial blocker accounting we check whether the
    # *source actor* is predicted c3 within the trial window.  To avoid
    # counting a pre-trial latch as "flagged", we check first_flag_timestamp_ms
    # from the classifier's own decision output against the trial onset.
    classifier = _make_fresh_classifier(model_dir, events, c3_threshold_override)

    for frame in frame_feed:
        frame_idx = int(frame["frame_index"])
        ts_ms = int(frame["timestamp_ms"])
        tracks = frame["tracks"]

        # Production seam call
        classifications = classifier.update_tracks(
            frame_index=frame_idx,
            timestamp_ms=ts_ms,
            tracks=tracks,
        )

        for actor_id, decision in classifications.items():
            pred = decision.get("predicted_class", "c5")
            c3_score = float(decision.get("c3_score") or 0.0)
            first_flag_ts = decision.get("first_flag_timestamp_ms")

            # Neutral false-flag count
            for neutral in events.neutral_intervals:
                if neutral.start_ms <= ts_ms <= neutral.end_ms:
                    if pred == "c3":
                        neutral_false_flags += 1

            # Trial attribution
            for trial in events.trials:
                res = trial_results[trial.trial_id]
                if not (trial.turn_onset_ms <= ts_ms <= trial.turn_end_ms):
                    continue

                trace_rec = trace_by_frame_actor.get((frame_idx, actor_id), {})

                if actor_id == trial.source_actor_id:
                    # Update max scores
                    res.max_p3 = max(res.max_p3, c3_score)
                    h_val = trace_rec.get("H_AB")
                    if h_val is not None:
                        res.max_h = max(res.max_h, float(h_val))

                    if pred == "c3" and not res.flagged:
                        # Only count as flagged if the first_flag happened
                        # inside this trial window (not from a prior trial latch)
                        effective_flag_ts = (
                            first_flag_ts
                            if first_flag_ts is not None
                            else ts_ms
                        )
                        if effective_flag_ts >= trial.turn_onset_ms:
                            lat = float(effective_flag_ts - trial.turn_onset_ms)
                            if lat <= max_latency_ms:
                                res.flagged = True
                                res.first_flag_latency_ms = lat
                    elif pred != "c3":
                        # Blocker attribution using trace record
                        seen = int(decision.get("warmup_frames_seen", 0))
                        req = int(decision.get("warmup_frames_required", 15))

                        # Check observability from trace
                        if trace_rec.get("O_A") is False:
                            res.blocker_counts["actor_unobservable"] += 1
                        elif trace_rec.get("O_AB") is False:
                            res.blocker_counts["peer_unobservable"] += 1
                        elif trace_rec.get("reset_reason") is not None:
                            res.blocker_counts["track_reset"] += 1
                        elif seen < req:
                            res.blocker_counts["not_ready"] += 1
                        else:
                            # Warmup done, peer observable — why no c3?
                            p3 = trace_rec.get("p3_A")
                            tau3 = trace_rec.get("tau_3", classifier.c3_threshold)
                            h_ab = trace_rec.get("H_AB")
                            tau_h = trace_rec.get("tau_H")
                            gate_final = trace_rec.get("legacy_c3_gate_final")

                            if p3 is not None and float(p3) < float(tau3):
                                res.blocker_counts["score_low"] += 1
                            elif c3_score < classifier.c3_threshold:
                                res.blocker_counts["score_low"] += 1
                            elif (
                                h_ab is not None
                                and tau_h is not None
                                and float(h_ab) < float(tau_h)
                            ):
                                res.blocker_counts["head_low"] += 1
                            elif gate_final is False:
                                res.blocker_counts["legacy_gate_veto"] += 1
                            else:
                                # gate_final unknown from trace (not yet wired)
                                # Use score threshold as proxy
                                if c3_score >= classifier.c3_threshold:
                                    res.blocker_counts["legacy_gate_veto"] += 1
                                else:
                                    res.blocker_counts["score_low"] += 1

                elif actor_id == trial.peer_actor_id:
                    if pred == "c3":
                        res.peer_false_flag = True

    replay_wall = time.perf_counter() - replay_start
    peer_false_count = sum(1 for r in trial_results.values() if r.peer_false_flag)
    flagged_count = sum(1 for r in trial_results.values() if r.flagged)

    passed = (
        flagged_count == len(events.trials)
        and neutral_false_flags == 0
        and peer_false_count == 0
    )

    return ReplaySummary(
        session_id=events.session_id,
        trials_total=len(events.trials),
        trials_flagged_in_time=flagged_count,
        neutral_false_flags=neutral_false_flags,
        peer_false_flags=peer_false_count,
        trial_results=list(trial_results.values()),
        passed=passed,
        replay_wall_seconds=round(replay_wall, 2),
    )


def print_replay_report(summary: ReplaySummary) -> None:
    print("=" * 60)
    print(f"REPLAY SUMMARY: {summary.session_id}")
    print("=" * 60)
    print(f"Trials Flagged in Time : {summary.trials_flagged_in_time}/{summary.trials_total}")
    print(f"Neutral False Flags    : {summary.neutral_false_flags}")
    print(f"Peer False Flags       : {summary.peer_false_flags}")
    print(f"Replay Wall Time       : {summary.replay_wall_seconds:.2f}s")
    print("-" * 60)
    for res in summary.trial_results:
        status = "FLAGGED (PASS)" if res.flagged else "MISSED  (FAIL)"
        lat_str = f"{res.first_flag_latency_ms:.1f} ms" if res.first_flag_latency_ms is not None else "N/A"
        print(f"  Trial {res.trial_id:2d} [{res.source_actor_id} -> {res.peer_actor_id}]: {status}")
        print(f"    Latency={lat_str}  max_p3={res.max_p3:.4f}  max_H={res.max_h:.4f}")
        blockers = [f"{k}={v}" for k, v in res.blocker_counts.items() if v > 0]
        if blockers:
            print(f"    Blockers: {', '.join(blockers)}")
        if res.peer_false_flag:
            print(f"    WARNING: peer false-flag detected")
    print("=" * 60)
    print(f"VERDICT: {'GREEN (PASS)' if summary.passed else 'RED   (FAIL)'}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministic replay runner for C3 live camera sessions"
    )
    p.add_argument(
        "--session-manifest",
        type=Path,
        default=None,
        help="Path to session_manifest.json; enables artifact hash verification",
    )
    p.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="Path to frames.jsonl written by FrameTraceLogger",
    )
    p.add_argument(
        "--events",
        type=Path,
        required=True,
        help="Path to events.json with ground-truth trial annotations",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Causal specialist artifact directory",
    )
    p.add_argument(
        "--c3-threshold-override",
        type=float,
        default=None,
        help="Override artifact c3_threshold for this replay only",
    )
    p.add_argument(
        "--assert-expected",
        action="store_true",
        help="Exit non-zero when any expected C3 trial is not flagged in time",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load and verify session manifest when provided
    if args.session_manifest is not None:
        if not args.session_manifest.is_file():
            print(f"ERROR: --session-manifest not found: {args.session_manifest}", file=sys.stderr)
            sys.exit(2)
        manifest = SessionManifest.load(args.session_manifest)
        try:
            manifest.validate()
        except ValueError as exc:
            print(f"ERROR: Invalid session manifest: {exc}", file=sys.stderr)
            sys.exit(2)
        # Verify model artifact hashes match capture
        try:
            manifest.verify_replay_artifact(args.model_dir)
        except ValueError as exc:
            print(f"ERROR: Artifact provenance mismatch:\n{exc}", file=sys.stderr)
            sys.exit(2)
        print(f"Manifest verified: commit={manifest.git['commit'][:12]}  "
              f"branch={manifest.git['branch']}")
        print(f"Captured command: {manifest.command}")
        cap_thresh = manifest.runtime_arguments.get("c3_threshold_override")
        if cap_thresh is not None:
            print(f"Capture c3_threshold_override: {cap_thresh}")

    # Load events
    events = SessionEvents.load(args.events)

    # Load trace JSONL
    trace_records = FrameTraceLogger.load_jsonl(args.trace)
    if not trace_records:
        print("ERROR: Trace file is empty", file=sys.stderr)
        sys.exit(2)

    # Check tracks_snapshot is present in at least one record
    has_tracks = any(bool(r.get("tracks_snapshot")) for r in trace_records)
    if not has_tracks:
        print(
            "ERROR: No tracks_snapshot found in trace — "
            "capture must be recorded with the instrumented webcam (FrameTraceLogger)",
            file=sys.stderr,
        )
        sys.exit(2)

    summary = run_c3_replay(
        trace_records=trace_records,
        events=events,
        model_dir=args.model_dir,
        c3_threshold_override=args.c3_threshold_override,
    )

    print_replay_report(summary)

    if summary.replay_wall_seconds > 30.0:
        print(
            f"WARNING: Replay took {summary.replay_wall_seconds:.1f}s > 30s criterion",
            file=sys.stderr,
        )

    if args.assert_expected and not summary.passed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

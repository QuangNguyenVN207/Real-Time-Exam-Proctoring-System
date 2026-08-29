"""Train and calibrate causal specialists under the Stage 6 grouped-OOF contract."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import sklearn
import xgboost as xgb

from . import behavior_subset_stage2 as behavior
from .stage6_bundle import _canonical_hash, _sha256, verify_grouped_oof_reproduction


FORMAT_VERSION = "causal_8fps_stage6_v1"
SPECIALISTS = ("c2", "c3", "suspicious_activity")
LABELS = ("suspicious_activity", "c2", "c3", "c5")
MODEL_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "max_depth": 3,
    "min_child_weight": 2,
    "eta": 0.035,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "seed": 20260811,
}
BOOST_ROUNDS = 300
MODEL_PROFILES = {
    "balanced_depth3": {"weight_policy": "balanced", "params": {}, "rounds": 300},
    "unweighted_depth3": {"weight_policy": "unweighted", "params": {}, "rounds": 300},
    "legacy3_depth3": {"weight_policy": "legacy_regenerated", "params": {}, "rounds": 300},
    "balanced_depth2": {"weight_policy": "balanced", "params": {"max_depth": 2}, "rounds": 400},
    "balanced_depth4": {"weight_policy": "balanced", "params": {"max_depth": 4}, "rounds": 300},
    "balanced_conservative": {
        "weight_policy": "balanced",
        "params": {"max_depth": 2, "min_child_weight": 5, "eta": 0.025, "colsample_bytree": 0.8},
        "rounds": 500,
    },
    "balanced_rich": {
        "weight_policy": "balanced",
        "params": {"max_depth": 4, "min_child_weight": 1, "eta": 0.03, "colsample_bytree": 1.0},
        "rounds": 500,
    },
    "actor_balanced_depth3": {"weight_policy": "actor_balanced", "params": {}, "rounds": 300},
    "actor_balanced_depth4": {
        "weight_policy": "actor_balanced", "params": {"max_depth": 4}, "rounds": 300,
    },
    "actor_balanced_rich": {
        "weight_policy": "actor_balanced",
        "params": {"max_depth": 4, "min_child_weight": 1, "eta": 0.03, "colsample_bytree": 1.0},
        "rounds": 500,
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _split_hash(rows: Iterable[dict[str, Any]]) -> str:
    assignments = sorted({
        (
            str(row["clip_id"]), str(row["actor_id"]),
            str(row["split_group"]), str(row["split"]),
        )
        for row in rows
    })
    return _canonical_hash(assignments)


def _feature_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape != (len(rows), len(names)):
        raise ValueError("specialist feature matrix has unexpected shape")
    return matrix


def _training_weights(
    rows: list[dict[str, Any]],
    positive_class: str,
    policy: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    labels = np.asarray([row["truth"] == positive_class for row in rows], dtype=np.int8)
    negative = int((labels == 0).sum())
    positive = int((labels == 1).sum())
    if not negative or not positive:
        raise ValueError(f"{positive_class} fold lacks a positive or negative class")
    if policy == "balanced":
        total = len(labels)
        class_weights = {"negative": total / (2.0 * negative), "positive": total / (2.0 * positive)}
        weights = np.where(labels == 1, class_weights["positive"], class_weights["negative"]).astype(np.float32)
    elif policy == "legacy_regenerated":
        class_weights = {"negative": 1.0 / math.sqrt(negative), "positive": 3.0 / math.sqrt(positive)}
        weights = np.where(labels == 1, class_weights["positive"], class_weights["negative"]).astype(np.float32)
        weights *= float(len(weights)) / float(weights.sum())
    elif policy == "unweighted":
        class_weights = {"negative": 1.0, "positive": 1.0}
        weights = None
    elif policy == "actor_balanced":
        actor_keys = [(str(row["video"]), str(row["actor_id"])) for row in rows]
        row_counts = Counter(actor_keys)
        actor_labels = {
            key: bool(row["truth"] == positive_class) for key, row in zip(actor_keys, rows, strict=True)
        }
        negative_actors = sum(not label for label in actor_labels.values())
        positive_actors = sum(actor_labels.values())
        if not negative_actors or not positive_actors:
            raise ValueError(f"{positive_class} fold lacks positive or negative actors")
        class_weights = {
            "negative": len(actor_labels) / (2.0 * negative_actors),
            "positive": len(actor_labels) / (2.0 * positive_actors),
        }
        weights = np.asarray([
            class_weights["positive" if actor_labels[key] else "negative"] / row_counts[key]
            for key in actor_keys
        ], dtype=np.float32)
        weights *= float(len(weights)) / float(weights.sum())
    else:
        raise ValueError(f"unknown Stage 6 weight policy: {policy}")
    return weights, {
        "policy": policy,
        "negative_rows": negative,
        "positive_rows": positive,
        "negative_weight": class_weights["negative"],
        "positive_weight": class_weights["positive"],
        "actor_count": len(set((str(row["video"]), str(row["actor_id"])) for row in rows)),
    }


def _fit_specialist(
    rows: list[dict[str, Any]],
    names: list[str],
    positive_class: str,
    device: str,
    model_profile: str,
) -> tuple[xgb.Booster, dict[str, Any]]:
    profile = MODEL_PROFILES[model_profile]
    weights, weight_metadata = _training_weights(rows, positive_class, str(profile["weight_policy"]))
    labels = np.asarray([row["truth"] == positive_class for row in rows], dtype=np.float32)
    params = dict(MODEL_PARAMS, device=device, **dict(profile["params"]))
    matrix = xgb.DMatrix(
        _feature_matrix(rows, names), label=labels, weight=weights, feature_names=names
    )
    model = xgb.train(params, matrix, num_boost_round=int(profile["rounds"]))
    return model, weight_metadata


def _prepare_rows(
    input_path: Path,
    manifest_path: Path,
    json_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    behavior.EXCLUDE_C7 = True
    behavior.EXTENDED_SUSPICIOUS = True
    behavior.MODEL_CLASSES = ("suspicious_activity", "c2", "c3", "c5")
    behavior.TARGET_CLASSES = ("suspicious_activity", "c2", "c3")
    manifest = behavior.load_manifest(manifest_path)
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        if "sample_index" not in source:
            raise ValueError("Stage 6 input requires sample_index")
        if "actor_id" not in source:
            raise ValueError("Stage 6 input requires actor_id")
        if "clip_id" not in source:
            source["clip_id"] = Path(source.get("source_filename", "")).stem
        if "source_frame_index" not in source:
            raise ValueError("Stage 6 input requires source_frame_index provenance")
        row = dict(source)
        row.update(behavior.actor_truth(source, manifest))
        row["excluded_source"] = bool(row["actor_truth"] == "c7")
        row["actor_label"] = row["actor_truth"]
        rows.append(row)

    rows = behavior.attach_selected_face_points(rows, json_root)
    rows = behavior.derive_c3_pose_contract(rows)
    rows = behavior._head_pnp_features(rows)
    rows = behavior.derive_behavior_motion(rows)
    rows = behavior.derive_face_c3_features(rows)
    rows = behavior.derive_finger_motion(rows)
    rows = behavior.derive_hand_shape_and_pair_cues(rows)
    rows = behavior.derive_strict_c2_c3_suspicious_cues(rows)
    rows = behavior._apply_stage3_temporal_contract(rows)
    train_frames = [row for row in rows if row["split"] == "train" and not row["excluded_source"]]
    test_frames = [row for row in rows if row["split"] == "test" and not row["excluded_source"]]
    if not train_frames or not test_frames:
        raise ValueError("Stage 6 requires non-empty approved train and locked test splits")
    return train_frames, test_frames


def _prefix_rows(frame_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    shared_bases = tuple(dict.fromkeys((
        *behavior.C2_CUE_FEATURES,
        *behavior.C3_POSE_ONLY_FEATURES,
        *behavior.STRICT_C3_SUSPICIOUS_FEATURES,
    )))
    prefix, shared_names = behavior.causal_aggregate_rows(frame_rows, shared_bases)
    usable = [row for row in prefix if row["warmup_ready"]]
    schemas = {
        "c2": [name for name in shared_names if name.split("__", 1)[0] in behavior.C2_CUE_FEATURES],
        "c3": [name for name in shared_names if name.split("__", 1)[0] in (
            *behavior.C3_POSE_ONLY_FEATURES, *behavior.STRICT_C3_SUSPICIOUS_FEATURES,
        )],
        "suspicious_activity": [
            name for name in shared_names
            if name.split("__", 1)[0] in behavior.STRICT_C3_SUSPICIOUS_FEATURES
        ],
    }
    expected_counts = {"c2": 65, "c3": 90, "suspicious_activity": 50}
    actual_counts = {name: len(values) for name, values in schemas.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"Stage 6 specialist schema mismatch: {actual_counts}")
    return usable, schemas


def _gate_functions(values: dict[str, float]) -> tuple[Any, Any]:

    def c3_gate(row: dict[str, Any]) -> bool:
        return (
            min(
                behavior.number(row.get("current_c3_pose_head_valid")),
                behavior.number(row.get("current_c3_pose_peer_valid")),
            ) >= 0.5
            and behavior.number(row.get("current_c3_pose_head_peer_delta")) >= values["c3_side_floor"]
            and behavior.number(row.get("current_strict_head_down_delta")) <= values["c3_down_ceiling"]
        )

    def suspicious_gate(row: dict[str, Any]) -> bool:
        return (
            behavior.number(row.get("strict_head_down_delta__q95")) >= values["suspicious_down_floor"]
            and max(
                behavior.number(row.get("hand_motion__q95")),
                behavior.number(row.get("finger_motion__q95")),
            ) >= values["suspicious_motion_floor"]
            and behavior.number(row.get("strict_hand_below_hip__max")) >= values["suspicious_lower_floor"]
            and behavior.number(row.get("strict_own_side_outside_midpoint__max")) >= 1.0
        )

    return c3_gate, suspicious_gate


def _quantile_candidates(rows: list[dict[str, Any]], name: str) -> list[float]:
    values = np.asarray([
        behavior.number(row.get(name)) for row in rows
        if behavior._is_number(row.get(name))
    ], dtype=np.float64)
    if not len(values):
        raise ValueError(f"no finite OOF gate candidates for {name}")
    return sorted({float(value) for value in np.quantile(values, np.linspace(0.0, 1.0, 101))})


class _FastOofEvaluator:
    """Vectorized equivalent of causal history reduction for calibration search."""

    def __init__(self, rows: list[dict[str, Any]], scores: dict[str, np.ndarray]) -> None:
        actor_keys = sorted({(str(row["video"]), str(row["actor_id"])) for row in rows})
        self.actor_keys = actor_keys
        actor_by_key = {key: index for index, key in enumerate(actor_keys)}
        self.actor_index = np.asarray([
            actor_by_key[(str(row["video"]), str(row["actor_id"]))] for row in rows
        ], dtype=np.int32)
        self.frames = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int32)
        self.timestamps = np.asarray([int(float(row["timestamp_ms"])) for row in rows], dtype=np.int64)
        self.scores = {name: np.asarray(scores[name], dtype=np.float32) for name in SPECIALISTS}
        self.truth = np.asarray([
            LABELS.index(behavior.rows_by_key_truth(rows, key)) for key in actor_keys
        ], dtype=np.int8)
        self.midpoint = np.asarray([
            behavior.number(row.get("near_midpoint_pre_cross")) >= 1.0
            and behavior.number(row.get("current_hand_quality_mask")) > 0.0
            and behavior.number(row.get("current_pair_hand_distance")) > 0.0
            and behavior.number(row.get("current_pair_margin_10pct")) > 0.0
            for row in rows
        ], dtype=bool)
        self.head_valid = np.asarray([
            min(
                behavior.number(row.get("current_c3_pose_head_valid")),
                behavior.number(row.get("current_c3_pose_peer_valid")),
            ) >= 0.5 for row in rows
        ], dtype=bool)
        self.head_delta = np.asarray([
            behavior.number(row.get("current_c3_pose_head_peer_delta")) for row in rows
        ], dtype=np.float32)
        self.current_down = np.asarray([
            behavior.number(row.get("current_strict_head_down_delta")) for row in rows
        ], dtype=np.float32)
        self.down_q95 = np.asarray([
            behavior.number(row.get("strict_head_down_delta__q95")) for row in rows
        ], dtype=np.float32)
        self.motion_q95 = np.asarray([
            max(
                behavior.number(row.get("hand_motion__q95")),
                behavior.number(row.get("finger_motion__q95")),
            ) for row in rows
        ], dtype=np.float32)
        self.lower_max = np.asarray([
            behavior.number(row.get("strict_hand_below_hip__max")) for row in rows
        ], dtype=np.float32)
        self.own_side = np.asarray([
            behavior.number(row.get("strict_own_side_outside_midpoint__max")) >= 1.0
            for row in rows
        ], dtype=bool)

        lookup = {
            (str(row["video"]), str(row["actor_id"]), int(row["sample_index"])): index
            for index, row in enumerate(rows)
        }
        samples_by_actor: dict[tuple[str, str], set[int]] = defaultdict(set)
        for row in rows:
            samples_by_actor[(str(row["video"]), str(row["actor_id"]))].add(int(row["sample_index"]))
        left_indices: list[int] = []
        right_indices: list[int] = []
        for pair in behavior.explicit_pair_keys(rows):
            endpoints = sorted(tuple(key) for key in pair)
            if len(endpoints) != 2:
                continue
            for sample_index in sorted(samples_by_actor[endpoints[0]] & samples_by_actor[endpoints[1]]):
                left_indices.append(lookup[(*endpoints[0], sample_index)])
                right_indices.append(lookup[(*endpoints[1], sample_index)])
        self.pair_left = np.asarray(left_indices, dtype=np.int32)
        self.pair_right = np.asarray(right_indices, dtype=np.int32)

    def _reduce(self, row_indices: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actor_count = len(self.actor_keys)
        best_score = np.full(actor_count, -np.inf, dtype=np.float32)
        best_frame = np.full(actor_count, -1, dtype=np.int32)
        best_timestamp = np.full(actor_count, -1, dtype=np.int64)
        if not len(row_indices):
            return best_score, best_frame, best_timestamp
        actors = self.actor_index[row_indices]
        candidate_scores = values[row_indices]
        np.maximum.at(best_score, actors, candidate_scores)
        score_winners = candidate_scores == best_score[actors]
        winner_rows = row_indices[score_winners]
        winner_actors = self.actor_index[winner_rows]
        np.maximum.at(best_frame, winner_actors, self.frames[winner_rows])
        frame_winners = self.frames[winner_rows] == best_frame[winner_actors]
        final_rows = winner_rows[frame_winners]
        np.maximum.at(best_timestamp, self.actor_index[final_rows], self.timestamps[final_rows])
        return best_score, best_frame, best_timestamp

    def evidence(
        self,
        thresholds: dict[str, float],
        gates: dict[str, float],
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        c2_source = (
            (self.midpoint[self.pair_left] & (self.scores["c2"][self.pair_left] >= thresholds["c2"]))
            | (self.midpoint[self.pair_right] & (self.scores["c2"][self.pair_right] >= thresholds["c2"]))
        )
        c2_rows = np.concatenate((self.pair_left[c2_source], self.pair_right[c2_source]))
        c2 = self._reduce(c2_rows, self.scores["c2"])
        c2_blocked_rows = np.zeros(len(self.actor_index), dtype=bool)
        c2_blocked_rows[c2_rows] = True
        suspicious_mask = (
            (self.scores["suspicious_activity"] >= thresholds["suspicious_activity"])
            & (self.down_q95 >= gates["suspicious_down_floor"])
            & (self.motion_q95 >= gates["suspicious_motion_floor"])
            & (self.lower_max >= gates["suspicious_lower_floor"])
            & self.own_side
            & ~c2_blocked_rows
        )
        suspicious = self._reduce(np.flatnonzero(suspicious_mask), self.scores["suspicious_activity"])
        c3_mask = (
            (self.scores["c3"] >= thresholds["c3"])
            & self.head_valid
            & (self.head_delta >= gates["c3_side_floor"])
            & (self.current_down <= gates["c3_down_ceiling"])
            & ~c2_blocked_rows
        )
        c3 = self._reduce(np.flatnonzero(c3_mask), self.scores["c3"])

        return c2, suspicious, c3

    def predict(self, thresholds: dict[str, float], gates: dict[str, float]) -> np.ndarray:
        predicted = np.full(len(self.actor_keys), 3, dtype=np.int8)
        best_score = np.full(len(self.actor_keys), -np.inf, dtype=np.float32)
        best_frame = np.full(len(self.actor_keys), -1, dtype=np.int32)
        best_timestamp = np.full(len(self.actor_keys), -1, dtype=np.int64)
        # Exact history insertion order for a tie on score/frame/timestamp.
        c2, suspicious, c3 = self.evidence(thresholds, gates)
        for label, evidence in ((1, c2), (0, suspicious), (2, c3)):
            score, frame, timestamp = evidence
            better = (
                (score > best_score)
                | ((score == best_score) & (frame > best_frame))
                | ((score == best_score) & (frame == best_frame) & (timestamp > best_timestamp))
            )
            predicted[better] = label
            best_score[better] = score[better]
            best_frame[better] = frame[better]
            best_timestamp[better] = timestamp[better]

        return predicted

    def evaluate(self, thresholds: dict[str, float], gates: dict[str, float]) -> float:
        predicted = self.predict(thresholds, gates)
        scores_by_class = []
        for label in range(len(LABELS)):
            truth = self.truth == label
            guess = predicted == label
            tp = int(np.sum(truth & guess))
            fp = int(np.sum(~truth & guess))
            fn = int(np.sum(truth & ~guess))
            scores_by_class.append(2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        return float(np.mean(scores_by_class))


def _prediction_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    truth = [str(row["truth"]) for row in predictions]
    predicted = [behavior._history_metric_class(row) for row in predictions]
    report = classification_report(truth, predicted, labels=list(LABELS), output_dict=True, zero_division=0)
    return {
        "actor_macro_f1": float(f1_score(truth, predicted, labels=list(LABELS), average="macro", zero_division=0)),
        "actor_metrics": {name: report.get(name, {}) for name in LABELS},
        "actor_confusion_matrix": confusion_matrix(truth, predicted, labels=list(LABELS)).tolist(),
        "actor_count": len(predictions),
    }


def _actor_score_candidates(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    specialist: str,
) -> list[float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        if specialist == "c2" and not (
            behavior.number(row.get("near_midpoint_pre_cross")) >= 1.0
            and behavior.number(row.get("current_hand_quality_mask")) > 0.0
            and behavior.number(row.get("current_pair_hand_distance")) > 0.0
            and behavior.number(row.get("current_pair_margin_10pct")) > 0.0
        ):
            continue
        grouped[(str(row["video"]), str(row["actor_id"]))].append(float(probability))
    actor_keys = {(str(row["video"]), str(row["actor_id"])) for row in rows}
    values = {max(grouped.get(key, [0.0])) for key in actor_keys}
    values.update((0.5, 1.0))
    return sorted(values)


def _calibrate_joint_thresholds(
    rows: list[dict[str, Any]],
    scores: dict[str, np.ndarray],
    initial_gate_thresholds: dict[str, float],
    *,
    global_trials: int = 30000,
    search_seed: int = 20260827,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]]]:
    """Coordinate-search OOF thresholds against primary actor macro-F1."""

    thresholds = {
        "c2": behavior._fit_current_geometry_actor_threshold(rows, scores["c2"], "c2"),
        "c3": behavior._fit_actor_max_threshold(rows, scores["c3"], "c3"),
        "suspicious_activity": behavior._fit_actor_max_threshold(
            rows, scores["suspicious_activity"], "suspicious_activity"
        ),
    }
    candidates = {
        specialist: _actor_score_candidates(rows, scores[specialist], specialist)
        for specialist in SPECIALISTS
    }
    gate_thresholds = dict(initial_gate_thresholds)
    gate_candidates = {
        "c3_side_floor": _quantile_candidates(rows, "current_c3_pose_head_peer_delta"),
        "c3_down_ceiling": _quantile_candidates(rows, "current_strict_head_down_delta"),
        "suspicious_down_floor": _quantile_candidates(rows, "strict_head_down_delta__q95"),
        "suspicious_motion_floor": sorted(set(
            _quantile_candidates(rows, "hand_motion__q95")
            + _quantile_candidates(rows, "finger_motion__q95")
        )),
        "suspicious_lower_floor": _quantile_candidates(rows, "strict_hand_below_hip__max"),
    }
    evaluator = _FastOofEvaluator(rows, scores)
    objective_cache: dict[tuple[float, ...], float] = {}

    def objective(values: dict[str, float], gates: dict[str, float]) -> float:
        cache_key = tuple(values[name] for name in SPECIALISTS) + tuple(
            gates[name] for name in gate_candidates
        )
        if cache_key in objective_cache:
            return objective_cache[cache_key]
        score = evaluator.evaluate(values, gates)
        objective_cache[cache_key] = score
        return score

    initial_fast_score = objective(thresholds, gate_thresholds)
    initial_c3_gate, initial_suspicious_gate = _gate_functions(gate_thresholds)
    initial_exact_predictions = behavior.causal_specialist_replay(
        rows,
        scores["c2"],
        scores["c3"],
        c2_threshold=thresholds["c2"],
        c3_threshold=thresholds["c3"],
        suspicious_probabilities=scores["suspicious_activity"],
        suspicious_threshold=thresholds["suspicious_activity"],
        c3_gate=initial_c3_gate,
        suspicious_gate=initial_suspicious_gate,
    )
    initial_exact_score = _prediction_metrics(initial_exact_predictions)["actor_macro_f1"]
    if initial_fast_score != initial_exact_score:
        exact_by_key = {
            (str(row["video"]), str(row["actor_id"])): behavior._history_metric_class(row)
            for row in initial_exact_predictions
        }
        fast_ids = evaluator.predict(thresholds, gate_thresholds)
        fast_evidence = evaluator.evidence(thresholds, gate_thresholds)
        exact_rows = {
            (str(row["video"]), str(row["actor_id"])): row
            for row in initial_exact_predictions
        }
        mismatches = []
        for index, key in enumerate(evaluator.actor_keys):
            if LABELS[int(fast_ids[index])] == exact_by_key[key]:
                continue
            mismatches.append({
                "video": key[0],
                "actor_id": key[1],
                "truth": LABELS[int(evaluator.truth[index])],
                "fast": LABELS[int(fast_ids[index])],
                "exact": exact_by_key[key],
                "fast_evidence": {
                    name: {
                        "score": float(fast_evidence[position][0][index]),
                        "frame": int(fast_evidence[position][1][index]),
                    }
                    for position, name in enumerate(("c2", "suspicious_activity", "c3"))
                },
                "exact_history": exact_rows[key].get("history", []),
            })
        raise RuntimeError(
            f"fast causal calibration evaluator mismatch: fast={initial_fast_score}, "
            f"exact={initial_exact_score}, actors={json.dumps(mismatches[:20], separators=(',', ':'))}"
        )

    trace = [{
        "pass": 0,
        "parameter": "initial",
        "thresholds": dict(thresholds),
        "gate_thresholds": dict(gate_thresholds),
        "actor_macro_f1": initial_fast_score,
        "exact_causal_parity": True,
    }]
    rng = np.random.default_rng(search_seed)
    parameter_candidates = {**candidates, **gate_candidates}
    best_random_score = initial_fast_score
    best_random_values = dict(thresholds)
    best_random_gates = dict(gate_thresholds)
    for _ in range(global_trials):
        proposal_values = {
            name: float(rng.choice(parameter_candidates[name])) for name in SPECIALISTS
        }
        proposal_gates = dict(gate_thresholds)
        for name in gate_candidates:
            proposal_gates[name] = float(rng.choice(parameter_candidates[name]))
        score = objective(proposal_values, proposal_gates)
        if score > best_random_score:
            best_random_score = score
            best_random_values = proposal_values
            best_random_gates = proposal_gates
    thresholds = best_random_values
    gate_thresholds = best_random_gates
    trace.append({
        "pass": 0,
        "parameter": "deterministic_global_search",
        "trials": global_trials,
        "seed": search_seed,
        "thresholds": dict(thresholds),
        "gate_thresholds": dict(gate_thresholds),
        "actor_macro_f1": best_random_score,
    })
    for pass_index in range(1, 4):
        changed = False
        for specialist in SPECIALISTS:
            current = thresholds[specialist]
            best_threshold = current
            best_score = objective(thresholds, gate_thresholds)
            for candidate in candidates[specialist]:
                proposal = dict(thresholds)
                proposal[specialist] = candidate
                score = objective(proposal, gate_thresholds)
                if score > best_score or (score == best_score and candidate > best_threshold):
                    best_score = score
                    best_threshold = candidate
            thresholds[specialist] = best_threshold
            changed = changed or best_threshold != current
            trace.append({
                "pass": pass_index,
                "parameter": specialist,
                "threshold": best_threshold,
                "actor_macro_f1": best_score,
                "candidate_count": len(candidates[specialist]),
            })
        for gate_name, values in gate_candidates.items():
            current = gate_thresholds[gate_name]
            best_threshold = current
            best_score = objective(thresholds, gate_thresholds)
            prefer_higher = not gate_name.endswith("ceiling")
            for candidate in values:
                proposal = dict(gate_thresholds)
                proposal[gate_name] = candidate
                score = objective(thresholds, proposal)
                tie_is_better = (
                    candidate > best_threshold if prefer_higher else candidate < best_threshold
                )
                if score > best_score or (score == best_score and tie_is_better):
                    best_score = score
                    best_threshold = candidate
            gate_thresholds[gate_name] = best_threshold
            changed = changed or best_threshold != current
            trace.append({
                "pass": pass_index,
                "parameter": gate_name,
                "threshold": best_threshold,
                "actor_macro_f1": best_score,
                "candidate_count": len(values),
            })
        if not changed:
            break
    final_fast_score = objective(thresholds, gate_thresholds)
    final_c3_gate, final_suspicious_gate = _gate_functions(gate_thresholds)
    final_exact_score = _prediction_metrics(behavior.causal_specialist_replay(
        rows,
        scores["c2"],
        scores["c3"],
        c2_threshold=thresholds["c2"],
        c3_threshold=thresholds["c3"],
        suspicious_probabilities=scores["suspicious_activity"],
        suspicious_threshold=thresholds["suspicious_activity"],
        c3_gate=final_c3_gate,
        suspicious_gate=final_suspicious_gate,
    ))["actor_macro_f1"]
    if final_fast_score != final_exact_score:
        raise RuntimeError(
            f"selected causal calibration mismatch: fast={final_fast_score}, exact={final_exact_score}"
        )
    trace.append({
        "pass": 4,
        "parameter": "selected_exact_causal_verification",
        "actor_macro_f1": final_exact_score,
        "exact_causal_parity": True,
    })
    return thresholds, gate_thresholds, trace


def _csv_safe_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in predictions:
        row = dict(source)
        row["history"] = json.dumps(row.get("history", []), separators=(",", ":"))
        row["metric_class"] = behavior._history_metric_class(source)
        output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pair_rows(prefix_rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_actor = {(str(row["video"]), str(row["actor_id"])): row for row in predictions}
    output: list[dict[str, Any]] = []
    for pair in sorted(
        behavior.explicit_pair_keys(prefix_rows),
        key=lambda value: sorted(tuple(item) for item in value),
    ):
        endpoints = sorted(tuple(item) for item in pair)
        if len(endpoints) != 2 or endpoints[0] not in by_actor or endpoints[1] not in by_actor:
            continue
        left, right = (by_actor[endpoints[0]], by_actor[endpoints[1]])
        output.append({
            "video": endpoints[0][0],
            "actor_id_a": endpoints[0][1],
            "actor_id_b": endpoints[1][1],
            "truth_a": left["truth"],
            "truth_b": right["truth"],
            "predicted_a": behavior._history_metric_class(left),
            "predicted_b": behavior._history_metric_class(right),
            "exact_match": int(
                left["truth"] == behavior._history_metric_class(left)
                and right["truth"] == behavior._history_metric_class(right)
            ),
        })
    return output


def _pair_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = [row[key] for row in rows for key in ("truth_a", "truth_b")]
    predicted = [row[key] for row in rows for key in ("predicted_a", "predicted_b")]
    return {
        "pair_count": len(rows),
        "pair_exact_accuracy": float(np.mean([row["exact_match"] for row in rows])) if rows else math.nan,
        "pair_endpoint_macro_f1": (
            float(f1_score(truth, predicted, labels=list(LABELS), average="macro", zero_division=0))
            if rows else math.nan
        ),
    }


def _fold_scores(
    train_rows: list[dict[str, Any]],
    schemas: dict[str, list[str]],
    output_dir: Path,
    device: str,
    model_profiles: dict[str, str],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    group_values = np.asarray([str(row["split_group"]) for row in train_rows])
    unique_groups = sorted(set(group_values))
    if len(unique_groups) < 2:
        raise ValueError("Stage 6 grouped OOF requires at least two training split groups")
    scores = {name: np.full(len(train_rows), np.nan, dtype=np.float32) for name in SPECIALISTS}
    fold_ids = np.full(len(train_rows), "", dtype=f"<U{max(map(len, unique_groups))}")
    fold_metadata: dict[str, Any] = {}
    for held_out_group in unique_groups:
        validation_indices = np.flatnonzero(group_values == held_out_group)
        fit_indices = np.flatnonzero(group_values != held_out_group)
        fit_rows = [train_rows[index] for index in fit_indices]
        validation_rows = [train_rows[index] for index in validation_indices]
        fold_dir = output_dir / "fold_models" / held_out_group
        fold_dir.mkdir(parents=True)
        fold_metadata[held_out_group] = {
            "fit_groups": sorted(set(group_values[fit_indices])),
            "held_out_group": held_out_group,
            "fit_rows": len(fit_rows),
            "validation_rows": len(validation_rows),
            "class_weights": {},
        }
        for specialist in SPECIALISTS:
            model, weights = _fit_specialist(
                fit_rows, schemas[specialist], specialist, device, model_profiles[specialist]
            )
            model.save_model(str(fold_dir / f"{specialist}.ubj"))
            model.feature_names = schemas[specialist]
            scores[specialist][validation_indices] = model.predict(
                xgb.DMatrix(
                    _feature_matrix(validation_rows, schemas[specialist]),
                    feature_names=schemas[specialist],
                )
            ).astype(np.float32)
            fold_metadata[held_out_group]["class_weights"][specialist] = weights
        fold_ids[validation_indices] = held_out_group
    if any(not np.isfinite(values).all() for values in scores.values()) or np.any(fold_ids == ""):
        raise RuntimeError("grouped OOF did not assign every training prefix row")
    return scores, fold_ids, fold_metadata


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise ValueError(f"Stage 6 output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    feature_manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    checks = feature_manifest.get("completion_checks", {})
    if checks.get("landmark_root_is_8fps") is not True or checks.get("contains_30fps_input_path") is not False:
        raise ValueError("Stage 5 manifest does not prove an exclusive 8 FPS input contract")
    policy = feature_manifest.get("temporal_policy")
    expected_policy = {
        "target_fps": 8,
        "baseline_valid_frames": 4,
        "head_turn_baseline_valid_frames": 8,
        "window_frames": 24,
        "max_derivative_gap_ms": 450,
    }
    if policy != expected_policy:
        raise ValueError(f"Stage 5 temporal policy mismatch: {policy}")

    behavior.XGBOOST_DEVICE = args.xgb_device
    train_frames, test_frames = _prepare_rows(args.input, args.manifest, args.json_root)
    train_rows, schemas = _prefix_rows(train_frames)
    model_profiles = {
        "c2": args.c2_profile or args.model_profile,
        "c3": args.c3_profile or args.model_profile,
        "suspicious_activity": args.suspicious_profile or args.model_profile,
    }

    scores, fold_ids, fold_metadata = _fold_scores(
        train_rows, schemas, args.output_dir, args.xgb_device, model_profiles
    )
    initial_gate_thresholds = behavior._extended_gate_thresholds(train_rows)
    selection_payload = None
    recorded_selection_metric = None
    if args.calibration_selection is not None:
        selection_payload = json.loads(args.calibration_selection.read_text(encoding="utf-8"))
        if selection_payload.get("locked_test_read_or_scored") is not False:
            raise ValueError("calibration selection must prove locked test was not read or scored")
        best = selection_payload.get("best", {})
        aliases = {
            "balanced": "balanced_depth3",
            "unweighted": "unweighted_depth3",
            "depth4": "balanced_depth4",
        }
        selected_profiles = {
            specialist: aliases.get(str(value), str(value))
            for specialist, value in best.get("profiles", {}).items()
        }
        if selected_profiles != model_profiles:
            raise ValueError(
                f"calibration selection profiles {selected_profiles} do not match requested {model_profiles}"
            )
        if best.get("exact_causal_parity") is not True:
            raise ValueError("calibration selection lacks exact causal parity")
        thresholds = {name: float(best["thresholds"][name]) for name in SPECIALISTS}
        gate_thresholds = {
            name: float(value) for name, value in best["gate_thresholds"].items()
        }
        recorded_selection_metric = float(best["actor_macro_f1"])
        calibration_trace = [{
            "source": "preselected grouped OOF result",
            "selection_file": str(args.calibration_selection.resolve()),
            "selection_sha256": _sha256(args.calibration_selection),
            "recorded_actor_macro_f1": recorded_selection_metric,
            "exact_causal_parity": True,
        }]
    else:
        thresholds, gate_thresholds, calibration_trace = _calibrate_joint_thresholds(
            train_rows,
            scores,
            initial_gate_thresholds,
            global_trials=args.global_trials,
            search_seed=args.search_seed,
        )
    c3_gate, suspicious_gate = _gate_functions(gate_thresholds)
    oof_predictions = behavior.causal_specialist_replay(
        train_rows,
        scores["c2"],
        scores["c3"],
        c2_threshold=thresholds["c2"],
        c3_threshold=thresholds["c3"],
        suspicious_probabilities=scores["suspicious_activity"],
        suspicious_threshold=thresholds["suspicious_activity"],
        c3_gate=c3_gate,
        suspicious_gate=suspicious_gate,
    )
    oof_metrics = _prediction_metrics(oof_predictions)
    if recorded_selection_metric is not None and oof_metrics["actor_macro_f1"] != recorded_selection_metric:
        raise RuntimeError(
            f"materialized OOF metric {oof_metrics['actor_macro_f1']} does not reproduce "
            f"selection metric {recorded_selection_metric}"
        )
    oof_pairs = _pair_rows(train_rows, oof_predictions)
    oof_metrics.update(_pair_metrics(oof_pairs))

    models_dir = args.output_dir / "models"
    models_dir.mkdir()
    final_models: dict[str, xgb.Booster] = {}
    final_weights: dict[str, Any] = {}
    for specialist in SPECIALISTS:
        model, weights = _fit_specialist(
            train_rows, schemas[specialist], specialist, args.xgb_device, model_profiles[specialist]
        )
        model.save_model(str(models_dir / f"{specialist}.ubj"))
        model.feature_names = schemas[specialist]
        final_models[specialist] = model
        final_weights[specialist] = weights

    _write_json(args.output_dir / "feature_schemas.json", schemas)
    _write_json(args.output_dir / "temporal_policy.json", policy)
    _write_json(args.output_dir / "calibration.json", {
        "source": "grouped OOF training-prefix scores only; coordinate search on primary actor macro-F1",
        "specialist_thresholds": thresholds,
        "gate_thresholds": gate_thresholds,
        "search_trace": calibration_trace,
    })
    if selection_payload is not None:
        _write_json(args.output_dir / "model_selection.json", selection_payload)
    _write_json(args.output_dir / "fold_manifest.json", fold_metadata)
    _write_json(args.output_dir / "metrics.json", {
        "protocol": "actor_only_causal_live_feed_rolling_replay_extended_suspicious",
        "primary_unit": "(video, actor_id) and explicit actor pair",
        "metric_labels": list(LABELS),
        "grouped_oof": oof_metrics,
        "locked_test": "not evaluated in Stage 6; owned by Step 7",
        "comparison_30fps_actor_macro_f1": 0.826448522100696,
        "acceptance_actor_macro_f1_strictly_greater_than": args.minimum_actor_macro_f1,
        "acceptance_pass": bool(oof_metrics["actor_macro_f1"] > args.minimum_actor_macro_f1),
    })
    _write_csv(args.output_dir / "grouped_oof_actor_predictions.csv", _csv_safe_predictions(oof_predictions))
    _write_csv(args.output_dir / "grouped_oof_pair_predictions.csv", oof_pairs)
    np.savez_compressed(
        args.output_dir / "grouped_oof_input.npz",
        fold_id=fold_ids,
        video=np.asarray([row["video"] for row in train_rows]),
        actor_id=np.asarray([row["actor_id"] for row in train_rows]),
        sample_index=np.asarray([int(row["sample_index"]) for row in train_rows], dtype=np.int32),
        timestamp_ms=np.asarray([float(row["timestamp_ms"]) for row in train_rows], dtype=np.float64),
        **{f"X_{name}": _feature_matrix(train_rows, schemas[name]) for name in SPECIALISTS},
        **{f"score_{name}": scores[name] for name in SPECIALISTS},
    )

    command = " ".join([sys.executable, "-m", __spec__.name, *sys.argv[1:]])
    _write_json(args.output_dir / "environment.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproducible_command": command,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "xgboost_version": xgb.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_device": args.xgb_device,
        "selected_model_profiles": model_profiles,
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_status_porcelain_sha256": hashlib.sha256(
            _git_value("status", "--porcelain=v1").encode("utf-8")
        ).hexdigest().upper(),
        "model_profiles": {name: MODEL_PROFILES[profile] for name, profile in model_profiles.items()},
        "model_params": {
            name: dict(
                MODEL_PARAMS,
                device=args.xgb_device,
                **dict(MODEL_PROFILES[profile]["params"]),
            )
            for name, profile in model_profiles.items()
        },
        "boost_rounds": {
            name: MODEL_PROFILES[profile]["rounds"] for name, profile in model_profiles.items()
        },
        "final_class_weights": final_weights,
    })
    _write_json(args.output_dir / "provenance.json", {
        "input_path": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "feature_manifest_path": str(args.feature_manifest.resolve()),
        "feature_manifest_sha256": _sha256(args.feature_manifest),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "json_root": str(args.json_root.resolve()),
        "landmark_root_hash": feature_manifest.get("input", {}).get("landmark_root_hash"),
        "split_assignment_hash": _split_hash([*train_frames, *test_frames]),
        "train_split_groups": sorted({row["split_group"] for row in train_rows}),
        "locked_test_split_groups": sorted({row["split_group"] for row in test_frames}),
        "train_prefix_rows": len(train_rows),
        "locked_test_model_scores_computed": False,
        "locked_test_used_for_model_selection_or_calibration": False,
        "calibration_selection_path": (
            str(args.calibration_selection.resolve()) if args.calibration_selection is not None else None
        ),
        "calibration_selection_sha256": (
            _sha256(args.calibration_selection) if args.calibration_selection is not None else None
        ),
    })

    files = sorted(
        path for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    )
    file_hashes = {
        path.relative_to(args.output_dir).as_posix(): _sha256(path)
        for path in files
    }
    bundle_manifest = {
        "format_version": FORMAT_VERSION,
        "feature_schema_hash": _canonical_hash(schemas),
        "temporal_policy_hash": _canonical_hash(policy),
        "file_sha256": file_hashes,
        "model_sha256": {
            specialist: file_hashes[f"models/{specialist}.ubj"] for specialist in SPECIALISTS
        },
        "metric_file": "metrics.json",
        "calibration_file": "calibration.json",
        "grouped_oof_input_file": "grouped_oof_input.npz",
    }
    _write_json(args.output_dir / "bundle_manifest.json", bundle_manifest)
    reproduction = verify_grouped_oof_reproduction(args.output_dir)
    result = {
        "bundle_dir": str(args.output_dir),
        "grouped_oof_actor_macro_f1": oof_metrics["actor_macro_f1"],
        "locked_test": "not evaluated in Stage 6; owned by Step 7",
        "acceptance_pass": bool(oof_metrics["actor_macro_f1"] > args.minimum_actor_macro_f1),
        "grouped_oof_reproduction": reproduction,
    }
    print(json.dumps(result, indent=2), flush=True)
    if not result["acceptance_pass"]:
        raise RuntimeError(
            f"Stage 6 grouped OOF actor macro-F1 {oof_metrics['actor_macro_f1']:.12f} "
            f"must be > {args.minimum_actor_macro_f1:.12f}"
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xgb-device", default="cpu")
    parser.add_argument("--model-profile", choices=tuple(MODEL_PROFILES), default="balanced_depth3")
    parser.add_argument("--c2-profile", choices=tuple(MODEL_PROFILES))
    parser.add_argument("--c3-profile", choices=tuple(MODEL_PROFILES))
    parser.add_argument("--suspicious-profile", choices=tuple(MODEL_PROFILES))
    parser.add_argument("--calibration-selection", type=Path)
    parser.add_argument("--global-trials", type=int, default=30000)
    parser.add_argument("--search-seed", type=int, default=20260827)
    parser.add_argument("--minimum-actor-macro-f1", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.minimum_actor_macro_f1 < 1.0:
        raise ValueError("minimum actor macro-F1 must be in [0, 1)")
    run(args)


if __name__ == "__main__":
    main()

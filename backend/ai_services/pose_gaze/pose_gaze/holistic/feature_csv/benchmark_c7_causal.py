"""Actor-level causal C7/C5 benchmark using persistent same-hand evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WARMUP_FRAMES = 30
ROLLING_FRAMES = 90
STATS = ("last", "mean", "max", "min", "std")
BASE_FEATURES = (
    "selected_hand_valid",
    "selected_hand_gap_frames",
    "selected_hand_raise",
    "selected_hand_raise_speed",
    "selected_hand_tip_speed",
    "selected_hand_shape_change",
    "selected_hand_coherence",
    "selected_hand_extension_mean",
    "selected_hand_own_side",
    "selected_hand_not_midpoint",
)


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _point(row: dict[str, str], prefix: str, index: int):
    if row.get(f"{prefix}_{index}_valid") != "1":
        return None
    x = _float(row.get(f"{prefix}_{index}_frame_x"))
    y = _float(row.get(f"{prefix}_{index}_frame_y"))
    return (x, y) if x is not None and y is not None else None


def _distance(first, second) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


class C7Stream:
    """Consume one actor frame at a time; retain one coherent gesture hand."""

    def __init__(self, *, baseline_frames: int = WARMUP_FRAMES) -> None:
        self.baseline_frames = int(baseline_frames)
        self.frame_count = 0
        self.histories: dict[str, deque[dict[str, float]]] = {
            side: deque(maxlen=ROLLING_FRAMES) for side in ("left", "right")
        }
        self.state = {
            side: {
                "baseline_y": [], "valid_count": 0, "previous_wrist": None,
                "previous_tips": None, "previous_shape": None,
                "baseline_shapes": [], "previous_timestamp": None, "gap": 0,
                "strongest_evidence": 0.0,
            }
            for side in ("left", "right")
        }
        self.selected_side: str | None = None

    def update(self, row: dict[str, str]) -> dict[str, float] | None:
        self.frame_count += 1
        shoulder_left = _point(row, "pose", 11)
        shoulder_right = _point(row, "pose", 12)
        if shoulder_left is None or shoulder_right is None:
            return None
        scale = max(_distance(shoulder_left, shoulder_right), 1.0)
        timestamp = _float(row.get("timestamp_ms")) or 0.0
        side_features: dict[str, dict[str, float]] = {}
        for side in ("left", "right"):
            hand = f"{side}_hand"
            wrist = _point(row, hand, 0)
            palm = _point(row, hand, 9) or _point(row, hand, 5)
            tips = [_point(row, hand, index) for index in (4, 8, 12, 16, 20)]
            valid = wrist is not None and palm is not None and all(tips)
            state = self.state[side]
            if not valid:
                state["gap"] += 1
                side_features[side] = {name: 0.0 for name in BASE_FEATURES}
                side_features[side]["selected_hand_gap_frames"] = float(state["gap"])
                continue
            state["gap"] = 0
            state["valid_count"] += 1
            if self.frame_count <= self.baseline_frames:
                state["baseline_y"].append(wrist[1])
            baseline_y = float(np.median(state["baseline_y"])) if state["baseline_y"] else wrist[1]
            dt = max((timestamp - state["previous_timestamp"]) / 1000.0, 1e-3) if state["previous_timestamp"] is not None else None
            raise_speed = max(0.0, state["previous_wrist"][1] - wrist[1]) / scale / dt if dt and state["previous_wrist"] else 0.0
            tip_speed = max((_distance(before, after) for before, after in zip(state["previous_tips"], tips)), default=0.0) / scale / dt if dt and state["previous_tips"] else 0.0
            palm_scale = max(_distance(wrist, palm), scale * 0.05)
            shape = np.asarray([_distance(wrist, tip) / palm_scale for tip in tips], dtype=np.float64)
            if self.frame_count <= self.baseline_frames:
                state["baseline_shapes"].append(shape)
            baseline_shape = np.median(np.asarray(state["baseline_shapes"]), axis=0) if state["baseline_shapes"] else shape
            shape_change = float(np.mean(np.abs(shape - baseline_shape)))
            coherence = 0.0
            if state["previous_wrist"] is not None and state["previous_tips"] is not None:
                wrist_dx = wrist[0] - state["previous_wrist"][0]
                wrist_dy = wrist[1] - state["previous_wrist"][1]
                wrist_norm = math.hypot(wrist_dx, wrist_dy)
                values = []
                if wrist_norm > 1e-6:
                    for before, after in zip(state["previous_tips"], tips):
                        tip_dx, tip_dy = after[0] - before[0], after[1] - before[1]
                        tip_norm = math.hypot(tip_dx, tip_dy)
                        if tip_norm > 1e-6:
                            values.append((tip_dx * wrist_dx + tip_dy * wrist_dy) / (tip_norm * wrist_norm))
                coherence = float(np.mean([value >= 0.25 for value in values])) if values else 0.0
            midpoint = _float(row.get("pair_mid_x_0"))
            actor_side = _float(row.get("actor_side")) or 0.0
            margin = (_float(row.get("pair_margin_10pct")) or 0.0) / scale
            selected_point = min(tips, key=lambda point: abs(point[0] - midpoint)) if midpoint is not None else wrist
            own_side = actor_side * (selected_point[0] - midpoint) / scale if midpoint is not None and actor_side else 0.0
            not_midpoint = own_side > margin if margin > 0.0 else own_side > 0.0
            raise_displacement = max(0.0, baseline_y - wrist[1]) / scale
            side_features[side] = {
                "selected_hand_valid": 1.0,
                "selected_hand_gap_frames": 0.0,
                "selected_hand_raise": raise_displacement,
                "selected_hand_raise_speed": raise_speed,
                "selected_hand_tip_speed": tip_speed,
                "selected_hand_shape_change": shape_change,
                "selected_hand_coherence": coherence,
                "selected_hand_extension_mean": float(np.mean(shape)),
                "selected_hand_own_side": own_side,
                "selected_hand_not_midpoint": float(not_midpoint),
            }
            evidence = float(not_midpoint) * (
                raise_displacement
                + shape_change
                + 0.10 * min(tip_speed, 3.0)
                + 0.20 * coherence
            )
            state["strongest_evidence"] = max(state["strongest_evidence"], evidence)
            state["previous_wrist"] = wrist
            state["previous_tips"] = tips
            state["previous_shape"] = shape
            state["previous_timestamp"] = timestamp

        candidate = max(
            ("left", "right"),
            key=lambda side: (
                self.state[side]["strongest_evidence"],
                self.state[side]["valid_count"],
                side == "left",
            ),
        )
        if self.selected_side is None or (
            self.state[candidate]["strongest_evidence"]
            > self.state[self.selected_side]["strongest_evidence"]
        ):
            self.selected_side = candidate
        selected = self.selected_side
        for side in ("left", "right"):
            if side_features[side]["selected_hand_valid"] > 0.0:
                self.histories[side].append(side_features[side])
        history = self.histories[selected]
        if not history:
            return {
                f"{name}__{stat}": 0.0
                for name in BASE_FEATURES for stat in STATS
            }
        output: dict[str, float] = {}
        for name in BASE_FEATURES:
            values = [frame[name] for frame in history]
            output[f"{name}__last"] = values[-1]
            output[f"{name}__mean"] = float(np.mean(values))
            output[f"{name}__max"] = max(values)
            output[f"{name}__min"] = min(values)
            output[f"{name}__std"] = float(np.std(values))
        output["selected_hand_valid__last"] = side_features[selected]["selected_hand_valid"]
        output["selected_hand_gap_frames__last"] = side_features[selected]["selected_hand_gap_frames"]
        return output


def _truth(manifest_row: dict[str, str], actor_id: str) -> str:
    actors = {str(item) for item in json.loads(manifest_row.get("action_actor_ids") or "[]")}
    return manifest_row["class_code"] if actor_id in actors else "c5"


def _pairs(manifest_row: dict[str, str]) -> list[tuple[str, str]]:
    output = []
    for item in json.loads(manifest_row.get("interaction_pairs") or "[]"):
        pair = (str(item["source"]), str(item["peer"]))
        if frozenset(pair) not in {frozenset(value) for value in output}:
            output.append(pair)
    return output


def _load(input_path: Path, manifest_path: Path):
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = {row["clip_id"]: row for row in csv.DictReader(handle)}
    actors = defaultdict(list)
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = manifest.get(row["clip_id"])
            if source is None or source["class_code"] not in {"c7", "c5"}:
                continue
            truth = _truth(source, row["actor_id"])
            if truth not in {"c7", "c5"}:
                continue
            row["truth"] = truth
            actors[(row["clip_id"], row["actor_id"])].append(row)
    return manifest, actors


def _prefix_rows(actors):
    output = []
    for (clip_id, actor_id), rows in sorted(actors.items()):
        rows.sort(key=lambda row: int(row["source_frame_index"]))
        stream = C7Stream()
        for row in rows:
            features = stream.update(row)
            if features is None or stream.frame_count < WARMUP_FRAMES:
                continue
            output.append({
                "clip_id": clip_id, "actor_id": actor_id, "truth": row["truth"],
                "split": row["split"], "split_group": row["split_group"],
                "frame_index": int(row["source_frame_index"]),
                "timestamp_ms": int(float(row["timestamp_ms"])), **features,
            })
    return output


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=0.5, class_weight="balanced", max_iter=500, random_state=42
    ))


def _actor_balanced_weights(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[(row["clip_id"], row["actor_id"])] += 1
    return np.asarray([
        1.0 / counts[(row["clip_id"], row["actor_id"])] for row in rows
    ], dtype=np.float64)


def _oof(rows, names):
    x = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float32)
    y = np.asarray([row["truth"] == "c7" for row in rows], dtype=np.int8)
    groups = np.asarray([row["split_group"] for row in rows])
    output = np.full(len(rows), np.nan)
    splitter = GroupKFold(n_splits=len(set(groups)))
    for fit, validation in splitter.split(x, y, groups):
        fit_rows = [rows[index] for index in fit]
        model = _model().fit(
            x[fit], y[fit],
            logisticregression__sample_weight=_actor_balanced_weights(fit_rows),
        )
        output[validation] = model.predict_proba(x[validation])[:, 1]
    if not np.isfinite(output).all():
        raise RuntimeError("OOF C7 probabilities are incomplete")
    return output


def _actor_predictions(rows, probabilities, threshold, manifest, *, propagate_pairs=True):
    grouped = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        grouped[(row["clip_id"], row["actor_id"])].append((row, float(probability)))
    pair_scores = {}
    for clip_id in {key[0] for key in grouped} if propagate_pairs else ():
        source = manifest[clip_id]
        for left, right in _pairs(source):
            keys = [(clip_id, left), (clip_id, right)]
            observed = [key for key in keys if key in grouped]
            if len(observed) == 2:
                score = max(max(value for _, value in grouped[key]) for key in observed)
                for key in observed:
                    pair_scores[key] = max(pair_scores.get(key, 0.0), score)
    predictions = []
    for key, sequence in sorted(grouped.items()):
        own_score = max(score for _, score in sequence)
        score = max(own_score, pair_scores.get(key, 0.0))
        first_flag = ""
        evidence_frame = max(sequence, key=lambda item: item[1])[0]["frame_index"]
        if score >= threshold:
            if pair_scores.get(key, 0.0) >= own_score:
                pair_keys = [candidate for candidate in grouped if candidate[0] == key[0] and candidate != key]
                candidates = sequence + [item for pair_key in pair_keys for item in grouped[pair_key]]
            else:
                candidates = sequence
            first_flag = min(
                (row["frame_index"] for row, value in candidates if value >= threshold),
                default=sequence[0][0]["frame_index"],
            )
        predictions.append({
            "clip_id": key[0], "actor_id": key[1], "truth": sequence[0][0]["truth"],
            "prediction": "c7" if score >= threshold else "c5", "score": score,
            "own_score": own_score, "pair_propagated": int(pair_scores.get(key, 0.0) > own_score),
            "first_flag_frame": first_flag, "evidence_frame": evidence_frame,
        })
    return predictions


def _threshold(rows, probabilities, manifest, *, propagate_pairs=True):
    grouped = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        grouped[(row["clip_id"], row["actor_id"])].append(float(probability))
    actor_scores = {key: max(values) for key, values in grouped.items()}
    candidates = set(actor_scores.values())
    for clip_id in {key[0] for key in grouped} if propagate_pairs else ():
        for left, right in _pairs(manifest[clip_id]):
            keys = ((clip_id, left), (clip_id, right))
            if all(key in actor_scores for key in keys):
                candidates.add(max(actor_scores[key] for key in keys))
    candidates = sorted(candidates)
    best_score, best_threshold = -1.0, 0.5
    for threshold in candidates:
        predictions = _actor_predictions(
            rows, probabilities, threshold, manifest,
            propagate_pairs=propagate_pairs,
        )
        score = f1_score(
            [row["truth"] for row in predictions], [row["prediction"] for row in predictions],
            labels=["c7", "c5"], average="macro",
        )
        if score > best_score or (score == best_score and threshold < best_threshold):
            best_score, best_threshold = float(score), float(threshold)
    return best_threshold, best_score


def run(
    input_path: Path, manifest_path: Path, output_dir: Path, *,
    propagate_pairs: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, actors = _load(input_path, manifest_path)
    prefixes = _prefix_rows(actors)
    names = [name for name in prefixes[0] if "__" in name]
    train = [row for row in prefixes if row["split"] == "train"]
    test = [row for row in prefixes if row["split"] == "test"]
    oof = _oof(train, names)
    threshold, train_macro = _threshold(
        train, oof, manifest, propagate_pairs=propagate_pairs
    )
    model = _model().fit(
        np.asarray([[row[name] for name in names] for row in train], dtype=np.float32),
        np.asarray([row["truth"] == "c7" for row in train], dtype=np.int8),
        logisticregression__sample_weight=_actor_balanced_weights(train),
    )
    probabilities = model.predict_proba(
        np.asarray([[row[name] for name in names] for row in test], dtype=np.float32)
    )[:, 1]
    predictions = _actor_predictions(
        test, probabilities, threshold, manifest,
        propagate_pairs=propagate_pairs,
    )
    truth = [row["truth"] for row in predictions]
    predicted = [row["prediction"] for row in predictions]
    report = classification_report(truth, predicted, labels=["c7", "c5"], output_dict=True, zero_division=0)
    metrics = {
        "protocol": "actor_only_c7_causal_live_feed_same_hand_pair_replay",
        "primary_unit": "(video, actor_id)", "metric_labels": ["c7", "c5"],
        "causal": True, "future_frames_used_for_decision": False,
        "active_feature_scope": "pose_and_hand_landmarks_only",
        "persistent_same_hand": True,
        "training_frame_weights": "equal_total_weight_per_train_actor",
        "actor_identity_used_as_model_feature": False,
        "actor_side_contract": "geometry_only_from_pair_layout_not_actor_identity",
        "offline_pair_layout_future_rows": 0,
        "explicit_pair_propagation": propagate_pairs,
        "warmup_frames": WARMUP_FRAMES, "rolling_frames": ROLLING_FRAMES,
        "threshold_train_only": threshold,
        "threshold_calibration": (
            "leave_one_split_group_out_oof_train_actor_max_prefix_with_explicit_pair_propagation"
            if propagate_pairs else
            "leave_one_split_group_out_oof_train_actor_max_prefix_without_pair_propagation"
        ),
        "train_oof_actor_macro_f1": train_macro,
        "test_actor_macro_f1_c7_c5": f1_score(truth, predicted, labels=["c7", "c5"], average="macro"),
        "actor_metrics": {label: report[label] for label in ("c7", "c5")},
        "actor_confusion_matrix": confusion_matrix(truth, predicted, labels=["c7", "c5"]).tolist(),
        "train_actor_count": len({(row["clip_id"], row["actor_id"]) for row in train}),
        "test_actor_count": len(predictions),
        "forbidden_feature_fields": [
            "truth", "actor_truth", "class_code", "manifest_class_code", "action_start_s",
            "action_end_s", "within_action_interval", "actor_id", "clip_id", "split_group",
        ],
    }
    with (output_dir / "c7_actor_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    (output_dir / "c7_actor_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "c7_feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    with (output_dir / "c7_hand_specialist.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-pair-propagation", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(
        args.input, args.manifest, args.output_dir,
        propagate_pairs=not args.no_pair_propagation,
    ), indent=2))


if __name__ == "__main__":
    main()

"""Actor-level causal, pose-only C1 benchmark.

The specialist is trained on original-label C1/C5 train actors only. Object
detections are deliberately outside this entrypoint because the current
checkpoint is not stable enough for C1 evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = (
    "head_below_shoulders",
    "head_delta",
    "left_wrist_to_hip",
    "right_wrist_to_hip",
    "left_wrist_below_hip",
    "right_wrist_below_hip",
    "left_wrist_speed",
    "right_wrist_speed",
    "left_lower_path",
    "right_lower_path",
    "wrist_pair_ratio",
    "elbow_pair_ratio",
)
ROLLING_FRAMES = 90
WARMUP_FRAMES = 30


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _point(row: dict[str, str], index: int) -> tuple[float, float] | None:
    if row.get(f"pose_{index}_valid") != "1":
        return None
    x = _float(row.get(f"pose_{index}_frame_x"))
    y = _float(row.get(f"pose_{index}_frame_y"))
    return (x, y) if x is not None and y is not None else None


def _mid(first: tuple[float, float] | None, second: tuple[float, float] | None):
    if first is None or second is None:
        return None
    return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)


def _dist(first: tuple[float, float] | None, second: tuple[float, float] | None):
    if first is None or second is None:
        return None
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _truth(manifest_row: dict[str, str], actor_id: str) -> str:
    action = {str(item) for item in json.loads(manifest_row.get("action_actor_ids") or "[]")}
    return manifest_row["class_code"] if actor_id in action else "c5"


@dataclass
class PoseStream:
    baseline_frames: int = WARMUP_FRAMES

    def __post_init__(self):
        self.frame_count = 0
        self.head_baseline: list[float] = []
        self.previous: dict[str, tuple[float, tuple[float, float]] | None] = {"left": None, "right": None}
        self.lower_paths: dict[str, deque[tuple[float, tuple[float, float], float]]] = {
            "left": deque(), "right": deque()
        }
        self.history: deque[dict[str, float]] = deque(maxlen=ROLLING_FRAMES)

    def update(self, row: dict[str, str]) -> dict[str, float] | None:
        self.frame_count += 1
        points = {index: _point(row, index) for index in (0, 11, 12, 13, 14, 15, 16, 23, 24)}
        shoulder_mid, hip_mid = _mid(points[11], points[12]), _mid(points[23], points[24])
        sw, torso = _dist(points[11], points[12]), _dist(shoulder_mid, hip_mid)
        scales = [value for value in (sw, torso) if value and value > 1e-6]
        if shoulder_mid is None or hip_mid is None or not scales:
            return None
        scale = float(np.median(scales))
        timestamp = _float(row.get("timestamp_ms")) or 0.0
        head = (points[0][1] - shoulder_mid[1]) / scale if points[0] else 0.0
        baseline_head = float(np.median(self.head_baseline)) if self.head_baseline else head
        current: dict[str, float] = {
            "head_below_shoulders": head,
            "head_delta": head - baseline_head,
        }
        for side, wrist_index, hip_index in (("left", 15, 23), ("right", 16, 24)):
            wrist, hip = points[wrist_index], points[hip_index]
            if wrist is None or hip is None:
                current[f"{side}_wrist_to_hip"] = 0.0
                current[f"{side}_wrist_below_hip"] = 0.0
                current[f"{side}_wrist_speed"] = 0.0
                current[f"{side}_lower_path"] = 0.0
                continue
            distance = (_dist(wrist, hip) or 0.0) / scale
            below = (wrist[1] - hip[1]) / scale
            previous = self.previous[side]
            speed = 0.0
            if previous is not None:
                dt = max((timestamp - previous[0]) / 1000.0, 1e-3)
                speed = (_dist(wrist, previous[1]) or 0.0) / scale / dt
            self.previous[side] = (timestamp, wrist)
            lower = distance <= 0.55 and below >= -0.25
            path = self.lower_paths[side]
            if lower:
                path.append((timestamp, wrist, scale))
            else:
                path.clear()
            while path and path[0][0] < timestamp - 750.0:
                path.popleft()
            path_length = sum(
                (_dist(before[1], after[1]) or 0.0)
                for before, after in zip(path, list(path)[1:])
            ) / float(np.median([item[2] for item in path])) if len(path) >= 2 else 0.0
            current[f"{side}_wrist_to_hip"] = distance
            current[f"{side}_wrist_below_hip"] = below
            current[f"{side}_wrist_speed"] = speed
            current[f"{side}_lower_path"] = path_length
        current["wrist_pair_ratio"] = (_dist(points[15], points[16]) or 0.0) / max(sw or 0.0, 1e-6)
        current["elbow_pair_ratio"] = (_dist(points[13], points[14]) or 0.0) / max(sw or 0.0, 1e-6)
        if self.frame_count <= self.baseline_frames:
            self.head_baseline.append(head)
        self.history.append(current)
        output: dict[str, float] = {}
        for name in FEATURES:
            values = [frame[name] for frame in self.history]
            output[f"{name}__last"] = values[-1]
            output[f"{name}__mean"] = float(np.mean(values))
            output[f"{name}__max"] = max(values)
            output[f"{name}__min"] = min(values)
            output[f"{name}__std"] = float(np.std(values))
        return output


def _load_inputs(input_path: Path, manifest_path: Path, target_class: str = "c1"):
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = {row["clip_id"]: row for row in csv.DictReader(handle)}
    actors: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = manifest.get(row["clip_id"])
            if source is None or source["class_code"] not in {target_class, "c5"}:
                continue
            truth = _truth(source, row["actor_id"])
            if truth not in {target_class, "c5"}:
                continue
            row["actor_truth"] = truth
            actors[(row["clip_id"], row["actor_id"])].append(row)
    return manifest, actors


def _prefix_rows(actors):
    prefixes = []
    for (clip_id, actor_id), rows in sorted(actors.items()):
        rows.sort(key=lambda item: int(item["source_frame_index"]))
        stream = PoseStream()
        for row in rows:
            features = stream.update(row)
            if features is None or stream.frame_count < WARMUP_FRAMES:
                continue
            prefixes.append({
                "clip_id": clip_id,
                "actor_id": actor_id,
                "split": row["split"],
                "split_group": row["split_group"],
                "truth": row["actor_truth"],
                "frame_index": int(row["source_frame_index"]),
                "timestamp_ms": int(float(row["timestamp_ms"])),
                **features,
            })
    return prefixes


def _threshold(train_rows, probabilities, target_class: str):
    actor_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    actor_truth: dict[tuple[str, str], str] = {}
    for row, probability in zip(train_rows, probabilities, strict=True):
        key = (row["clip_id"], row["actor_id"])
        actor_scores[key].append(float(probability))
        actor_truth[key] = row["truth"]
    candidates = sorted({score for scores in actor_scores.values() for score in scores})
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        truth = [actor_truth[key] for key in actor_scores]
        prediction = [target_class if max(actor_scores[key]) >= threshold else "c5" for key in actor_scores]
        objective = f1_score(truth, prediction, labels=[target_class, "c5"], average="macro")
        candidate = (objective, threshold)
        if candidate > best:
            best = candidate
    return float(best[1]), float(best[0])


def _model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=500,
            random_state=42,
        ),
    )


def _oof_train_probabilities(train_rows, feature_names, target_class: str):
    x = np.asarray(
        [[row[name] for name in feature_names] for row in train_rows],
        dtype=np.float32,
    )
    y = np.asarray([row["truth"] == target_class for row in train_rows], dtype=np.int8)
    groups = np.asarray([row["split_group"] for row in train_rows])
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError("C1 calibration requires at least two train split groups")
    output = np.full(len(train_rows), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=len(unique_groups))
    for fit_indices, validation_indices in splitter.split(x, y, groups):
        fold_model = _model()
        fold_model.fit(x[fit_indices], y[fit_indices])
        output[validation_indices] = fold_model.predict_proba(x[validation_indices])[:, 1]
    if not np.isfinite(output).all():
        raise RuntimeError("OOF calibration left non-finite probabilities")
    return output


def run(
    input_path: Path,
    manifest_path: Path,
    output_dir: Path,
    feature_family: str = "combined",
    target_class: str = "c1",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if target_class not in {"c1", "c4"}:
        raise ValueError("target_class must be c1 or c4")
    _, actors = _load_inputs(input_path, manifest_path, target_class)
    prefixes = _prefix_rows(actors)
    feature_names = [key for key in prefixes[0] if "__" in key]
    if feature_family == "head_only":
        feature_names = [name for name in feature_names if name.startswith("head_")]
    elif feature_family == "hands_only":
        feature_names = [name for name in feature_names if not name.startswith("head_")]
    elif feature_family != "combined":
        raise ValueError(f"Unknown feature family: {feature_family}")
    train = [row for row in prefixes if row["split"] == "train"]
    test = [row for row in prefixes if row["split"] == "test"]
    x_train = np.asarray([[row[name] for name in feature_names] for row in train], dtype=np.float32)
    y_train = np.asarray([row["truth"] == target_class for row in train], dtype=np.int8)
    oof_probability = _oof_train_probabilities(train, feature_names, target_class)
    threshold, train_macro = _threshold(train, oof_probability, target_class)
    model = _model()
    model.fit(x_train, y_train)
    x_test = np.asarray([[row[name] for name in feature_names] for row in test], dtype=np.float32)
    probabilities = model.predict_proba(x_test)[:, 1]
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, probability in zip(test, probabilities, strict=True):
        grouped[(row["clip_id"], row["actor_id"])].append((row, float(probability)))
    predictions = []
    for key, sequence in sorted(grouped.items()):
        best_score = 0.0
        predicted = "c5"
        first_flag = None
        evidence_frame = None
        decision_source = "pose_specialist"
        for row, probability in sequence:
            best_score = max(best_score, probability)
            if probability >= threshold and predicted == "c5":
                predicted, first_flag = target_class, row["frame_index"]
            if probability >= best_score:
                evidence_frame = row["frame_index"]
        predictions.append({
            "clip_id": key[0], "actor_id": key[1], "truth": sequence[0][0]["truth"],
            "prediction": predicted, "score": best_score, "first_flag_frame": first_flag if first_flag is not None else "",
            "evidence_frame": evidence_frame if evidence_frame is not None else "", "decision_source": decision_source,
        })
    truth = [row["truth"] for row in predictions]
    pred = [row["prediction"] for row in predictions]
    report = classification_report(truth, pred, labels=[target_class, "c5"], output_dict=True, zero_division=0)
    metrics = {
        "protocol": f"actor_only_{target_class}_causal_live_feed_rolling_replay",
        "primary_unit": "(video, actor_id)",
        "metric_labels": [target_class, "c5"],
        "causal": True,
        "future_frames_used_for_decision": False,
        "manifest_truth_used_as_model_feature": False,
        "active_feature_scope": "pose_only",
        "pose_feature_family": feature_family,
        "object_features_used": False,
        "object_override": None,
        "pose_threshold_train_only": threshold,
        "threshold_calibration": "leave_one_split_group_out_oof_train_actor_max_prefix",
        "train_oof_actor_macro_f1_at_threshold": train_macro,
        "train_split_groups": sorted({row["split_group"] for row in train}),
        f"test_actor_macro_f1_{target_class}_c5": f1_score(truth, pred, labels=[target_class, "c5"], average="macro"),
        "actor_metrics": {label: report[label] for label in (target_class, "c5")},
        "confusion_labels": [target_class, "c5"],
        "actor_confusion_matrix": confusion_matrix(truth, pred, labels=[target_class, "c5"]).tolist(),
        "train_actor_count": len({(row["clip_id"], row["actor_id"]) for row in train}),
        "test_actor_count": len(predictions),
        "object_cue_file": None,
    }
    with (output_dir / f"{target_class}_actor_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    (output_dir / f"{target_class}_actor_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / f"{target_class}_feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    with (output_dir / f"{target_class}_pose_specialist.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-family",
        choices=("combined", "head_only", "hands_only"),
        default="combined",
    )
    parser.add_argument("--target-class", choices=("c1", "c4"), default="c1")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.manifest, args.output_dir, args.feature_family, args.target_class), indent=2))


if __name__ == "__main__":
    main()

"""Actor-level XGBoost classification adapter for the test_media JSON contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from ..feature_csv import behavior_subset_stage2 as behavior
from ..feature_csv.canonical_behavior_features import _frame_rows
from ..feature_csv.temporal_geometry import enrich

# V3 current demo: c7 excluded. To re-enable c7, uncomment the marked C7
# blocks below and point --xgboost-model-dir at an artifact containing its UBJ.
USE_C7 = False


def _attach_face_points(rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    lookup: dict[tuple[int, str], dict[int, tuple[float, float]]] = {}
    for frame in payload.get("frames", []):
        source_index = int(frame.get("source_frame_index", 0))
        for track in frame.get("tracks", []):
            points = {}
            for point in track.get("selected_face_landmarks", []):
                if point.get("frame_x") is not None and point.get("frame_y") is not None:
                    points[int(point["index"])] = (
                        float(point["frame_x"]), float(point["frame_y"])
                    )
            lookup[(source_index, str(track.get("track_id")))] = points
    for row in rows:
        row["_selected_face_points"] = lookup.get(
            (int(row.get("source_frame_index", 0)), str(row.get("track_id", ""))), {}
        )


def _aggregate(rows: list[dict[str, Any]], selected_features: tuple[str, ...]):
    actor_rows, feature_names = behavior.aggregate_actor_rows(rows, selected_features)
    matrix = np.asarray(
        [[behavior.number(row.get(name)) for name in feature_names] for row in actor_rows],
        dtype=np.float32,
    )
    return actor_rows, feature_names, matrix


def classify_landmarks(
    landmarks_path: Path,
    model_dir: Path,
    output_path: Path,
) -> dict[str, dict[str, Any]]:
    """Classify every tracked actor in one full video; no video-level label."""
    payload = json.loads(landmarks_path.read_text(encoding="utf-8"))
    tracks_by_id = {
        str(track.get("track_id"))
        for frame in payload.get("frames", [])
        for track in frame.get("tracks", [])
        if track.get("track_id") is not None
    }
    if not tracks_by_id:
        raise ValueError("No tracked actors found in landmark JSON")

    clip_id = landmarks_path.stem.removesuffix("_landmarks")
    manifest = {
        "clip_id": clip_id,
        "filename": payload.get("input", clip_id),
        "split": "demo",
        "split_group": "demo",
        "class_code": "c2",
        "action_actor_ids": [],
    }
    mapping = {
        track_id: {
            "actor_id": f"student_{int(track_id):02d}",
            "track_id": track_id,
            "spatial_role": "demo_actor",
            "confidence": "demo",
        }
        for track_id in sorted(tracks_by_id, key=lambda value: int(value))
    }
    rows = list(_frame_rows(manifest, mapping, payload))
    for row in rows:
        # canonical CSV consumers read validity flags as strings. Keep the
        # adapter identical to that contract before temporal geometry runs.
        for name, value in list(row.items()):
            if name.endswith("_valid"):
                row[name] = str(int(bool(value)))
    for row in rows:
        # Aggregate helper expects training metadata, but these fields are
        # audit-only placeholders and never enter any XGBoost feature matrix.
        row["actor_truth"] = "c5"
        row["source_actor"] = 0
        row["manifest_class_code"] = "c2"
        row["interaction_peer_ids"] = []
    _attach_face_points(rows, payload)
    rows = enrich(rows, baseline_frames=30)
    rows = behavior.derive_behavior_motion(rows)
    rows = behavior.derive_face_c3_features(rows)
    rows = behavior.derive_finger_motion(rows)
    rows = behavior.derive_hand_shape_and_pair_cues(rows)

    c2_rows, c2_names, c2_matrix = _aggregate(rows, behavior.C2_CUE_FEATURES)
    c3_rows, c3_names, c3_matrix = _aggregate(rows, behavior.C3_FEATURES)
    # C7 disabled for current V3 benchmark.
    # c7_rows, c7_names, c7_matrix = _aggregate(rows, behavior.C7_CUE_FEATURES)

    models = {
        "c2": ("actor_c2_hand_middle_cue.ubj", c2_names, c2_matrix, c2_rows),
        "c3": ("actor_c3_face_tilt_cue.ubj", c3_names, c3_matrix, c3_rows),
    }
    # To enable C7:
    # models["c7"] = ("actor_c7_finger_hand_cue.ubj", c7_names, c7_matrix, c7_rows)
    scores: dict[tuple[str, str], dict[str, float]] = {
        (clip_id, row["actor_id"]): {} for row in c2_rows
    }
    c2_probabilities = None
    for class_code, (filename, names, matrix, actor_rows) in models.items():
        model_path = model_dir / filename
        if not model_path.is_file():
            raise FileNotFoundError(f"XGBoost model not found: {model_path}")
        model = xgb.Booster()
        model.load_model(str(model_path))
        probabilities = model.predict(xgb.DMatrix(matrix, feature_names=names))
        if class_code == "c2":
            c2_probabilities = probabilities
        for row, probability in zip(actor_rows, probabilities, strict=True):
            scores[(clip_id, row["actor_id"])][class_code] = float(probability)

    thresholds = {"c2": 0.7725166082, "c3": 0.6995993257}
    # To enable C7:
    # thresholds["c7"] = 0.7362594604
    if c2_probabilities is None:
        raise RuntimeError("c2 classifier did not produce probabilities")
    pair_c2_keys = behavior.pair_c2_event_keys(c2_rows, c2_probabilities, threshold=0.5)
    c2_event_frames = [
        int(row.get("source_frame_index", 0))
        for row in rows
        if behavior.number(row.get("near_midpoint_pre_cross")) >= 1.0
        or behavior.number(row.get("crossing_indicator")) >= 1.0
    ]
    activation_frame = min(c2_event_frames) if c2_event_frames else 0
    output: dict[str, dict[str, Any]] = {}
    for key, class_scores in scores.items():
        predicted = (
            "c2" if key in pair_c2_keys else max(
                (
                    class_code for class_code, score in class_scores.items()
                    if score >= thresholds[class_code]
                ),
                key=lambda class_code: class_scores[class_code],
                default="c5",
            )
        )
        actor_id = key[1]
        output[actor_id] = {
            "video": clip_id,
            "actor_id": actor_id,
            "predicted_class": predicted,
            "c2_score": class_scores["c2"],
            "c3_score": class_scores["c3"],
            "c7_score": class_scores.get("c7", 0.0),
            "pair_c2_event": int(key in pair_c2_keys),
            "activation_frame": activation_frame if key in pair_c2_keys else 0,
            "frames_scanned": sum(row["actor_id"] == actor_id for row in rows),
            "thresholds_source": "v3 specialist_thresholds_train_only",
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output

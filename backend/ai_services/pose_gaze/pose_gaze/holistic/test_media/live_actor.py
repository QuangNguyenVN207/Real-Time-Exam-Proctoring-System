"""Causal actor classifier for camera/replay inference.

This adapter is deliberately stateful.  It consumes one observed frame at a
time, keeps only a bounded prefix/history, and emits the current actor state.
It never loads a completed landmark JSON or waits for the end of a video.
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..feature_csv.causal_stream import CausalActorWindow, CausalSpecialistState
from ..feature_csv.benchmark_c1_causal import PoseStream
from ..feature_csv.benchmark_c7_causal import C7Stream


class CausalLiveActorClassifier:
    """Run the causal c2/c3 specialist on one live stream."""

    def __init__(
        self,
        model_dir: Path,
        *,
        clip_id: str = "live",
        student_prefix: str = "student_",
        explicit_pairs: Iterable[tuple[str, str]] = (),
        warmup_frames: int = 15,
        window_frames: int = 90,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.clip_id = str(clip_id)
        self.student_prefix = student_prefix
        self.warmup_frames = int(warmup_frames)
        self.window_frames = int(window_frames)
        metrics_path = self.model_dir / "causal_actor_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"causal live artifact is missing: {metrics_path}"
            )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("future_frames_used_for_decision") is not False:
            raise ValueError("model artifact is not certified causal")
        thresholds = metrics.get("specialist_thresholds_train_only", {})
        self.c3_threshold = float(metrics.get("c3_threshold_train_only", thresholds.get("c3", 1.0)))
        self.suspicious_threshold = thresholds.get("suspicious_activity")
        # Extended suspicious artifacts predate the explicit metadata field,
        # but their C3 schema still requires this causal pose contract.
        self.c3_pose_only = (
            metrics.get("c3_feature_family") == "pose_only_contract"
            or "c3_pose_head_peer_delta__mean" in json.loads(
                (self.model_dir / "causal_c3_feature_names.json").read_text(encoding="utf-8")
            )
        )
        self.c2_model, self.c2_names = self._load_model(
            "causal_c2_specialist.ubj", "causal_c2_feature_names.json"
        )
        self.c3_model, self.c3_names = self._load_model(
            "causal_c3_specialist.ubj", "causal_c3_feature_names.json"
        )
        self.suspicious_model = self.suspicious_names = None
        if self.suspicious_threshold is not None:
            self.suspicious_model, self.suspicious_names = self._load_model(
                "causal_suspicious_activity_specialist.ubj",
                "causal_suspicious_activity_feature_names.json",
            )
        self.c2_bases = tuple(name.rsplit("__", 1)[0] for name in self.c2_names[::5])
        self.c3_bases = tuple(name.rsplit("__", 1)[0] for name in self.c3_names[::5])
        self.suspicious_bases = tuple(name.rsplit("__", 1)[0] for name in self.suspicious_names[::5]) if self.suspicious_names else ()
        self._windows_c2: dict[str, CausalActorWindow] = {}
        self._windows_c3: dict[str, CausalActorWindow] = {}
        self._windows_suspicious: dict[str, CausalActorWindow] = {}
        self.gates = metrics.get("gate_thresholds_train_only", {})
        self._state = CausalSpecialistState(
            (), c3_threshold=self.c3_threshold,
            suspicious_threshold=self.suspicious_threshold,
            c3_gate=lambda values: bool(values.get("c3_gate", True)),
            suspicious_gate=lambda values: bool(values.get("suspicious_gate", True)),
        )
        self._explicit_pairs = tuple(
            (str(left), str(right)) for left, right in explicit_pairs
        )
        # Keep only causal baseline plus rolling tail. Never grow with video.
        self._baseline_rows: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.warmup_frames)
        )
        self._tail_rows: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.window_frames)
        )
        self._last_frame: int | None = None
        self._last_timestamp: int | None = None
        self._latest_scores: dict[str, dict[str, float]] = {}
        self._window_sizes: dict[str, int] = {}

    def _load_model(self, model_name: str, names_name: str):
        import xgboost as xgb

        model_path = self.model_dir / model_name
        names_path = self.model_dir / names_name
        if not model_path.is_file() or not names_path.is_file():
            raise FileNotFoundError(
                f"causal model files are missing: {model_path} / {names_path}"
            )
        model = xgb.Booster()
        model.load_model(str(model_path))
        names = tuple(json.loads(names_path.read_text(encoding="utf-8")))
        if len(names) % 5 != 0:
            raise ValueError(f"invalid causal feature schema: {names_path}")
        return model, names

    def _make_rows(
        self,
        *,
        frame_index: int,
        timestamp_ms: int,
        results: Iterable[Any],
    ) -> list[dict[str, Any]]:
        from ..feature_csv.canonical_behavior_features import _frame_rows

        tracks = [result.to_dict() for result in results]
        mapping: dict[str, dict[str, str]] = {}
        for track in tracks:
            track_id = str(track["track_id"])
            student_id = track.get("student_id") or f"{self.student_prefix}{int(track_id):02d}"
            mapping[student_id] = {
                "actor_id": student_id,
                "track_id": track_id,
                "spatial_role": "live_actor",
                "confidence": "live",
            }
        manifest = {
            "clip_id": self.clip_id,
            "filename": self.clip_id,
            "split": "live",
            "split_group": "live",
            "class_code": "c5",
            "action_actor_ids": [],
        }
        payload = {
            "frames": [{
                "source_frame_index": int(frame_index),
                "frame_id": int(frame_index) + 1,
                "timestamp_ms": int(timestamp_ms),
                "tracks": tracks,
            }]
        }
        rows = list(_frame_rows(manifest, mapping, payload))
        for row in rows:
            actor_id = str(row["actor_id"])
            peers = []
            for left, right in self._explicit_pairs:
                if actor_id == left:
                    peers.append(right)
                elif actor_id == right:
                    peers.append(left)
            for name, value in list(row.items()):
                if name.endswith("_valid"):
                    row[name] = str(int(bool(value)))
            row.update({
                "actor_truth": "c5",
                "actor_label": "c5",
                "source_actor": 0,
                "manifest_class_code": "c5",
                "interaction_peer_ids": json.dumps(peers),
                "excluded_source": False,
            })
        # The face points are transient model input metadata, matching the
        # offline feature contract; they are derived from this frame only.
        lookup: dict[tuple[int, str], dict[int, tuple[float, float]]] = {}
        for track in tracks:
            lookup[(frame_index, str(track["track_id"]))] = {
                int(point["index"]): (
                    float(point["frame_x"]), float(point["frame_y"])
                )
                for point in track.get("selected_face_landmarks", [])
                if point.get("frame_x") is not None and point.get("frame_y") is not None
            }
        for row in rows:
            row["_selected_face_points"] = lookup.get(
                (frame_index, str(row.get("track_id", ""))), {}
            )
        return rows

    def _bounded_prefix(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for actor_id in self._tail_rows:
            keep = list(self._baseline_rows[actor_id]) + list(self._tail_rows[actor_id])
            seen: set[int] = set()
            for row in keep:
                frame = int(row["source_frame_index"])
                if frame not in seen:
                    output.append(dict(row))
                    seen.add(frame)
        return output

    @staticmethod
    def _latest(rows: list[dict[str, Any]], frame_index: int) -> dict[str, dict[str, Any]]:
        return {
            str(row["actor_id"]): row
            for row in rows
            if int(row["source_frame_index"]) == int(frame_index)
        }

    def update(self, *, frame_index: int, timestamp_ms: int, results: Iterable[Any]):
        import xgboost as xgb
        from ..feature_csv import behavior_subset_stage2 as behavior
        from ..feature_csv.temporal_geometry import enrich

        frame_index, timestamp_ms = int(frame_index), int(timestamp_ms)
        if self._last_frame is not None and frame_index <= self._last_frame:
            raise ValueError("live classifier received a non-increasing frame")
        if self._last_timestamp is not None and timestamp_ms < self._last_timestamp:
            raise ValueError("live classifier received a decreasing timestamp")
        self._last_frame, self._last_timestamp = frame_index, timestamp_ms
        current_rows = self._make_rows(
            frame_index=frame_index, timestamp_ms=timestamp_ms, results=results
        )
        for row in current_rows:
            actor_id = str(row["actor_id"])
            if len(self._baseline_rows[actor_id]) < self.warmup_frames:
                self._baseline_rows[actor_id].append(dict(row))
            self._tail_rows[actor_id].append(dict(row))
        prefix = enrich(self._bounded_prefix(), baseline_frames=self.warmup_frames)
        if self.c3_pose_only:
            prefix = behavior.derive_c3_pose_contract(prefix, baseline_frames=self.warmup_frames)
        prefix = behavior._head_pnp_features(prefix)
        prefix = behavior.derive_behavior_motion(prefix)
        prefix = behavior.derive_face_c3_features(prefix)
        prefix = behavior.derive_finger_motion(prefix)
        prefix = behavior.derive_hand_shape_and_pair_cues(prefix)
        if self.suspicious_names:
            prefix = behavior.derive_strict_c2_c3_suspicious_cues(prefix, baseline_frames=self.warmup_frames)
        latest = self._latest(prefix, frame_index)
        scores, midpoint = {}, {}
        for actor_id, row in latest.items():
            self._state.register_actor(actor_id)
            c2_window = self._windows_c2.setdefault(actor_id, CausalActorWindow(actor_id, self.c2_bases, max_frames=self.window_frames))
            c3_window = self._windows_c3.setdefault(actor_id, CausalActorWindow(actor_id, self.c3_bases, max_frames=self.window_frames))
            c2_state = c2_window.update(frame_index=frame_index, timestamp_ms=timestamp_ms, features=row)
            c3_state = c3_window.update(frame_index=frame_index, timestamp_ms=timestamp_ms, features=row)
            suspicious_state = None
            if self.suspicious_names:
                suspicious_state = self._windows_suspicious.setdefault(actor_id, CausalActorWindow(actor_id, self.suspicious_bases, max_frames=self.window_frames)).update(frame_index=frame_index, timestamp_ms=timestamp_ms, features=row)
            self._window_sizes[actor_id] = min(c2_state.window_size, c3_state.window_size)
            if min(c2_state.window_size, c3_state.window_size) < self.warmup_frames:
                continue
            c2 = float(self.c2_model.predict(xgb.DMatrix(np.asarray([[c2_state.features[name] for name in self.c2_names]], dtype=np.float32)))[0])
            c3 = float(self.c3_model.predict(xgb.DMatrix(np.asarray([[c3_state.features[name] for name in self.c3_names]], dtype=np.float32)))[0])
            scores[actor_id] = {"c2": c2, "c3": c3}
            scores[actor_id]["c3_gate"] = (
                row.get("hand_motion", 0.0) <= self.gates.get("c3_motion_ceiling", float("inf"))
                and row.get("c3_pose_head_peer_delta", 0.0) >= self.gates.get("c3_side_floor", float("-inf"))
                and row.get("strict_head_down_delta", 0.0) <= self.gates.get("c3_down_ceiling", float("inf"))
            )
            if suspicious_state is not None:
                scores[actor_id]["suspicious_activity"] = float(self.suspicious_model.predict(xgb.DMatrix(np.asarray([[suspicious_state.features[name] for name in self.suspicious_names]], dtype=np.float32)))[0])
                scores[actor_id]["suspicious_gate"] = (
                    row.get("strict_head_down_delta", 0.0) >= self.gates.get("suspicious_down_floor", float("inf"))
                    and max(row.get("hand_motion", 0.0), row.get("finger_motion", 0.0)) >= self.gates.get("suspicious_motion_floor", float("inf"))
                    and row.get("strict_hand_below_hip", 0.0) >= self.gates.get("suspicious_lower_floor", float("inf"))
                    and row.get("strict_own_side_outside_midpoint", 0.0) >= 1.0
                )
            midpoint[actor_id] = row.get("near_midpoint_pre_cross", 0.0)
        self._latest_scores.update(scores)
        self._state.update(frame_index=frame_index, timestamp_ms=timestamp_ms, scores_by_actor=scores, explicit_pairs=self._explicit_pairs, near_midpoint_by_actor=midpoint)
        return self._decision_output(scores)

    def _decision_output(self, scores):
        return {
            actor_id: {
                "actor_id": actor_id, "predicted_class": decision.class_code,
                "c2_score": scores.get(actor_id, self._latest_scores.get(actor_id, {})).get("c2", ""),
                "c3_score": scores.get(actor_id, self._latest_scores.get(actor_id, {})).get("c3", ""),
                "suspicious_activity_score": scores.get(actor_id, self._latest_scores.get(actor_id, {})).get("suspicious_activity", ""),
                "warmup_frames_seen": self._window_sizes.get(actor_id, 0),
                "warmup_frames_required": self.warmup_frames,
                "evidence_score": decision.evidence_score if decision.evidence_score is not None else "",
                "evidence_frame_index": decision.evidence_frame_index if decision.evidence_frame_index is not None else "",
                "first_flag_frame_index": decision.first_flag_frame_index if decision.first_flag_frame_index is not None else "",
                "first_flag_timestamp_ms": decision.first_flag_timestamp_ms if decision.first_flag_timestamp_ms is not None else "",
                "causal": True,
            }
            for actor_id, decision in self._state.decisions().items()
        }

    def final_decisions(self):
        return self._decision_output({})

    def reset(self) -> None:
        """Start a new causal observation without reopening the camera."""
        self._windows_c2.clear()
        self._windows_c3.clear()
        self._windows_suspicious.clear()
        self._state = CausalSpecialistState((), c3_threshold=self.c3_threshold)
        self._baseline_rows.clear()
        self._tail_rows.clear()
        self._last_frame = None
        self._last_timestamp = None
        self._latest_scores.clear()
        self._window_sizes.clear()


class CausalPoseActorClassifier:
    """Causal pose-only suspicious-activity specialist.

    C1 and C4 remain internal evidence specialists. Pose does not identify the
    object needed to distinguish those classes, so the public decision is the
    shared ``suspicious_activity`` label until object detection is fused.
    """

    def __init__(
        self,
        model_dirs: dict[str, Path],
        *,
        student_prefix: str = "student_",
        warmup_frames: int = 30,
    ) -> None:
        self.student_prefix = student_prefix
        self.warmup_frames = int(warmup_frames)
        self._specialists = {
            class_code: self._load(Path(model_dir), class_code)
            for class_code, model_dir in model_dirs.items()
        }
        self._streams: dict[tuple[str, str], PoseStream] = {}
        self._decisions: dict[str, dict[str, Any]] = {}
        self._actors: set[str] = set()
        self._last_frame: int | None = None
        self._last_timestamp: int | None = None

    @staticmethod
    def _load(model_dir: Path, class_code: str) -> dict[str, Any]:
        metrics_path = model_dir / f"{class_code}_actor_metrics.json"
        names_path = model_dir / f"{class_code}_feature_names.json"
        model_path = model_dir / f"{class_code}_pose_specialist.pkl"
        for path in (metrics_path, names_path, model_path):
            if not path.is_file():
                raise FileNotFoundError(f"causal pose artifact is missing: {path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("causal") is not True or metrics.get("future_frames_used_for_decision") is not False:
            raise ValueError(f"{class_code} artifact is not certified causal")
        with warnings.catch_warnings(), model_path.open("rb") as handle:
            # We read stable numeric scaler/logistic attributes immediately;
            # inference does not call the unpickled sklearn estimators.
            warnings.simplefilter("ignore")
            pipeline = pickle.load(handle)
        scaler, classifier = pipeline.steps[0][1], pipeline.steps[-1][1]
        return {
            "threshold": float(metrics["pose_threshold_train_only"]),
            "feature_names": tuple(json.loads(names_path.read_text(encoding="utf-8"))),
            "mean": np.asarray(scaler.mean_, dtype=np.float64),
            "scale": np.asarray(scaler.scale_, dtype=np.float64),
            "coef": np.asarray(classifier.coef_[0], dtype=np.float64),
            "intercept": float(classifier.intercept_[0]),
        }

    @staticmethod
    def _row(track: dict[str, Any], timestamp_ms: int) -> dict[str, str]:
        row = {"timestamp_ms": str(int(timestamp_ms))}
        for point in track.get("pose_landmarks", []):
            index = int(point["index"])
            x, y = point.get("frame_x"), point.get("frame_y")
            valid = x is not None and y is not None
            row[f"pose_{index}_valid"] = "1" if valid else "0"
            if valid:
                row[f"pose_{index}_frame_x"] = str(x)
                row[f"pose_{index}_frame_y"] = str(y)
        return row

    @staticmethod
    def _probability(spec: dict[str, Any], features: dict[str, float]) -> float:
        values = np.asarray([features[name] for name in spec["feature_names"]], dtype=np.float64)
        standardized = (values - spec["mean"]) / np.where(spec["scale"] == 0, 1.0, spec["scale"])
        logit = float(np.dot(standardized, spec["coef"]) + spec["intercept"])
        return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, logit))))

    def update(
        self, *, frame_index: int, timestamp_ms: int, results: Iterable[Any]
    ) -> dict[str, dict[str, Any]]:
        return self.update_tracks(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            tracks=(result.to_dict() for result in results),
        )

    def update_tracks(
        self,
        *,
        frame_index: int,
        timestamp_ms: int,
        tracks: Iterable[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        frame_index, timestamp_ms = int(frame_index), int(timestamp_ms)
        if self._last_frame is not None and frame_index <= self._last_frame:
            raise ValueError("pose classifier received a non-increasing frame")
        if self._last_timestamp is not None and timestamp_ms < self._last_timestamp:
            raise ValueError("pose classifier received a decreasing timestamp")
        self._last_frame, self._last_timestamp = frame_index, timestamp_ms
        current_scores: dict[str, dict[str, float]] = defaultdict(dict)
        seen: set[str] = set()
        for track in tracks:
            track_id = str(track["track_id"])
            actor_id = str(track.get("student_id") or f"{self.student_prefix}{int(track_id):02d}")
            seen.add(actor_id)
            self._actors.add(actor_id)
            row = self._row(track, timestamp_ms)
            for class_code, spec in self._specialists.items():
                stream = self._streams.setdefault((actor_id, class_code), PoseStream())
                features = stream.update(row)
                if features is None or stream.frame_count < self.warmup_frames:
                    continue
                score = self._probability(spec, features)
                current_scores[actor_id][f"{class_code}_score"] = score
                if score < spec["threshold"]:
                    continue
                decision = self._decisions.get(actor_id)
                if decision is None:
                    self._decisions[actor_id] = {
                        "actor_id": actor_id,
                        "predicted_class": "suspicious_activity",
                        "source_specialist": class_code,
                        "evidence_score": score,
                        "evidence_frame_index": frame_index,
                        "first_flag_frame_index": frame_index,
                        "first_flag_timestamp_ms": timestamp_ms,
                        "causal": True,
                    }
                elif score > float(decision["evidence_score"]):
                    decision["source_specialist"] = class_code
                    decision["evidence_score"] = score
                    decision["evidence_frame_index"] = frame_index
        output: dict[str, dict[str, Any]] = {}
        for actor_id in seen:
            selected = self._decisions.get(actor_id, {
                "actor_id": actor_id, "predicted_class": "c5", "evidence_score": "",
                "evidence_frame_index": "", "first_flag_frame_index": "",
                "first_flag_timestamp_ms": "", "causal": True,
            })
            output[actor_id] = {**selected, **current_scores.get(actor_id, {})}
        return output

    def final_decisions(self) -> dict[str, dict[str, Any]]:
        return {
            actor_id: dict(self._decisions.get(actor_id, {
                "actor_id": actor_id, "predicted_class": "c5", "evidence_score": "",
                "evidence_frame_index": "", "first_flag_frame_index": "",
                "first_flag_timestamp_ms": "", "causal": True,
            }))
            for actor_id in self._actors
        }

    def reset(self) -> None:
        """Discard evidence and warmup history while retaining loaded models."""
        self._streams.clear()
        self._decisions.clear()
        self._actors.clear()
        self._last_frame = None
        self._last_timestamp = None


class CausalC7ActorClassifier:
    """Causal per-hand C7 specialist with explicit-pair propagation."""

    def __init__(
        self, model_dir: Path, *, explicit_pairs: Iterable[tuple[str, str]],
        student_prefix: str = "student_", warmup_frames: int = 30,
    ) -> None:
        self.student_prefix = student_prefix
        self.warmup_frames = int(warmup_frames)
        self.explicit_pairs = tuple((str(a), str(b)) for a, b in explicit_pairs)
        if not self.explicit_pairs:
            raise ValueError("C7 live inference requires at least one explicit --live-pair")
        model_dir = Path(model_dir)
        metrics = json.loads((model_dir / "c7_actor_metrics.json").read_text(encoding="utf-8"))
        if metrics.get("causal") is not True or metrics.get("future_frames_used_for_decision") is not False:
            raise ValueError("C7 artifact is not certified causal")
        self.threshold = float(metrics["threshold_train_only"])
        self.names = tuple(json.loads((model_dir / "c7_feature_names.json").read_text(encoding="utf-8")))
        with warnings.catch_warnings(), (model_dir / "c7_hand_specialist.pkl").open("rb") as handle:
            warnings.simplefilter("ignore")
            pipeline = pickle.load(handle)
        scaler, classifier = pipeline.steps[0][1], pipeline.steps[-1][1]
        self.mean = np.asarray(scaler.mean_, dtype=np.float64)
        self.scale = np.asarray(scaler.scale_, dtype=np.float64)
        self.coef = np.asarray(classifier.coef_[0], dtype=np.float64)
        self.intercept = float(classifier.intercept_[0])
        self.streams: dict[str, C7Stream] = {}
        self.decisions: dict[str, dict[str, Any]] = {}
        self.actors: set[str] = set()
        self.layout: dict[frozenset[str], tuple[float, dict[str, int], float]] = {}
        self.last_frame: int | None = None

    def _actor_id(self, track):
        return str(track.get("student_id") or f"{self.student_prefix}{int(track['track_id']):02d}")

    def _lock_layout(self, tracks):
        by_actor = {self._actor_id(track): track for track in tracks}
        for left, right in self.explicit_pairs:
            key = frozenset((left, right))
            if key in self.layout or left not in by_actor or right not in by_actor:
                continue
            centers = {}
            for actor in (left, right):
                bbox = by_actor[actor].get("bbox_xyxy", [])
                if len(bbox) == 4:
                    centers[actor] = (float(bbox[0]) + float(bbox[2])) / 2.0
            if len(centers) != 2:
                continue
            ordered = sorted(centers, key=centers.get)
            midpoint = (centers[ordered[0]] + centers[ordered[1]]) / 2.0
            margin = 0.10 * abs(centers[ordered[1]] - centers[ordered[0]])
            self.layout[key] = (midpoint, {ordered[0]: -1, ordered[1]: 1}, margin)

    @staticmethod
    def _row(track, timestamp_ms, layout):
        row = {"timestamp_ms": str(timestamp_ms)}
        for field, prefix in (("pose_landmarks", "pose"), ("left_hand_landmarks", "left_hand"), ("right_hand_landmarks", "right_hand")):
            for point in track.get(field, []):
                index, x, y = int(point["index"]), point.get("frame_x"), point.get("frame_y")
                valid = x is not None and y is not None
                row[f"{prefix}_{index}_valid"] = "1" if valid else "0"
                if valid:
                    row[f"{prefix}_{index}_frame_x"] = str(x)
                    row[f"{prefix}_{index}_frame_y"] = str(y)
        if layout is not None:
            midpoint, actor_side, margin = layout
            row.update({"pair_mid_x_0": str(midpoint), "actor_side": str(actor_side), "pair_margin_10pct": str(margin)})
        return row

    def _probability(self, features):
        values = np.asarray([features[name] for name in self.names], dtype=np.float64)
        standardized = (values - self.mean) / np.where(self.scale == 0, 1.0, self.scale)
        logit = float(np.dot(standardized, self.coef) + self.intercept)
        return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, logit))))

    def update(self, *, frame_index: int, timestamp_ms: int, results: Iterable[Any]):
        return self.update_tracks(frame_index=frame_index, timestamp_ms=timestamp_ms, tracks=(result.to_dict() for result in results))

    def update_tracks(self, *, frame_index: int, timestamp_ms: int, tracks: Iterable[dict[str, Any]]):
        frame_index, timestamp_ms, tracks = int(frame_index), int(timestamp_ms), list(tracks)
        if self.last_frame is not None and frame_index <= self.last_frame:
            raise ValueError("C7 classifier received a non-increasing frame")
        self.last_frame = frame_index
        self._lock_layout(tracks)
        current_scores = {}
        by_actor = {self._actor_id(track): track for track in tracks}
        for actor_id, track in by_actor.items():
            self.actors.add(actor_id)
            layout = None
            for pair in self.explicit_pairs:
                if actor_id in pair and frozenset(pair) in self.layout:
                    midpoint, sides, margin = self.layout[frozenset(pair)]
                    layout = (midpoint, sides[actor_id], margin)
                    break
            features = self.streams.setdefault(actor_id, C7Stream()).update(self._row(track, timestamp_ms, layout))
            if features is not None and self.streams[actor_id].frame_count >= self.warmup_frames:
                current_scores[actor_id] = self._probability(features)
        for left, right in self.explicit_pairs:
            pair_scores = [(actor, current_scores[actor]) for actor in (left, right) if actor in current_scores]
            if not pair_scores:
                continue
            source, score = max(pair_scores, key=lambda item: item[1])
            if score < self.threshold:
                continue
            for actor_id in (left, right):
                previous = self.decisions.get(actor_id)
                if previous is None:
                    self.decisions[actor_id] = {
                        "actor_id": actor_id, "predicted_class": "c7", "evidence_score": score,
                        "evidence_frame_index": frame_index, "first_flag_frame_index": frame_index,
                        "first_flag_timestamp_ms": timestamp_ms, "source_actor_id": source, "causal": True,
                    }
                elif score > float(previous["evidence_score"]):
                    previous.update({"evidence_score": score, "evidence_frame_index": frame_index, "source_actor_id": source})
        return {
            actor_id: {**self.decisions.get(actor_id, {
                "actor_id": actor_id, "predicted_class": "c5", "evidence_score": "",
                "evidence_frame_index": "", "first_flag_frame_index": "",
                "first_flag_timestamp_ms": "", "causal": True,
            }), "c7_score": current_scores.get(actor_id, "")}
            for actor_id in self.actors
        }

    def final_decisions(self):
        return {
            actor_id: dict(self.decisions.get(actor_id, {
                "actor_id": actor_id, "predicted_class": "c5", "evidence_score": "",
                "evidence_frame_index": "", "first_flag_frame_index": "",
                "first_flag_timestamp_ms": "", "causal": True,
            })) for actor_id in self.actors
        }

    def reset(self) -> None:
        """Discard evidence and causal hand history while retaining the model."""
        self.streams.clear()
        self.decisions.clear()
        self.actors.clear()
        self.layout.clear()
        self.last_frame = None

class CombinedCausalActorClassifier:
    """Merge enabled specialists by strongest accepted actor evidence."""

    def __init__(self, classifiers: Iterable[Any]) -> None:
        self.classifiers = tuple(classifiers)

    @staticmethod
    def _merge(outputs: Iterable[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for output in outputs:
            for actor_id, decision in output.items():
                previous = merged.get(actor_id)
                score = float(decision.get("evidence_score") or 0.0)
                previous_score = float(previous.get("evidence_score") or 0.0) if previous else -1.0
                if previous is None or score > previous_score:
                    merged[actor_id] = dict(decision)
                elif previous is not None:
                    for key, value in decision.items():
                        if key.endswith("_score"):
                            previous[key] = value
        return merged

    def update(self, **kwargs) -> dict[str, dict[str, Any]]:
        return self._merge(classifier.update(**kwargs) for classifier in self.classifiers)

    def update_tracks(self, **kwargs) -> dict[str, dict[str, Any]]:
        return self._merge(classifier.update_tracks(**kwargs) for classifier in self.classifiers)

    def reset(self) -> None:
        for classifier in self.classifiers:
            classifier.reset()

    def final_decisions(self) -> dict[str, dict[str, Any]]:
        return self._merge(classifier.final_decisions() for classifier in self.classifiers)


__all__ = [
    "CausalLiveActorClassifier",
    "CausalPoseActorClassifier",
    "CausalC7ActorClassifier",
    "CombinedCausalActorClassifier",
]

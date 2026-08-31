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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..feature_csv.causal_stream import CausalActorWindow, CausalSpecialistState
from ..feature_csv.benchmark_c1_causal import PoseStream
from ..feature_csv.benchmark_c7_causal import C7Stream
from ..feature_csv.stage6_bundle import load_stage6_bundle


_STAGE6_SCHEMA_LENGTHS = {"c2": 65, "c3": 90, "suspicious_activity": 50}
_STAGE6_TEMPORAL_POLICY = {
    "target_fps": 8,
    "baseline_valid_frames": 4,
    "head_turn_baseline_valid_frames": 8,
    "window_frames": 24,
    "max_derivative_gap_ms": 450,
}
_STAGE6_GATE_NAMES = {
    "c3_motion_ceiling",
    "c3_side_floor",
    "c3_down_ceiling",
    "suspicious_down_floor",
    "suspicious_motion_floor",
    "suspicious_lower_floor",
}


@dataclass(frozen=True)
class CausalLiveActorLoadResult:
    """Startup boundary that keeps invalid Stage 6 bundles out of camera loop."""

    available: bool
    classifier: "CausalLiveActorClassifier | None"
    requested_model_dir: Path
    error: str | None
    provenance: Mapping[str, Any]


class CausalLiveActorClassifier:
    """Run the Stage 6 causal specialists on one live stream."""

    def __init__(
        self,
        model_dir: Path,
        *,
        clip_id: str = "live",
        student_prefix: str = "student_",
        explicit_pairs: Iterable[tuple[str, str]] = (),
        c3_threshold_override: float | None = None,
        xgboost_device: str = "cpu",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.clip_id = str(clip_id)
        self.student_prefix = student_prefix
        self.xgboost_device = self._normalize_xgboost_device(xgboost_device)
        bundle = load_stage6_bundle(
            self.model_dir,
            expected_temporal_policy=_STAGE6_TEMPORAL_POLICY,
        )
        schemas = bundle["feature_schemas"]
        if set(schemas) != set(_STAGE6_SCHEMA_LENGTHS):
            raise ValueError("Stage 6 runtime specialist schema set mismatch")
        if set(bundle["hashes"]["models"]) != set(_STAGE6_SCHEMA_LENGTHS):
            raise ValueError("Stage 6 runtime model hash set mismatch")
        for specialist, expected_length in _STAGE6_SCHEMA_LENGTHS.items():
            names = schemas[specialist]
            if not isinstance(names, list) or len(names) != expected_length:
                raise ValueError(
                    f"Stage 6 runtime schema length mismatch for {specialist}: "
                    f"expected {expected_length}, got {len(names) if isinstance(names, list) else 'invalid'}"
                )

        calibration = bundle["calibration"]
        thresholds = calibration.get("specialist_thresholds")
        gates = calibration.get("gate_thresholds")
        if not isinstance(thresholds, dict) or set(thresholds) != set(_STAGE6_SCHEMA_LENGTHS):
            raise ValueError("Stage 6 specialist calibration set mismatch")
        if not isinstance(gates, dict) or not _STAGE6_GATE_NAMES.issubset(gates):
            raise ValueError("Stage 6 gate calibration is incomplete")

        self.bundle_manifest = bundle["manifest"]
        self.feature_schemas = schemas
        self.temporal_policy = bundle["temporal_policy"]
        self.target_fps = int(self.temporal_policy["target_fps"])
        self.warmup_frames = int(self.temporal_policy["baseline_valid_frames"])
        self.head_turn_baseline_frames = int(
            self.temporal_policy["head_turn_baseline_valid_frames"]
        )
        self.window_frames = int(self.temporal_policy["window_frames"])
        self.max_derivative_gap_ms = int(
            self.temporal_policy["max_derivative_gap_ms"]
        )
        self.calibration = calibration
        self.bundle_hashes = bundle["hashes"]
        self.c2_threshold = float(thresholds["c2"])
        artifact_c3_threshold = float(thresholds["c3"])
        self.c3_threshold = (
            float(c3_threshold_override)
            if c3_threshold_override is not None
            else artifact_c3_threshold
        )
        self.suspicious_threshold = float(thresholds["suspicious_activity"])
        self.c3_pose_only = "c3_pose_head_peer_delta__mean" in schemas["c3"]
        self.c2_model, self.c2_names = self._load_model("c2", schemas["c2"])
        self.c3_model, self.c3_names = self._load_model("c3", schemas["c3"])
        self.suspicious_model, self.suspicious_names = self._load_model(
            "suspicious_activity", schemas["suspicious_activity"]
        )
<<<<<<< HEAD
        self.suspicious_model = self.suspicious_names = None
        if self.suspicious_threshold is not None:
            self.suspicious_model, self.suspicious_names = self._load_model(
                "causal_suspicious_activity_specialist.ubj",
                "causal_suspicious_activity_feature_names.json",
            )
=======
>>>>>>> 1168bab (Revert "Merge branch 'main' into develop")
        self.c2_bases = tuple(name.rsplit("__", 1)[0] for name in self.c2_names[::5])
        self.c3_bases = tuple(name.rsplit("__", 1)[0] for name in self.c3_names[::5])
        self.suspicious_bases = tuple(name.rsplit("__", 1)[0] for name in self.suspicious_names[::5]) if self.suspicious_names else ()
        self.shared_bases = tuple(dict.fromkeys((*self.c2_bases, *self.c3_bases, *self.suspicious_bases)))
        self._windows_c2: dict[str, CausalActorWindow] = {}
        self._windows_c3: dict[str, CausalActorWindow] = {}
        self._windows_suspicious: dict[str, CausalActorWindow] = {}
        self.gates = gates
        self._state = CausalSpecialistState(
            (), c3_threshold=self.c3_threshold,
            c2_threshold=self.c2_threshold,
            suspicious_threshold=self.suspicious_threshold,
            c3_gate=lambda values: bool(values.get("c3_gate", True)),
            suspicious_gate=lambda values: bool(values.get("suspicious_gate", True)),
        )
        self._explicit_pairs = tuple(
            (str(left), str(right)) for left, right in explicit_pairs
        )
        # Retain a bounded union of the first valid rows for every baseline
        # family; raw invalid startup rows must not displace later valid seeds.
        self._baseline_rows: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        self._baseline_feature_samples: dict[str, dict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._tail_rows: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.window_frames)
        )
        self._actor_mapping: dict[str, dict[str, str]] = {}
        self._next_sample_index = 0
        self._last_source_frame_index: int | None = None
        self._last_timestamp: float | None = None
        self._latest_scores: dict[str, dict[str, float]] = {}
        self._latest_feature_rows: dict[str, dict[str, Any]] = {}
        self._window_sizes: dict[str, int] = {}

    def _load_model(self, specialist: str, names: list[str]):
        import xgboost as xgb

        if (
            self.xgboost_device.startswith("cuda")
            and not xgb.build_info().get("USE_CUDA")
        ):
            raise RuntimeError("XGBoost was built without CUDA support")

        model_path = self.model_dir / "models" / f"{specialist}.ubj"
        model = xgb.Booster()
        model.load_model(str(model_path))
        model.set_param({"device": self.xgboost_device})
        if len(names) % 5 != 0:
            raise ValueError(f"invalid Stage 6 causal feature schema: {specialist}")
        model.feature_names = list(names)
        return model, tuple(names)

    @staticmethod
    def _normalize_xgboost_device(device: object) -> str:
        value = str(device).strip().lower()
        if value in {"gpu", "cuda", "0"}:
            return "cuda:0"
        if value == "cpu" or value.startswith("cuda:"):
            return value
        raise ValueError(f"unsupported XGBoost device: {device}")

    def set_compute_device(self, device: object) -> str:
        """Move all specialists together; any failure restores every model to CPU."""

        import xgboost as xgb

        target = self._normalize_xgboost_device(device)
        if target == self.xgboost_device:
            return target
        models = (
            (self.c2_model, self.c2_names),
            (self.c3_model, self.c3_names),
            (self.suspicious_model, self.suspicious_names),
        )
        try:
            if target.startswith("cuda") and not xgb.build_info().get("USE_CUDA"):
                raise RuntimeError("XGBoost was built without CUDA support")
            for model, names in models:
                model.set_param({"device": target})
                model.predict(
                    xgb.DMatrix(
                        np.zeros((1, len(names)), dtype=np.float32),
                        feature_names=list(names),
                    )
                )
        except Exception as error:
            for model, _ in models:
                model.set_param({"device": "cpu"})
            self.xgboost_device = "cpu"
            raise RuntimeError(f"XGBoost device switch failed: {error}") from error
        self.xgboost_device = target
        return target

    def _make_rows(
        self,
        *,
        sample_index: int,
        source_frame_index: int,
        timestamp_ms: float,
        results: Iterable[Any],
    ) -> list[dict[str, Any]]:
        from ..feature_csv.canonical_behavior_features import _frame_rows

        tracks = [result.to_dict() for result in results]
        for track in tracks:
            track_id = str(track["track_id"])
            student_id = track.get("student_id") or f"{self.student_prefix}{int(track_id):02d}"
            self._actor_mapping[student_id] = {
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
                "sample_index": int(sample_index),
                "source_frame_index": int(source_frame_index),
                "frame_id": int(source_frame_index) + 1,
                "timestamp_ms": float(timestamp_ms),
                "tracks": tracks,
            }]
        }
        rows = list(_frame_rows(manifest, self._actor_mapping, payload))
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
            lookup[(source_frame_index, str(track["track_id"]))] = {
                int(point["index"]): (
                    float(point["frame_x"]), float(point["frame_y"])
                )
                for point in track.get("selected_face_landmarks", [])
                if point.get("frame_x") is not None and point.get("frame_y") is not None
            } if track.get("face_valid") and not track.get("face_predicted") else {}
        for row in rows:
            row["_selected_face_points"] = lookup.get(
                (source_frame_index, str(row.get("track_id", ""))), {}
            )
        return rows

    @staticmethod
    def _valid_number(row: dict[str, Any], name: str) -> bool:
        try:
            value = float(row.get(name))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
        valid = row.get(f"{name}_valid")
        return valid is None or str(valid).strip().lower() not in {
            "", "0", "false", "no", "nan",
        }

    def _retain_frozen_baseline_rows(self, derived_rows: list[dict[str, Any]]) -> None:
        """Keep first-valid raw baseline seeds without retaining full history."""

        pose_indices = (0, 1, 4, 9, 10, 11, 12)
        for row in sorted(derived_rows, key=lambda value: int(value["sample_index"])):
            actor_id = str(row["actor_id"])
            sample_index = int(row["sample_index"])
            raw = next(
                (
                    candidate for candidate in self._tail_rows[actor_id]
                    if int(candidate["sample_index"]) == sample_index
                ),
                None,
            )
            if raw is None:
                continue
            families = []
            if bool(int(float(row.get("track_present", 0)))) and all(
                str(row.get(f"pose_{index}_valid", "0")) == "1"
                for index in pose_indices
            ):
                families.append("pose_geometry")
            if (
                bool(int(float(row.get("face_valid", 0))))
                and not bool(int(float(row.get("face_predicted", 0))))
            ):
                families.append("face_geometry")
            if self._valid_number(row, "hand_quality_mask") and float(
                row["hand_quality_mask"]
            ) > 0.0:
                families.append("hand_geometry")
            families.extend(
                f"feature:{name}"
                for name in self.shared_bases
                if self._valid_number(row, name)
            )
            for family in families:
                samples = self._baseline_feature_samples[actor_id][family]
                if len(samples) >= self.head_turn_baseline_frames:
                    continue
                samples.add(sample_index)
                self._baseline_rows[actor_id][sample_index] = dict(raw)

    def _bounded_prefix(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for actor_id in self._tail_rows:
            keep = [
                self._baseline_rows[actor_id][sample]
                for sample in sorted(self._baseline_rows[actor_id])
            ] + list(self._tail_rows[actor_id])
            seen: set[int] = set()
            for row in keep:
                frame = int(row["sample_index"])
                if frame not in seen:
                    output.append(dict(row))
                    seen.add(frame)
        return output

    def _legacy_c3_gate(self, row: dict[str, Any]) -> bool:
        """Pre-B4 C3 gate retained for one-release diagnostic tracing."""
        return (
            row.get("strict_hand_quality__mean", 0.0) > 0.0
            and row.get("hand_motion__q95", 0.0) <= self.gates.get("c3_motion_ceiling", 0.124559)
            and row.get("finger_motion__q95", 0.0) <= self.gates.get("c3_motion_ceiling", 0.124559)
            and row.get("c3_pose_head_peer_delta__max", 0.0) >= self.gates.get("c3_side_floor", 0.05)
            and row.get("strict_head_down_delta__q95", 0.0) <= self.gates.get("c3_down_ceiling", 0.05)
        )

    def _b4_c3_gate(self, row: dict[str, Any]) -> bool:
        """Current-frame B4 C3 gate; rolling and hand features are diagnostic only."""
        head_reliability = min(
            float(row.get("current_c3_pose_head_valid", 0.0)),
            float(row.get("current_c3_pose_peer_valid", 0.0)),
        )
        return (
            head_reliability >= 0.5
            and float(row.get("current_c3_pose_head_peer_delta", 0.0))
            >= float(self.gates["c3_side_floor"])
            and float(row.get("current_strict_head_down_delta", 0.0))
            <= float(self.gates["c3_down_ceiling"])
        )

    @staticmethod
    def _latest(rows: list[dict[str, Any]], sample_index: int) -> dict[str, dict[str, Any]]:
        return {
            str(row["actor_id"]): row
            for row in rows
            if int(row["sample_index"]) == int(sample_index)
        }

    def _prepare_live_pair_inputs(
        self,
        scores: dict[str, dict[str, float]],
        midpoint: dict[str, float],
        endpoint_present: dict[str, bool],
    ) -> None:
        """Permit one C2 source only while both configured bboxes are current."""

        for left, right in self._explicit_pairs:
            pair_ready = endpoint_present.get(left, False) and endpoint_present.get(
                right, False
            )
            if pair_ready:
                scores.setdefault(left, {}).setdefault("c2", 0.0)
                scores.setdefault(right, {}).setdefault("c2", 0.0)
            else:
                midpoint[left] = 0.0
                midpoint[right] = 0.0
            source_ready = {
                actor_id: (
                    pair_ready
                    and float(midpoint.get(actor_id, 0.0)) >= 1.0
                    and float(scores.get(actor_id, {}).get("c2", float("nan")))
                    >= self.c2_threshold
                )
                for actor_id in (left, right)
            }
            pair_gate = any(source_ready.values())
            for actor_id in (left, right):
                scores.setdefault(actor_id, {})["c2_gate"] = pair_gate
                scores[actor_id]["c2_source_gate"] = source_ready[actor_id]

    def update(self, *, frame_index: int, timestamp_ms: int, results: Iterable[Any]):
        import xgboost as xgb
        from ..feature_csv import behavior_subset_stage2 as behavior
        from ..feature_csv.temporal_geometry import enrich

        source_frame_index, timestamp_ms = int(frame_index), float(timestamp_ms)
        if (
            self._last_source_frame_index is not None
            and source_frame_index <= self._last_source_frame_index
        ):
            raise ValueError("live classifier received a non-increasing frame")
        if self._last_timestamp is not None and timestamp_ms <= self._last_timestamp:
            raise ValueError("live classifier received a non-increasing timestamp")
        sample_index = self._next_sample_index
        self._next_sample_index += 1
        self._last_source_frame_index = source_frame_index
        self._last_timestamp = timestamp_ms
        current_rows = self._make_rows(
            sample_index=sample_index,
            source_frame_index=source_frame_index,
            timestamp_ms=timestamp_ms,
            results=results,
        )
        endpoint_present: dict[str, bool] = {}
        for row in current_rows:
            actor_id = str(row["actor_id"])
            try:
                bbox_present = all(
                    math.isfinite(float(row[name]))
                    for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
                )
            except (KeyError, TypeError, ValueError):
                bbox_present = False
            endpoint_present[actor_id] = bool(row.get("track_present")) and bbox_present
            self._tail_rows[actor_id].append(dict(row))
        prefix = enrich(self._bounded_prefix(), baseline_frames=self.warmup_frames)
        if self.c3_pose_only:
            prefix = behavior.derive_c3_pose_contract(
                prefix,
                baseline_frames=self.warmup_frames,
                max_derivative_gap_ms=self.max_derivative_gap_ms,
            )
        prefix = behavior._head_pnp_features(
            prefix, baseline_frames=self.warmup_frames
        )
        prefix = behavior.derive_behavior_motion(
            prefix,
            baseline_frames=self.warmup_frames,
            max_derivative_gap_ms=self.max_derivative_gap_ms,
        )
        prefix = behavior.derive_face_c3_features(
            prefix, baseline_frames=self.warmup_frames
        )
        prefix = behavior.derive_finger_motion(
            prefix, max_derivative_gap_ms=self.max_derivative_gap_ms
        )
        prefix = behavior.derive_hand_shape_and_pair_cues(
            prefix,
            head_turn_baseline_frames=self.head_turn_baseline_frames,
            max_derivative_gap_ms=self.max_derivative_gap_ms,
        )
        if self.suspicious_names:
            prefix = behavior.derive_strict_c2_c3_suspicious_cues(prefix, baseline_frames=self.warmup_frames)
        prefix = behavior._apply_stage3_temporal_contract(
            prefix, max_derivative_gap_ms=self.max_derivative_gap_ms
        )
        self._retain_frozen_baseline_rows(prefix)
        aggregate_rows, _ = behavior.causal_aggregate_rows(
            prefix,
            self.shared_bases,
            warmup_frames=self.warmup_frames,
            window_frames=self.window_frames,
            max_derivative_gap_ms=self.max_derivative_gap_ms,
        )
        latest = self._latest(aggregate_rows, sample_index)
        self._latest_feature_rows = {
            str(actor_id): dict(row) for actor_id, row in latest.items()
        }
        scores, midpoint = {}, {}
        for actor_id, row in latest.items():
            self._state.register_actor(actor_id)
            self._window_sizes[actor_id] = int(row.get("prefix_frames", 0))
            if not endpoint_present.get(actor_id, False):
                continue
            if not int(row.get("warmup_ready", 0)):
                continue
            c2 = float(self.c2_model.predict(xgb.DMatrix(np.asarray([[row[name] for name in self.c2_names]], dtype=np.float32), feature_names=list(self.c2_names)))[0])
            c3 = float(self.c3_model.predict(xgb.DMatrix(np.asarray([[row[name] for name in self.c3_names]], dtype=np.float32), feature_names=list(self.c3_names)))[0])
            scores[actor_id] = {"c2": c2, "c3": c3}
            scores[actor_id]["c3_gate"] = self._b4_c3_gate(row)
            scores[actor_id]["legacy_c3_gate"] = self._legacy_c3_gate(row)
            if self.suspicious_names:
                scores[actor_id]["suspicious_activity"] = float(self.suspicious_model.predict(xgb.DMatrix(np.asarray([[row[name] for name in self.suspicious_names]], dtype=np.float32), feature_names=list(self.suspicious_names)))[0])
                scores[actor_id]["suspicious_gate"] = (
                    row.get("strict_head_down_delta__q95", 0.0) >= self.gates.get("suspicious_down_floor", 0.028485)
                    and max(row.get("hand_motion__q95", 0.0), row.get("finger_motion__q95", 0.0)) >= self.gates.get("suspicious_motion_floor", 0.037612)
                    and row.get("strict_hand_below_hip__max", 0.0) >= self.gates.get("suspicious_lower_floor", -0.187859)
                    and row.get("strict_own_side_outside_midpoint__max", 0.0) >= 1.0
                )
            midpoint[actor_id] = (
                row.get("near_midpoint_pre_cross", 0.0)
                if behavior.number(row.get("current_hand_quality_mask")) > 0.0
                and behavior.number(row.get("current_pair_hand_distance")) > 0.0
                and behavior.number(row.get("current_pair_margin_10pct")) > 0.0
                else 0.0
            )
        self._prepare_live_pair_inputs(scores, midpoint, endpoint_present)
        self._latest_scores = {
            actor_id: dict(values) for actor_id, values in scores.items()
        }
        self._state.update(frame_index=sample_index, timestamp_ms=timestamp_ms, scores_by_actor=scores, explicit_pairs=self._explicit_pairs, near_midpoint_by_actor=midpoint)
        return self._decision_output(scores)

    def update_tracks(
        self,
        *,
        frame_index: int,
        timestamp_ms: int,
        tracks: Iterable[dict[str, Any]],
    ):
        """Replay adapter using the same live update path and causal state."""
        class TrackResult:
            def __init__(self, payload: dict[str, Any]) -> None:
                self.payload = payload

            def to_dict(self) -> dict[str, Any]:
                return self.payload

        return self.update(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            results=(TrackResult(dict(track)) for track in tracks),
        )

    def _decision_output(self, scores=None):
        scores = scores or {}
        return {
            actor_id: {
                "actor_id": actor_id,
                "predicted_class": decision.class_code,
                "candidate_class": decision.class_code,
                "current_scores": {
                    name: value
                    for name, value in scores.get(actor_id, {}).items()
                    if name in {"c2", "c3", "suspicious_activity"}
                },
                "current_gates": {
                    name: bool(scores.get(actor_id, {}).get(f"{name}_gate", False))
                    for name in ("c2", "c3", "suspicious_activity")
                },
                "c2_score": scores.get(actor_id, self._latest_scores.get(actor_id, {})).get("c2", ""),
                "c3_score": scores.get(actor_id, self._latest_scores.get(actor_id, {})).get("c3", ""),
                "suspicious_activity_score": scores.get(actor_id, self._latest_scores.get(actor_id, {})).get("suspicious_activity", ""),
                "warmup_frames_seen": self._window_sizes.get(actor_id, 0),
                "warmup_frames_required": self.warmup_frames,
                "history": [
                    {
                        "class_code": evidence.class_code,
                        "frame_index": evidence.frame_index,
                        "timestamp_ms": evidence.timestamp_ms,
                        "score": evidence.score,
                        "source_actor_id": evidence.source_actor_id or "",
                        "source_score": evidence.source_score,
                    }
                    for evidence in decision.history
                ],
                "evidence_class": decision.evidence_class or "",
                "evidence_score": decision.evidence_score if decision.evidence_score is not None else "",
                "evidence_frame_index": decision.evidence_frame_index if decision.evidence_frame_index is not None else "",
                "evidence_timestamp_ms": decision.evidence_timestamp_ms if decision.evidence_timestamp_ms is not None else "",
                "evidence_source_score": decision.evidence_source_score if decision.evidence_source_score is not None else "",
                "evidence_source_actor_id": decision.source_actor_id or "",
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
        self._state = CausalSpecialistState(
            (), c3_threshold=self.c3_threshold,
            c2_threshold=self.c2_threshold,
            suspicious_threshold=self.suspicious_threshold,
            c3_gate=lambda values: bool(values.get("c3_gate", True)),
            suspicious_gate=lambda values: bool(values.get("suspicious_gate", True)),
        )
        self._baseline_rows.clear()
        self._baseline_feature_samples.clear()
        self._tail_rows.clear()
        self._actor_mapping.clear()
        self._next_sample_index = 0
        self._last_source_frame_index = None
        self._last_timestamp = None
        self._latest_scores.clear()
        self._latest_feature_rows.clear()
        self._window_sizes.clear()

    def diagnostic_snapshot(self, actor_id: str) -> dict[str, Any]:
        """Return raw current-frame aggregate features for Stage A trace."""
        return dict(self._latest_feature_rows.get(str(actor_id), {}))


def load_causal_live_actor_classifier(
    model_dir: Path,
    **classifier_options: Any,
) -> CausalLiveActorLoadResult:
    """Load Stage 6 runtime without letting startup failure escape to camera."""

    requested_model_dir = Path(model_dir)
    try:
        classifier = CausalLiveActorClassifier(
            requested_model_dir,
            **classifier_options,
        )
    except Exception as error:
        return CausalLiveActorLoadResult(
            available=False,
            classifier=None,
            requested_model_dir=requested_model_dir,
            error=f"{type(error).__name__}: {error}",
            provenance={},
        )
    return CausalLiveActorLoadResult(
        available=True,
        classifier=classifier,
        requested_model_dir=requested_model_dir,
        error=None,
        provenance=dict(classifier.bundle_hashes),
    )


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
    "CausalLiveActorLoadResult",
    "CausalLiveActorClassifier",
    "load_causal_live_actor_classifier",
    "CausalPoseActorClassifier",
    "CausalC7ActorClassifier",
    "CombinedCausalActorClassifier",
]

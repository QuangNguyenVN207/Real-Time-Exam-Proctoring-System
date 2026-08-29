from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xgboost as xgb

from pose_gaze.holistic.feature_csv import behavior_subset_stage2 as behavior
from pose_gaze.holistic.feature_csv.canonical_behavior_features import _frame_rows
from pose_gaze.holistic.feature_csv.stage6_bundle import load_stage6_bundle
from pose_gaze.holistic.feature_csv.behavior_subset_stage2 import derive_c3_pose_contract
from pose_gaze.holistic.feature_csv.causal_stream import (
    CausalActorWindow,
    CausalSpecialistState,
)
from pose_gaze.holistic.feature_csv.temporal_geometry import enrich
from pose_gaze.holistic.test_media.live_actor import (
    CausalLiveActorClassifier,
    load_causal_live_actor_classifier,
)
from pose_gaze.holistic.landmark.landmarks import (
    HolisticLandmarkExtractor,
)
from pose_gaze.holistic.debug.session_manifest import (
    SessionManifestRecorder,
    compute_model_artifacts_hashes,
    verify_model_artifacts,
)
from pose_gaze.holistic.test_webcam.test_webcam import (
    DemoWebcamInteractionController,
    switch_compute_to_cuda,
)
from pose_gaze.holistic.test_webcam import test_webcam as webcam_module
from pose_gaze.tracking.schemas import BoundingBox


REPO_ROOT = Path(__file__).resolve().parents[6]
BUNDLE_DIR = REPO_ROOT / "tmp" / "causal_8fps_stage6_mixed_084699_final_20260827"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class _Track:
    def __init__(self, track_id: int = 1) -> None:
        self.track_id = track_id

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "student_id": f"student_{self.track_id:02d}",
            "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
        }


class Stage6RuntimeWP6Tests(unittest.TestCase):
    @staticmethod
    def _pose_row(
        actor_id: str,
        sample_index: int,
        *,
        peer_id: str,
        nose_x: float,
        own_valid: bool = True,
        peer_bbox: bool = True,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "clip_id": "clip",
            "actor_id": actor_id,
            "sample_index": sample_index,
            "source_frame_index": sample_index * 2,
            "timestamp_ms": sample_index * 125,
            "track_id": actor_id,
            "track_present": "1",
            "interaction_peer_ids": json.dumps([peer_id]),
            "bbox_x1": 100 if actor_id == "s1" else 500,
            "bbox_x2": 200 if actor_id == "s1" else 600,
        }
        if not peer_bbox:
            row["bbox_x1"] = row["bbox_x2"] = ""
        points = {
            0: (nose_x, 100), 1: (nose_x - 5, 90), 4: (nose_x + 5, 90),
            11: (100, 120), 12: (140, 120), 23: (105, 180), 24: (135, 180),
        }
        if actor_id == "s2":
            points = {index: (x + 400, y) for index, (x, y) in points.items()}
        for index, (x, y) in points.items():
            row[f"pose_{index}_valid"] = "1" if own_valid else "0"
            row[f"pose_{index}_frame_x"] = x
            row[f"pose_{index}_frame_y"] = y
        return row

    def _bundle_copy(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        target = Path(temporary.name) / "bundle"
        shutil.copytree(BUNDLE_DIR, target)
        return temporary, target

    def test_bundle_loads_and_hash_schema_policy_mutations_reject(self) -> None:
        loaded = load_stage6_bundle(BUNDLE_DIR)
        self.assertEqual(
            {name: len(schema) for name, schema in loaded["feature_schemas"].items()},
            {"c2": 65, "c3": 90, "suspicious_activity": 50},
        )

        temporary, bundle = self._bundle_copy()
        self.addCleanup(temporary.cleanup)
        model_path = bundle / "models" / "c2.ubj"
        model_path.write_bytes(model_path.read_bytes() + b"mutated")
        rejected = load_causal_live_actor_classifier(bundle, xgboost_device="cpu")
        self.assertFalse(rejected.available)
        self.assertIn("hash mismatch", rejected.error or "")

        temporary, bundle = self._bundle_copy()
        self.addCleanup(temporary.cleanup)
        schema_path = bundle / "feature_schemas.json"
        schemas = json.loads(schema_path.read_text(encoding="utf-8"))
        schemas["c2"] = schemas["c2"][:-1]
        schema_path.write_text(json.dumps(schemas), encoding="utf-8")
        manifest_path = bundle / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["file_sha256"]["feature_schemas.json"] = _sha256(schema_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "feature-schema hash mismatch"):
            load_stage6_bundle(bundle)

        temporary, bundle = self._bundle_copy()
        self.addCleanup(temporary.cleanup)
        policy_path = bundle / "temporal_policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy["window_frames"] = 23
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        manifest_path = bundle / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["file_sha256"]["temporal_policy.json"] = _sha256(policy_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "temporal-policy hash mismatch"):
            load_stage6_bundle(bundle)

    def test_missing_bundle_returns_structured_unavailable(self) -> None:
        missing = REPO_ROOT / "tmp" / "__wp6_missing_bundle__"
        result = load_causal_live_actor_classifier(missing, xgboost_device="cpu")
        self.assertFalse(result.available)
        self.assertIsNone(result.classifier)
        self.assertEqual(result.requested_model_dir, missing)
        self.assertIn("Stage 6 bundle manifest missing", result.error or "")

    def test_missing_bundle_keeps_camera_tracking_and_landmarks_running(self) -> None:
        calls = {"tracking": 0, "landmarks": 0, "released": 0}
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        packet = SimpleNamespace(frame_id=1, timestamp_ms=1000, tracks=())

        class FakeCapture:
            def read(self):
                return True, frame.copy()

            def get(self, prop):
                return {3: 1280, 4: 720, 5: 30.0}.get(prop, 0)

            def release(self):
                calls["released"] += 1

        class FakeWriter:
            def isOpened(self):
                return True

            def write(self, _frame):
                pass

            def release(self):
                pass

        manager = SimpleNamespace(
            get_packet=lambda _session: packet,
            generate_final_output=lambda _session: REPO_ROOT / "tmp" / "wp6.json",
        )

        class FakeTracking:
            def __init__(self, _config):
                self.manager = manager
                self.detector = SimpleNamespace(_device="cpu", _model=SimpleNamespace())

            def open_webcam(self, *_args, **_kwargs):
                return FakeCapture()

            def process_frame(self, _frame):
                calls["tracking"] += 1
                return packet

            def draw_tracks(self, *_args, **_kwargs):
                pass

        class FakeHolistic:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def process_packet(self, _frame, _packet):
                calls["landmarks"] += 1
                return ()

            def draw_results(self, *_args, **_kwargs):
                pass

            def reset(self):
                pass

        fake_cv2 = SimpleNamespace(
            CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5,
            FONT_HERSHEY_SIMPLEX=0, LINE_AA=0,
            VideoWriter_fourcc=lambda *_args: 0,
            VideoWriter=lambda *_args, **_kwargs: FakeWriter(),
            putText=lambda *_args, **_kwargs: None,
            imshow=lambda *_args, **_kwargs: None,
            waitKeyEx=lambda _delay: ord("q"),
            waitKey=lambda _delay: ord("q"),
            destroyAllWindows=lambda: None,
        )
        missing = REPO_ROOT / "tmp" / "__wp6_missing_camera_bundle__"
        argv = [
            "test_webcam", "--causal-model-dir", str(missing),
            "--actions", "c2,c3,suspicious_activity",
        ]
        with patch("sys.argv", argv):
            real_args = webcam_module.parse_args()
        with (
            patch.object(webcam_module, "_DEBUG_AVAILABLE", False),
            patch.object(webcam_module, "PersonTrackingModule", FakeTracking),
            patch.object(webcam_module, "HolisticLandmarkExtractor", FakeHolistic),
            patch.object(webcam_module.TrackingManager, "generate_session_id", return_value="wp6"),
            patch.object(webcam_module, "parse_args", return_value=real_args),
            patch.object(webcam_module, "pump_keyboard_until_frame_deadline", side_effect=lambda _cv2, **kwargs: kwargs["interaction"].handle_key(ord("q"))),
            patch.dict("sys.modules", {"cv2": fake_cv2}),
        ):
            webcam_module.main()

        self.assertEqual(calls["tracking"], 1)
        self.assertEqual(calls["landmarks"], 1)
        self.assertEqual(calls["released"], 1)

    def test_live_observations_use_monotonic_sample_indices(self) -> None:
        classifier = CausalLiveActorClassifier(BUNDLE_DIR, xgboost_device="cpu")
        classifier.update(frame_index=10, timestamp_ms=1000, results=[_Track()])
        first = classifier.diagnostic_snapshot("student_01")
        classifier.update(frame_index=20, timestamp_ms=1125, results=[_Track()])
        second = classifier.diagnostic_snapshot("student_01")

        self.assertEqual((first["sample_index"], first["source_frame_index"]), (0, 10))
        self.assertEqual((second["sample_index"], second["source_frame_index"]), (1, 20))

    def test_actor_absence_remains_in_exact_latest_24_sample_window(self) -> None:
        classifier = CausalLiveActorClassifier(BUNDLE_DIR, xgboost_device="cpu")
        classifier.update(frame_index=0, timestamp_ms=0.0, results=[_Track()])
        for sample in range(1, 31):
            classifier.update_tracks(
                frame_index=sample * 2,
                timestamp_ms=sample * 125.0,
                tracks=[],
            )
        row = classifier.diagnostic_snapshot("student_01")
        decision = classifier.final_decisions()["student_01"]
        self.assertEqual(row["sample_index"], 30)
        self.assertEqual(row["window_start_frame"], 7)
        self.assertEqual(row["window_end_frame"], 30)
        self.assertEqual(row["prefix_frames"], 24)
        self.assertEqual(decision["predicted_class"], "c5")

    def test_first_valid_baseline_survives_rollover_and_matches_offline(self) -> None:
        frames = []
        for sample in range(61):
            points = []
            if sample >= 30:
                nose_x = 120.0 + min(sample - 30, 8)
                coordinates = {
                    0: (nose_x, 100.0), 1: (nose_x - 5, 90.0),
                    4: (nose_x + 5, 90.0), 9: (115.0, 108.0),
                    10: (125.0, 108.0), 11: (100.0, 120.0),
                    12: (140.0, 120.0), 23: (105.0, 180.0),
                    24: (135.0, 180.0),
                }
                points = [
                    {
                        "index": index, "frame_x": x, "frame_y": y,
                        "x": x / 640.0, "y": y / 480.0,
                    }
                    for index, (x, y) in coordinates.items()
                ]
            frames.append({
                "sample_index": sample,
                "source_frame_index": sample * 2,
                "frame_id": sample * 2 + 1,
                "timestamp_ms": sample * 125.0,
                "tracks": [{
                    "track_id": 1, "student_id": "s1",
                    "bbox_xyxy": [50.0, 0.0, 250.0, 300.0],
                    "pose_landmarks": points,
                }],
            })
        mapping = {
            "s1": {
                "actor_id": "s1", "track_id": "1",
                "spatial_role": "live_actor", "confidence": "fixture",
            }
        }
        manifest = {
            "clip_id": "rollover", "filename": "rollover", "split": "live",
            "split_group": "live", "class_code": "c5", "action_actor_ids": [],
        }
        offline = list(_frame_rows(manifest, mapping, {"frames": frames}))
        for row in offline:
            row.update({
                "actor_truth": "c5", "actor_label": "c5", "source_actor": 0,
                "manifest_class_code": "c5", "interaction_peer_ids": "[]",
                "excluded_source": False, "_selected_face_points": {},
            })
            for name, value in list(row.items()):
                if name.endswith("_valid"):
                    row[name] = str(int(bool(value)))
        offline = enrich(offline, baseline_frames=4)
        offline = behavior.derive_c3_pose_contract(offline, baseline_frames=4)
        offline = behavior._head_pnp_features(offline, baseline_frames=4)
        offline = behavior.derive_behavior_motion(offline, baseline_frames=4)
        offline = behavior.derive_face_c3_features(offline, baseline_frames=4)
        offline = behavior.derive_finger_motion(offline)
        offline = behavior.derive_hand_shape_and_pair_cues(
            offline, head_turn_baseline_frames=8
        )
        offline = behavior.derive_strict_c2_c3_suspicious_cues(
            offline, baseline_frames=4
        )
        offline = behavior._apply_stage3_temporal_contract(offline)

        live = CausalLiveActorClassifier(BUNDLE_DIR, xgboost_device="cpu")
        for frame in frames:
            live.update_tracks(
                frame_index=int(frame["source_frame_index"]),
                timestamp_ms=float(frame["timestamp_ms"]),
                tracks=frame["tracks"],
            )
        aggregates, _ = behavior.causal_aggregate_rows(
            offline, live.shared_bases, warmup_frames=4, window_frames=24
        )
        expected = next(
            row for row in aggregates
            if row["actor_id"] == "s1" and int(row["sample_index"]) == 60
        )
        actual = live.diagnostic_snapshot("s1")
        for names in (live.c2_names, live.c3_names, live.suspicious_names):
            np.testing.assert_allclose(
                [expected[name] for name in names],
                [actual[name] for name in names],
                rtol=0.0, atol=1e-9, equal_nan=True,
            )

    def test_c2_accepted_output_exposes_matching_pair_and_source_gates(self) -> None:
        classifier = CausalLiveActorClassifier(
            BUNDLE_DIR,
            explicit_pairs=(("s1", "s2"),),
            xgboost_device="cpu",
        )
        scores = {"s1": {"c2": 0.8}, "s2": {"c2": 0.0}}
        midpoint = {"s1": 1.0, "s2": 0.0}
        classifier._prepare_live_pair_inputs(
            scores, midpoint, {"s1": True, "s2": True}
        )
        classifier._state.update(
            frame_index=0,
            timestamp_ms=1000,
            scores_by_actor=scores,
            explicit_pairs=classifier._explicit_pairs,
            near_midpoint_by_actor=midpoint,
        )
        output = classifier._decision_output(scores)
        for actor in ("s1", "s2"):
            self.assertEqual(output[actor]["predicted_class"], "c2")
            self.assertTrue(output[actor]["current_gates"]["c2"])
        self.assertTrue(scores["s1"]["c2_source_gate"])
        self.assertFalse(scores["s2"]["c2_source_gate"])

    def test_recursive_stage6_hashes_verify_and_detect_nested_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").mkdir()
            model = root / "models" / "c2.ubj"
            model.write_bytes(b"model-v1")
            (root / "calibration.json").write_text("{}", encoding="utf-8")
            stored = compute_model_artifacts_hashes(root)
            self.assertEqual(
                set(stored), {"calibration.json", "models/c2.ubj"}
            )
            verify_model_artifacts(root, stored)
            model.write_bytes(b"model-v2")
            with self.assertRaisesRegex(ValueError, "HASH MISMATCH models/c2.ubj"):
                verify_model_artifacts(root, stored)

    def test_c2_one_valid_source_propagates_but_no_source_does_not(self) -> None:
        state = CausalSpecialistState((), c3_threshold=1.0, c2_threshold=0.5)
        decisions = state.update(
            frame_index=0,
            timestamp_ms=1000,
            scores_by_actor={"s1": {"c2": 0.8}, "s2": {"c2": 0.0}},
            explicit_pairs=(("s1", "s2"),),
            near_midpoint_by_actor={"s1": 1.0, "s2": 0.0},
        )
        self.assertEqual(decisions["s1"].class_code, "c2")
        self.assertEqual(decisions["s2"].class_code, "c2")
        self.assertEqual(decisions["s2"].source_actor_id, "s1")

        state = CausalSpecialistState((), c3_threshold=1.0, c2_threshold=0.5)
        decisions = state.update(
            frame_index=0,
            timestamp_ms=1000,
            scores_by_actor={"s1": {"c2": 0.8}, "s2": {"c2": 0.7}},
            explicit_pairs=(("s1", "s2"),),
            near_midpoint_by_actor={"s1": 0.0, "s2": 0.0},
        )
        self.assertEqual(decisions["s1"].class_code, "c5")
        self.assertEqual(decisions["s2"].class_code, "c5")

    def test_c3_peer_landmark_dropout_keeps_eligibility_but_own_dropout_blocks(self) -> None:
        rows = []
        for sample, nose_x in ((0, 120), (1, 120), (2, 120), (3, 140)):
            rows.append(self._pose_row("s1", sample, peer_id="s2", nose_x=nose_x))
            rows.append(self._pose_row(
                "s2", sample, peer_id="s1", nose_x=520,
                own_valid=False, peer_bbox=True,
            ))
        derived = derive_c3_pose_contract(rows, baseline_frames=3)
        current = next(
            row for row in derived
            if row["actor_id"] == "s1" and row["sample_index"] == 3
        )
        self.assertEqual(current["c3_pose_peer_valid"], 1.0)
        self.assertEqual(current["c3_pose_head_valid"], 1.0)

        invalid = derive_c3_pose_contract([
            self._pose_row("s1", 0, peer_id="s2", nose_x=120, own_valid=False),
            self._pose_row("s2", 0, peer_id="s1", nose_x=520),
        ])
        own = next(row for row in invalid if row["actor_id"] == "s1")
        self.assertEqual(own["c3_pose_head_valid"], 0.0)

    def test_shared_c3_offline_live_parity_with_peer_present_and_dropout(self) -> None:
        for peer_dropout in (False, True):
            with self.subTest(peer_dropout=peer_dropout):
                raw_frames = []
                offline_rows = []
                mapping = {
                    "s1": {"actor_id": "s1", "track_id": "1", "spatial_role": "left", "confidence": "fixture"},
                    "s2": {"actor_id": "s2", "track_id": "2", "spatial_role": "right", "confidence": "fixture"},
                }
                manifest = {
                    "clip_id": "c3", "filename": "c3.mp4", "split": "live",
                    "split_group": "live", "class_code": "c5", "action_actor_ids": [],
                }
                for sample, nose_x in ((0, 120), (1, 120), (2, 120), (3, 120), (4, 140)):
                    tracks = []
                    for actor, track_id, actor_nose in (
                        ("s1", 1, nose_x), ("s2", 2, 520),
                    ):
                        row = self._pose_row(
                            actor, sample,
                            peer_id="s2" if actor == "s1" else "s1",
                            nose_x=actor_nose,
                            own_valid=not (actor == "s2" and peer_dropout),
                        )
                        points = [
                            {
                                "index": index,
                                "frame_x": row[f"pose_{index}_frame_x"],
                                "frame_y": row[f"pose_{index}_frame_y"],
                                "x": float(row[f"pose_{index}_frame_x"]) / 640.0,
                                "y": float(row[f"pose_{index}_frame_y"]) / 480.0,
                            }
                            for index in (0, 1, 4, 11, 12, 23, 24)
                            if row[f"pose_{index}_valid"] == "1"
                        ]
                        tracks.append({
                            "track_id": track_id,
                            "student_id": actor,
                            "bbox_xyxy": [
                                float(row["bbox_x1"]), 0.0,
                                float(row["bbox_x2"]), 300.0,
                            ],
                            "pose_landmarks": points,
                        })
                    raw_frames.append({
                        "sample_index": sample,
                        "source_frame_index": sample * 2,
                        "frame_id": sample * 2 + 1,
                        "timestamp_ms": sample * 125.0,
                        "tracks": tracks,
                    })
                offline_rows = list(_frame_rows(
                    manifest, mapping, {"frames": raw_frames}
                ))
                for row in offline_rows:
                    actor = str(row["actor_id"])
                    row.update({
                        "interaction_peer_ids": json.dumps([
                            "s2" if actor == "s1" else "s1"
                        ]),
                    })
                    for name, value in list(row.items()):
                        if name.endswith("_valid"):
                            row[name] = str(int(bool(value)))
                offline = derive_c3_pose_contract(offline_rows, baseline_frames=4)
                expected = next(
                    row for row in offline
                    if row["actor_id"] == "s1" and int(row["sample_index"]) == 4
                )

                live = CausalLiveActorClassifier(
                    BUNDLE_DIR,
                    explicit_pairs=(("s1", "s2"),),
                    xgboost_device="cpu",
                )
                for frame in raw_frames:
                    live.update_tracks(
                        frame_index=int(frame["source_frame_index"]),
                        timestamp_ms=float(frame["timestamp_ms"]),
                        tracks=frame["tracks"],
                    )
                actual = live.diagnostic_snapshot("s1")
                for offline_name, live_name in (
                    ("c3_pose_head_valid", "current_c3_pose_head_valid"),
                    ("c3_pose_peer_valid", "current_c3_pose_peer_valid"),
                    ("c3_pose_head_peer_delta", "current_c3_pose_head_peer_delta"),
                    ("c3_pose_torso_peer_delta", "c3_pose_torso_peer_delta__max"),
                ):
                    self.assertAlmostEqual(
                        float(expected[offline_name]), float(actual[live_name]), places=9
                    )

    def test_current_returns_c5_while_history_remains(self) -> None:
        state = CausalSpecialistState(("s1",), c3_threshold=0.5)
        state.update(
            frame_index=0,
            timestamp_ms=1000,
            scores_by_actor={"s1": {"c3": 0.8, "c3_gate": True}},
        )
        decision = state.update(
            frame_index=1,
            timestamp_ms=1125,
            scores_by_actor={"s1": {"c3": 0.1, "c3_gate": True}},
        )["s1"]
        self.assertEqual(decision.class_code, "c5")
        self.assertEqual(decision.evidence_class, "c3")
        self.assertEqual(len(decision.history), 1)

    def test_over_gap_and_epoch_change_mask_derivative_history(self) -> None:
        window = CausalActorWindow(
            "s1", ("hand_motion",),
            derivative_feature_names=("hand_motion",),
            max_derivative_gap_ms=450,
        )
        window.update(
            sample_index=0, timestamp_ms=1000,
            features={"hand_motion": 1.0}, validity={"hand_motion": 1},
            continuity_epoch=0,
        )
        over_gap = window.update(
            sample_index=1, timestamp_ms=1501,
            features={"hand_motion": 9.0}, validity={"hand_motion": 1},
            continuity_epoch=0,
        )
        changed = window.update(
            sample_index=2, timestamp_ms=1626,
            features={"hand_motion": 7.0}, validity={"hand_motion": 1},
            continuity_epoch=1,
        )
        self.assertEqual(over_gap.valid_counts["hand_motion"], 0)
        self.assertEqual(changed.valid_counts["hand_motion"], 0)

    def test_default_classifier_starts_all_specialists_on_cpu(self) -> None:
        classifier = CausalLiveActorClassifier(BUNDLE_DIR)
        self.assertEqual(classifier.xgboost_device, "cpu")
        for model in (
            classifier.c2_model,
            classifier.c3_model,
            classifier.suspicious_model,
        ):
            self.assertEqual(model.attributes().get("device"), None)

    def test_compute_hotkey_prompt_and_atomic_failure(self) -> None:
        interaction = DemoWebcamInteractionController(None, "session", None)
        interaction.handle_key(ord("C"))
        self.assertTrue(interaction.consume_compute_switch())
        interaction.prompt_mode = "student_id"
        interaction.input_buffer = ""
        interaction.handle_key(ord("c"))
        self.assertEqual(interaction.input_buffer, "c")
        self.assertFalse(interaction.consume_compute_switch())

        classifier = SimpleNamespace(device="cpu")
        classifier.set_compute_device = lambda device: (
            setattr(classifier, "device", device) or device
        )
        model = SimpleNamespace(device="cpu")
        model.to = lambda device: (
            (_ for _ in ()).throw(RuntimeError("simulated CUDA failure"))
            if device == "cuda" else setattr(model, "device", device)
        )
        tracking = SimpleNamespace(
            detector=SimpleNamespace(_model=model, _device="cpu")
        )
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True)
        )
        success_classifier = SimpleNamespace(device="cpu")
        success_classifier.set_compute_device = lambda device: (
            setattr(success_classifier, "device", device) or device
        )
        success_model = SimpleNamespace(device="cpu")
        success_model.to = lambda device: setattr(success_model, "device", device)
        success_tracking = SimpleNamespace(
            detector=SimpleNamespace(_model=success_model, _device="cpu")
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            success, error = switch_compute_to_cuda(
                success_tracking, success_classifier
            )
        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertEqual(success_classifier.device, "cuda:0")
        self.assertEqual(success_model.device, "cuda")
        self.assertEqual(success_tracking.detector._device, "0")

        recorder = SessionManifestRecorder(
            session_id="wp6",
            working_directory=REPO_ROOT,
            model_dir=BUNDLE_DIR,
            runtime_arguments={"camera": 0, "target_fps": 8},
            camera_config={"index": 0, "width": 1280, "height": 720},
        )
        recorder.record_compute_switch(
            timestamp_ms=1000,
            requested_mode="CUDA",
            active_mode="CUDA",
            result="success",
            devices_before={
                "yolo": "cpu", "c2": "cpu", "c3": "cpu",
                "suspicious_activity": "cpu",
            },
            devices_after={
                "yolo": "0", "c2": "cuda:0", "c3": "cuda:0",
                "suspicious_activity": "cuda:0",
            },
            error=None,
        )
        transition = recorder.compute_switches[0]
        self.assertEqual(transition["result"], "success")
        self.assertEqual(set(transition["devices_after"]), {
            "yolo", "c2", "c3", "suspicious_activity",
        })

        with patch.dict("sys.modules", {"torch": fake_torch}):
            success, error = switch_compute_to_cuda(tracking, classifier)
        self.assertFalse(success)
        self.assertIn("simulated CUDA failure", error or "")
        self.assertEqual(classifier.device, "cpu")
        self.assertEqual(model.device, "cpu")
        self.assertEqual(tracking.detector._device, "cpu")

    def test_face_hold_emits_predicted_then_expires_and_reset_clears(self) -> None:
        observed_face = [
            SimpleNamespace(x=0.5, y=0.5, visibility=1.0, presence=1.0)
            for _ in range(455)
        ]
        results = [
            SimpleNamespace(
                pose_landmarks=[], pose_world_landmarks=[],
                left_hand_landmarks=[], left_hand_world_landmarks=[],
                right_hand_landmarks=[], right_hand_world_landmarks=[],
                face_landmarks=observed_face,
            ),
            *[
                SimpleNamespace(
                    pose_landmarks=[], pose_world_landmarks=[],
                    left_hand_landmarks=[], left_hand_world_landmarks=[],
                    right_hand_landmarks=[], right_hand_world_landmarks=[],
                    face_landmarks=[],
                )
                for _ in range(4)
            ],
        ]
        processor = SimpleNamespace(
            detect_for_video=lambda _image, _timestamp: results.pop(0),
            close=lambda: None,
        )
        extractor = HolisticLandmarkExtractor(
            face_hold_frames=3,
            task_model_path=REPO_ROOT / "weights" / "mediapipe" / "holistic_landmarker.task",
            processor_factory=lambda: processor,
        )
        track = SimpleNamespace(
            track_id=1,
            student_id="student_01",
            bbox=BoundingBox(0, 0, 100, 100),
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        current = extractor.process_track(frame, track, timestamp_ms=1000)
        held = [
            extractor.process_track(frame, track, timestamp_ms=1125 + index * 125)
            for index in range(3)
        ]
        expired = extractor.process_track(frame, track, timestamp_ms=1500)

        self.assertTrue(current.face_valid)
        self.assertFalse(current.face_predicted)
        self.assertTrue(all(not item.face_valid and item.face_predicted for item in held))
        self.assertTrue(all(item.selected_face_landmarks for item in held))
        self.assertFalse(expired.face_valid)
        self.assertFalse(expired.face_predicted)
        self.assertFalse(expired.selected_face_landmarks)
        self.assertNotIn(1, extractor._last_face_landmarks)

        results.append(SimpleNamespace(
            pose_landmarks=[], pose_world_landmarks=[],
            left_hand_landmarks=[], left_hand_world_landmarks=[],
            right_hand_landmarks=[], right_hand_world_landmarks=[],
            face_landmarks=observed_face,
        ))
        track.is_present = True
        seeded = extractor.process_packet(
            frame, SimpleNamespace(tracks=[track], timestamp_ms=1625)
        )[0]
        self.assertTrue(seeded.face_valid)
        self.assertIn(1, extractor._last_face_landmarks)

        track.is_present = False
        self.assertEqual(
            extractor.process_packet(
                frame, SimpleNamespace(tracks=[track], timestamp_ms=1750)
            ),
            (),
        )
        self.assertNotIn(1, extractor._last_face_landmarks)

        results.append(SimpleNamespace(
            pose_landmarks=[], pose_world_landmarks=[],
            left_hand_landmarks=[], left_hand_world_landmarks=[],
            right_hand_landmarks=[], right_hand_world_landmarks=[],
            face_landmarks=[],
        ))
        track.is_present = True
        reacquired = extractor.process_packet(
            frame, SimpleNamespace(tracks=[track], timestamp_ms=1875)
        )[0]
        self.assertFalse(reacquired.face_valid)
        self.assertFalse(reacquired.face_predicted)
        self.assertFalse(reacquired.selected_face_landmarks)

        extractor._last_face_landmarks[1] = current.selected_face_landmarks
        extractor.reset()
        self.assertNotIn(1, extractor._last_face_landmarks)

    def test_saved_sequence_offline_and_live_matrices_scores_gates_decisions_match(self) -> None:
        clip_id = "1786173698008_53534665179931303_9030090460038115400"
        source = json.loads(
            (REPO_ROOT / "data" / "raw_video" / "processed" / "holistic_output_8fps" / f"{clip_id}.json")
            .read_text(encoding="utf-8")
        )
        frames = source["frames"][:8]
        mapping = {
            "s9": {"actor_id": "s9", "track_id": "2", "spatial_role": "left", "confidence": "fixture"},
            "s10": {"actor_id": "s10", "track_id": "1", "spatial_role": "right", "confidence": "fixture"},
        }
        manifest = {
            "clip_id": clip_id, "filename": f"{clip_id}.mp4", "split": "live",
            "split_group": "live", "class_code": "c5", "action_actor_ids": [],
        }
        offline_rows = list(_frame_rows(manifest, mapping, {"frames": frames}))
        face_lookup = {
            (int(frame["source_frame_index"]), str(track["track_id"])): {
                int(point["index"]): (float(point["frame_x"]), float(point["frame_y"]))
                for point in track.get("selected_face_landmarks", [])
                if point.get("frame_x") is not None and point.get("frame_y") is not None
            }
            for frame in frames for track in frame.get("tracks", [])
            if track.get("face_valid") and not track.get("face_predicted")
        }
        for row in offline_rows:
            actor = str(row["actor_id"])
            row.update({
                "actor_truth": "c5", "actor_label": "c5", "source_actor": 0,
                "manifest_class_code": "c5", "excluded_source": False,
                "interaction_peer_ids": json.dumps(["s10" if actor == "s9" else "s9"]),
                "_selected_face_points": face_lookup.get(
                    (int(row["source_frame_index"]), str(row["track_id"])), {}
                ),
            })
            for name, value in list(row.items()):
                if name.endswith("_valid"):
                    row[name] = str(int(bool(value)))

        offline = enrich(offline_rows, baseline_frames=4)
        offline = behavior.derive_c3_pose_contract(offline, baseline_frames=4)
        offline = behavior._head_pnp_features(offline, baseline_frames=4)
        offline = behavior.derive_behavior_motion(offline, baseline_frames=4)
        offline = behavior.derive_face_c3_features(offline, baseline_frames=4)
        offline = behavior.derive_finger_motion(offline)
        offline = behavior.derive_hand_shape_and_pair_cues(
            offline, head_turn_baseline_frames=8
        )
        offline = behavior.derive_strict_c2_c3_suspicious_cues(
            offline, baseline_frames=4
        )
        offline = behavior._apply_stage3_temporal_contract(offline)

        live = CausalLiveActorClassifier(
            BUNDLE_DIR,
            explicit_pairs=(("s9", "s10"),),
            xgboost_device="cpu",
        )
        live_outputs = {}
        live_rows = {}
        live_snapshots: dict[int, dict[str, object]] = {}
        for frame in frames:
            tracks = []
            for track in frame.get("tracks", []):
                copied = dict(track)
                copied["student_id"] = {"1": "s10", "2": "s9"}[str(track["track_id"])]
                tracks.append(copied)
            live_outputs = live.update_tracks(
                frame_index=int(frame["source_frame_index"]),
                timestamp_ms=float(frame["timestamp_ms"]),
                tracks=tracks,
            )
            live_rows = {
                actor: live.diagnostic_snapshot(actor) for actor in ("s9", "s10")
            }
            live_snapshots[int(frame["sample_index"])] = {
                "outputs": json.loads(json.dumps(live_outputs)),
                "rows": {actor: dict(row) for actor, row in live_rows.items()},
            }

        aggregates, _ = behavior.causal_aggregate_rows(
            offline, live.shared_bases, warmup_frames=4, window_frames=24
        )
        final_sample = int(frames[-1]["sample_index"])
        offline_final = {
            str(row["actor_id"]): row
            for row in aggregates if int(row["sample_index"]) == final_sample
        }
        offline_state = CausalSpecialistState(
            ("s9", "s10"),
            c2_threshold=live.c2_threshold,
            c3_threshold=live.c3_threshold,
            suspicious_threshold=live.suspicious_threshold,
            c3_gate=lambda values: bool(values.get("c3_gate")),
            suspicious_gate=lambda values: bool(values.get("suspicious_gate")),
        )
        for sample in sorted({int(row["sample_index"]) for row in aggregates}):
            current = [row for row in aggregates if int(row["sample_index"]) == sample]
            scores = {}
            midpoint = {}
            for row in current:
                actor = str(row["actor_id"])
                if int(row["warmup_ready"]):
                    scores[actor] = {
                        "c2": float(live.c2_model.predict(xgb.DMatrix(
                            np.asarray([[row[name] for name in live.c2_names]], dtype=np.float32),
                            feature_names=list(live.c2_names),
                        ))[0]),
                        "c3": float(live.c3_model.predict(xgb.DMatrix(
                            np.asarray([[row[name] for name in live.c3_names]], dtype=np.float32),
                            feature_names=list(live.c3_names),
                        ))[0]),
                        "suspicious_activity": float(live.suspicious_model.predict(xgb.DMatrix(
                            np.asarray([[row[name] for name in live.suspicious_names]], dtype=np.float32),
                            feature_names=list(live.suspicious_names),
                        ))[0]),
                        "c3_gate": live._b4_c3_gate(row),
                        "suspicious_gate": (
                            row.get("strict_head_down_delta__q95", 0.0)
                            >= live.gates["suspicious_down_floor"]
                            and max(
                                row.get("hand_motion__q95", 0.0),
                                row.get("finger_motion__q95", 0.0),
                            ) >= live.gates["suspicious_motion_floor"]
                            and row.get("strict_hand_below_hip__max", 0.0)
                            >= live.gates["suspicious_lower_floor"]
                            and row.get("strict_own_side_outside_midpoint__max", 0.0) >= 1.0
                        ),
                    }
                midpoint[actor] = (
                    row.get("near_midpoint_pre_cross", 0.0)
                    if behavior.number(row.get("current_hand_quality_mask")) > 0.0
                    and behavior.number(row.get("current_pair_hand_distance")) > 0.0
                    and behavior.number(row.get("current_pair_margin_10pct")) > 0.0
                    else 0.0
                )
            offline_decisions = offline_state.update(
                frame_index=sample,
                timestamp_ms=min(float(row["timestamp_ms"]) for row in current),
                scores_by_actor=scores,
                explicit_pairs=(("s9", "s10"),),
                near_midpoint_by_actor=midpoint,
            )
            snapshot = live_snapshots[sample]
            snapshot_outputs = snapshot["outputs"]
            snapshot_rows = snapshot["rows"]
            for row in current:
                if not int(row["warmup_ready"]):
                    continue
                actor = str(row["actor_id"])
                for names in (live.c2_names, live.c3_names, live.suspicious_names):
                    np.testing.assert_allclose(
                        [row[name] for name in names],
                        [snapshot_rows[actor][name] for name in names],
                        rtol=0.0, atol=1e-9, equal_nan=True,
                        err_msg=f"sample={sample} actor={actor}",
                    )
                for specialist in ("c2", "c3", "suspicious_activity"):
                    self.assertAlmostEqual(
                        scores[actor][specialist],
                        snapshot_outputs[actor]["current_scores"][specialist],
                        places=7,
                        msg=(
                            f"sample={sample} actor={actor} specialist={specialist} "
                            f"output={snapshot_outputs[actor]} "
                            f"warmup={snapshot_rows[actor].get('warmup_ready')}"
                        ),
                    )
                self.assertEqual(
                    bool(scores[actor]["c3_gate"]),
                    snapshot_outputs[actor]["current_gates"]["c3"],
                )
                self.assertEqual(
                    offline_decisions[actor].class_code,
                    snapshot_outputs[actor]["predicted_class"],
                )
                self.assertEqual(
                    [item.class_code for item in offline_decisions[actor].history],
                    [item["class_code"] for item in snapshot_outputs[actor]["history"]],
                )
        for actor in ("s9", "s10"):
            for names in (live.c2_names, live.c3_names, live.suspicious_names):
                mismatches = [
                    name for name in names
                    if not np.isclose(
                        offline_final[actor][name], live_rows[actor][name],
                        rtol=0.0, atol=1e-9, equal_nan=True,
                    )
                ]
                np.testing.assert_allclose(
                    [offline_final[actor][name] for name in names],
                    [live_rows[actor][name] for name in names],
                    rtol=0.0,
                    atol=1e-9,
                    equal_nan=True,
                    err_msg=f"{actor} mismatches={mismatches}",
                )
            expected_scores = {
                "c2": float(live.c2_model.predict(xgb.DMatrix(
                    np.asarray([[offline_final[actor][name] for name in live.c2_names]], dtype=np.float32),
                    feature_names=list(live.c2_names),
                ))[0]),
                "c3": float(live.c3_model.predict(xgb.DMatrix(
                    np.asarray([[offline_final[actor][name] for name in live.c3_names]], dtype=np.float32),
                    feature_names=list(live.c3_names),
                ))[0]),
                "suspicious_activity": float(live.suspicious_model.predict(xgb.DMatrix(
                    np.asarray([[offline_final[actor][name] for name in live.suspicious_names]], dtype=np.float32),
                    feature_names=list(live.suspicious_names),
                ))[0]),
            }
            for specialist, score in expected_scores.items():
                self.assertAlmostEqual(
                    score, live_outputs[actor]["current_scores"][specialist], places=7
                )
            self.assertEqual(
                live._b4_c3_gate(offline_final[actor]),
                live_outputs[actor]["current_gates"]["c3"],
            )
            self.assertEqual(
                live_outputs[actor]["predicted_class"],
                offline_state.decisions()[actor].class_code,
            )
            self.assertEqual(
                live_outputs[actor]["history"],
                [
                    {
                        "class_code": item.class_code,
                        "frame_index": item.frame_index,
                        "timestamp_ms": item.timestamp_ms,
                        "score": item.score,
                        "source_actor_id": item.source_actor_id or "",
                        "source_score": item.source_score,
                    }
                    for item in offline_state.decisions()[actor].history
                ],
            )


if __name__ == "__main__":
    unittest.main()

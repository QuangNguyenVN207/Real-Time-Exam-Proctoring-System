"""Verify the exact causal bundle and supporting pose/gaze model files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
RELEASE_CONFIG = MODULE_DIR / "POSE_GAZE_RELEASE.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _verify_file(path: Path, expected_hash: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{label} checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )


def verify_pose_gaze_release(*, load_models: bool = True) -> dict[str, Any]:
    config = json.loads(RELEASE_CONFIG.read_text(encoding="utf-8"))
    bundle_config = config["bundle"]
    runtime = config["runtime"]
    bundle_dir = REPO_ROOT / bundle_config["path"]
    manifest_path = bundle_dir / "bundle_manifest.json"
    _verify_file(
        manifest_path,
        bundle_config["bundle_manifest_sha256"],
        "Stage 6 bundle manifest",
    )

    from backend.ai_services.pose_gaze.pose_gaze.holistic.feature_csv.stage6_bundle import (
        load_stage6_bundle,
    )
    from backend.ai_services.pose_gaze.pose_gaze.settings import (
        DEFAULT_CAUSAL_MODEL_DIR,
        DEFAULT_CAUSAL_TARGET_FPS,
    )

    bundle = load_stage6_bundle(bundle_dir)
    if bundle["manifest"]["format_version"] != bundle_config["format_version"]:
        raise ValueError("Stage 6 format version drift")
    if bundle["hashes"]["models"] != bundle_config["models"]:
        raise ValueError("Stage 6 final model hashes drift")
    actual_counts = {
        name: len(features) for name, features in bundle["feature_schemas"].items()
    }
    if actual_counts != bundle_config["feature_counts"]:
        raise ValueError(
            f"Stage 6 feature counts drift: {actual_counts}"
        )
    policy = bundle["temporal_policy"]
    expected_policy = {
        key: runtime[key]
        for key in (
            "target_fps",
            "baseline_valid_frames",
            "head_turn_baseline_valid_frames",
            "window_frames",
            "max_derivative_gap_ms",
        )
    }
    if policy != expected_policy:
        raise ValueError(
            f"Stage 6 temporal policy drift: release={expected_policy}, bundle={policy}"
        )
    calibration = bundle["calibration"]
    for key in ("specialist_thresholds", "gate_thresholds"):
        if calibration.get(key) != runtime[key]:
            raise ValueError(f"Stage 6 {key} drift")
    if DEFAULT_CAUSAL_MODEL_DIR.resolve() != bundle_dir.resolve():
        raise ValueError(
            "Pose/gaze default model directory does not point to the locked bundle"
        )
    if DEFAULT_CAUSAL_TARGET_FPS != runtime["target_fps"]:
        raise ValueError("Pose/gaze default FPS does not match the locked bundle")

    supporting = config["supporting_models"]
    for label, item in supporting.items():
        _verify_file(REPO_ROOT / item["path"], item["sha256"], label)

    result: dict[str, Any] = {
        "verdict": "PASS",
        "bundle_dir": str(bundle_dir),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "model_sha256": bundle["hashes"]["models"],
        "feature_counts": actual_counts,
        "temporal_policy": policy,
        "specialist_thresholds": calibration["specialist_thresholds"],
        "gate_thresholds": calibration["gate_thresholds"],
        "models_loaded": False,
        "cpu_forward_pass": False,
        "mediapipe_task_loaded": False,
    }
    if not load_models:
        return result

    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(REPO_ROOT / ".cache" / "ultralytics"),
    )
    import numpy as np
    import xgboost as xgb
    from ultralytics import YOLO

    from backend.ai_services.pose_gaze.pose_gaze.holistic.landmark import (
        HolisticLandmarkExtractor,
    )
    from backend.ai_services.pose_gaze.pose_gaze.holistic.test_media.live_actor import (
        CausalLiveActorClassifier,
    )

    classifier = CausalLiveActorClassifier(bundle_dir, xgboost_device="cpu")
    predictions: dict[str, float] = {}
    for specialist, model, names in (
        ("c2", classifier.c2_model, classifier.c2_names),
        ("c3", classifier.c3_model, classifier.c3_names),
        (
            "suspicious_activity",
            classifier.suspicious_model,
            classifier.suspicious_names,
        ),
    ):
        score = float(
            model.predict(
                xgb.DMatrix(
                    np.zeros((1, len(names)), dtype=np.float32),
                    feature_names=list(names),
                )
            )[0]
        )
        if not math.isfinite(score):
            raise RuntimeError(f"{specialist} returned a non-finite CPU score")
        predictions[specialist] = score

    person_model = YOLO(
        str(REPO_ROOT / supporting["person_detector"]["path"]),
        task="detect",
    )
    person_result = person_model.predict(
        source=np.zeros((640, 640, 3), dtype=np.uint8),
        imgsz=640,
        classes=[0],
        device="cpu",
        verbose=False,
    )
    if len(person_result) != 1:
        raise RuntimeError("YOLO person detector CPU forward pass failed")

    extractor = HolisticLandmarkExtractor(
        task_model_path=REPO_ROOT / supporting["holistic_landmarker"]["path"]
    )
    processor = extractor._new_processor()
    processor.close()
    extractor.close()

    result.update(
        {
            "models_loaded": True,
            "cpu_forward_pass": True,
            "mediapipe_task_loaded": True,
            "zero_feature_scores": predictions,
            "verified_versions": {
                "mediapipe": __import__("mediapipe").__version__,
                "numpy": np.__version__,
                "xgboost": xgb.__version__,
                "ultralytics": __import__("ultralytics").__version__,
            },
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="Skip loading XGBoost, YOLO, and MediaPipe models.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify_pose_gaze_release(load_models=not args.integrity_only),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

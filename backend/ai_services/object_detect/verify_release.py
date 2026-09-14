"""Verify the locked object checkpoint, configuration, and CPU forward pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from backend.core.config import settings


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
RELEASE_CONFIG = MODULE_DIR / "OBJECT_RELEASE.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _names_list(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(value) for value in names]


def verify_object_release(*, load_model: bool = True) -> dict[str, Any]:
    config = json.loads(RELEASE_CONFIG.read_text(encoding="utf-8"))
    checkpoint = config["checkpoint"]
    runtime = config["runtime"]
    model_path = REPO_ROOT / checkpoint["path"]
    if not model_path.is_file():
        raise FileNotFoundError(f"Locked object checkpoint missing: {model_path}")

    actual_hash = sha256_file(model_path)
    if actual_hash != checkpoint["sha256"]:
        raise ValueError(
            "best (1).pt checksum mismatch: "
            f"expected {checkpoint['sha256']}, got {actual_hash}"
        )
    if model_path.stat().st_size != int(checkpoint["size_bytes"]):
        raise ValueError("best (1).pt size mismatch")

    expected_runtime = {
        "input_size": settings.object_inference_size,
        "default_confidence_threshold": settings.yolo_confidence_threshold,
        "class_confidence_thresholds": settings.object_class_confidence_thresholds,
        "paper_confidence_threshold": settings.paper_detection_confidence_threshold,
        "confirm_frames": settings.object_confirm_frames,
        "confirm_window": settings.object_confirm_window,
        "detect_every_n_frames": settings.object_detect_every_n_frames,
        "smartphone_fallback_enabled": settings.smartphone_fallback_enabled,
    }
    drift = {
        key: {"release": runtime.get(key), "runtime": value}
        for key, value in expected_runtime.items()
        if runtime.get(key) != value
    }
    if Path(settings.yolo_model_path).resolve() != model_path.resolve():
        drift["checkpoint_path"] = {
            "release": str(model_path),
            "runtime": settings.yolo_model_path,
        }
    if runtime.get("openvino_enabled") is not False:
        drift["openvino_enabled"] = {
            "release": runtime.get("openvino_enabled"),
            "runtime": False,
        }
    if drift:
        raise ValueError(f"Locked object runtime configuration drift: {drift}")

    result: dict[str, Any] = {
        "verdict": "PASS",
        "checkpoint": str(model_path),
        "checkpoint_sha256": actual_hash,
        "checkpoint_size_bytes": model_path.stat().st_size,
        "classes": list(checkpoint["classes"]),
        "runtime": runtime,
        "model_loaded": False,
        "cpu_forward_pass": False,
    }
    if not load_model:
        return result

    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(REPO_ROOT / ".cache" / "ultralytics"),
    )
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    actual_names = _names_list(model.names)
    if actual_names != checkpoint["classes"]:
        raise ValueError(
            f"Object class order mismatch: expected {checkpoint['classes']}, "
            f"got {actual_names}"
        )
    predictions = model.predict(
        source=np.zeros((runtime["input_size"], runtime["input_size"], 3), dtype=np.uint8),
        imgsz=runtime["input_size"],
        conf=runtime["default_confidence_threshold"],
        device="cpu",
        verbose=False,
    )
    if len(predictions) != 1 or tuple(predictions[0].orig_shape) != (
        runtime["input_size"],
        runtime["input_size"],
    ):
        raise RuntimeError("best (1).pt CPU forward pass returned an invalid result")
    for value in runtime["class_confidence_thresholds"].values():
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("Invalid object confidence threshold")

    result.update(
        {
            "model_loaded": True,
            "cpu_forward_pass": True,
            "verified_versions": {
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="Skip importing Ultralytics and the CPU forward pass.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify_object_release(load_model=not args.integrity_only),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

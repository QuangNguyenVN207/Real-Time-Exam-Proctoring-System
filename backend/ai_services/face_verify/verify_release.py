"""Verify the locked Face P1 artifacts, runtime settings, and CPU models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from backend.core.config import settings


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
RELEASE_CONFIG = MODULE_DIR / "FACE_P1_RELEASE.json"
MODEL_ROOT = REPO_ROOT / "data" / "insightface"
GALLERY_PATH = REPO_ROOT / "data" / "student_faces" / "gallery_embeddings.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _verify_file(path: Path, expected_hash: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash.lower():
        raise ValueError(
            f"{label} checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return actual_hash


def verify_face_release(*, load_models: bool = True) -> dict[str, Any]:
    config = json.loads(RELEASE_CONFIG.read_text(encoding="utf-8"))
    if config.get("status") != "locked":
        raise ValueError("Face release is not marked as locked")

    runtime = config["runtime"]
    expected_runtime = {
        "model_pack": settings.face_model_name,
        "detector_input_size": list(settings.face_det_size),
        "detector_confidence_threshold": settings.face_detection_threshold,
        "similarity_threshold": settings.face_similarity_threshold,
        "runner_up_margin_threshold": settings.face_identity_margin_threshold,
        "identity_scan_every_n_frames": settings.face_scan_every_n_frames,
        "identity_assignment_confirmations": settings.face_assignment_confirmations,
        "identity_mismatch_confirmations": settings.face_mismatch_confirmations,
    }
    drift = {
        key: {"release": runtime.get(key), "runtime": value}
        for key, value in expected_runtime.items()
        if runtime.get(key) != value
    }
    if runtime.get("embedding_dimensions") != 512:
        drift["embedding_dimensions"] = {
            "release": runtime.get("embedding_dimensions"),
            "runtime": 512,
        }
    if runtime.get("similarity") != "cosine_on_L2_normalized_vectors":
        drift["similarity"] = {
            "release": runtime.get("similarity"),
            "runtime": "cosine_on_L2_normalized_vectors",
        }
    if drift:
        raise ValueError(f"Locked Face P1 runtime configuration drift: {drift}")

    model_dir = MODEL_ROOT / "models" / runtime["model_pack"]
    detector_path = model_dir / runtime["detector_file"]
    recognizer_path = model_dir / runtime["recognizer_file"]
    expected_hashes = config["sha256"]
    artifact_hashes = {
        runtime["detector_file"]: _verify_file(
            detector_path,
            expected_hashes[runtime["detector_file"]],
            "SCRFD detector",
        ),
        runtime["recognizer_file"]: _verify_file(
            recognizer_path,
            expected_hashes[runtime["recognizer_file"]],
            "ArcFace recognizer",
        ),
        GALLERY_PATH.name: _verify_file(
            GALLERY_PATH,
            expected_hashes[GALLERY_PATH.name],
            "Face gallery",
        ),
    }

    import numpy as np

    with np.load(GALLERY_PATH, allow_pickle=False) as gallery:
        required_arrays = {
            "actor_ids",
            "centroids",
            "template_actor_ids",
            "template_vectors",
        }
        missing_arrays = sorted(required_arrays.difference(gallery.files))
        if missing_arrays:
            raise ValueError(f"Face gallery arrays missing: {missing_arrays}")
        actor_ids = [str(value) for value in gallery["actor_ids"].tolist()]
        centroids = np.asarray(gallery["centroids"], dtype=np.float32)

    dimensions = int(runtime["embedding_dimensions"])
    if centroids.ndim != 2 or centroids.shape != (len(actor_ids), dimensions):
        raise ValueError(
            "Face gallery shape mismatch: "
            f"actor_ids={len(actor_ids)}, centroids={centroids.shape}"
        )
    if not actor_ids or len(set(actor_ids)) != len(actor_ids):
        raise ValueError("Face gallery actor_ids must be non-empty and unique")
    if not np.isfinite(centroids).all():
        raise ValueError("Face gallery contains non-finite centroid values")
    centroid_norms = np.linalg.norm(centroids, axis=1)
    if not np.allclose(centroid_norms, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("Face gallery centroids are not L2-normalized")

    result: dict[str, Any] = {
        "verdict": "PASS",
        "release_id": config["release_id"],
        "artifacts": {
            name: {"path": str(path), "sha256": artifact_hashes[name]}
            for name, path in (
                (runtime["detector_file"], detector_path),
                (runtime["recognizer_file"], recognizer_path),
                (GALLERY_PATH.name, GALLERY_PATH),
            )
        },
        "runtime": runtime,
        "gallery_actor_count": len(actor_ids),
        "gallery_embedding_dimensions": dimensions,
        "models_loaded": False,
        "cpu_forward_pass": False,
    }
    if not load_models:
        return result

    import cv2
    import insightface
    import onnxruntime as ort
    from insightface.model_zoo import model_zoo

    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")

    detector = model_zoo.get_model(
        str(detector_path), providers=["CPUExecutionProvider"]
    )
    detector.prepare(
        ctx_id=-1,
        input_size=tuple(runtime["detector_input_size"]),
        det_thresh=float(runtime["detector_confidence_threshold"]),
    )
    blank_frame = np.zeros(
        (
            int(runtime["detector_input_size"][1]),
            int(runtime["detector_input_size"][0]),
            3,
        ),
        dtype=np.uint8,
    )
    bboxes, _ = detector.detect(blank_frame, max_num=1, metric="default")
    if bboxes.ndim != 2 or bboxes.shape[1] != 5:
        raise RuntimeError(f"SCRFD CPU forward pass returned shape {bboxes.shape}")

    recognizer = model_zoo.get_model(
        str(recognizer_path), providers=["CPUExecutionProvider"]
    )
    recognizer.prepare(ctx_id=-1)
    embedding = np.asarray(
        recognizer.get_feat(np.zeros((112, 112, 3), dtype=np.uint8))
    )
    if embedding.shape != (1, dimensions) or not np.isfinite(embedding).all():
        raise RuntimeError(
            f"ArcFace CPU forward pass returned invalid shape {embedding.shape}"
        )

    result.update(
        {
            "models_loaded": True,
            "cpu_forward_pass": True,
            "verified_versions": {
                "insightface": insightface.__version__,
                "onnxruntime": ort.__version__,
                "opencv": cv2.__version__,
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
        help="Skip loading SCRFD/ArcFace and their CPU forward passes.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify_face_release(load_models=not args.integrity_only),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

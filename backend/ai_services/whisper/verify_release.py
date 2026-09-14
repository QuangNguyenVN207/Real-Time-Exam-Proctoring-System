"""Verify the exact Whisper P1 package and run local-only CPU smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
EXTERNAL_RELEASE_MANIFEST = (
    MODULE_DIR / "releases" / "whisper_p1_20260914.manifest.json"
)
EMBEDDED_RELEASE_MANIFEST = MODULE_DIR / "WHISPER_P1_RELEASE.json"
RELEASE_MANIFEST = (
    EXTERNAL_RELEASE_MANIFEST
    if EXTERNAL_RELEASE_MANIFEST.is_file()
    else EMBEDDED_RELEASE_MANIFEST
)
TEXT_SUFFIXES = {".codes", ".csv", ".json", ".md", ".py", ".txt"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _manifest_payload(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return payload


def _manifest_file_sha256(path: Path) -> str:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return sha256_file(path)
    return hashlib.sha256(_manifest_payload(path)).hexdigest().lower()


def _manifest_path(relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe path in Whisper release manifest: {relative_name}")
    candidate = (MODULE_DIR / relative).resolve()
    try:
        candidate.relative_to(MODULE_DIR)
    except ValueError as exc:
        raise ValueError(
            f"Whisper release path escapes module directory: {relative_name}"
        ) from exc
    return candidate


def _verify_manifest_files(manifest: dict[str, Any]) -> dict[str, str]:
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("Whisper release manifest files must be a list")
    if manifest.get("file_count") != len(entries):
        raise ValueError(
            "Whisper release file_count mismatch: "
            f"declared={manifest.get('file_count')}, entries={len(entries)}"
        )

    declared_paths = [str(entry.get("path", "")) for entry in entries]
    duplicates = sorted(
        path for path in set(declared_paths) if declared_paths.count(path) > 1
    )
    if "" in declared_paths or duplicates:
        raise ValueError(
            f"Invalid or duplicate Whisper manifest paths: duplicates={duplicates}"
        )

    verified: dict[str, str] = {}
    problems: list[str] = []
    for entry in entries:
        relative_name = str(entry["path"])
        path = _manifest_path(relative_name)
        if not path.is_file():
            problems.append(f"missing: {relative_name}")
            continue
        actual_size = (
            len(_manifest_payload(path))
            if path.suffix.lower() in TEXT_SUFFIXES
            else path.stat().st_size
        )
        expected_size = int(entry["bytes"])
        if actual_size != expected_size:
            problems.append(
                f"size mismatch: {relative_name}: expected {expected_size}, "
                f"got {actual_size}"
            )
            continue
        actual_hash = _manifest_file_sha256(path)
        expected_hash = str(entry["sha256"]).lower()
        if actual_hash != expected_hash:
            problems.append(
                f"checksum mismatch: {relative_name}: expected {expected_hash}, "
                f"got {actual_hash}"
            )
            continue
        verified[relative_name] = actual_hash

    if problems:
        raise ValueError(
            "Whisper P1 release manifest drift:\n- " + "\n- ".join(problems)
        )
    return verified


def _verify_archive_if_present(manifest: dict[str, Any]) -> dict[str, Any]:
    archive = manifest.get("archive", {})
    if not archive:
        return {"present": False, "declared": False}
    archive_name = str(archive.get("path", ""))
    if not archive_name or Path(archive_name).name != archive_name:
        raise ValueError(f"Invalid Whisper archive name: {archive_name}")

    candidates = (
        MODULE_DIR / "releases" / archive_name,
        REPO_ROOT / "dist" / archive_name,
    )
    archive_path = next((path for path in candidates if path.is_file()), None)
    if archive_path is None:
        return {
            "present": False,
            "expected_name": archive_name,
            "expected_sha256": str(archive.get("sha256", "")).lower(),
        }

    actual_size = archive_path.stat().st_size
    expected_size = int(archive["bytes"])
    if actual_size != expected_size:
        raise ValueError(
            f"Whisper archive size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_hash = sha256_file(archive_path)
    expected_hash = str(archive["sha256"]).lower()
    if actual_hash != expected_hash:
        raise ValueError(
            f"Whisper archive checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return {
        "present": True,
        "path": str(archive_path),
        "bytes": actual_size,
        "sha256": actual_hash,
    }


def verify_whisper_release(*, load_models: bool = True) -> dict[str, Any]:
    manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("release") != "whisper-p1":
        raise ValueError(f"Unexpected Whisper release: {manifest.get('release')}")

    verified_files = _verify_manifest_files(manifest)
    archive_result = _verify_archive_if_present(manifest)

    from backend.ai_services.whisper.config import (
        PACKAGED_PHOWHISPER_DIR,
        PHOBERT_CHEATING_THRESHOLD,
        PHOBERT_MODEL_DIR,
        PHOWHISPER_MODEL,
        PHOWHISPER_REPO_ID,
        PHOWHISPER_REVISION,
    )

    asr_spec = str(manifest["pretrained_asr"])
    try:
        expected_asr_model, expected_asr_revision = asr_spec.rsplit("@", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid pinned pretrained_asr: {asr_spec}") from exc
    if manifest.get("self_contained_asr") is not True:
        raise ValueError("Whisper release does not declare a self-contained ASR model")
    packaged_asr_relative = str(manifest["pretrained_asr_packaged_path"])
    expected_asr_dir = _manifest_path(packaged_asr_relative)
    expected_asr_runtime = str(expected_asr_dir.resolve())

    pipeline_results_path = (
        MODULE_DIR / "outputs" / "p1_audio_pipeline" / "pipeline_results.json"
    )
    pipeline_results = json.loads(pipeline_results_path.read_text(encoding="utf-8"))
    release_runtime = pipeline_results["configuration"]
    expected_phobert_dir = (MODULE_DIR / "phobert" / "weights").resolve()
    runtime_drift: dict[str, dict[str, Any]] = {}
    comparisons = {
        "phowhisper_repository": (expected_asr_model, PHOWHISPER_REPO_ID),
        "phowhisper_model": (
            expected_asr_runtime,
            str(Path(PHOWHISPER_MODEL).expanduser().resolve()),
        ),
        "packaged_phowhisper_dir": (
            expected_asr_runtime,
            str(PACKAGED_PHOWHISPER_DIR.resolve()),
        ),
        "phowhisper_revision": (expected_asr_revision, PHOWHISPER_REVISION),
        "phobert_cheating_threshold": (
            float(release_runtime["phobert_cheating_threshold"]),
            PHOBERT_CHEATING_THRESHOLD,
        ),
        "phobert_model_dir": (str(expected_phobert_dir), str(PHOBERT_MODEL_DIR.resolve())),
    }
    for key, (expected, actual) in comparisons.items():
        values_match = (
            math.isclose(expected, actual, abs_tol=1e-12)
            if isinstance(expected, float) and isinstance(actual, float)
            else expected == actual
        )
        if not values_match:
            runtime_drift[key] = {"release": expected, "runtime": actual}

    if release_runtime.get("phowhisper_model") != expected_asr_model:
        runtime_drift["benchmark_phowhisper_model"] = {
            "release": expected_asr_model,
            "benchmark": release_runtime.get("phowhisper_model"),
        }
    if release_runtime.get("phowhisper_revision") != expected_asr_revision:
        runtime_drift["benchmark_phowhisper_revision"] = {
            "release": expected_asr_revision,
            "benchmark": release_runtime.get("phowhisper_revision"),
        }
    if runtime_drift:
        raise ValueError(f"Locked Whisper P1 runtime configuration drift: {runtime_drift}")

    weight_path = expected_phobert_dir / "model.safetensors"
    weight_relative = "phobert/weights/model.safetensors"
    config_relative = "phobert/weights/config.json"
    asr_weight_relative = f"{packaged_asr_relative}/pytorch_model.bin"
    model_config = json.loads(
        (expected_phobert_dir / "config.json").read_text(encoding="utf-8")
    )
    if model_config.get("model_type") != "roberta":
        raise ValueError("Packaged PhoBERT config model_type is not roberta")
    if model_config.get("architectures") != ["RobertaForSequenceClassification"]:
        raise ValueError("Packaged PhoBERT classifier architecture drift")

    result: dict[str, Any] = {
        "verdict": "PASS",
        "release": manifest["release"],
        "source_commit": manifest["develop_source_commit"],
        "pretrained_asr": asr_spec,
        "fine_tuned_model": manifest["fine_tuned_model"],
        "best_checkpoint": manifest["best_checkpoint"],
        "verified_file_count": len(verified_files),
        "critical_artifacts": {
            weight_relative: verified_files[weight_relative],
            config_relative: verified_files[config_relative],
            asr_weight_relative: verified_files[asr_weight_relative],
            "keywords.json": verified_files["keywords.json"],
        },
        "archive": archive_result,
        "runtime": {
            "phowhisper_model": PHOWHISPER_MODEL,
            "phowhisper_repository": PHOWHISPER_REPO_ID,
            "phowhisper_revision": PHOWHISPER_REVISION,
            "phobert_model_dir": str(PHOBERT_MODEL_DIR),
            "phobert_cheating_threshold": PHOBERT_CHEATING_THRESHOLD,
        },
        "models_loaded": False,
        "cpu_forward_pass": False,
        "packaged_asr_verified": False,
    }
    if not load_models:
        return result

    import numpy as np
    import torch
    import transformers

    from backend.ai_services.whisper.keyword_detector import KeywordDetector
    from backend.ai_services.whisper.phobert.decision_fusion import (
        DecisionFusionService,
    )
    from backend.ai_services.whisper.phobert.phobert_service import PhobertService
    from backend.ai_services.whisper.phowhisper_service import PhoWhisperService

    phobert = PhobertService(model_path=weight_path.parent, device="cpu")
    phobert_result = phobert.predict("hôm nay trời nóng quá")
    probabilities = phobert_result.get("all_probs", {})
    if set(probabilities) != {"Normal", "Cheating"}:
        raise RuntimeError(f"PhoBERT returned invalid labels: {probabilities}")
    probability_sum = sum(float(value) for value in probabilities.values())
    if not math.isfinite(probability_sum) or not math.isclose(
        probability_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5
    ):
        raise RuntimeError(f"PhoBERT returned invalid probabilities: {probabilities}")

    keyword_result = KeywordDetector().detect("hôm nay trời nóng quá", timestamp=0.0)
    if keyword_result.get("risk") not in {"safe", "low", "medium", "high"}:
        raise RuntimeError(f"Keyword detector returned invalid risk: {keyword_result}")
    fusion = DecisionFusionService(
        ai_cheating_threshold=PHOBERT_CHEATING_THRESHOLD
    ).fuse("hôm nay trời nóng quá", keyword_result["risk"], phobert_result)
    if fusion.get("final_label") not in {"Normal", "Cheating"}:
        raise RuntimeError(f"Decision fusion returned invalid label: {fusion}")

    try:
        phowhisper = PhoWhisperService(
            model_name=str(expected_asr_dir),
            revision=expected_asr_revision,
            device="cpu",
            local_files_only=True,
        )
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(
            "Packaged PhoWhisper files are incomplete or cannot be loaded from "
            f"{expected_asr_dir}. Release verification will not download them."
        ) from exc
    asr_result = phowhisper.transcribe(np.zeros(1600, dtype=np.float32))
    if asr_result.get("language") != "vi" or not isinstance(
        asr_result.get("text"), str
    ):
        raise RuntimeError(
            f"PhoWhisper CPU forward pass returned invalid output: {asr_result}"
        )

    result.update(
        {
            "models_loaded": True,
            "cpu_forward_pass": True,
            "packaged_asr_verified": True,
            "smoke_predictions": {
                "phobert": phobert_result,
                "keyword_risk": keyword_result["risk"],
                "fusion_label": fusion["final_label"],
                "silent_asr_text": asr_result["text"],
            },
            "verified_versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
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
        help="Skip loading PhoBERT/PhoWhisper and their CPU forward passes.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify_whisper_release(load_models=not args.integrity_only),
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

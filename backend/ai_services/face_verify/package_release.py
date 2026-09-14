"""Build the self-contained academic Face P1 hand-off package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

from backend.core.config import settings


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
RELEASE_CONFIG = MODULE_DIR / "FACE_P1_RELEASE.json"
MODEL_DIR = REPO_ROOT / "data" / "insightface" / "models" / "buffalo_l"
GALLERY_PATH = REPO_ROOT / "data" / "student_faces" / "gallery_embeddings.npz"
RESULTS_DIR = REPO_ROOT / "data" / "face_benchmark_p1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, release_root: Path, relative: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = release_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def validate_locked_config(config: dict, metrics: dict) -> None:
    runtime = config["runtime"]
    expected = {
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
        for key, value in expected.items()
        if runtime.get(key) != value
    }
    selected = metrics["policies"]["selected_operating_point"]
    if not math.isclose(
        selected["threshold"], runtime["similarity_threshold"], abs_tol=1e-12
    ):
        drift["benchmark_threshold"] = {
            "release": runtime["similarity_threshold"],
            "metrics": selected["threshold"],
        }
    if not math.isclose(
        selected["margin_threshold"],
        runtime["runner_up_margin_threshold"],
        abs_tol=1e-12,
    ):
        drift["benchmark_margin"] = {
            "release": runtime["runner_up_margin_threshold"],
            "metrics": selected["margin_threshold"],
        }
    if drift:
        raise ValueError(f"Locked Face P1 configuration drift: {drift}")


def build_package(
    output_dir: Path,
    manifest_path: Path,
    include_models: bool,
) -> tuple[Path, Path, Path]:
    config = json.loads(RELEASE_CONFIG.read_text(encoding="utf-8"))
    metrics_path = RESULTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validate_locked_config(config, metrics)

    release_id = config["release_id"]
    output_dir = output_dir.resolve()
    release_root = (output_dir / release_id).resolve()
    if release_root.parent != output_dir or not release_root.name.startswith("face-p1-"):
        raise ValueError(f"Unsafe release target: {release_root}")
    if release_root.exists():
        shutil.rmtree(release_root)
    release_root.mkdir(parents=True)

    tracked_files = [
        MODULE_DIR / "face_verify.py",
        MODULE_DIR / "identity_guard.py",
        MODULE_DIR / "benchmark_video.py",
        MODULE_DIR / "package_release.py",
        MODULE_DIR / "test_face_verify_logic.py",
        MODULE_DIR / "test_identity_guard.py",
        MODULE_DIR / "test_benchmark_video.py",
        MODULE_DIR / "test_webcam.py",
        MODULE_DIR / "BENCHMARK_P1.md",
        MODULE_DIR / "README_PACKAGE.md",
        MODULE_DIR / "FACE_P1_RELEASE.json",
        MODULE_DIR / "requirements-benchmark.txt",
        REPO_ROOT / "backend" / "core" / "config.py",
        REPO_ROOT / "backend" / "ai_services" / "pose_gaze" / "paper_pipeline.py",
        REPO_ROOT / "requirements.txt",
    ]
    for source in tracked_files:
        copy_file(source, release_root, source.relative_to(REPO_ROOT))

    for source in sorted(RESULTS_DIR.iterdir()):
        if source.is_file():
            copy_file(source, release_root, Path("benchmark_results") / source.name)

    copy_file(
        GALLERY_PATH,
        release_root,
        Path("runtime_data/student_faces/gallery_embeddings.npz"),
    )
    if include_models:
        for model_name in ("det_10g.onnx", "w600k_r50.onnx"):
            copy_file(
                MODEL_DIR / model_name,
                release_root,
                Path("runtime_data/insightface/models/buffalo_l") / model_name,
            )

    copied_manifest = copy_file(
        manifest_path,
        release_root,
        Path("benchmark_input") / manifest_path.name,
    )
    checksums_csv = release_root / "benchmark_input" / "source_checksums.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with checksums_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["data_path", "exists", "size_bytes", "sha256"],
        )
        writer.writeheader()
        for row in rows:
            source = Path(row["data_path"])
            writer.writerow(
                {
                    "data_path": str(source),
                    "exists": source.is_file(),
                    "size_bytes": source.stat().st_size if source.is_file() else "",
                    "sha256": sha256(source) if source.is_file() else "",
                }
            )

    expected_hashes = config["sha256"]
    actual_locked = {
        "det_10g.onnx": sha256(MODEL_DIR / "det_10g.onnx"),
        "w600k_r50.onnx": sha256(MODEL_DIR / "w600k_r50.onnx"),
        "gallery_embeddings.npz": sha256(GALLERY_PATH),
    }
    if actual_locked != expected_hashes:
        raise ValueError(
            f"Locked model/gallery checksum mismatch: expected={expected_hashes}, actual={actual_locked}"
        )

    inventory = []
    for path in sorted(item for item in release_root.rglob("*") if item.is_file()):
        inventory.append(
            {
                "path": path.relative_to(release_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    package_manifest = {
        "release_id": release_id,
        "includes_pretrained_models": include_models,
        "raw_videos_included": False,
        "source_manifest_sha256": sha256(copied_manifest),
        "files": inventory,
    }
    (release_root / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(package_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    archive = output_dir / f"{release_id}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(item for item in release_root.rglob("*") if item.is_file()):
            bundle.write(path, Path(release_id) / path.relative_to(release_root))
    archive_checksum = archive.with_suffix(".zip.sha256")
    archive_checksum.write_text(
        f"{sha256(archive)}  {archive.name}\n",
        encoding="ascii",
    )
    return release_root, archive, archive_checksum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "dist")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--without-models", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    folder, zip_path, checksum_path = build_package(
        args.output_dir,
        args.manifest,
        include_models=not args.without_models,
    )
    print(
        json.dumps(
            {
                "folder": str(folder),
                "archive": str(zip_path),
                "archive_sha256": str(checksum_path),
            },
            indent=2,
        )
    )

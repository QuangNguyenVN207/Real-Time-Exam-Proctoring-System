from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
DEFAULT_OUTPUT = MODULE_DIR / "releases" / "whisper_p1_20260914.zip"

REQUIRED_FILES = [
    "P1_README.md",
    "requirements-p1.txt",
    "config.py",
    "audio_pipeline.py",
    "audio_utils.py",
    "audio_logger.py",
    "vad_service.py",
    "phowhisper_service.py",
    "evaluate_vivos.py",
    "keyword_detector.py",
    "keywords.json",
    "context_words.py",
    "negative_rules.py",
    "rules.py",
    "text_utils.py",
    "phobert/train_phobert.py",
    "phobert/evaluate_p1.py",
    "phobert/phobert_service.py",
    "phobert/decision_fusion.py",
    "phobert/dataset/cheating.csv",
    "phobert/weights/config.json",
    "phobert/weights/model.safetensors",
    "phobert/weights/tokenizer_config.json",
    "phobert/weights/vocab.txt",
    "phobert/weights/bpe.codes",
    "phobert/weights/added_tokens.json",
    "phobert/weights/checkpoint-235/config.json",
    "phobert/weights/checkpoint-235/trainer_state.json",
    "phobert/weights/checkpoint-235/rng_state.pth",
    "phobert/weights/checkpoint-235/scheduler.pt",
    "phobert/weights/checkpoint-235/training_args.bin",
    "phobert/weights/checkpoint-423/config.json",
    "phobert/weights/checkpoint-423/trainer_state.json",
    "tests/test_pipeline.py",
    "tests/test_p1_logic.py",
]

OPTIONAL_FILES = [
    "phobert/weights/checkpoint-235/optimizer.pt",
]

OUTPUT_DIRS = [
    "outputs/p1_phobert",
    "outputs/p1_audio_pipeline",
    "samples",
]

EXPECTED_SHA256 = {
    "phobert/weights/model.safetensors": "9f321d440ae1112a89a1618ab34202e3185583d620c6c1c01b786b45e7d9c2f0",
    "phobert/weights/config.json": "e95f5e1f59392822a51f295c9c8bb2ad3cb9a40ea2489ec0e06e55d1e81f3246",
    "phobert/weights/checkpoint-235/trainer_state.json": "84428ed9bc04786cf6507310a83a1c31bf305d5a48c65ed1cb14ede0f458acf6",
    "phobert/weights/checkpoint-423/trainer_state.json": "223b624f0d5144e636234a85994a05ad2de1002f4bc7336ec90fa458c2c6f52e",
    "phobert/dataset/cheating.csv": "02df077da283948aace169058f0b665619d6cce46bc0eb705b83324a9451a8d8",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_files() -> list[Path]:
    files = [MODULE_DIR / relative for relative in REQUIRED_FILES]
    files.extend(
        MODULE_DIR / relative
        for relative in OPTIONAL_FILES
        if (MODULE_DIR / relative).is_file()
    )
    for relative_dir in OUTPUT_DIRS:
        directory = MODULE_DIR / relative_dir
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Required release directory is missing: {directory}. "
                "Run both P1 evaluators before packaging."
            )
        files.extend(path for path in directory.rglob("*") if path.is_file())

    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required P1 files:\n" + "\n".join(str(path) for path in missing)
        )
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the self-contained Whisper P1 release archive.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files()
    for relative, expected_hash in EXPECTED_SHA256.items():
        actual_hash = sha256_file(MODULE_DIR / relative)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Integrity check failed for {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    manifest_files = []
    for path in files:
        relative = path.relative_to(MODULE_DIR).as_posix()
        manifest_files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    optimizer_included = any(
        item["path"] == "phobert/weights/checkpoint-235/optimizer.pt"
        for item in manifest_files
    )

    manifest = {
        "release": "whisper-p1",
        "develop_source_commit": "9b22e939543193c6d9eef6306cf7a626d063051b",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_layout": "Files are rooted at backend/ai_services/whisper inside the ZIP.",
        "fine_tuned_model": "vinai/phobert-base-v2",
        "pretrained_asr": "vinai/PhoWhisper-small@a86b604c346caf7148c37512eafe783a16420adb",
        "best_checkpoint": "checkpoint-235 (epoch 5)",
        "packaged_weight": "model.safetensors recovered from checkpoint-235",
        "early_stop_checkpoint": "checkpoint-423 (epoch 9)",
        "optimizer_state": (
            "included for exact training resume"
            if optimizer_included
            else "not required for inference; optimizer.pt is currently absent and optional"
        ),
        "file_count": len(manifest_files),
        "files": manifest_files,
    }
    manifest_path = output.with_suffix(".manifest.json")

    zip_root = Path("backend/ai_services/whisper")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, (zip_root / path.relative_to(MODULE_DIR)).as_posix())
        archive.writestr(
            (zip_root / "WHISPER_P1_RELEASE.json").as_posix(),
            json.dumps(manifest, indent=2),
        )

    archive_hash = sha256_file(output)
    external_manifest = {
        **manifest,
        "archive": {
            "path": output.name,
            "bytes": output.stat().st_size,
            "sha256": archive_hash,
        },
    }
    manifest_path.write_text(
        json.dumps(external_manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": archive_hash,
        "manifest": str(manifest_path),
        "file_count": len(files),
    }, indent=2))


if __name__ == "__main__":
    main()

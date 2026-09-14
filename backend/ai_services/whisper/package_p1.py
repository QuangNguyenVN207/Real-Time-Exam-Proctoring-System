from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from backend.ai_services.whisper.config import (
    PACKAGED_PHOWHISPER_DIR,
    PHOWHISPER_REPO_ID,
    PHOWHISPER_REVISION,
)


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
DEFAULT_OUTPUT = MODULE_DIR / "releases" / "whisper_p1_20260914.zip"
ZIP_TIMESTAMP = (2026, 9, 14, 0, 0, 0)
RELEASE_CREATED_AT_UTC = "2026-09-14T08:04:35.358491+00:00"
TEXT_SUFFIXES = {".codes", ".csv", ".json", ".md", ".py", ".txt"}

REQUIRED_PHOWHISPER_FILES = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "normalizer.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]

REQUIRED_FILES = [
    "P1_README.md",
    "verify_release.py",
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

# Text entries use LF-canonical hashes so Windows core.autocrlf cannot change
# release identity. Binary entries always use their exact raw SHA-256.
EXPECTED_SHA256 = {
    "phobert/weights/model.safetensors": "9f321d440ae1112a89a1618ab34202e3185583d620c6c1c01b786b45e7d9c2f0",
    "phobert/weights/config.json": "d5784bc547fa8d157f9391a1253a925b989029415d95e564fa1a498b2105e17c",
    "phobert/weights/checkpoint-235/trainer_state.json": "022cf64c8725b2acbd977e022466bb0aef684e075c2c6b0d3457fe76e1af240d",
    "phobert/weights/checkpoint-423/trainer_state.json": "a5d5afa619c94b7e63a5570c504bba6ee9ff72b8001cd4d3749d56f1b628eff2",
    "phobert/dataset/cheating.csv": "9d08b18316c4d5cc84b65e509e24e869732025a37f68d3331cd70c072dfe1868",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_payload(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return payload


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload, compresslevel=6)


def write_path(archive: zipfile.ZipFile, name: str, path: Path) -> None:
    if path.suffix.lower() in TEXT_SUFFIXES:
        write_member(archive, name, release_payload(path))
        return
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def collect_files() -> list[Path]:
    files = [MODULE_DIR / relative for relative in REQUIRED_FILES]
    files.extend(PACKAGED_PHOWHISPER_DIR / name for name in REQUIRED_PHOWHISPER_FILES)
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
    parser = argparse.ArgumentParser(
        description="Create the self-contained Whisper P1 release archive."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files()
    for relative, expected_hash in EXPECTED_SHA256.items():
        locked_path = MODULE_DIR / relative
        actual_hash = (
            sha256_bytes(release_payload(locked_path))
            if locked_path.suffix.lower() in TEXT_SUFFIXES
            else sha256_file(locked_path)
        )
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Integrity check failed for {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    manifest_files = []
    for path in files:
        relative = path.relative_to(MODULE_DIR).as_posix()
        if path.suffix.lower() in TEXT_SUFFIXES:
            payload = release_payload(path)
            byte_count = len(payload)
            file_hash = sha256_bytes(payload)
        else:
            byte_count = path.stat().st_size
            file_hash = sha256_file(path)
        manifest_files.append(
            {
                "path": relative,
                "bytes": byte_count,
                "sha256": file_hash,
            }
        )
    optimizer_included = any(
        item["path"] == "phobert/weights/checkpoint-235/optimizer.pt"
        for item in manifest_files
    )

    manifest = {
        "release": "whisper-p1",
        "develop_source_commit": "9b22e939543193c6d9eef6306cf7a626d063051b",
        "created_at_utc": RELEASE_CREATED_AT_UTC,
        "archive_layout": "Files are rooted at backend/ai_services/whisper inside the ZIP.",
        "fine_tuned_model": "vinai/phobert-base-v2",
        "pretrained_asr": f"{PHOWHISPER_REPO_ID}@{PHOWHISPER_REVISION}",
        "pretrained_asr_packaged_path": PACKAGED_PHOWHISPER_DIR.relative_to(
            MODULE_DIR
        ).as_posix(),
        "self_contained_asr": True,
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
    with zipfile.ZipFile(output, "w") as archive:
        for path in files:
            write_path(
                archive,
                (zip_root / path.relative_to(MODULE_DIR)).as_posix(),
                path,
            )
        write_member(
            archive,
            (zip_root / "WHISPER_P1_RELEASE.json").as_posix(),
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
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
        json.dumps(external_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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

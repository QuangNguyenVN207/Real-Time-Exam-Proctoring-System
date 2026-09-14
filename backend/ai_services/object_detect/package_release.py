"""Build a deterministic, self-contained object-detection runtime archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from .verify_release import RELEASE_CONFIG, REPO_ROOT, sha256_file, verify_object_release


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = MODULE_DIR / "releases" / "object_detection_runtime_20260914.zip"
ZIP_TIMESTAMP = (2026, 9, 14, 0, 0, 0)


def _release_files() -> list[Path]:
    allowed_suffixes = {".py", ".md", ".json", ".txt"}
    files = [
        path
        for path in MODULE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed_suffixes
        and "releases" not in path.parts
        and "__pycache__" not in path.parts
    ]
    files.extend(
        [
            REPO_ROOT / "backend" / "core" / "config.py",
            REPO_ROOT / "backend" / "ai_services" / "webcam_utils.py",
            REPO_ROOT / "weights" / "best (1).pt",
        ]
    )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing object release files: " + ", ".join(map(str, missing)))
    return sorted(set(files), key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def _write_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload, compresslevel=9)


def build_package(output: Path) -> tuple[Path, Path]:
    verification = verify_object_release(load_model=True)
    config = json.loads(RELEASE_CONFIG.read_text(encoding="utf-8"))
    files = _release_files()
    inventory = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    package_manifest = {
        "release_id": config["release_id"],
        "archive_layout": "Extract at the repository root.",
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "verification": verification,
        "files": inventory,
    }
    manifest_bytes = json.dumps(
        package_manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w") as archive:
        for path in files:
            _write_member(
                archive,
                path.relative_to(REPO_ROOT).as_posix(),
                path.read_bytes(),
            )
        _write_member(archive, "OBJECT_PACKAGE_MANIFEST.json", manifest_bytes)

    external_manifest = output.with_suffix(".manifest.json")
    external_manifest.write_text(
        json.dumps(
            {
                **package_manifest,
                "archive": {
                    "path": output.name,
                    "size_bytes": output.stat().st_size,
                    "sha256": sha256_file(output),
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output, external_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    archive, manifest = build_package(args.output)
    print(
        json.dumps(
            {
                "archive": str(archive),
                "archive_sha256": sha256_file(archive),
                "manifest": str(manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

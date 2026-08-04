from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from .common import processed_root
from .extract import build_frames
from .manifest import build_manifest, discover_raw_root
from .negatives import mine_negatives
from .report import generate_report
from .split import build_splits
from .windows import build_windows


def _stage_manifest(raw_root: Path, output_dir: Path, limit: int | None) -> None:
    build_manifest(raw_root=raw_root, output_path=output_dir / "manifest.parquet", limit=limit, strict=limit is None)


def _stage_extract(output_dir: Path, limit: int | None) -> None:
    build_frames(manifest_path=output_dir / "manifest.parquet", output_path=output_dir / "frames.parquet", limit=limit)


def _stage_split(output_dir: Path, limit: int | None) -> None:
    build_splits(manifest_path=output_dir / "manifest.parquet", output_path=output_dir / "splits.parquet", strict=limit is None)


def _stage_windows(output_dir: Path) -> None:
    build_windows(frames_path=output_dir / "frames.parquet", output_path=output_dir / "windows.parquet", scaler_path=output_dir / "scaler.json", vocab_path=output_dir / "vocab.json")


def _stage_negatives(output_dir: Path) -> None:
    mine_negatives(frames_path=output_dir / "frames.parquet", output_csv=output_dir / "mined_negatives.csv", review_dir=output_dir / "review")


def _stage_report(output_dir: Path) -> None:
    generate_report(processed_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pose/gaze dataset artifacts")
    parser.add_argument("--stage", choices=["all", "manifest", "extract", "split", "windows", "negatives", "report"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=processed_root())
    args = parser.parse_args()

    raw_root = args.raw_root or discover_raw_root()
    output_dir = args.processed_root
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "manifest":
        _stage_manifest(raw_root, output_dir, args.limit)
        return
    if args.stage == "extract":
        _stage_manifest(raw_root, output_dir, args.limit)
        _stage_extract(output_dir, args.limit)
        return
    if args.stage == "split":
        _stage_split(output_dir, args.limit)
        return
    if args.stage == "windows":
        _stage_windows(output_dir)
        return
    if args.stage == "negatives":
        _stage_negatives(output_dir)
        return
    if args.stage == "report":
        _stage_report(output_dir)
        return

    _stage_manifest(raw_root, output_dir, args.limit)
    _stage_extract(output_dir, args.limit)
    _stage_split(output_dir, args.limit)
    _stage_windows(output_dir)
    _stage_negatives(output_dir)
    _stage_report(output_dir)


if __name__ == "__main__":
    main()
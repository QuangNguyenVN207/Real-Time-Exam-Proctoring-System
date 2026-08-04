"""Validate that train, val, test windows have no source overlap."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_windows(root: Path) -> pd.DataFrame:
    path = root / "windows.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    frame = pd.read_parquet(path)
    required = {"window_id", "clip_id", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} thiếu cột: {sorted(missing)}")
    return frame


def validate_split_roots(processed_root: Path, splits: tuple[str, ...]) -> pd.DataFrame:
    parts = []
    for split in splits:
        frame = _load_windows(processed_root / split).copy()
        frame["source_split"] = split
        parts.append(frame)
    windows = pd.concat(parts, ignore_index=True)

    inconsistent = windows.loc[windows["split"].astype(str) != windows["source_split"]]
    if not inconsistent.empty:
        raise ValueError("windows.parquet có split không khớp thư mục chứa nó")

    duplicate_windows = windows[windows.duplicated("window_id", keep=False)]
    if not duplicate_windows.empty:
        ids = sorted(duplicate_windows["window_id"].astype(str).unique())
        raise ValueError(f"window_id xuất hiện ở nhiều split: {ids[:10]}")

    clip_split_count = windows.groupby("clip_id")["source_split"].nunique()
    shared_clips = clip_split_count[clip_split_count > 1]
    if not shared_clips.empty:
        raise ValueError(
            f"clip_id xuất hiện ở nhiều split: {sorted(shared_clips.index.astype(str))}"
        )

    start_col = next((c for c in ("start_frame", "start_frame_idx", "source_start_frame") if c in windows), None)
    end_col = next((c for c in ("end_frame", "end_frame_idx", "source_end_frame") if c in windows), None)
    if start_col and end_col:
        ordered = windows.sort_values(["clip_id", start_col, end_col])
        for clip_id, group in ordered.groupby("clip_id", sort=False):
            previous_end = None
            previous_split = None
            for _, row in group.iterrows():
                start = int(row[start_col])
                end = int(row[end_col])
                if previous_end is not None and start <= previous_end and row["source_split"] != previous_split:
                    raise ValueError(f"window frame overlap giữa split tại clip_id={clip_id}")
                previous_end = max(previous_end or end, end)
                previous_split = row["source_split"]

    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra split train/val/test không chồng lấn")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    windows = validate_split_roots(args.processed_root, tuple(args.splits))
    print(f"OK: {len(windows)} windows; splits={sorted(windows['source_split'].unique())}")


if __name__ == "__main__":
    main()
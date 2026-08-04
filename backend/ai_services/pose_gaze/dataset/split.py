from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import processed_root, split_for_subjects


def assign_split(subject_ids: tuple[str, ...] | list[str] | set[str]) -> str | None:
    return split_for_subjects(subject_ids)


def build_splits(*, manifest_path: Path | None = None, output_path: Path | None = None, strict: bool = True) -> pd.DataFrame:
    manifest_path = manifest_path or (processed_root() / "manifest.parquet")
    output_path = output_path or (processed_root() / "splits.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(manifest_path)
    subject_ids = manifest["subject_ids"].fillna("").map(lambda value: tuple(token for token in str(value).split(",") if token))
    splits = subject_ids.map(assign_split)

    split_frame = manifest.copy()
    split_frame["split"] = splits
    split_frame["subject_ids_tuple"] = subject_ids
    split_frame["kept"] = split_frame["split"].notna()
    kept = split_frame.loc[split_frame["kept"]].copy()

    kept.to_parquet(output_path, index=False)

    if strict:
        observed_splits = kept.groupby("split")["video_stem"].nunique().to_dict()
        if set(observed_splits) != {"train", "val", "test"}:
            raise AssertionError(f"Split coverage is incomplete: {sorted(observed_splits)}")

        class_codes = set(kept["class_code"].unique())
        if class_codes != {"c1", "c2", "c3", "c4", "c5"}:
            raise AssertionError(f"All five classes must be preserved, got {sorted(class_codes)}")

    return kept

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import processed_root


def _maybe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table(name: str, frame: pd.DataFrame, output_dir: Path) -> None:
    table_path = output_dir / f"{name}.csv"
    print(f"\n[{name}]")
    print(frame.to_string(index=False) if not frame.empty else "(empty)")
    frame.to_csv(table_path, index=False)


def _save_figure(path: Path, figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def generate_report(*, processed_dir: Path | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    processed_dir = processed_dir or processed_root()
    figures_dir = processed_dir / "figures"
    tables_dir = processed_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    manifest = _maybe_read(processed_dir / "manifest.parquet")
    frames = _maybe_read(processed_dir / "frames.parquet")
    windows = _maybe_read(processed_dir / "windows.parquet")
    splits = _maybe_read(processed_dir / "splits.parquet")
    negatives = _maybe_read(processed_dir / "mined_negatives.csv")

    tables = {
        "manifest_counts": manifest.groupby("class_code").size().reset_index(name="rows") if not manifest.empty else pd.DataFrame(columns=["class_code", "rows"]),
        "frame_quality": frames.groupby("class_code").agg(rows=("frame_id", "size"), quality_rate=("quality_ok", "mean"), mean_quality=("quality_score", "mean")).reset_index() if not frames.empty else pd.DataFrame(columns=["class_code", "rows", "quality_rate", "mean_quality"]),
        "window_summary": windows.groupby("split").agg(rows=("window_index", "size"), quality_rate=("quality_ok", "mean")).reset_index() if not windows.empty else pd.DataFrame(columns=["split", "rows", "quality_rate"]),
        "split_summary": splits.groupby("split").size().reset_index(name="rows") if not splits.empty else pd.DataFrame(columns=["split", "rows"]),
        "negative_summary": negatives.groupby("class_code").size().reset_index(name="rows") if not negatives.empty else pd.DataFrame(columns=["class_code", "rows"]),
    }

    for name, frame in tables.items():
        _write_table(name, frame, tables_dir)

    sns.set_theme(style="whitegrid")

    if not manifest.empty:
        figure, axis = plt.subplots(figsize=(8, 4))
        sns.countplot(data=manifest, x="class_code", ax=axis)
        axis.set_title("Video count by class")
        _save_figure(figures_dir / "manifest_class_counts.png", figure)
        plt.close(figure)

    if not frames.empty:
        figure, axis = plt.subplots(figsize=(10, 4))
        sns.barplot(data=frames.groupby("class_code").quality_ok.mean().reset_index(name="quality_rate"), x="class_code", y="quality_rate", ax=axis)
        axis.set_ylim(0, 1)
        axis.set_title("Quality rate by class")
        _save_figure(figures_dir / "frame_quality_rate.png", figure)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10, 4))
        sns.histplot(data=frames, x="delta_yaw", hue="class_code", element="step", stat="density", common_norm=False, ax=axis)
        axis.set_title("Delta yaw distribution")
        _save_figure(figures_dir / "delta_yaw_distribution.png", figure)
        plt.close(figure)

    if not windows.empty:
        figure, axis = plt.subplots(figsize=(8, 4))
        sns.countplot(data=windows, x="split", ax=axis)
        axis.set_title("Window count by split")
        _save_figure(figures_dir / "window_counts_by_split.png", figure)
        plt.close(figure)

    if not negatives.empty:
        figure, axis = plt.subplots(figsize=(8, 4))
        sns.countplot(data=negatives, x="class_code", ax=axis)
        axis.set_title("Mined negative onsets by class")
        _save_figure(figures_dir / "negative_onsets_by_class.png", figure)
        plt.close(figure)

    if not splits.empty:
        figure, axis = plt.subplots(figsize=(8, 4))
        sns.barplot(data=splits.groupby("split").size().reset_index(name="rows"), x="split", y="rows", ax=axis)
        axis.set_title("Rows kept by split")
        _save_figure(figures_dir / "kept_rows_by_split.png", figure)
        plt.close(figure)

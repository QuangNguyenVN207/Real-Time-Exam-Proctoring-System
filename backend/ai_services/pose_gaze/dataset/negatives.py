from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import processed_root


def _rolling_onset_score(window: pd.DataFrame) -> float:
    deltas = pd.concat(
        [
            pd.to_numeric(window["delta_yaw"], errors="coerce").abs(),
            pd.to_numeric(window["delta_pitch"], errors="coerce").abs(),
            pd.to_numeric(window["delta_roll"], errors="coerce").abs(),
        ],
        axis=1,
    ).fillna(0.0)
    return float(deltas.mean(axis=1).mean())


def _render_review_sheet(review_dir: Path, row: dict[str, object]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    review_dir.mkdir(parents=True, exist_ok=True)
    figure_path = review_dir / f"{row['video_stem']}_track_{row['track_id']}_review.png"
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.axis("off")
    axis.text(0.02, 0.95, f"video: {row['video_stem']}", va="top", fontsize=14)
    axis.text(0.02, 0.82, f"track_id: {row['track_id']}", va="top", fontsize=14)
    axis.text(0.02, 0.69, f"action_start_ms: {row['action_start_ms']}", va="top", fontsize=14)
    axis.text(0.02, 0.56, f"onset_score: {row['onset_score']:.3f}", va="top", fontsize=14)
    axis.text(0.02, 0.43, f"class_code: {row['class_code']}", va="top", fontsize=14)
    axis.text(0.02, 0.30, f"reviewed: {row['reviewed']}", va="top", fontsize=14)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    return figure_path


def mine_negatives(*, frames_path: Path | None = None, output_csv: Path | None = None, review_dir: Path | None = None, min_onset_score: float = 0.35) -> pd.DataFrame:
    frames_path = frames_path or (processed_root() / "frames.parquet")
    output_csv = output_csv or (processed_root() / "mined_negatives.csv")
    review_dir = review_dir or (processed_root() / "review")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    frames = pd.read_parquet(frames_path)
    records: list[dict[str, object]] = []

    for (video_stem, track_id), group in frames.sort_values(["timestamp_ms", "frame_id"]).groupby(["video_stem", "track_id"]):
        group = group.reset_index(drop=True)
        start_index = 0
        window_rows: list[dict[str, object]] = []
        while start_index < len(group):
            start_ms = int(group.loc[start_index, "timestamp_ms"])
            window = group.loc[(group["timestamp_ms"] >= start_ms) & (group["timestamp_ms"] < start_ms + 2000)]
            if len(window) < 3:
                break
            onset_score = _rolling_onset_score(window)
            sustained = False
            if onset_score >= min_onset_score:
                next_window = group.loc[(group["timestamp_ms"] >= start_ms + 1000) & (group["timestamp_ms"] < start_ms + 3000)]
                third_window = group.loc[(group["timestamp_ms"] >= start_ms + 2000) & (group["timestamp_ms"] < start_ms + 4000)]
                sustained = len(next_window) > 0 and len(third_window) > 0 and _rolling_onset_score(next_window) >= min_onset_score and _rolling_onset_score(third_window) >= min_onset_score
            if sustained:
                row = {
                    "video_stem": video_stem,
                    "track_id": int(track_id),
                    "class_code": str(group.loc[0, "class_code"]),
                    "action_start_ms": start_ms,
                    "onset_score": onset_score,
                    "reviewed": False,
                }
                records.append(row)
                window_rows.append(row)
                _render_review_sheet(review_dir, row)
                break
            start_index += max(1, len(window) // 2)

    negatives = pd.DataFrame(records, columns=["video_stem", "track_id", "class_code", "action_start_ms", "onset_score", "reviewed"])
    negatives.to_csv(output_csv, index=False)

    if not negatives.empty:
        output_csv.with_suffix(".json").write_text(json.dumps({"rows": len(negatives)}, ensure_ascii=False, indent=2), encoding="utf-8")

    return negatives

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .build import _classify_window, _window_bounds


def main() -> None:
    root = Path("data/processed/stage1")
    manifest = pd.read_csv("data/processed/stage0_manifest.csv")
    sampled = pd.read_parquet(root / "sampled_frames.parquet")
    rows: list[dict[str, object]] = []
    for clip_id, group in sampled.groupby("clip_id", sort=False):
        group = group.sort_values("frame_index").reset_index(drop=True)
        meta = manifest.loc[manifest["filename"] == group.loc[0, "filename"]].iloc[0]
        action_start = float(meta["action_start_s"]) if pd.notna(meta["action_start_s"]) else None
        action_end = float(meta["action_end_s"]) if pd.notna(meta["action_end_s"]) else None
        duration = float(meta["duration_s"])
        for window_index, (start, end) in enumerate(_window_bounds(len(group))):
            window_start = float(group.loc[start, "timestamp_s"])
            window_end = float(group.loc[end - 1, "timestamp_s"])
            label, include = _classify_window(window_start, window_end, action_start, action_end, duration)
            if label == "mixed":
                continue
            frame_dir = root / "frames" / str(clip_id) / f"w{window_index:03d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for local, source_index in enumerate(range(start, end)):
                source = root / "frames" / str(clip_id) / str(group.loc[source_index, "frame_path"]).split("/", 1)[-1]
                target = frame_dir / f"frame_{local:04d}.png"
                if not target.exists():
                    shutil.copyfile(source, target)
            rows.append({
                "window_id": f"{clip_id}_w{window_index:03d}", "clip_id": clip_id,
                "filename": group.loc[0, "filename"], "class_code": meta["class_code"],
                "label": label, "include_in_training": include,
                "window_start_s": round(window_start, 3), "window_end_s": round(window_end, 3),
                "window_size_frames": 30, "window_overlap_frames": 15, "target_fps": 10.0,
                "frame_count": end - start, "frame_dir": str(Path(str(clip_id)) / f"w{window_index:03d}").replace("\\", "/"),
                "action_start_s": action_start, "action_end_s": action_end,
            })
    windows = pd.DataFrame(rows)
    windows.to_parquet(root / "windows.parquet", index=False)
    report = pd.DataFrame([{
        "clips": sampled["clip_id"].nunique(), "sampled_frames": len(sampled), "windows": len(windows),
        "action_windows": int((windows["label"] == "action").sum()),
        "negative_pre_windows": int((windows["label"] == "negative_pre").sum()),
        "negative_post_windows": int((windows["label"] == "negative_post").sum()),
        "target_fps": 10.0, "window_frames": 30, "window_overlap_frames": 15,
    }])
    report.to_csv(root / "stage1_report.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()

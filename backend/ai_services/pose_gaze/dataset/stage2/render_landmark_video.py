"""Render video từ frame Stage 1 và landmark NPZ Stage 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from .artifact import load_raw_npz
from .export_csv import _draw_track, _select_front_tracks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--clip", required=True, help="Tên clip WIN")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stage1 = Path("data/processed") / args.split
    stage2 = Path("data/processed/stage2_landmarks") / args.split
    windows = pd.read_parquet(stage2 / "features" / "windows.parquet")
    windows = windows[windows["clip_id"].astype(str) == args.clip].sort_values("window_id")
    selected = pd.read_parquet(stage1 / "selected_frames.parquet")
    selected = selected[selected["clip_id"].astype(str) == args.clip]
    if selected.empty or windows.empty:
        raise FileNotFoundError(f"Không có frame/landmark mới cho {args.split}/{args.clip}")

    output_frames = []
    for _, window in windows.iterrows():
        raw_dir = stage2 / "raw" / args.clip / str(window["window_id"])
        paths = sorted(raw_dir.glob("track_*.npz"))
        if not paths:
            continue
        tracks = [load_raw_npz(path) for path in paths]
        for frame_idx in range(min(len(track["pose_frame_lm"]) for track in tracks)):
            frame_row = selected[
                (selected["window_id"].astype(str) == str(window["window_id"]))
                & (selected["window_local_index"] == frame_idx)
            ]
            if frame_row.empty:
                continue
            path = Path(str(frame_row.iloc[0]["frame_path"]))
            if not path.is_absolute():
                path = stage1 / "frames" / path
            image = cv2.imread(str(path))
            if image is None:
                continue
            tracks_now = _select_front_tracks(tracks, frame_idx)
            for index, track in enumerate(tracks_now):
                _draw_track(image, track, frame_idx, [(40, 210, 40), (255, 255, 255)][index], f"Track {int(track['track_id'][frame_idx])}")
            output_frames.append(image)

    if not output_frames:
        raise RuntimeError("Không tạo được frame visualization")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    height, width = output_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (width, height))
    for image in output_frames:
        writer.write(image)
    writer.release()
    print(f"OK: {args.output} ({len(output_frames)} frames, {args.split}/{args.clip})")


if __name__ == "__main__":
    main()
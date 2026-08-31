"""Extract diverse video frames for manual paper-instance annotation.

Every physical sheet must receive its own polygon, including authorized exam
papers.  The count policy, not the model class name, decides which instance is
a cheat sheet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames for a one-class paper segmentation dataset."
    )
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/paper_segmentation_annotation/images"),
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Base sampling rate over each complete video.",
    )
    parser.add_argument(
        "--hard-fps",
        type=float,
        default=3.0,
        help="Sampling rate inside --hard-interval ranges.",
    )
    parser.add_argument(
        "--hard-interval",
        action="append",
        default=[],
        metavar="VIDEO_STEM:START:END",
        help=(
            "Increase sampling in a difficult interval, for example "
            "cheatsheet:40:68. May be supplied multiple times."
        ),
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def _parse_intervals(values: list[str]) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = {}
    for value in values:
        try:
            stem, raw_start, raw_end = value.rsplit(":", 2)
            start = float(raw_start)
            end = float(raw_end)
        except ValueError as error:
            raise ValueError(
                f"Invalid hard interval {value!r}; expected STEM:START:END"
            ) from error
        if start < 0 or end <= start:
            raise ValueError(f"Invalid hard interval bounds: {value!r}")
        intervals.setdefault(stem.lower(), []).append((start, end))
    return intervals


def _in_hard_interval(
    stem: str,
    second: float,
    intervals: dict[str, list[tuple[float, float]]],
) -> bool:
    return any(
        start <= second <= end
        for start, end in intervals.get(stem.lower(), [])
    )


def extract_video(
    video_path: Path,
    *,
    output_dir: Path,
    sample_fps: float,
    hard_fps: float,
    hard_intervals: dict[str, list[tuple[float, float]]],
    jpeg_quality: int,
) -> list[dict[str, Any]]:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        capture.release()
        raise RuntimeError(f"Video has invalid FPS: {video_path}")

    base_step = max(1, round(source_fps / sample_fps))
    hard_step = max(1, round(source_fps / hard_fps))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    frame_id = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_id += 1
            second = frame_id / source_fps
            is_hard = _in_hard_interval(
                video_path.stem,
                second,
                hard_intervals,
            )
            step = hard_step if is_hard else base_step
            if frame_id % step != 0:
                continue
            filename = (
                f"{video_path.stem}__f{frame_id:06d}__"
                f"t{second:08.3f}.jpg"
            )
            output_path = output_dir / filename
            written = cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if not written:
                raise RuntimeError(f"Could not write extracted frame: {output_path}")
            manifest.append(
                {
                    "image": filename,
                    "source_video": str(video_path.resolve()),
                    "source_frame_id": frame_id,
                    "second": round(second, 3),
                    "hard_interval": is_hard,
                }
            )
    finally:
        capture.release()
    return manifest


def main() -> None:
    args = _arguments()
    if args.sample_fps <= 0 or args.hard_fps <= 0:
        raise ValueError("sample FPS values must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg quality must be between 1 and 100")
    hard_intervals = _parse_intervals(args.hard_interval)
    manifest = []
    for video in args.videos:
        rows = extract_video(
            video,
            output_dir=args.output_dir,
            sample_fps=args.sample_fps,
            hard_fps=args.hard_fps,
            hard_intervals=hard_intervals,
            jpeg_quality=args.jpeg_quality,
        )
        manifest.extend(rows)
        print(f"[paper-dataset] {video.name}: extracted {len(rows)} frames")
    manifest_path = args.output_dir.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps({"frames": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[paper-dataset] Total: {len(manifest)} frames")
    print(f"[paper-dataset] Images: {args.output_dir.resolve()}")
    print(f"[paper-dataset] Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()

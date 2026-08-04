"""Finalize an interrupted Stage 1 run from already written base frames."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from .build import (
    _classify_window,
    _collect_clip_specs,
    _label_confidence,
    _quality_metadata,
    _read_manifest,
    _selected_frame_indices,
    _video_hash,
    _window_bounds,
    _probe_video,
)
from .common import SELECTED_FRAMES_PER_WINDOW, TARGET_FPS, WINDOW_FRAMES, WINDOW_OVERLAP_FRAMES


def finalize(*, manifest: Path, raw_root: Path, output_root: Path) -> pd.DataFrame:
    if not (output_root / "frames").is_dir():
        raise FileNotFoundError(f"Existing frames directory not found: {output_root / 'frames'}")
    specs = [s for s in _collect_clip_specs(_read_manifest(manifest)) if not s.exclude_from_training]
    segments, sampled, selected, windows = [], [], [], []
    for spec in specs:
        clip_id = str(spec.raw_row.get("clip_id") or Path(spec.filename).stem).strip()
        clip_root = output_root / "frames" / clip_id
        base = sorted(clip_root.glob("frame_*.png"))
        if not base:
            raise FileNotFoundError(f"Missing base frames for clip {clip_id}: {clip_root}")
        video_path = raw_root / spec.filename
        metadata = _probe_video(video_path)
        first = cv2.imread(str(base[0]), cv2.IMREAD_UNCHANGED)
        if first is None:
            raise RuntimeError(f"Cannot read existing frame: {base[0]}")
        session_id = spec.raw_row.get("session_id") or spec.raw_row.get("session")
        subject_id = spec.raw_row.get("subject_id") or spec.raw_row.get("student_id") or spec.raw_row.get("subject")
        group_id = spec.raw_row.get("group_id") or spec.raw_row.get("group")
        subject_ids = str(spec.raw_row.get("subject_ids", subject_id or session_id or group_id or ""))
        split = str(spec.raw_row.get("split", "")).strip().lower() or None
        split_status = "ready" if split in {"train", "val", "test"} else "unassigned"
        segments.append({"clip_id": clip_id, "filename": spec.filename, "video_path": str(video_path),
                         "video_hash": _video_hash(video_path), "class_code": spec.class_code,
                         "segment_type": "action" if spec.action_start_s is not None else "full_clip",
                         "action_start_s": spec.action_start_s, "action_end_s": spec.action_end_s,
                         "session_id": session_id, "subject_id": subject_id, "group_id": group_id,
                         "subject_ids": subject_ids, "split": split, "split_status": split_status,
                         "source_width": metadata["width"], "source_height": metadata["height"], **metadata})
        rows = []
        for i, path in enumerate(base):
            ts = i / TARGET_FPS
            rows.append({"clip_id": clip_id, "filename": spec.filename, "frame_index": i,
                         "timestamp_s": round(ts, 3), "timestamp_ms": round(ts * 1000, 1),
                         "actual_timestamp_s": round(ts, 3), "frame_path": str(path.relative_to(output_root)).replace("\\", "/"),
                         "decode_error": False, "frame_label": "action" if spec.action_start_s is not None and spec.action_start_s <= ts <= (spec.action_end_s or ts) else "background",
                         "width": first.shape[1], "height": first.shape[0], "analysis_width": first.shape[1],
                         "analysis_height": first.shape[0], "target_fps": TARGET_FPS})
        sampled.extend(rows)
        for wi, (start, end) in enumerate(_window_bounds(len(rows))):
            start_s, end_s = rows[start]["timestamp_s"], rows[end - 1]["timestamp_s"]
            label, include = _classify_window(window_start=start_s, window_end=end_s,
                                               action_start=spec.action_start_s, action_end=spec.action_end_s,
                                               duration_s=float(metadata["duration_s"]))
            wid = f"{clip_id}_w{wi:03d}"
            picks = _selected_frame_indices(pd.DataFrame(rows), start, end)
            for idx, reason in picks:
                selected.append({"window_id": wid, "clip_id": clip_id, "filename": spec.filename,
                                 "source_frame_index": idx, "window_local_index": idx - start,
                                 "selection_reason": reason, "frame_path": rows[idx]["frame_path"],
                                 "timestamp_s": rows[idx]["timestamp_s"],
                                 "frame_quality_score": rows[idx].get("frame_quality_score")})
            windows.append({"window_id": wid, "clip_id": clip_id, "filename": spec.filename,
                            "video_hash": segments[-1]["video_hash"], "class_code": spec.class_code,
                            "session_id": session_id, "subject_id": subject_id, "group_id": group_id,
                            "subject_ids": subject_ids, "split": split, "split_status": split_status,
                            "label": label, "window_label": label, "include_in_training": include,
                            "review_required": label == "mixed", "window_start_s": start_s,
                            "window_end_s": end_s, "window_size_frames": end - start,
                            "window_overlap_frames": WINDOW_OVERLAP_FRAMES,
                            "window_stride_frames": WINDOW_FRAMES - WINDOW_OVERLAP_FRAMES,
                            "target_fps": TARGET_FPS, "frame_count": end - start,
                            "frame_dir": f"{clip_id}/w{wi:03d}", "action_start_s": spec.action_start_s,
                            "action_end_s": spec.action_end_s,
                            "label_confidence": _label_confidence(label, 0.0),
                            "selected_frame_count": len(picks), "sampled_frame_count": end - start})
    pd.DataFrame(segments).to_parquet(output_root / "segments.parquet", index=False)
    pd.DataFrame(sampled).to_parquet(output_root / "sampled_frames.parquet", index=False)
    pd.DataFrame(selected).to_parquet(output_root / "selected_frames.parquet", index=False)
    result = pd.DataFrame(windows)
    result.to_parquet(output_root / "windows.parquet", index=False)
    pd.DataFrame([{"clips": len(segments), "sampled_frames": len(sampled), "selected_frames": len(selected),
                   "windows": len(windows), "action_windows": int((result.label == "action").sum()),
                   "negative_windows": int(result.label.isin(["negative_pre", "negative_post"]).sum()),
                   "mixed_windows": int((result.label == "mixed").sum()), "split_status": "ready" if all(s["split_status"] == "ready" for s in segments) else "insufficient_metadata"}]).to_csv(output_root / "stage1_report.csv", index=False)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    result = finalize(manifest=args.manifest, raw_root=args.raw_root, output_root=args.output_root)
    print(f"Finalized {result['clip_id'].nunique()} clips, {len(result)} windows")


if __name__ == "__main__":
    main()

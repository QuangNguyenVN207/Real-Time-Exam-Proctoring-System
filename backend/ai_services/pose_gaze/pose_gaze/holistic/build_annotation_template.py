"""Create a reviewable front-v4 gaze/temporal annotation template."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .annotation_schema import ANNOTATION_COLUMNS, PAIR_EVENTS, SUBJECT_EVENTS


def build(manifest_path: Path, output_path: Path) -> int:
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    output_rows: list[dict[str, str]] = []
    for source in source_rows:
        if source.get("camera_view_id") != "front":
            continue
        try:
            actors = [str(value) for value in json.loads(source.get("action_actor_ids") or source.get("actor_ids") or "[]")]
        except json.JSONDecodeError:
            actors = []
        actors = [actor for actor in actors if actor]
        common = {
            "source_filename": source.get("filename", ""),
            "clip_id": source.get("clip_id", ""),
            "split": source.get("split", ""),
            "split_group": source.get("split_group", ""),
            "camera_view_id": "front",
            "video_class_code": source.get("class_code", ""),
            "duration_s": source.get("duration_s", source.get("actual_duration_s", "")),
            "actual_fps": source.get("actual_fps", ""),
            "action_start_s": source.get("action_start_s", ""),
            "action_end_s": source.get("action_end_s", ""),
        }
        for actor in actors:
            for event_type in SUBJECT_EVENTS:
                output_rows.append({**common, "actor_id": actor, "track_id": "", "target_actor_id": "", "event_scope": "subject", "event_type": event_type, "interval_index": "1", "start_frame": "", "end_frame": "", "start_time_ms": "", "end_time_ms": "", "status": "unannotated", "annotator": "", "confidence": "", "notes": ""})
            for target in actors:
                if target == actor:
                    continue
                for event_type in PAIR_EVENTS:
                    output_rows.append({**common, "actor_id": actor, "track_id": "", "target_actor_id": target, "event_scope": "pair", "event_type": event_type, "interval_index": "1", "start_frame": "", "end_frame": "", "start_time_ms": "", "end_time_ms": "", "status": "unannotated", "annotator": "", "confidence": "", "notes": ""})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print({"rows": build(args.manifest, args.output), "output": str(args.output)})


if __name__ == "__main__":
    main()

"""Build the video-level manifest consumed by the Holistic batch pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import re
from zipfile import ZipFile
from xml.etree import ElementTree
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}
BASE_COLUMNS = (
    "clip_id", "filename", "source", "recording_session", "actor_ids",
    "action_actor_ids", "class_code", "layout", "action_start_s",
    "action_end_s", "reviewed", "duration_s", "split_group", "split",
    "exclude_from_training", "quality", "note",
)
ADDITIVE_COLUMNS = (
    "media_type", "camera_view_id", "fps_verified", "annotation_confidence",
    "interaction_pairs", "media_path", "frame_count", "frame_width",
    "frame_height", "actual_fps", "actual_duration_s", "media_readable",
)
FILENAME_RE = re.compile(r"^v_(c\d+)_(.+)$", re.IGNORECASE)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _probe(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    readable = capture.isOpened()
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if readable else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) if readable else 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if readable else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if readable else 0
    capture.release()
    duration = count / fps if count and fps > 0 else 0.0
    return {
        "frame_count": count, "frame_width": width, "frame_height": height,
        "actual_fps": round(fps, 6), "actual_duration_s": round(duration, 6),
        "media_readable": readable and count > 0 and fps > 0,
        "fps_verified": readable and count > 0 and fps > 0,
    }


def _split(subjects: list[str]) -> str:
    values = set(subjects)
    if values and values <= {"s1", "s2", "s3", "s4"}:
        return "train"
    if values and values <= {"s5", "s6"}:
        return "val"
    if values and values <= {"s7", "s8"}:
        return "test"
    return ""


def _read_annotations(path: Path) -> dict[str, dict[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return {row["filename"].lower(): row for row in csv.DictReader(handle)}

    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.text or "" for node in item.findall(".//m:t", namespace))
                  for item in shared_root.findall("m:si", namespace)]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = sheet.findall(".//m:sheetData/m:row", namespace)
    values = []
    for row in rows:
        cells = {}
        for cell in row.findall("m:c", namespace):
            reference = cell.attrib["r"]
            column = re.match(r"[A-Z]+", reference).group(0)
            value = cell.find("m:v", namespace)
            text = "" if value is None else value.text or ""
            if cell.attrib.get("t") == "s":
                text = shared[int(text)]
            cells[column] = text
        values.append(cells)
    header_by_column = values[0]
    return {
        row.get("B", "").lower(): {
            header_by_column[column]: row.get(column, "")
            for column in header_by_column
        }
        for row in values[1:]
        if row.get("B")
    }


def build_manifest(raw_root: Path, source_manifest: Path, output: Path) -> int:
    annotations = _read_annotations(source_manifest)

    rows = []
    for path in sorted(raw_root.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        source = annotations.get(path.name.lower())
        match = FILENAME_RE.fullmatch(path.stem)
        if source:
            row = {column: source.get(column, "") for column in BASE_COLUMNS}
            row["media_type"] = "video"
            row["annotation_confidence"] = "reviewed"
            row["interaction_pairs"] = _json([
                {"source": actor, "peer": peer}
                for actor in json.loads(row["action_actor_ids"] or "[]")
                for peer in json.loads(row["actor_ids"] or "[]")
                if actor != peer
            ])
        elif match:
            class_code = match.group(1).lower()
            subjects = [part.lower() for part in match.group(2).split("_") if re.fullmatch(r"s\d+", part, re.I)]
            row = {column: "" for column in BASE_COLUMNS}
            row.update({
                "clip_id": path.stem, "filename": path.name, "source": "raw_video",
                "recording_session": "raw_video", "actor_ids": _json(subjects),
                "action_actor_ids": "[]" if class_code == "c5" else _json(subjects[:1]),
                "class_code": class_code, "layout": "", "split_group": "_".join(subjects),
                "split": _split(subjects), "exclude_from_training": "false",
                "quality": "unreviewed", "annotation_confidence": "filename_inferred",
                "interaction_pairs": "[]" if class_code in {"c1", "c4", "c5"} else _json([
                    {"source": subjects[0], "peer": subjects[1]} for _ in [0]
                    if len(subjects) > 1
                ]),
            })
        else:
            row = {column: "" for column in BASE_COLUMNS}
            row.update({"clip_id": path.stem, "filename": path.name, "source": "raw_video",
                        "recording_session": "raw_video", "class_code": "",
                        "exclude_from_training": "true", "quality": "unreviewed",
                        "annotation_confidence": "unresolved", "interaction_pairs": "[]",
                        "note": "No stage0 annotation; label required before training"})
        row.update({"media_type": row.get("media_type", "video"), "camera_view_id": "default",
                    "media_path": str(path.resolve()), **_probe(path)})
        rows.append(row)

    if not rows:
        raise ValueError(f"No videos found in {raw_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_COLUMNS + ADDITIVE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"wrote {build_manifest(args.raw_root, args.source_manifest, args.output)} rows to {args.output}")


if __name__ == "__main__":
    main()
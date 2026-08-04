from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "clip_id", "filename", "source", "recording_session", "actor_ids",
    "action_actor_ids", "class_code", "layout", "action_start_s",
    "action_end_s", "reviewed", "duration_s", "split_group", "split",
    "exclude_from_training", "quality",
]
ALLOWED_LAYOUTS = {"same_row_two_seats_apart", "different_row_nearby"}
KNOWN_CLASS_CODES = {f"c{index}" for index in range(1, 8)}
ALLOWED_SOURCES = {"author", "self_recorded"}
ALLOWED_SPLITS = {"train", "val", "test"}
ACTOR_ID_PATTERN = re.compile(r"s\d+")


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def _parse_seconds(value: object) -> float | None:
    text = _text(value)
    if not text or text in {"bỏ", "bo", "nan", "none"}:
        return None
    minute_second = re.fullmatch(r"(\d+)p(\d+)s", text)
    if minute_second:
        return float(int(minute_second.group(1)) * 60 + int(minute_second.group(2)))
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_bool(value: object) -> str:
    text = _text(value)
    if isinstance(value, bool) or text in {"true", "1", "yes", "y"}:
        return "true"
    if text in {"false", "0", "no", "n"}:
        return "false"
    return text


def _parse_actor_ids(value: object) -> list[str] | None:
    text = _text(value)
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    actor_ids = [_text(item) for item in parsed]
    if any(not ACTOR_ID_PATTERN.fullmatch(actor_id) for actor_id in actor_ids):
        return None
    if len(set(actor_ids)) != len(actor_ids):
        return None
    return actor_ids


def validate_manifest(
    manifest_path: Path,
    raw_root: Path | None = None,
) -> pd.DataFrame:
    frame = _read_table(manifest_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw_files: set[str] = set()
    if raw_root is not None:
        raw_files = {
            path.name.casefold()
            for path in raw_root.rglob("*")
            if path.is_file()
        }

    issues: list[dict[str, object]] = []
    split_by_key: dict[str, set[str]] = {}
    for index, row in frame.iterrows():
        filename = _text(row.get("filename", ""))
        clip_id = _text(row.get("clip_id", ""))
        source = _text(row.get("source", ""))
        recording_session = _text(row.get("recording_session", ""))
        class_code = _text(row.get("class_code", ""))
        actor_ids = _parse_actor_ids(row.get("actor_ids"))
        action_actor_ids = _parse_actor_ids(row.get("action_actor_ids"))
        layout = str(row.get("layout", "")).strip()
        reviewed = _normalize_bool(row.get("reviewed", ""))
        duration_s = _parse_seconds(row.get("duration_s"))
        start_s = _parse_seconds(row.get("action_start_s"))
        end_s = _parse_seconds(row.get("action_end_s"))
        exclude = _normalize_bool(row.get("exclude_from_training", "")) == "true"
        split_group = _text(row.get("split_group", ""))
        split = _text(row.get("split", ""))

        row_issues: list[str] = []
        if not clip_id:
            row_issues.append("missing_clip_id")
        if not filename:
            row_issues.append("missing_filename")
        elif raw_root is not None and filename not in raw_files:
            row_issues.append("missing_source_file")
        if source not in ALLOWED_SOURCES:
            row_issues.append("invalid_source")
        if not recording_session:
            row_issues.append("missing_recording_session")
        if not class_code:
            row_issues.append("missing_class_code")
        if class_code and class_code not in KNOWN_CLASS_CODES and class_code != "excluded":
            row_issues.append("unknown_class_code")
        if actor_ids is None:
            row_issues.append("invalid_actor_ids")
        if action_actor_ids is None:
            row_issues.append("invalid_action_actor_ids")
        elif actor_ids is not None and not set(action_actor_ids).issubset(actor_ids):
            row_issues.append("action_actor_not_in_actor_ids")
        if reviewed != "true":
            row_issues.append("not_reviewed")

        if layout not in ALLOWED_LAYOUTS:
            row_issues.append("invalid_layout")
        if duration_s is None:
            row_issues.append("invalid_duration")
        elif duration_s < 1.0 and not exclude:
            row_issues.append("short_clip_not_excluded")
        if class_code == "c5":
            if start_s is not None or end_s is not None:
                row_issues.append("normal_has_action_window")
        elif start_s is None or end_s is None:
            row_issues.append("missing_action_window")
        elif duration_s is not None and (end_s <= start_s or end_s > duration_s):
            row_issues.append("invalid_action_window")
        if not split_group:
            row_issues.append("missing_split_group")
        if split and split not in ALLOWED_SPLITS:
            row_issues.append("invalid_split")
        if not exclude and not split:
            row_issues.append("missing_split")

        if split:
            split_keys = [
                f"session:{recording_session}",
                f"split_group:{split_group}",
            ]
            if actor_ids is not None:
                split_keys.extend(f"subject:{actor_id}" for actor_id in actor_ids)
            for key in split_keys:
                split_by_key.setdefault(key, set()).add(split)

        if row_issues:
            issues.append({
                "row": index + 2,
                "filename": filename,
                "class_code": class_code,
                "issues": ";".join(row_issues),
                "reviewed": reviewed,
                "duration_s": duration_s if duration_s is not None else "",
                "exclude_from_training": exclude,
            })

    duplicate_columns = ("clip_id", "filename")
    for column in duplicate_columns:
        duplicated = frame[column].astype(str).duplicated(keep=False)
        for index in frame.index[duplicated]:
            issues.append({
                "row": int(index) + 2,
                "filename": _text(frame.at[index, "filename"]),
                "class_code": _text(frame.at[index, "class_code"]),
                "issues": f"duplicate_{column}",
                "reviewed": _normalize_bool(frame.at[index, "reviewed"]),
                "duration_s": _parse_seconds(frame.at[index, "duration_s"]) or "",
                "exclude_from_training": _normalize_bool(frame.at[index, "exclude_from_training"]) == "true",
            })

    for key, splits in split_by_key.items():
        if len(splits) > 1:
            issues.append({
                "row": "",
                "filename": "",
                "class_code": "",
                "issues": f"split_leakage:{key}:{','.join(sorted(splits))}",
                "reviewed": "",
                "duration_s": "",
                "exclude_from_training": "",
            })
    return pd.DataFrame(issues)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a reviewed Stage 0 manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_video"))
    args = parser.parse_args()
    issues = validate_manifest(args.manifest, raw_root=args.raw_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        issues.to_csv(args.output, index=False)
    if issues.empty:
        print("OK: no validation issues found")
        return
    print(issues.to_string(index=False))
    raise SystemExit(1)


if __name__ == "__main__":
    main()

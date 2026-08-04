from __future__ import annotations

from pathlib import Path

CLASS_COUNTS: dict[str, int] = {
    "c1": 13,
    "c2": 4,
    "c3": 8,
    "c4": 8,
    "c5": 4,
}

TRAIN_SUBJECTS = {"s1", "s2", "s3", "s4"}
VAL_SUBJECTS = {"s5", "s6"}
TEST_SUBJECTS = {"s7", "s8"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def processed_root() -> Path:
    return repo_root() / "data" / "processed"


def default_raw_roots() -> tuple[Path, ...]:
    home = Path.home()
    return (
        repo_root() / "data" / "raw" / "Dataset_cheating",
        repo_root().parent / "archive (1)" / "Dataset_cheating",
        home / "Downloads" / "archive (1)" / "Dataset_cheating",
        home / "Downloads" / "archive" / "Dataset_cheating",
    )


def split_for_subjects(subject_ids: tuple[str, ...] | list[str] | set[str]) -> str | None:
    subject_set = set(subject_ids)
    if not subject_set:
        return None
    if subject_set <= TRAIN_SUBJECTS:
        return "train"
    if subject_set <= VAL_SUBJECTS:
        return "val"
    if subject_set <= TEST_SUBJECTS:
        return "test"
    return None


def validate_class_counts(class_counts: dict[str, int]) -> None:
    missing = set(CLASS_COUNTS) - set(class_counts)
    if missing:
        raise AssertionError(f"Missing classes: {sorted(missing)}")
    for class_code, expected_count in CLASS_COUNTS.items():
        actual_count = class_counts[class_code]
        if actual_count != expected_count:
            raise AssertionError(f"{class_code} expected {expected_count} rows, got {actual_count}")

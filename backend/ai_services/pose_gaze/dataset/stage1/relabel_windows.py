"""Tạo nhãn train mới từ Stage 1 đã hoàn thành, không decode lại video.

Artifact gốc được giữ nguyên. Script chỉ ghi windows parquet/report dẫn xuất.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .build import _binary_label, _classify_window, _label_confidence, _action_overlap_ratio


def relabel_windows(source: Path, output: Path) -> pd.DataFrame:
    windows = pd.read_parquet(source)
    required = {"window_start_s", "window_end_s", "action_start_s", "action_end_s", "clip_id", "class_code"}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"windows.parquet thiếu cột: {sorted(missing)}")

    result = windows.copy()
    phases: list[str] = []
    binary_labels: list[str] = []
    include: list[bool] = []
    confidence: list[float] = []
    for row in result.itertuples(index=False):
        label, eligible = _classify_window(
            float(row.window_start_s), float(row.window_end_s),
            None if pd.isna(row.action_start_s) else float(row.action_start_s),
            None if pd.isna(row.action_end_s) else float(row.action_end_s),
            float(row.window_end_s),
            str(row.class_code),
        )
        ratio = _action_overlap_ratio(
            float(row.window_start_s), float(row.window_end_s),
            None if pd.isna(row.action_start_s) else float(row.action_start_s),
            None if pd.isna(row.action_end_s) else float(row.action_end_s),
        )
        phases.append(label)
        binary_labels.append(_binary_label(str(row.class_code)))
        include.append(eligible)
        confidence.append(_label_confidence(label, ratio))

    result["label"] = phases
    result["window_label"] = phases
    result["window_phase"] = phases
    result["binary_label"] = binary_labels
    result["include_in_training"] = include
    result["review_required"] = False
    result["rejected_reason"] = ""
    result["label_confidence"] = confidence
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel Stage 1 windows without decoding video")
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    source = args.stage1_root / "windows.parquet"
    output = args.output or args.stage1_root / "windows_training_ready.parquet"
    result = relabel_windows(source, output)
    report = pd.DataFrame([{
        "windows": len(result),
        "cheating_windows": int((result["binary_label"] == "cheating").sum()),
        "non_cheating_windows": int((result["binary_label"] == "non_cheating").sum()),
        "action_phase_windows": int((result["window_phase"] == "action").sum()),
        "non_action_phase_windows": int((result["window_phase"] == "non_action").sum()),
        "trainable_windows": int(result["include_in_training"].sum()),
    }])
    report_path = output.with_name("stage1_relabel_report.csv")
    report.to_csv(report_path, index=False)
    print(f"Wrote {output}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

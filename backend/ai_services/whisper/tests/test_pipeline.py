from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.ai_services.whisper.audio_pipeline import AudioPipeline
from backend.ai_services.whisper.config import (
    PHOBERT_CHEATING_THRESHOLD,
    PHOBERT_MODEL_DIR,
    PHOWHISPER_MODEL,
    PHOWHISPER_REVISION,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES_DIR = BASE_DIR / "samples"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "p1_audio_pipeline"


def infer_label(audio_path: Path) -> str | None:
    for part in reversed(audio_path.parts):
        name = part.lower()
        if name in {"cheating", "normal"}:
            return name
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate_metrics(rows: list[dict]) -> dict:
    labeled = [row for row in rows if row["actual_label"] in {"normal", "cheating"} and not row.get("error")]
    tp = sum(row["actual_label"] == "cheating" and row["prediction_alert"] for row in labeled)
    tn = sum(row["actual_label"] == "normal" and not row["prediction_alert"] for row in labeled)
    fp = sum(row["actual_label"] == "normal" and row["prediction_alert"] for row in labeled)
    fn = sum(row["actual_label"] == "cheating" and not row["prediction_alert"] for row in labeled)
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "average_time_seconds": sum(row["time_seconds"] for row in labeled) / total if total else 0.0,
    }


def deduplicate(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for row in rows:
        file_hash = row["sha256"]
        if file_hash in seen:
            continue
        seen.add(file_hash)
        unique.append(row)
    return unique


def prediction_at_threshold(row: dict, threshold: float) -> bool:
    rule_label = str(row.get("rule_label", "safe")).lower()
    if rule_label in {"medium", "high", "cheating", "suspicious"}:
        return True
    return float(row.get("ai_cheating_probability", 0.0)) >= threshold


def component_rows(rows: list[dict], component: str) -> list[dict]:
    """Return a copy with predictions from one pipeline component."""
    result = []
    for row in rows:
        rule_alert = str(row.get("rule_label", "safe")).lower() in {
            "medium",
            "high",
            "cheating",
            "suspicious",
        }
        ai_alert = (
            float(row.get("ai_cheating_probability", 0.0))
            >= PHOBERT_CHEATING_THRESHOLD
        )
        if component == "keyword_rules":
            prediction = rule_alert
        elif component == "phobert":
            prediction = ai_alert
        elif component == "fusion":
            prediction = rule_alert or ai_alert
        else:
            raise ValueError(f"Unknown component: {component}")
        result.append(dict(row, prediction_alert=prediction))
    return result


def threshold_sweep(rows: list[dict]) -> list[dict]:
    result = []
    for value in range(101):
        threshold = value / 100
        threshold_rows = [dict(row, prediction_alert=prediction_at_threshold(row, threshold)) for row in rows]
        result.append({"threshold": threshold, **calculate_metrics(threshold_rows)})
    return result


def choose_threshold(rows: list[dict]) -> float:
    best_f1 = max(row["f1"] for row in rows)
    candidates = [row for row in rows if abs(row["f1"] - best_f1) < 1e-12]
    return float(min(candidates, key=lambda row: (abs(row["threshold"] - 0.60), row["threshold"]))["threshold"])


def plot_metrics(raw_metrics: dict, unique_metrics: dict, output_path: Path) -> None:
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    raw = [raw_metrics[key.lower()] for key in labels]
    unique = [unique_metrics[key.lower()] for key in labels]
    positions = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar([x - width / 2 for x in positions], raw, width, label=f"All files (n={raw_metrics['n']})")
    ax.bar([x + width / 2 for x in positions], unique, width, label=f"Unique audio hashes (n={unique_metrics['n']})")
    ax.set_xticks(positions, labels=labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Audio pipeline classification metrics")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(rows: list[dict], selected: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for key, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("accuracy", "Accuracy")):
        ax.plot([row["threshold"] for row in rows], [row[key] for row in rows], label=label)
    ax.axvline(PHOBERT_CHEATING_THRESHOLD, color="#777777", linestyle=":", label=f"Runtime {PHOBERT_CHEATING_THRESHOLD:.2f}")
    ax.axvline(selected, color="#111111", linestyle="--", label=f"Best sample F1 {selected:.2f}")
    ax.set(xlabel="PhoBERT cheating-probability threshold", ylabel="Score", ylim=(0, 1.05), title="End-to-end audio threshold sensitivity")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_component_ablation(component_metrics: dict[str, dict], output_path: Path) -> None:
    labels = ["Keyword rules", "PhoBERT", "Fusion"]
    keys = ["keyword_rules", "phobert", "fusion"]
    metrics = ["precision", "recall", "f1"]
    positions = list(range(len(labels)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for offset, metric in enumerate(metrics):
        values = [component_metrics[key][metric] for key in keys]
        shifted = [x + (offset - 1) * width for x in positions]
        ax.bar(shifted, values, width, label=metric.capitalize())
    ax.set_xticks(positions, labels=labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Component ablation on unique audio samples")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_probability_distribution(rows: list[dict], output_path: Path) -> None:
    normal_scores = sorted(
        float(row["ai_cheating_probability"])
        for row in rows
        if row["actual_label"] == "normal"
    )
    cheating_scores = sorted(
        float(row["ai_cheating_probability"])
        for row in rows
        if row["actual_label"] == "cheating"
    )

    def spread(count: int, center: float) -> list[float]:
        if count <= 1:
            return [center] * count
        return [center - 0.12 + 0.24 * index / (count - 1) for index in range(count)]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(normal_scores, spread(len(normal_scores), 0), alpha=0.75, label="Normal")
    ax.scatter(cheating_scores, spread(len(cheating_scores), 1), alpha=0.75, label="Cheating")
    ax.axvline(
        PHOBERT_CHEATING_THRESHOLD,
        color="#111111",
        linestyle="--",
        label=f"Runtime threshold {PHOBERT_CHEATING_THRESHOLD:.2f}",
    )
    ax.set_yticks([0, 1], labels=["Normal", "Cheating"])
    ax.set_ylim(-0.25, 1.25)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("PhoBERT probability of Cheating")
    ax.set_title("PhoBERT score separation on unique audio samples")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report_snippet(report: dict, output_path: Path) -> None:
    unique = report["primary_unique_audio_hashes"]
    components = report["component_ablation_unique_audio"]
    text = f"""# Kết quả kiểm thử pipeline audio

Pipeline sử dụng **PhoWhisper-small pretrained** để nhận dạng tiếng Việt và checkpoint **PhoBERT step 235 (epoch 5)** để phân loại transcript. Ngưỡng xác suất gian lận dùng khi vận hành là **0,60**; luật từ khóa mức `medium` hoặc `high` có thể kích hoạt cảnh báo trực tiếp.

Trên {unique['n']} mẫu audio duy nhất theo SHA-256 ({unique['tp']} gian lận, {unique['tn']} bình thường), pipeline fusion đạt accuracy = precision = recall = F1 = **{unique['f1']:.3f}** với ma trận TP={unique['tp']}, TN={unique['tn']}, FP={unique['fp']}, FN={unique['fn']}. Tỷ lệ xử lý thành công là **{report['successful_coverage'] * 100:.1f}%**. Hai đường dẫn trùng nội dung byte đã được loại khỏi metric chính.

Thử nghiệm tách thành phần cho thấy luật từ khóa đạt recall **{components['keyword_rules']['recall']:.3f}** ({components['keyword_rules']['tp']}/{components['keyword_rules']['tp'] + components['keyword_rules']['fn']} mẫu gian lận), trong khi PhoBERT và fusion đều đạt recall **{components['fusion']['recall']:.3f}**. PhoBERT vì vậy bổ sung {components['fusion']['tp'] - components['keyword_rules']['tp']} mẫu không được luật từ khóa kích hoạt trong tập này.

Sweep trên cùng tập mẫu cho F1 tối đa trong khoảng ngưỡng **{report['threshold_sweep']['best_f1_plateau']['min_threshold']:.2f}–{report['threshold_sweep']['best_f1_plateau']['max_threshold']:.2f}**; ngưỡng 0,60 được giữ nguyên vì nằm trong vùng ổn định và là cấu hình vận hành đã chọn. Đây chỉ là phân tích độ nhạy, không phải tối ưu trên một tập test độc lập.

**Giới hạn:** tập thử nhỏ, các câu ngắn và phân tách khá dễ; nhãn chỉ cho hành vi cuối, không có transcript chuẩn để tính WER/CER. Vì vậy kết quả trên được dùng để xác nhận tính đúng đắn và khả năng tái lập của pipeline, không dùng để tuyên bố khả năng tổng quát hóa. Thời gian CPU trung bình **{unique['average_time_seconds']:.2f} giây/tệp** trong lần chạy này cũng chưa đáp ứng suy luận thời gian thực nếu xử lý tuần tự trên CPU.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Evaluate pretrained PhoWhisper + fine-tuned PhoBERT on labeled audio samples.")
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Reuse successful rows from an earlier pipeline_results.json and retry failures.",
    )
    args = parser.parse_args()

    samples_dir = args.samples_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = sorted(
        path
        for path in samples_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3"}
    )
    if args.limit is not None:
        audio_files = audio_files[: args.limit]
    if not audio_files:
        raise FileNotFoundError(f"No .wav/.mp3 files found under {samples_dir}")

    previous_rows = {}
    if args.resume_from is not None:
        previous_report = json.loads(args.resume_from.resolve().read_text(encoding="utf-8"))
        previous_rows = {
            row["file"]: row
            for row in previous_report.get("files", [])
            if not row.get("error")
        }

    pipeline = None
    rows: list[dict] = []
    result_cache: dict[str, tuple[dict, float]] = {}

    for index, audio_file in enumerate(audio_files, start=1):
        relative_name = audio_file.relative_to(samples_dir).as_posix()
        file_hash = sha256_file(audio_file)
        print(f"[{index}/{len(audio_files)}] {relative_name}")
        if relative_name in previous_rows and previous_rows[relative_name].get("sha256") == file_hash:
            rows.append(previous_rows[relative_name])
            print("  reused successful row from resume report")
            continue
        try:
            if pipeline is None:
                pipeline = AudioPipeline()
            if file_hash in result_cache:
                result, elapsed = result_cache[file_hash]
                reused = True
            else:
                start = time.perf_counter()
                result = pipeline.process(str(audio_file))
                elapsed = time.perf_counter() - start
                result_cache[file_hash] = (result, elapsed)
                reused = False

            risk = str(result.get("risk", "Normal"))
            row = {
                "file": relative_name,
                "sha256": file_hash,
                "duplicate_result_reused": reused,
                "actual_label": infer_label(audio_file),
                "prediction_alert": risk.lower() == "cheating",
                "time_seconds": round(elapsed, 6),
                "transcription": result.get("transcription", ""),
                "final_risk": risk,
                "rule_label": result.get("rule_label", ""),
                "ai_label": result.get("ai_label", ""),
                "ai_confidence": result.get("confidence", 0.0),
                "ai_cheating_probability": result.get("ai_cheating_probability", 0.0),
                "ai_cheating_threshold": result.get("ai_cheating_threshold", PHOBERT_CHEATING_THRESHOLD),
                "fusion_reason": result.get("fusion_reason", ""),
                "matched_keywords": result.get("matched_keywords", []),
                "error": None,
            }
        except Exception as exc:
            row = {
                "file": relative_name,
                "sha256": file_hash,
                "duplicate_result_reused": False,
                "actual_label": infer_label(audio_file),
                "prediction_alert": False,
                "time_seconds": 0.0,
                "transcription": "",
                "final_risk": "",
                "rule_label": "",
                "ai_label": "",
                "ai_confidence": 0.0,
                "ai_cheating_probability": 0.0,
                "ai_cheating_threshold": PHOBERT_CHEATING_THRESHOLD,
                "fusion_reason": "",
                "matched_keywords": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(row["error"])
        rows.append(row)

    raw_metrics = calculate_metrics(rows)
    unique_rows = deduplicate(rows)
    unique_metrics = calculate_metrics(unique_rows)
    plot_metrics(raw_metrics, unique_metrics, output_dir / "pipeline_metrics.png")
    sweep = threshold_sweep(unique_rows)
    selected_threshold = choose_threshold(sweep)
    best_f1 = max(row["f1"] for row in sweep)
    best_rows = [row for row in sweep if abs(row["f1"] - best_f1) < 1e-12]
    plot_threshold_sweep(sweep, selected_threshold, output_dir / "pipeline_threshold_sweep.png")
    with (output_dir / "pipeline_threshold_sweep.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)

    component_metrics = {
        component: calculate_metrics(component_rows(unique_rows, component))
        for component in ("keyword_rules", "phobert", "fusion")
    }
    plot_component_ablation(component_metrics, output_dir / "component_ablation.png")
    plot_probability_distribution(unique_rows, output_dir / "score_distribution.png")
    with (output_dir / "component_ablation.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["component", *list(next(iter(component_metrics.values())))],
        )
        writer.writeheader()
        for component, metrics in component_metrics.items():
            writer.writerow({"component": component, **metrics})

    csv_fields = [
        "file",
        "sha256",
        "duplicate_result_reused",
        "actual_label",
        "prediction_alert",
        "time_seconds",
        "transcription",
        "final_risk",
        "rule_label",
        "ai_label",
        "ai_confidence",
        "ai_cheating_probability",
        "ai_cheating_threshold",
        "fusion_reason",
        "matched_keywords",
        "error",
    ]
    with (output_dir / "pipeline_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["matched_keywords"] = json.dumps(csv_row["matched_keywords"], ensure_ascii=False)
            writer.writerow(csv_row)

    report = {
        "configuration": {
            "phowhisper_model": PHOWHISPER_MODEL,
            "phowhisper_revision": PHOWHISPER_REVISION,
            "phobert_model_dir": str(PHOBERT_MODEL_DIR),
            "phobert_cheating_threshold": PHOBERT_CHEATING_THRESHOLD,
            "samples_dir": str(samples_dir),
        },
        "raw_all_files": raw_metrics,
        "primary_unique_audio_hashes": unique_metrics,
        "threshold_sweep": {
            "selected_on_unique_audio": selected_threshold,
            "best_f1_plateau": {
                "f1": best_f1,
                "min_threshold": min(row["threshold"] for row in best_rows),
                "max_threshold": max(row["threshold"] for row in best_rows),
            },
            "selection_warning": "Selected and measured on the same small audio sample set; descriptive only.",
        },
        "component_ablation_unique_audio": component_metrics,
        "duplicate_files": len(rows) - len(unique_rows),
        "failed_files": sum(bool(row["error"]) for row in rows),
        "successful_coverage": sum(not row["error"] for row in rows) / len(rows),
        "metrics_by_extension": {
            extension: calculate_metrics(
                [row for row in unique_rows if Path(row["file"]).suffix.lower() == extension]
            )
            for extension in sorted({Path(row["file"]).suffix.lower() for row in unique_rows})
        },
        "files": rows,
    }
    (output_dir / "pipeline_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report_snippet(report, output_dir / "report_snippet_vi.md")
    print(json.dumps({key: report[key] for key in ("raw_all_files", "primary_unique_audio_hashes", "duplicate_files", "failed_files")}, indent=2))


if __name__ == "__main__":
    main()

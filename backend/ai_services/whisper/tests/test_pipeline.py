from __future__ import annotations

import sys
import json
import time
from datetime import datetime
from pathlib import Path

# =====================================================================
# FIX LỖI IMPORT: Chỉ định rõ thư mục ai_services cho Python
# =====================================================================
# Lấy đường dẫn tuyệt đối lên 3 cấp: tests -> whisper -> ai_services
AI_SERVICES_DIR = Path(__file__).resolve().parent.parent.parent
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))
# =====================================================================

# Lúc này Python đã biết thư mục phobert nằm ở đâu, sẽ không báo lỗi nữa
from whisper.audio_pipeline import AudioPipeline

# --- CÁC ĐƯỜNG DẪN BÊN DƯỚI GIỮ NGUYÊN CỦA BẠN ---
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_TXT = OUTPUTS_DIR / f"pipeline_report_{STAMP}.txt"
REPORT_JSON = OUTPUTS_DIR / f"pipeline_report_{STAMP}.json"


def infer_label(audio_path: Path) -> str | None:
    for part in reversed(audio_path.parts):
        name = part.lower()
        if name in {"cheating", "normal"}:
            return name
    return None


def main():
    pipeline = AudioPipeline()

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    total_time = 0.0

    audio_files = sorted(SAMPLES_DIR.rglob("*.wav"))

    print(f"Tìm thấy {len(audio_files)} file.\n")

    report_lines = []
    file_results = []

    report_lines.append("=" * 80)
    report_lines.append("AUDIO PIPELINE EVALUATION REPORT (AI FUSION VERSION)")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Samples dir : {SAMPLES_DIR}")
    report_lines.append(f"Total files : {len(audio_files)}")
    report_lines.append("")

    for audio_file in audio_files:
        relative_name = audio_file.relative_to(SAMPLES_DIR)
        actual_label = infer_label(audio_file)

        print("=" * 70)
        print(f"Đang xử lý: {relative_name}")

        start = time.perf_counter()
        result = pipeline.process(str(audio_file))
        elapsed = time.perf_counter() - start
        total_time += elapsed

        # --- TRÍCH XUẤT DỮ LIỆU TỪ JSON MỚI ---
        risk = result.get("risk", "Normal")
        
        # Đánh giá: Nếu hệ thống báo Cheating hoặc Suspicious thì coi là phát hiện có biến (True)
        prediction = risk in ["Cheating", "Suspicious"] 
        
        transcription = result.get("transcription", "")
        confidence = result.get("confidence", 0.0)
        
        rule_label = result.get("rule_label", "Normal")
        ai_label = result.get("ai_label", "Normal")
        fusion_reason = result.get("fusion_reason", "")
        matched_keywords = result.get("matched_keywords", [])

        # --- IN RA TERMINAL ---
        print("Transcript :", transcription)
        print(f"Time       : {elapsed:.3f}s")
        print("-" * 30)
        print("Rule-based :", rule_label)
        print("PhoBERT AI :", f"{ai_label} (Conf: {confidence:.2f})")
        print("Final Risk :", risk)
        print("Reason     :", fusion_reason)
        print("-" * 30)

        if matched_keywords:
            print("Matched Keywords:")
            for item in matched_keywords:
                # Tùy thuộc vào cấu trúc item của KeywordDetector, có thể điều chỉnh lại nếu cần
                print(f"  - {item}") 
        else:
            print("Matched Keywords: None")

        # ==========================
        # Evaluation (Đánh giá TP, TN, FP, FN)
        # ==========================
        if actual_label == "cheating":
            if prediction:
                TP += 1
                eval_result = "TP (Bắt Trúng)"
            else:
                FN += 1
                eval_result = "FN (Bỏ Lọt)"
        elif actual_label == "normal":
            if prediction:
                FP += 1
                eval_result = "FP (Bắt Nhầm)"
            else:
                TN += 1
                eval_result = "TN (An Toàn)"
        else:
            eval_result = "SKIP"

        print(f"Label Thực Tế : {actual_label.upper()}")
        print(f"Đánh giá Eval : {eval_result}\n")

        # --- ĐÓNG GÓI DỮ LIỆU BÁO CÁO ---
        file_results.append({
            "file": str(relative_name).replace("\\", "/"),
            "actual_label": actual_label,
            "prediction_alert": prediction,
            "eval_result": eval_result,
            "time_seconds": round(elapsed, 4),
            "transcription": transcription,
            "final_risk": risk,
            "rule_label": rule_label,
            "ai_label": ai_label,
            "ai_confidence": confidence,
            "fusion_reason": fusion_reason,
            "matched_keywords": matched_keywords,
        })

        report_lines.append(f"File       : {relative_name}")
        report_lines.append(f"Label      : {actual_label}")
        report_lines.append(f"Transcript : {transcription}")
        report_lines.append(f"Rule-based : {rule_label}")
        report_lines.append(f"PhoBERT AI : {ai_label} ({confidence:.2f})")
        report_lines.append(f"Final Risk : {risk} -> Eval: {eval_result}")
        report_lines.append(f"Reason     : {fusion_reason}")
        report_lines.append("-" * 80)

    # --- TÍNH TOÁN METRICS TỔNG QUÁT ---
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    avg_time = total_time / total if total else 0

    summary_lines = [
        "",
        "=" * 80,
        "SUMMARY (FUSION METRICS)",
        "=" * 80,
        f"Total Files  : {total}",
        "",
        f"True Positives  (TP) : {TP} (Gian lận -> Bắt trúng)",
        f"True Negatives  (TN) : {TN} (Bình thường -> An toàn)",
        f"False Positives (FP) : {FP} (Bình thường -> Báo động nhầm)",
        f"False Negatives (FN) : {FN} (Gian lận -> Bỏ lọt)",
        "",
        f"Accuracy  : {accuracy * 100:.2f}%",
        f"Precision : {precision * 100:.2f}%",
        f"Recall    : {recall * 100:.2f}%",
        f"F1-score  : {f1 * 100:.2f}%",
        f"Avg Time  : {avg_time:.3f}s / file",
        "=" * 80,
    ]

    for line in summary_lines:
        print(line)
        report_lines.append(line)

    # --- LƯU REPORT RA FILE ---
    report_text = "\n".join(report_lines)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total_files": total,
                    "tp": TP,
                    "tn": TN,
                    "fp": FP,
                    "fn": FN,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "average_time_seconds": avg_time,
                },
                "files": file_results,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    print(f"\nSaved report: {REPORT_TXT}")
    print(f"Saved json  : {REPORT_JSON}")


if __name__ == "__main__":
    main()
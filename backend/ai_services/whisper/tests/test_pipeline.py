from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from whisper.audio_pipeline import AudioPipeline


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


def format_segments(segments):
    if not segments:
        return "None"

    lines = []
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")

        lines.append(
            f"  - {start:.2f}s -> {end:.2f}s : {text}"
            if isinstance(start, (int, float)) and isinstance(end, (int, float))
            else f"  - {seg}"
        )
    return "\n".join(lines)


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
    report_lines.append("AUDIO PIPELINE EVALUATION REPORT")
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

        prediction = bool(result.get("keyword_detected", False))
        confidence = result.get("confidence", 0)
        risk = result.get("risk", "safe")
        transcription = result.get("transcription", "")
        matched = result.get("matched", [])
        matched_rules = result.get("matched_rules", [])
        matched_context = result.get("matched_context", [])
        matched_negative = result.get("matched_negative", [])

        print("Language   :", result.get("language", ""))
        print("Transcript :", transcription)
        print(f"Time       : {elapsed:.3f}s")
        print("Risk       :", risk)
        print("Confidence :", confidence)
        print("Keyword    :", result.get("keyword_score", 0))
        print("Rule Bonus :", result.get("rule_bonus", 0))
        print("Context    :", result.get("context_bonus", 0))
        print("Penalty    :", result.get("penalty", 0))

        print("\nSpeech Segments:")
        for seg in result.get("speech_segments", []):
            start_s = seg.get("start", 0) / 16000
            end_s = seg.get("end", 0) / 16000
            print(f"  - {start_s:.2f}s -> {end_s:.2f}s : {seg.get('text', '')}")

        print("Alert     :", prediction)

        if matched:
            print("Matched:")
            for item in matched:
                print(
                    f"  - {item['keyword']} "
                    f"(candidate={item['candidate']}, "
                    f"score={item['score']:.1f}, "
                    f"severity={item['severity']}, "
                    f"category={item['category']})"
                )
        else:
            print("Matched: None")

        print("Rules    :", matched_rules if matched_rules else "None")
        print("Context  :", matched_context if matched_context else "None")
        print("Negative :", matched_negative if matched_negative else "None")

        # ==========================
        # Evaluation
        # ==========================
        if actual_label == "cheating":
            if prediction:
                TP += 1
                eval_result = "TP"
            else:
                FN += 1
                eval_result = "FN"

        elif actual_label == "normal":
            if prediction:
                FP += 1
                eval_result = "FP"
            else:
                TN += 1
                eval_result = "TN"
        else:
            eval_result = "SKIP"

        print(f"Label     : {actual_label}")
        print(f"Eval      : {eval_result}")
        print()

        file_results.append({
            "file": str(relative_name).replace("\\", "/"),
            "label": actual_label,
            "prediction": prediction,
            "eval": eval_result,
            "time_seconds": round(elapsed, 4),
            "language": result.get("language", ""),
            "transcription": transcription,
            "risk": risk,
            "confidence": confidence,
            "keyword_score": result.get("keyword_score", 0),
            "rule_bonus": result.get("rule_bonus", 0),
            "context_bonus": result.get("context_bonus", 0),
            "penalty": result.get("penalty", 0),
            "matched": matched,
            "matched_rules": matched_rules,
            "matched_context": matched_context,
            "matched_negative": matched_negative,
        })

        report_lines.append(f"File       : {relative_name}")
        report_lines.append(f"Label      : {actual_label}")
        report_lines.append(f"Prediction : {prediction}")
        report_lines.append(f"Eval       : {eval_result}")
        report_lines.append(f"Time       : {elapsed:.3f}s")
        report_lines.append(f"Risk       : {risk}")
        report_lines.append(f"Confidence : {confidence}")
        report_lines.append(f"Transcript : {transcription}")
        report_lines.append(f"Matched    : {matched if matched else 'None'}")
        report_lines.append(f"Rules      : {matched_rules if matched_rules else 'None'}")
        report_lines.append(f"Context    : {matched_context if matched_context else 'None'}")
        report_lines.append(f"Negative   : {matched_negative if matched_negative else 'None'}")
        report_lines.append("-" * 80)

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )
    avg_time = total_time / total if total else 0

    summary_lines = [
        "",
        "=" * 80,
        "SUMMARY",
        "=" * 80,
        f"Total Files : {total}",
        "",
        f"TP : {TP}",
        f"TN : {TN}",
        f"FP : {FP}",
        f"FN : {FN}",
        "",
        f"Accuracy  : {accuracy * 100:.2f}%",
        f"Precision : {precision * 100:.2f}%",
        f"Recall    : {recall * 100:.2f}%",
        f"F1-score   : {f1 * 100:.2f}%",
        f"Average Time : {avg_time:.3f}s/file",
        "=" * 80,
    ]

    for line in summary_lines:
        print(line)
        report_lines.append(line)

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
                    "f1": f1,
                    "average_time_seconds": avg_time,
                },
                "files": file_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved report: {REPORT_TXT}")
    print(f"Saved json  : {REPORT_JSON}")


if __name__ == "__main__":
    main()
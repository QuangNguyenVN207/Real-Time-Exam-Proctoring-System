from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jiwer import cer, wer

from backend.ai_services.whisper.config import PHOWHISPER_MODEL, PHOWHISPER_REVISION
from backend.ai_services.whisper.phowhisper_service import PhoWhisperService


def load_manifest(test_dir: Path) -> list[tuple[str, str, Path]]:
    prompts_file = test_dir / "prompts.txt"
    waves_dir = test_dir / "waves"
    if not prompts_file.is_file():
        raise FileNotFoundError(f"Missing VIVOS prompts file: {prompts_file}")
    if not waves_dir.is_dir():
        raise FileNotFoundError(f"Missing VIVOS waves directory: {waves_dir}")

    rows = []
    for line in prompts_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        audio_id, reference = parts
        speaker_id = audio_id.split("_", 1)[0]
        audio_path = waves_dir / speaker_id / f"{audio_id}.wav"
        if audio_path.is_file():
            rows.append((audio_id, reference.strip().lower(), audio_path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pretrained PhoWhisper on a VIVOS test folder.")
    parser.add_argument("--vivos-test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "p1_vivos")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    test_dir = args.vivos_test_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(test_dir)
    if args.max_files is not None:
        manifest = manifest[: args.max_files]
    if not manifest:
        raise RuntimeError(f"No matching VIVOS audio files found under {test_dir}")

    service = PhoWhisperService()
    rows = []
    references = []
    hypotheses = []
    for index, (audio_id, reference, audio_path) in enumerate(manifest, start=1):
        start = time.perf_counter()
        hypothesis = service.transcribe(str(audio_path))["text"].strip().lower()
        elapsed = time.perf_counter() - start
        references.append(reference)
        hypotheses.append(hypothesis)
        row = {
            "audio_id": audio_id,
            "speaker_id": audio_id.split("_", 1)[0],
            "reference": reference,
            "hypothesis": hypothesis,
            "utterance_wer": wer(reference, hypothesis),
            "utterance_cer": cer(reference, hypothesis),
            "time_seconds": elapsed,
        }
        rows.append(row)
        print(f"[{index}/{len(manifest)}] {audio_id}: WER={row['utterance_wer']:.3f}")

    summary = {
        "model": PHOWHISPER_MODEL,
        "revision": PHOWHISPER_REVISION,
        "files": len(rows),
        "speakers": len({row["speaker_id"] for row in rows}),
        "corpus_wer": wer(references, hypotheses),
        "corpus_cer": cer(references, hypotheses),
        "average_time_seconds": sum(row["time_seconds"] for row in rows) / len(rows),
        "warning": "WER/CER are valid only for the exact manifest and max-files setting recorded here; 1-WER is not reported as accuracy.",
    }
    (output_dir / "asr_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "asr_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = ["WER", "CER"]
    values = [summary["corpus_wer"], summary["corpus_cer"]]
    bars = ax.bar(labels, values, color=["#C44E52", "#4C72B0"])
    ax.set_ylabel("Error rate")
    ax.set_title(f"PhoWhisper ASR errors on VIVOS (n={len(rows)})")
    ax.set_ylim(0, max(1.0, max(values) * 1.15))
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
    fig.tight_layout()
    fig.savefig(output_dir / "asr_error_rates.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Bộ thực nghiệm ASR phục vụ báo cáo Eureka.

Vì sao không có bảng CER so với văn bản chuẩn: `data/phong_thi_data.csv` KHÔNG phải tập câu
đã sinh ra `backend/ai_services/whisper/samples/*` (đối chiếu tự động: 0/43 file khớp CER<=0.2,
xem offline_out/asr_ablation/align.json). Nên ở đây dùng 3 phép đo không cần text chuẩn:

  asr --model {small|medium|large} : phiên âm 62 file có nhãn -> văn bản + độ trễ + RAM đỉnh.
  fusion                           : chạy keyword + PhoBERT trên văn bản đã lưu -> P/R/F1 theo
                                     TỪNG model ASR (cùng một bộ phân loại, chỉ đổi ASR).
  robust                           : nói thật trộn 2 nền của data/video_test_demo.mp4 theo SNR,
                                     reference = chính kết quả large-v3 của bản sạch.
  gate                             : tỉ lệ sinh ra câu khi đầu vào chỉ có nhiễu (cổng BẬT/TẮT).

Mỗi stage chạy ở tiến trình riêng vì máy 15 GB RAM không chứa 2 model Whisper lớn đồng thời.

  .venv/bin/python scripts/asr_ablation.py --stage asr --model small
  .venv/bin/python scripts/asr_ablation.py --stage fusion
  .venv/bin/python scripts/asr_ablation.py --stage robust
  .venv/bin/python scripts/asr_ablation.py --stage gate
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "backend/ai_services/whisper/samples"
OUT_DIR = ROOT / "offline_out/asr_ablation"
DEMO_CLIP = ROOT / "data/video_test_demo.mp4"
SR = 16000
SNR_LEVELS = (6, 3, 0, -3, -6)

_PUNCT = re.compile(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.UNICODE)


def norm_text(text: str) -> str:
    text = (text or "").lower().replace("_", " ")
    return " ".join(_PUNCT.sub(" ", text).split())


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ac in enumerate(a, 1):
        cur = [i]
        for j, bc in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ac != bc)))
        prev = cur
    return prev[-1]


def error_rate(hypo: str, ref: str, chars: bool = False) -> float:
    h = list(norm_text(hypo)) if chars else norm_text(hypo).split()
    r = list(norm_text(ref)) if chars else norm_text(ref).split()
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(h, r) / len(r)


def corpus() -> list[dict]:
    items = []
    for sub, label in (("cheating", 1), ("normal", 0)):
        files = sorted((SAMPLES / sub).glob("*.wav"), key=lambda p: (len(p.stem), p.stem))
        for path in files:
            items.append({"id": f"{sub}/{path.name}", "path": str(path), "label": label})
    return items


def load_wav(path: str) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=SR, mono=True)
    return np.ascontiguousarray(audio, dtype=np.float32)


def rms(a) -> float:
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float32) ** 2)))


def demo_beds() -> dict[str, np.ndarray]:
    cache = Path("/tmp/eureka_demo_beds.npz")
    if cache.exists():
        data = np.load(cache)
        return {k: data[k] for k in data.files}
    raw = Path("/tmp/eureka_demo_mono.f32")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(DEMO_CLIP), "-ac", "1", "-ar", str(SR),
                    "-f", "f32le", str(raw)], check=True)
    audio = np.fromfile(raw, dtype=np.float32)
    beds = {"room_noise": audio[:3 * SR], "music": audio[13 * SR:18 * SR]}
    np.savez(cache, **beds)
    return beds


def env_info() -> dict:
    import resource

    return {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
            "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)}


def build_asr(name: str):
    if name == "small":
        from backend.ai_services.whisper.phowhisper_service import PhoWhisperService

        service = PhoWhisperService()
        meta = {"model": "vinai/PhoWhisper-small", "runtime": "OpenVINO",
                "device": getattr(service, "device", "?"), "beams": 3}

        def asr(audio):
            t0 = time.perf_counter()
            text = service.transcribe(audio)["text"]
            return text, time.perf_counter() - t0

        return asr, meta

    from faster_whisper import WhisperModel

    if name == "medium":
        # faster-whisper chỉ đọc định dạng CTranslate2; `vinai/PhoWhisper-medium` là PyTorch
        # (lỗi "Unable to open file 'model.bin'"), nên dùng bản đã convert.
        model = WhisperModel("mad1999/pho-whisper-medium-ct2", device="cpu", compute_type="int8")
        meta = {"model": "mad1999/pho-whisper-medium-ct2 (= vinai/PhoWhisper-medium)",
                "runtime": "faster-whisper", "device": "cpu",
                "compute_type": "int8", "beams": 5}
    else:
        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        meta = {"model": "openai/whisper-large-v3", "runtime": "faster-whisper", "device": "cpu",
                "compute_type": "int8", "beams": 5}

    def asr(audio):
        t0 = time.perf_counter()
        segments, _ = model.transcribe(audio, language="vi", beam_size=5)
        return " ".join(s.text.strip() for s in segments).strip(), time.perf_counter() - t0

    return asr, meta


def stage_asr(name: str) -> None:
    asr, meta = build_asr(name)
    rows, lat = [], []
    for item in corpus():
        audio = load_wav(item["path"])
        text, seconds = asr(audio)
        lat.append(seconds)
        rows.append({**item, "hyp": text, "seconds": round(seconds, 3),
                     "audio_s": round(len(audio) / SR, 2)})
        print(f"  {item['id']:26s} {seconds:5.2f}s {text!r}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"stage": "asr", "model": meta, "env": env_info(), "n": len(rows),
           "latency_s": {"p50": round(statistics.median(lat), 3),
                         "p95": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 3),
                         "mean": round(statistics.mean(lat), 3)},
           "rtf": round(statistics.mean(r["seconds"] / max(r["audio_s"], 0.1) for r in rows), 3),
           "empty": sum(1 for r in rows if not r["hyp"].strip()), "rows": rows}
    (OUT_DIR / f"asr_{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[asr:{name}] n={len(rows)} p50={out['latency_s']['p50']}s p95={out['latency_s']['p95']}s "
          f"RTF={out['rtf']} RSS={out['env']['peak_rss_mb']}MB")


def stage_fusion() -> None:
    """Chạy keyword + PhoBERT trên văn bản của từng model -> P/R/F1 phân loại gian lận bằng lời nói."""
    from backend.ai_services.whisper.keyword_detector import KeywordDetector
    from backend.ai_services.whisper.phobert.phobert_service import PhobertService

    detector, phobert = KeywordDetector(), PhobertService()
    reports = {}
    for name in ("small", "medium", "large"):
        path = OUT_DIR / f"asr_{name}.json"
        if not path.exists():
            print(f"[fusion] bỏ {name}: chưa có {path.name}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        tp = fp = fn = tn = 0
        for row in data["rows"]:
            text = row["hyp"]
            kw = detector.detect(text, timestamp=0.0).get("risk", "Normal").capitalize()
            ai = phobert.predict(text)
            predicted = 1 if (kw in ("Suspicious", "Cheating")
                              or str(ai.get("label", "")).capitalize() in ("Suspicious", "Cheating")) else 0
            if row["label"] == 1 and predicted == 1:
                tp += 1
            elif row["label"] == 0 and predicted == 1:
                fp += 1
            elif row["label"] == 1 and predicted == 0:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        reports[name] = {"model": data["model"], "n": data["n"], "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                         "latency_s": data["latency_s"], "rtf": data["rtf"],
                         "peak_rss_mb": data["env"]["peak_rss_mb"]}
        print(f"[fusion:{name}] P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
              f"(tp={tp} fp={fp} fn={fn} tn={tn})")

    # Tham chiếu chéo: small/medium so với large-v3 trên cùng file.
    large = OUT_DIR / "asr_large.json"
    if large.exists():
        ref = {r["id"]: r["hyp"] for r in json.loads(large.read_text(encoding="utf-8"))["rows"]}
        for name in ("small", "medium"):
            path = OUT_DIR / f"asr_{name}.json"
            if not path.exists():
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
            scores = [error_rate(r["hyp"], ref.get(r["id"], "")) for r in rows]
            reports.setdefault("agreement_vs_large_v3", {})[name] = {
                "cer_mean": round(statistics.mean(error_rate(r["hyp"], ref.get(r["id"], ""), chars=True)
                                                  for r in rows), 4),
                "wer_mean": round(statistics.mean(scores), 4),
                "exact_match": round(sum(1 for s in scores if s == 0) / len(scores), 4)}
            print(f"[agreement {name}] {reports['agreement_vs_large_v3'][name]}")

    reports["env"] = env_info()
    (OUT_DIR / "fusion.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_robust() -> None:
    """Độ bền theo SNR của pipeline ĐANG TRIỂN KHAI (small + cổng).

    Reference = chính văn bản model nhỏ trả lời trên file SẠCH (self-consistency). Không dùng
    văn bản của model khác làm chuẩn: small và large-v3 chỉ khớp nhau 46.5% số câu, nên lấy
    large làm chuẩn sẽ lẫn "mô hình khác nhau" vào "nhiễu tăng".
    """
    from backend.ai_services.whisper.audio_pipeline import AudioPipeline

    ref_path = OUT_DIR / "asr_small.json"
    if not ref_path.exists():
        print("[robust] cần asr_small.json trước (chạy --stage asr --model small)")
        return
    refs = {r["id"]: r["hyp"] for r in json.loads(ref_path.read_text(encoding="utf-8"))["rows"]}
    pipe = AudioPipeline()
    beds = demo_beds()
    items = [c for c in corpus() if c["label"] == 1][:8]
    rows = []
    for bed_name in ("room_noise", "music"):
        bed = beds[bed_name]
        for snr in SNR_LEVELS:
            exact = 0
            cers = []
            passed = 0
            for item in items:
                speech = load_wav(item["path"])
                scaled = np.resize(bed, len(speech) + SR // 2) * ((rms(speech) / 10 ** (snr / 20)) / max(rms(bed), 1e-9))
                mix = scaled.astype(np.float32)
                mix[: len(speech)] += speech
                text = pipe.process_audio(mix, timestamp=0.0, source="robust").get("transcription", "")
                ref = refs.get(item["id"], "")
                if text.strip():
                    passed += 1
                if norm_text(text) == norm_text(ref) and text.strip():
                    exact += 1
                cers.append(error_rate(text, ref, chars=True))
            rows.append({"bed": bed_name, "snr_db": snr, "n": len(items), "gate_passed": passed,
                         "exact_match": exact, "cer_vs_clean": round(statistics.mean(cers), 4)})
            print(f"  {bed_name:10s} SNR={snr:+3d}dB  cổng_cho_qua={passed}/{len(items)} "
                  f"khung_chuan={exact}/{len(items)} cer={rows[-1]['cer_vs_clean']:.3f}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "robust.json").write_text(
        json.dumps({"stage": "robust", "env": env_info(),
                    "protocol": "self-consistency: reference = văn bản model nhỏ trên file sạch",
                    "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def demo_utterances() -> list:
    """Các 'câu' mà bộ cắt câu realtime thực sự sinh ra từ data/video_test_demo.mp4.

    Đây mới là đầu vào gây ảo giác trong thực tế: đoạn có RMS vượt ngưỡng cắt câu nhưng bên
    trong chỉ là nhiễu/va chạm. Im lặng thuần thì VAD đã chặn từ trước, không cần cổng.
    """
    import itertools

    import run_offline as ro
    from backend.ai_services.pipeline import audio_scan

    chunks = ro.decode_audio(DEMO_CLIP, SR, SR // 2, gain=1.95)
    first = next(chunks, None)
    if first is None:
        return []
    chunker = audio_scan.DynamicSpeechChunker(SR, noise_profile_seconds=0.0)
    out = []
    for index, chunk in enumerate(itertools.chain([first], chunks)):
        audio16 = chunker.feed(chunk)
        if audio16 is not None:
            lo, hi = chunker.last_span
            out.append((f"clip_demo_{lo:.0f}-{hi:.0f}s", audio16))
    tail = chunker.flush()
    if tail is not None:
        lo, hi = chunker.last_span
        out.append((f"clip_demo_{lo:.0f}-cuoi", tail))
    return out


def gate_inputs(beds: dict):
    """Đầu vào cho stage gate: đoạn nhiễu sinh ra từ clip + file nói thật làm đối chứng."""
    yield from demo_utterances()
    for name, bed in beds.items():
        yield (f"nen_{name}", np.concatenate([bed] * 3)[: 5 * SR])
    real = [c for c in corpus() if c["label"] == 1][:3] + [c for c in corpus() if c["label"] == 0][:3]
    for index, item in enumerate(real):
        yield (f"that_{index}_{Path(item['path']).stem}", load_wav(item["path"]))


def stage_gate() -> None:
    """Tỉ lệ câu + cảnh báo sai khi đầu vào KHÔNG phải tiếng người rõ: cổng BẬT vs TẮT.

    Đối chứng bắt buộc: 6 file nói thật phải được cổng CHO QUA hết — nếu không thì cổng đang
    chặn oan thay vì chặn ảo giác.
    """
    from backend.ai_services.whisper.audio_pipeline import AudioPipeline

    pipe = AudioPipeline()
    beds = demo_beds()
    rows = []
    for name, audio in gate_inputs(beds):
        for rep in range(2):
            off = pipe.process_audio(audio, timestamp=0.0, source="gate", bypass_gate=True)
            on = pipe.process_audio(audio, timestamp=0.0, source="gate", bypass_gate=False)
            rows.append({"input": name, "rep": rep,
                         "off_text": off.get("transcription", ""), "off_risk": off.get("risk"),
                         "on_text": on.get("transcription", ""), "on_risk": on.get("risk")})
            print(f"  {name:24s} rep{rep} tắt_cổng={off.get('transcription','')!r}({off.get('risk')}) "
                  f"bật_cổng={on.get('transcription','')!r}({on.get('risk')})", flush=True)
    noise = [r for r in rows if not r["input"].startswith("that_")]
    real = [r for r in rows if r["input"].startswith("that_")]
    bad = lambda v: str(v).lower() in ("cheating", "suspicious")
    out = {"stage": "gate", "env": env_info(), "n_noise": len(noise), "n_real": len(real),
           "invented_without_gate": sum(1 for r in noise if r["off_text"].strip()),
           "false_alerts_without_gate": sum(1 for r in noise if bad(r["off_risk"])),
           "invented_with_gate": sum(1 for r in noise if r["on_text"].strip()),
           "false_alerts_with_gate": sum(1 for r in noise if bad(r["on_risk"])),
           "real_kept_without_gate": sum(1 for r in real if r["off_text"].strip()),
           "real_kept_with_gate": sum(1 for r in real if r["on_text"].strip()),
           "rows": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gate.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gate] nhiễu ({out['n_noise']} phép): câu bịa {out['invented_without_gate']} -> "
          f"{out['invented_with_gate']}; cảnh báo sai {out['false_alerts_without_gate']} -> "
          f"{out['false_alerts_with_gate']} | nói thật giữ {out['real_kept_with_gate']}/{out['n_real']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["asr", "fusion", "robust", "gate"])
    parser.add_argument("--model", choices=["small", "medium", "large"], default="small")
    args = parser.parse_args()
    {"asr": lambda: stage_asr(args.model), "fusion": stage_fusion,
     "robust": stage_robust, "gate": stage_gate}[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

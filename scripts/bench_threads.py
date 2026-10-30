#!/usr/bin/env python
"""Ablation ngân sách thread cho YOLO — nguồn của dãy 145/68/76/60/118 ms trong báo cáo.

PyTorch/ONNX Runtime mặc định chiếm hết core của máy; khi bốn mô-đun AI cùng chạy thì "nhiều thread
hơn" không đồng nghĩa "nhanh hơn". Script đo lại phát biểu đó trên đúng khung hình 1080p.

  .venv/bin/python scripts/bench_threads.py                 # quét 1,2,4,8,16 thread (mỗi cấu hình 1 tiến trình)
  PROCTOR_CPU_THREADS=4 .venv/bin/python scripts/bench_threads.py --once   # đo một cấu hình
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FRAME_SIZES = {"1080p": (1920, 1080), "720p": (1280, 720), "540p": (960, 540)}


def measure_once(model_path: str, size: str, warmup: int, runs: int) -> tuple[int, float, float]:
    import numpy as np
    from ultralytics import YOLO

    from backend.core.device import CPU_THREADS, configure_cpu_threads

    configure_cpu_threads()
    height, width = FRAME_SIZES[size][1], FRAME_SIZES[size][0]
    frame = (np.random.rand(height, width, 3) * 255).astype("uint8")
    model = YOLO(model_path)
    for _ in range(warmup):
        model(frame, verbose=False)
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        model(frame, verbose=False)
        timings.append((time.perf_counter() - started) * 1000)
    return CPU_THREADS, statistics.mean(timings), statistics.pstdev(timings)


def sweep(model_path: str, size: str, warmup: int, runs: int) -> None:
    print(f"# YOLO {model_path} @ {size}, {runs} lần chạy sau {warmup} lần làm nóng")
    print(f"{'thread':>7} {'mean ms':>9} {'stddev':>8}")
    for threads in (1, 2, 4, 8, 16):
        env = dict(os.environ, PROCTOR_CPU_THREADS=str(threads))
        result = subprocess.run([sys.executable, __file__, "--once", "--model", model_path,
                                 "--size", size, "--warmup", str(warmup), "--runs", str(runs)],
                                env=env, capture_output=True, text=True, check=True)
        line = result.stdout.strip().splitlines()[-1]
        print(line, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="weights/yolov8n.pt")
    parser.add_argument("--size", default="1080p", choices=sorted(FRAME_SIZES))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--once", action="store_true", help="đo đúng cấu hình PROCTOR_CPU_THREADS hiện tại")
    args = parser.parse_args()

    if args.once:
        threads, mean_ms, std = measure_once(args.model, args.size, args.warmup, args.runs)
        print(f"{threads:>7} {mean_ms:>9.1f} {std:>8.1f}")
        return 0
    sweep(args.model, args.size, args.warmup, args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

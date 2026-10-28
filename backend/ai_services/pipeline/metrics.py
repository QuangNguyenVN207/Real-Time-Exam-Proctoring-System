"""Đo hiệu năng từng module ở luồng realtime — thuần Python, không phụ thuộc Streamlit.

Mục đích: trả lời "module nào đang tụt, chậm bao nhiêu, rớt bao nhiêu frame" bằng số thay vì cảm giác.
Mọi truy cập đều qua lock vì mỗi module chạy ở một thread khác nhau, còn UI đọc từ thread chính.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Optional

__all__ = ["METRICS", "MetricsRegistry", "StageStat"]

_WINDOW = 120  # số mẫu gần nhất để tính p50/p95 (~4s ở 30fps)


class _Timer:
    """Context manager: `with METRICS.time("face"): ...` — lỗi vẫn được ghi nhận rồi raise lại."""

    def __init__(self, stat: "StageStat") -> None:
        self._stat = stat
        self._t0 = 0.0

    def __enter__(self) -> "_Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.perf_counter() - self._t0
        if exc is None:
            self._stat.add(elapsed)
        else:
            self._stat.fail(exc)
        return False


class StageStat:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.runs = 0
        self.errors = 0
        self.drops = 0
        self.total_s = 0.0
        self.max_s = 0.0
        self.last_error: str = ""
        self._recent: Deque[float] = deque(maxlen=_WINDOW)
        self._first_ts: Optional[float] = None
        self._last_ts: Optional[float] = None

    def add(self, seconds: float) -> None:
        now = time.perf_counter()
        with self._lock:
            self.runs += 1
            self.total_s += seconds
            self.max_s = max(self.max_s, seconds)
            self._recent.append(seconds)
            self._first_ts = self._first_ts or now
            self._last_ts = now

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:160]

    def drop(self, n: int = 1) -> None:
        with self._lock:
            self.drops += n

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samples = sorted(self._recent)
            recent = self._recent

            def pct(q: float) -> float:
                if not samples:
                    return 0.0
                idx = min(len(samples) - 1, max(0, int(round(q * (len(samples) - 1)))))
                return samples[idx]

            span = (self._last_ts - self._first_ts) if self._first_ts and self._last_ts else 0.0
            return {
                "runs": self.runs,
                "errors": self.errors,
                "drops": self.drops,
                "avg_ms": round(1000 * (sum(recent) / len(recent)), 1) if recent else 0.0,
                "p50_ms": round(1000 * pct(0.50), 1),
                "p95_ms": round(1000 * pct(0.95), 1),
                "max_ms": round(1000 * self.max_s, 1),
                "fps": round(self.runs / span, 1) if span > 1e-6 else 0.0,
                "last_error": self.last_error,
            }


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stages: dict[str, StageStat] = {}

    def stage(self, name: str) -> StageStat:
        with self._lock:
            return self._stages.setdefault(name, StageStat())

    def time(self, name: str) -> _Timer:
        return _Timer(self.stage(name))

    def drop(self, name: str, n: int = 1) -> None:
        self.stage(name).drop(n)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            stages = dict(self._stages)
        return {name: stat.snapshot() for name, stat in sorted(stages.items())}

    def reset(self) -> None:
        with self._lock:
            self._stages.clear()


METRICS = MetricsRegistry()

"""Cắt câu theo RMS (dynamic VAD) + đóng gói kết quả AudioPipeline thành alert.

`DynamicSpeechChunker` là bản tách nguyên văn từ
`RealtimeAudioWorker._worker_loop` (realtime_audio_ubuntu.py:120-172,
realtime_audio_worker.py tương ứng trên Windows) để luồng offline đọc track audio
của file video cũng cắt câu y hệt. `to_alert()` sao y `hooked_process_audio`
trong app_webrtc.py:531-564.
"""
from __future__ import annotations

from math import gcd
from typing import Any, Optional

import numpy as np
from scipy.signal import resample_poly

__all__ = ["resample", "DynamicSpeechChunker", "is_alert", "to_alert"]


def resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample về `target_sr` rồi kẹp [-1, 1] — đúng thứ tự `_resample_to_target` + `_preprocess_audio` cũ."""
    audio = np.asarray(audio, dtype=np.float32)
    if source_sr != target_sr:
        g = gcd(source_sr, target_sr)
        audio = resample_poly(audio, target_sr // g, source_sr // g)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


class DynamicSpeechChunker:
    """Nhận từng chunk mẫu (float32, mono) -> trả về nguyên câu 16kHz khi đủ điều kiện dịch."""

    TARGET_SR = 16000
    MIN_PROCESS_SECONDS = 2.0
    MAX_PROCESS_SECONDS = 6.0
    SILENCE_TAIL_SECONDS = 0.8
    SILENCE_THRESHOLD = 0.02
    NOISE_PROFILE_SECONDS = 1.0
    OVERLAP_SECONDS = 1.5

    def __init__(
        self,
        source_sr: int,
        *,
        target_sr: int = TARGET_SR,
        noise_profile_seconds: float = NOISE_PROFILE_SECONDS,
        min_process_seconds: float = MIN_PROCESS_SECONDS,
        max_process_seconds: float = MAX_PROCESS_SECONDS,
        silence_tail_seconds: float = SILENCE_TAIL_SECONDS,
        silence_threshold: float = SILENCE_THRESHOLD,
        overlap_seconds: float = OVERLAP_SECONDS,
    ) -> None:
        self.source_sr = int(source_sr)
        self.target_sr = int(target_sr)
        self.min_samples = int(source_sr * min_process_seconds)
        self.max_samples = int(source_sr * max_process_seconds)
        self.tail_samples = max(1, int(source_sr * silence_tail_seconds))
        self.threshold = float(silence_threshold)
        self.overlap_samples = int(source_sr * overlap_seconds)
        self.noise_target_samples = int(source_sr * noise_profile_seconds)

        self._buffer = np.array([], dtype=np.float32)
        self._noise = np.array([], dtype=np.float32)
        self._fed = 0                      # số mẫu đã nạp, tính từ đầu nguồn audio
        self._anchor = 0                   # vị trí tuyệt đối của sample đầu tiên trong buffer
        self.last_span = (0.0, 0.0)        # (start_s, end_s) của câu vừa trả về lần gần nhất

    def feed(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """Nạp 1 chunk âm thanh ở `source_sr`. Trả về câu (audio 16kHz) cần đem đi dịch, hoặc None."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.size == 0:
            return None

        chunk_start = self._fed
        self._fed += chunk.size

        # 1s đầu bị ném đi làm noise profile — giống hệt bản realtime.
        if self.noise_target_samples > 0 and self._noise.size < self.noise_target_samples:
            self._noise = np.concatenate((self._noise, chunk))
            return None

        if self._buffer.size == 0:
            self._anchor = chunk_start
        self._buffer = np.concatenate((self._buffer, chunk))

        if self._buffer.size < self.min_samples:
            return None

        rms_tail = float(np.sqrt(np.mean(self._buffer[-self.tail_samples:] ** 2)))
        is_silence_tail = rms_tail < self.threshold

        if not is_silence_tail and self._buffer.size < self.max_samples:
            return None

        rms_total = float(np.sqrt(np.mean(self._buffer**2)))
        if rms_total < self.threshold:
            self._buffer = np.array([], dtype=np.float32)
            return None

        segment = self._buffer.copy()
        self.last_span = (self._anchor / self.source_sr, self._fed / self.source_sr)
        if is_silence_tail:
            self._buffer = np.array([], dtype=np.float32)
        else:
            self._buffer = self._buffer[-self.overlap_samples:]
            self._anchor = self._fed - self.overlap_samples

        return resample(segment, self.source_sr, self.target_sr)

    def flush(self) -> Optional[np.ndarray]:
        """Hết audio (offline): đem nốt phần đang đệm đi dịch nếu còn tiếng người."""
        if self._buffer.size < self.min_samples:
            return None
        rms_total = float(np.sqrt(np.mean(self._buffer**2)))
        segment = self._buffer.copy()
        self._buffer = np.array([], dtype=np.float32)
        if rms_total < self.threshold:
            return None
        self.last_span = (self._anchor / self.source_sr, self._fed / self.source_sr)
        return resample(segment, self.source_sr, self.target_sr)


def is_alert(result: dict[str, Any]) -> bool:
    risk = str(result.get("risk", "")).lower()
    status = str(result.get("status", "")).lower()
    return risk in ("cheating", "high", "medium") or status == "alert"


def to_alert(result: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Kết quả 1 câu từ AudioPipeline -> alert cho UI/log.

    Bản realtime chỉ đẩy lên UI khi transcription không rỗng (VAD cắt nhầm tạp âm),
    offline cũng giữ đúng quy tắc đó.
    """
    if not result:
        return None
    if not str(result.get("transcription", "")).strip():
        return None
    alert = dict(result)
    alert["module"] = "audio_phobert_pipeline"
    alert.setdefault("status", "alert" if is_alert(alert) else "info")
    return alert

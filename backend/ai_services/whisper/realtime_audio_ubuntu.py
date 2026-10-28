import time
import queue
import threading
import os
import sys
from contextlib import contextmanager

import numpy as np
import pyaudio

from backend.ai_services.whisper.audio_pipeline import AudioPipeline
from backend.ai_services.pipeline import audio_scan

# =========================
# Tắt log ALSA rác trên Linux
# =========================
@contextmanager
def suppress_alsa_logs():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)


class RealtimeAudioWorker:
    TARGET_SR = 16000
    CHANNELS = 1
    CHUNK_SECONDS = 0.5

    def __init__(self):
        self.pipeline = AudioPipeline()
        self.audio_queue = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

        self.audio_stream = None
        self.pyaudio_instance = None
        self.started = False

        self.input_sr = self.TARGET_SR

        # Thuật toán cắt câu theo RMS dùng chung với luồng offline (pipeline/audio_scan.py)
        self.chunker = audio_scan.DynamicSpeechChunker(self.input_sr)

        self.total_sent = 0
        self.total_alert = 0

    # =========================
    # Audio helpers (Lõi PyAudio cho Ubuntu)
    # =========================
    def _audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            print(f"[PyAudio] Cảnh báo luồng âm thanh: {status}")

        chunk = np.frombuffer(in_data, dtype=np.float32)
        if self.audio_queue.full():
            try:
                _ = self.audio_queue.get_nowait()
            except queue.Empty:
                pass
        self.audio_queue.put_nowait(chunk)
        return (in_data, pyaudio.paContinue)

    # =========================
    # Worker loop (Tích hợp logic VAD thông minh)
    # =========================
    def _worker_loop(self):
        print("[Worker] Started.")

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None:  # sentinel từ stop()
                break

            # 1..6. Gom sample -> đo RMS đuôi câu -> cắt câu 16kHz khi ngừng nói hoặc nói quá 6s
            clean_audio = self.chunker.feed(chunk)
            if clean_audio is None:
                continue

            try:
                timestamp = time.time()
                result = self.pipeline.process_audio(clean_audio, timestamp=timestamp, source="realtime-mic")

                if result is None:
                    continue

                transcription = result.get('transcription', '').strip()
                is_alert = audio_scan.is_alert(result)

                # Chặn log rác nếu không có tiếng và không có cảnh báo
                if not transcription and not is_alert:
                    continue

                print("\n" + "=" * 70)
                if is_alert:
                    reason_text = str(result.get('fusion_reason', '')).lower()
                    nguoi_bat = "🤖 AI PhoBERT (Bọc lót Keyword)" if 'ai catch' in reason_text else "🔑 Keyword (Bộ luật cứng)"
                    print(f"🚨 [CẢNH BÁO GIAN LẬN] - Phát hiện bởi: {nguoi_bat}")
                else:
                    print(f"✅ [AN TOÀN] - Cả Keyword và AI đều đồng ý an toàn")

                print(f"🗣️ Transcript : '{transcription}'")
                print(f"🧠 Lý do      : {result.get('fusion_reason', '')}")

                keywords = result.get('matched_keywords', [])
                if keywords:
                    print("🔑 Từ khóa bị bắt:")
                    for kw in keywords:
                        print(f"   - '{kw['keyword']}' (Mức độ: {kw['severity']})")

                self.total_sent += 1
                if is_alert:
                    self.total_alert += 1

                print(f"Thống kê Alert: {self.total_alert}/{self.total_sent}")
                print("=" * 70)

            except Exception as e:
                print(f"[Worker] Error: {e}")

    def stop(self):
        print("[Audio] 🛑 Đang dừng RealtimeAudioWorker...")
        self.stop_event.set()
        if self.audio_stream is not None:
            try:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
            except Exception as e:
                print(f"[Audio] Lỗi stop stream: {e}")

            try:
                self.audio_stream.close()
            except Exception as e:
                print(f"[Audio] Lỗi close stream: {e}")

            self.audio_stream = None

        if self.pyaudio_instance is not None:
            try:
                self.pyaudio_instance.terminate()
            except Exception as e:
                print(f"[Audio] Lỗi terminate PyAudio: {e}")

            self.pyaudio_instance = None

        # Đánh thức worker thread nếu đang chờ queue
        try:
            self.audio_queue.put_nowait(None)
        except Exception:
            pass

        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        self.started = False
        print("[Audio] ✅ RealtimeAudioWorker đã dừng.")

    # =========================
    # Start
    # =========================
    def start(self):
        if self.started: return
        self.started = True
        self.stop_event.clear()

        print("[Audio] 🔊 Khởi động PyAudio...")
        with suppress_alsa_logs():
            p = pyaudio.PyAudio()
        self.pyaudio_instance = p

        try:
            default_device = p.get_default_input_device_info()
            self.input_sr = int(default_device["defaultSampleRate"])
        except Exception:
            self.input_sr = self.TARGET_SR

        # Chunker làm việc theo sample rate thật của card âm thanh
        self.chunker = audio_scan.DynamicSpeechChunker(self.input_sr)
        self.worker_thread.start()

        blocksize = int(self.input_sr * self.CHUNK_SECONDS)

        print("=" * 70)
        print("Realtime audio worker started (PyAudio + Dynamic VAD)")
        print(f"Input SR      : {self.input_sr}")
        print(f"Target SR     : {self.TARGET_SR}")
        print("=" * 70)
        print("Listening... (Ctrl+C to stop)")
        print("Tip: giữ im lặng ~1 giây đầu để lấy noise profile.\n")

        try:
            with suppress_alsa_logs():
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=self.CHANNELS,
                    rate=self.input_sr,
                    input=True,
                    frames_per_buffer=blocksize,
                    stream_callback=self._audio_callback
                )
            self.audio_stream = stream
            stream.start_stream()

            while stream.is_active() and not self.stop_event.is_set():
                time.sleep(0.1)

        except Exception as e:
            print(f"[Audio] ❌ Audio stream error: {e}")
        finally:
            self.stop()

if __name__ == "__main__":
    RealtimeAudioWorker().start()
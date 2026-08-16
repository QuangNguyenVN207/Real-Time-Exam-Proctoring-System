import time
import queue
import threading
import os
import sys
from math import gcd
from contextlib import contextmanager

import numpy as np
import pyaudio
from scipy.signal import butter, sosfiltfilt, resample_poly

from backend.ai_services.whisper.audio_pipeline import AudioPipeline

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

    CHUNK_SECONDS = 0.5       # Ghi từng mảng 0.5s để queue chạy mượt
    PROCESS_INTERVAL = 3.5    # Cứ đúng 3 giây thì kích hoạt AI 1 lần
    BUFFER_SECONDS = 3.0      # Độ dài tối đa của file âm thanh đưa vào AI
    MIN_SECONDS = 2.0         # Phải tích đủ 3 giây mới bắt đầu chạy
    NOISE_PROFILE_SECONDS = 1.0

    def __init__(self):
        self.pipeline = AudioPipeline()
        self.audio_queue = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

        self.audio_stream = None
        self.pyaudio_instance = None
        self.started = False

        self.input_sr = self.TARGET_SR

        self.buffer_raw = np.array([], dtype=np.float32)
        self.noise_raw = np.array([], dtype=np.float32)

        self.last_process_time = 0.0
        self.total_sent = 0
        self.total_alert = 0

    # =========================
    # Audio helpers
    # =========================
    def _audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            print(f"[PyAudio] Cảnh báo luồng âm thanh: {status}")

        # PyAudio cấu hình paFloat32 sẽ trả về bytes có thể convert thẳng sang float32
        chunk = np.frombuffer(in_data, dtype=np.float32)

        if self.audio_queue.full():
            try:
                _ = self.audio_queue.get_nowait()
            except queue.Empty:
                pass

        self.audio_queue.put_nowait(chunk)
        return (in_data, pyaudio.paContinue)

    def _should_process(self):
        now = time.perf_counter()
        return (now - self.last_process_time) >= self.PROCESS_INTERVAL

    def _resample_to_target(self, audio):
        if self.input_sr == self.TARGET_SR:
            return audio

        audio = np.asarray(audio, dtype=np.float32)
        g = gcd(self.input_sr, self.TARGET_SR)
        up = self.TARGET_SR // g
        down = self.input_sr // g

        return resample_poly(audio, up, down).astype(np.float32)

    def _normalize_gain(self, audio, target_rms=0.08):
        audio = audio.astype(np.float32)
        rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
        gain = target_rms / (rms + 1e-8)
        gain = np.clip(gain, 0.5, 8.0)
        audio = audio * gain

        peak = np.max(np.abs(audio)) + 1e-8
        if peak > 1.0:
            audio = audio / peak

        return np.clip(audio, -1.0, 1.0)

    def _bandpass_filter(self, audio, sr, lowcut=80.0, highcut=7600.0, order=4):
        nyq = 0.5 * sr
        highcut = min(highcut, nyq - 100.0)

        if highcut <= lowcut:
            return audio

        sos = butter(
            order,
            [lowcut / nyq, highcut / nyq],
            btype="bandpass",
            output="sos",
        )
        return sosfiltfilt(sos, audio).astype(np.float32)

    def _noise_reduce(self, audio, sr, noise_profile=None):
        if noise_profile is None or len(noise_profile) < sr // 4:
            return audio

        try:
            import noisereduce as nr
            return nr.reduce_noise(
                y=audio,
                sr=sr,
                y_noise=noise_profile,
                stationary=False,
                prop_decrease=0.5,
            ).astype(np.float32)
        except Exception:
            return audio

    def _preprocess_audio(self, audio, sr, noise_profile=None):
        audio = audio.astype(np.float32)
        VOLUME_MULTIPLIER = 4.0
        audio = audio * VOLUME_MULTIPLIER
        audio = np.clip(audio, -1.0, 1.0)

        audio = self._noise_reduce(audio, sr, noise_profile=noise_profile)
        audio = self._bandpass_filter(audio, sr)
        audio = self._normalize_gain(audio, target_rms=0.08)

        return audio

    # =========================
    # Worker loop
    # =========================
    def _worker_loop(self):
        print("[Worker] Started.")

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if len(self.noise_raw) < int(self.input_sr * self.NOISE_PROFILE_SECONDS):
                self.noise_raw = np.concatenate((self.noise_raw, chunk))
                continue

            self.buffer_raw = np.concatenate((self.buffer_raw, chunk))

            max_raw_samples = int(self.input_sr * self.BUFFER_SECONDS)
            if len(self.buffer_raw) > max_raw_samples:
                self.buffer_raw = self.buffer_raw[-max_raw_samples:]

            if not self._should_process():
                continue

            min_raw_samples = int(self.input_sr * self.MIN_SECONDS)
            if len(self.buffer_raw) < min_raw_samples:
                continue

            self.last_process_time = time.perf_counter()

            samples_to_take = int(self.input_sr * self.BUFFER_SECONDS)
            raw_audio = self.buffer_raw[:samples_to_take]
            
            self.buffer_raw = self.buffer_raw[samples_to_take:]

            audio_16k = self._resample_to_target(raw_audio)
            noise_16k = self._resample_to_target(self.noise_raw[: int(self.input_sr * self.NOISE_PROFILE_SECONDS)])

            clean_audio = self._preprocess_audio(
                audio_16k,
                sr=self.TARGET_SR,
                noise_profile=noise_16k,
            )

            try:
                timestamp = time.time()
                result = self.pipeline.process_audio(
                    clean_audio,
                    timestamp=timestamp,
                    source="realtime-mic",
                )

                if result is None:
                    continue

                transcription = result.get('transcription', '').strip()
                risk_final = str(result.get('risk', '')).lower()
                status_final = str(result.get('status', '')).lower()
                
                is_alert = False
                if risk_final in ['cheating', 'high', 'medium'] or status_final == 'alert':
                    is_alert = True
                
                # 🛑 CHẶN LOG RÁC: NẾU KHÔNG CÓ TIẾNG NGƯỜI (CHUỖI RỖNG) VÀ KHÔNG PHẢI CẢNH BÁO -> BỎ QUA KHÔNG IN GÌ CẢ
                if not transcription and not is_alert:
                    continue

                print("\n" + "=" * 70)
                if is_alert:
                    reason_text = str(result.get('fusion_reason', '')).lower()
                    if 'ai catch' in reason_text:
                        nguoi_bat = "🤖 AI PhoBERT (Bọc lót Keyword)"
                    else:
                        nguoi_bat = "🔑 Keyword (Bộ luật cứng)"
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

        # Dừng PyAudio stream
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

        # Đóng PyAudio
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

        # Chờ worker kết thúc
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

        self.started = False

        print("[Audio] ✅ RealtimeAudioWorker đã dừng.")


    # =========================
    # Start
    # =========================
    def start(self):
        if self.started:
            return

        self.started = True
        self.stop_event.clear()

        self.worker_thread.start()

        print("[Audio] 🔊 Khởi động PyAudio...")

        with suppress_alsa_logs():
            p = pyaudio.PyAudio()

        self.pyaudio_instance = p

        # Lấy Sample Rate chuẩn của thiết bị
        try:
            default_device = p.get_default_input_device_info()
            self.input_sr = int(default_device["defaultSampleRate"])
        except Exception:
            self.input_sr = self.TARGET_SR

        blocksize = int(self.input_sr * self.CHUNK_SECONDS)

        print("=" * 70)
        print("Realtime audio worker started (PyAudio Engine)")
        print(f"Input SR      : {self.input_sr}")
        print(f"Target SR     : {self.TARGET_SR}")
        print(f"Channels      : {self.CHANNELS}")
        print(f"Chunk seconds : {self.CHUNK_SECONDS}")
        print(f"Process every : {self.PROCESS_INTERVAL}s")
        print("=" * 70)
        print("Listening...")

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

            # QUAN TRỌNG:
            # kiểm tra stop_event thay vì chỉ kiểm tra stream
            while (
                stream.is_active()
                and not self.stop_event.is_set()
            ):
                time.sleep(0.1)

        except Exception as e:
            print(f"[Audio] ❌ Audio stream error: {e}")

        finally:
            self.stop_event.set()

            if self.audio_stream is not None:
                try:
                    if self.audio_stream.is_active():
                        self.audio_stream.stop_stream()
                except Exception:
                    pass

                try:
                    self.audio_stream.close()
                except Exception:
                    pass

                self.audio_stream = None

            if self.pyaudio_instance is not None:
                try:
                    self.pyaudio_instance.terminate()
                except Exception:
                    pass

                self.pyaudio_instance = None

            if self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2.0)

            self.started = False

            print("[Audio] ✅ Audio worker stopped.")

if __name__ == "__main__":
    RealtimeAudioWorker().start()
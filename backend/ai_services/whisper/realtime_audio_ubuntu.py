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
    CHUNK_SECONDS = 0.5

    # 🔥 THUẬT TOÁN CẮT CÂU THÔNG MINH (DYNAMIC VAD CHUNKING TỪ CỘNG SỰ)
    MIN_PROCESS_SECONDS = 2.0   # Tối thiểu 2 giây mới đem đi dịch
    MAX_PROCESS_SECONDS = 6.0   # Tối đa 6 giây (nếu sinh viên nói liên tục không nghỉ)
    SILENCE_TAIL_SECONDS = 0.8  # Nếu phát hiện im lặng 0.8s ở cuối -> Cắt câu mang đi dịch ngay
    SILENCE_THRESHOLD = 0.01    # Ngưỡng âm lượng để tính là "im lặng"
    
    # Giữ lại thời gian lấy mẫu tiếng ồn cho Ubuntu
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

    def _resample_to_target(self, audio):
        if self.input_sr == self.TARGET_SR:
            return audio
        audio = np.asarray(audio, dtype=np.float32)
        g = gcd(self.input_sr, self.TARGET_SR)
        return resample_poly(audio, self.TARGET_SR // g, self.input_sr // g).astype(np.float32)

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
        if highcut <= lowcut: return audio
        sos = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass", output="sos")
        return sosfiltfilt(sos, audio).astype(np.float32)

    def _noise_reduce(self, audio, sr, noise_profile=None):
        if noise_profile is None or len(noise_profile) < sr // 4:
            return audio
        try:
            import noisereduce as nr
            return nr.reduce_noise(y=audio, sr=sr, y_noise=noise_profile, stationary=False, prop_decrease=0.5).astype(np.float32)
        except Exception:
            return audio

    def _preprocess_audio(self, audio, sr, noise_profile=None):
        audio = audio.astype(np.float32)
        audio = audio * 4.0 # Tăng Gain để AI nghe rõ
        audio = np.clip(audio, -1.0, 1.0)
        audio = self._noise_reduce(audio, sr, noise_profile=noise_profile)
        audio = self._bandpass_filter(audio, sr)
        audio = self._normalize_gain(audio, target_rms=0.08)
        return audio

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

            # Thu noise profile từ 1 giây đầu (Giữ nguyên của Ubuntu)
            if len(self.noise_raw) < int(self.input_sr * self.NOISE_PROFILE_SECONDS):
                self.noise_raw = np.concatenate((self.noise_raw, chunk))
                continue

            # 1. Liên tục gom âm thanh vào hộp chứa
            self.buffer_raw = np.concatenate((self.buffer_raw, chunk))
            buffer_len_sec = len(self.buffer_raw) / self.input_sr

            # 2. Chưa đủ 2 giây thì không làm gì cả
            if buffer_len_sec < self.MIN_PROCESS_SECONDS:
                continue

            # 3. Đo âm lượng của 0.8 giây cuối cùng (đuôi câu)
            tail_samples = int(self.input_sr * self.SILENCE_TAIL_SECONDS)
            tail_audio = self.buffer_raw[-tail_samples:]
            rms_tail = np.sqrt(np.mean(tail_audio ** 2))
            
            is_silence_tail = rms_tail < self.SILENCE_THRESHOLD

            # 4. Kích hoạt dịch nếu ngừng nói HOẶC nói liên tục quá 6 giây
            if is_silence_tail or buffer_len_sec >= self.MAX_PROCESS_SECONDS:
                
                # Kiểm tra âm lượng TOÀN BỘ hộp chứa
                rms_total = np.sqrt(np.mean(self.buffer_raw ** 2))
                if rms_total < self.SILENCE_THRESHOLD:
                    self.buffer_raw = np.array([], dtype=np.float32)
                    continue

                # 5. Có tiếng người -> Copy ra để chuẩn bị gửi AI
                raw_audio_to_process = self.buffer_raw.copy()

                # 🔥 6. QUYẾT ĐỊNH XÓA/GIỮ HỘP CHỨA
                if is_silence_tail:
                    self.buffer_raw = np.array([], dtype=np.float32)
                else:
                    overlap_samples = int(self.input_sr * 0.5)
                    self.buffer_raw = self.buffer_raw[-overlap_samples:]

                # --- BẮT ĐẦU DỊCH VÀ XỬ LÝ KẾT QUẢ ---
                audio_16k = self._resample_to_target(raw_audio_to_process)
                noise_16k = self._resample_to_target(self.noise_raw[: int(self.input_sr * self.NOISE_PROFILE_SECONDS)])
                
                clean_audio = self._preprocess_audio(audio_16k, sr=self.TARGET_SR, noise_profile=noise_16k)

                try:
                    timestamp = time.time()
                    result = self.pipeline.process_audio(clean_audio, timestamp=timestamp, source="realtime-mic")

                    if result is None:
                        continue

                    transcription = result.get('transcription', '').strip()
                    risk_final = str(result.get('risk', '')).lower()
                    status_final = str(result.get('status', '')).lower()
                    
                    is_alert = risk_final in ['cheating', 'high', 'medium'] or status_final == 'alert'
                    
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
        self.worker_thread.start()

        print("[Audio] 🔊 Khởi động PyAudio...")
        with suppress_alsa_logs():
            p = pyaudio.PyAudio()
        self.pyaudio_instance = p

        try:
            default_device = p.get_default_input_device_info()
            self.input_sr = int(default_device["defaultSampleRate"])
        except Exception:
            self.input_sr = self.TARGET_SR

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
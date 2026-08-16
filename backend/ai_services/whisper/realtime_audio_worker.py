import time
import queue
import threading
from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt, resample_poly

from backend.ai_services.whisper.audio_pipeline import AudioPipeline

class RealtimeAudioWorker:
    TARGET_SR = 16000
    CHANNELS = 1
    CHUNK_SECONDS = 0.5 

    # 🔥 THUẬT TOÁN CẮT CÂU THÔNG MINH (DYNAMIC VAD CHUNKING)
    MIN_PROCESS_SECONDS = 2.0   # Tối thiểu 2 giây mới đem đi dịch
    MAX_PROCESS_SECONDS = 6.0   # Tối đa 6 giây (nếu sinh viên nói liên tục không nghỉ)
    SILENCE_TAIL_SECONDS = 0.8  # Nếu phát hiện im lặng 0.8s ở cuối -> Cắt câu mang đi dịch ngay
    SILENCE_THRESHOLD = 0.01    # Ngưỡng âm lượng để tính là "im lặng" (tùy chỉnh theo mic)

    def __init__(self):
        self.pipeline = AudioPipeline()
        self.audio_queue = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

        self.device_id = None
        self.input_sr = self.TARGET_SR

        self.buffer_raw = np.array([], dtype=np.float32)

        self.total_sent = 0
        self.total_alert = 0

    # =========================
    # Device selection
    # =========================
    def list_input_devices(self):
        devices = sd.query_devices()
        print("=" * 90)
        print("INPUT DEVICES")
        print("=" * 90)
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                print(
                    f"[{idx}] {dev['name']} | "
                    f"in={dev['max_input_channels']} | "
                    f"default_sr={dev.get('default_samplerate', 'N/A')}"
                )

    def select_input_device(self):
        devices = sd.query_devices()
        keywords = ["lenovo", "realtek", "microphone", "array", "stereo mix"]
        candidates = []
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] <= 0:
                continue
            candidates.append(idx)
            name = dev["name"].lower()
            if any(k in name for k in keywords):
                return idx
        return candidates[0] if candidates else None

    def resolve_device_and_samplerate(self):
        self.device_id = self.select_input_device()
        if self.device_id is None:
            raise RuntimeError("No input device found.")

        info = sd.query_devices(self.device_id, "input")
        default_sr = int(info.get("default_samplerate", self.TARGET_SR))

        try:
            sd.check_input_settings(
                device=self.device_id, channels=self.CHANNELS,
                samplerate=self.TARGET_SR, dtype="float32",
            )
            self.input_sr = self.TARGET_SR
        except Exception:
            self.input_sr = default_sr
            sd.check_input_settings(
                device=self.device_id, channels=self.CHANNELS,
                samplerate=self.input_sr, dtype="float32",
            )

    # =========================
    # Audio helpers
    # =========================
    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata.copy().astype(np.float32).reshape(-1)
        if self.audio_queue.full():
            try: _ = self.audio_queue.get_nowait()
            except queue.Empty: pass
        self.audio_queue.put_nowait(chunk)

    def _resample_to_target(self, audio):
        if self.input_sr == self.TARGET_SR: return audio
        g = gcd(self.input_sr, self.TARGET_SR)
        return resample_poly(np.asarray(audio, dtype=np.float32), self.TARGET_SR // g, self.input_sr // g).astype(np.float32)

    def _normalize_audio(self, audio):
        audio = audio.astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak > 0.0:
            audio = audio * (0.9 / peak)
        return audio

    def _bandpass_filter(self, audio, sr, lowcut=80.0, highcut=7600.0, order=4):
        nyq = 0.5 * sr
        highcut = min(highcut, nyq - 100.0)
        if highcut <= lowcut: return audio
        sos = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass", output="sos")
        return sosfiltfilt(sos, audio).astype(np.float32)

    def _preprocess_audio(self, audio, sr):
        audio = self._bandpass_filter(audio, sr)
        audio = self._normalize_audio(audio)
        return audio

    # =========================
    # Worker loop (🔥 LOGIC CẮT CÂU MỚI NẰM Ở ĐÂY)
    # =========================
    def _worker_loop(self):
        print("[Worker] Started.")

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
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

            # 4. Kích hoạt dịch nếu: Người dùng ngừng nói (im lặng) HOẶC nói liên tục quá 6 giây
            if is_silence_tail or buffer_len_sec >= self.MAX_PROCESS_SECONDS:
                
                # Kiểm tra âm lượng TOÀN BỘ hộp chứa
                rms_total = np.sqrt(np.mean(self.buffer_raw ** 2))
                if rms_total < self.SILENCE_THRESHOLD:
                    # Nguyên đoạn vừa rồi toàn im lặng -> Xóa sạch hộp chứa, KHÔNG GỌI AI
                    self.buffer_raw = np.array([], dtype=np.float32)
                    continue

                # 5. Có tiếng người -> Copy ra để chuẩn bị gửi AI
                raw_audio_to_process = self.buffer_raw.copy()

                # 🔥 6. QUYẾT ĐỊNH XÓA/GIỮ HỘP CHỨA
                if is_silence_tail:
                    # Cắt tự nhiên tại khoảng lặng -> Chắc chắn trọn vẹn câu -> Xóa sạch (Chống lặp chữ)
                    self.buffer_raw = np.array([], dtype=np.float32)
                else:
                    # Bị ép cắt do quá 6 giây -> Cắt ngang câu nói -> Giữ lại 0.5s làm gối đầu cho câu sau (Chống mất chữ)
                    overlap_samples = int(self.input_sr * 0.5)
                    self.buffer_raw = self.buffer_raw[-overlap_samples:]

                # --- BẮT ĐẦU DỊCH VÀ XỬ LÝ KẾT QUẢ ---
                audio_16k = self._resample_to_target(raw_audio_to_process)
                clean_audio = self._preprocess_audio(audio_16k, sr=self.TARGET_SR)

                try:
                    result = self.pipeline.process_audio(clean_audio, timestamp=time.time(), source="realtime-mic")

                    if not result: continue

                    transcription = result.get('transcription', '').strip()
                    risk_final = str(result.get('risk', '')).lower()
                    status_final = str(result.get('status', '')).lower()
                    
                    is_alert = risk_final in ['cheating', 'high', 'medium'] or status_final == 'alert'
                    
                    if not transcription and not is_alert:
                        continue

                    print("\n" + "=" * 70)
                    if is_alert:
                        reason_text = str(result.get('fusion_reason', '')).lower()
                        nguoi_bat = "🤖 AI PhoBERT (Bọc lót Keyword)" if 'ai catch' in reason_text else "🔑 Keyword (Bộ luật cứng)"
                        print(f"🚨 [CẢNH BÁO GIAN LẬN] - Phát hiện bởi: {nguoi_bat}")
                    else:
                        print(f"✅ [AN TOÀN]")
                        
                    print(f"🗣️ Transcript : '{transcription}'")
                    
                    if is_alert:
                        print(f"🧠 Lý do      : {result.get('fusion_reason', '')}")
                        keywords = result.get('matched_keywords', [])
                        if keywords:
                            print("Từ khóa bị bắt:")
                            for kw in keywords:
                                print(f"   - '{kw['keyword']}' (Mức độ: {kw['severity']})")

                    self.total_sent += 1
                    if is_alert: self.total_alert += 1

                    print(f"Thống kê Alert: {self.total_alert}/{self.total_sent}")
                    print("=" * 70)

                except Exception as e:
                    print(f"[Worker] Error: {e}")

        print("[Worker] Stopped.")

    # =========================
    # Start 
    # =========================
    def start(self):
        self.list_input_devices()
        self.resolve_device_and_samplerate()
        self.worker_thread.start()

        blocksize = int(self.input_sr * self.CHUNK_SECONDS)

        print("=" * 70)
        print("Realtime audio worker started (DYNAMIC VAD CHUNKING OPTIMIZED)")
        print(f"Device        : {self.device_id}")
        print(f"Input SR      : {self.input_sr}")
        print(f"Target SR     : {self.TARGET_SR}")
        print("=" * 70)
        print("Listening... (Ctrl+C to stop)")

        try:
            with sd.InputStream(
                device=self.device_id,
                samplerate=self.input_sr,
                channels=self.CHANNELS,
                dtype="float32",
                blocksize=blocksize,
                callback=self._audio_callback,
            ):
                while True:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[Main] Stopping...")
        finally:
            self.stop_event.set()
            self.worker_thread.join(timeout=2.0)
            print("[Main] Done.")

if __name__ == "__main__":
    RealtimeAudioWorker().start()
import time
import queue
import threading
from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt, resample_poly

from whisper.audio_pipeline import AudioPipeline


class RealtimeAudioWorker:
    TARGET_SR = 16000
    CHANNELS = 1

    CHUNK_SECONDS = 0.5       # Ghi từng mảng 0.5s để queue chạy mượt
    PROCESS_INTERVAL = 3.0    # Cứ đúng 3 giây thì kích hoạt AI 1 lần
    BUFFER_SECONDS = 3.0      # Độ dài tối đa của file âm thanh đưa vào AI
    MIN_SECONDS = 2.0         # Phải tích đủ 3 giây mới bắt đầu chạy
    NOISE_PROFILE_SECONDS = 1.0

    def __init__(self):
        self.pipeline = AudioPipeline()
        self.audio_queue = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

        self.device_id = None
        self.input_sr = self.TARGET_SR

        self.buffer_raw = np.array([], dtype=np.float32)
        self.noise_raw = np.array([], dtype=np.float32)

        self.last_process_time = 0.0
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

        # Ưu tiên 16k nếu mic hỗ trợ
        try:
            sd.check_input_settings(
                device=self.device_id,
                channels=self.CHANNELS,
                samplerate=self.TARGET_SR,
                dtype="float32",
            )
            self.input_sr = self.TARGET_SR
        except Exception:
            # fallback về default samplerate của thiết bị
            self.input_sr = default_sr
            sd.check_input_settings(
                device=self.device_id,
                channels=self.CHANNELS,
                samplerate=self.input_sr,
                dtype="float32",
            )

    # =========================
    # Audio helpers
    # =========================
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[SoundDevice] {status}")

        chunk = indata.copy().astype(np.float32).reshape(-1)

        if self.audio_queue.full():
            try:
                _ = self.audio_queue.get_nowait()
            except queue.Empty:
                pass

        self.audio_queue.put_nowait(chunk)

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
                # 🔥 FIX: Hạ mức khử ồn từ 0.85 xuống 0.5 để không "ăn" mất giọng nói nhỏ
                prop_decrease=0.5,
            ).astype(np.float32)
        except Exception:
            return audio

    def _preprocess_audio(self, audio, sr, noise_profile=None):
        audio = audio.astype(np.float32)

        # 🔥 FIX: TĂNG GAIN NHƯ BẢN TEST (Để AI nghe rõ trước khi bị khử ồn)
        VOLUME_MULTIPLIER = 4.0  # Tăng gấp 4 lần
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

            # Thu noise profile từ 1 giây đầu
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

            # Lấy ĐÚNG 3 giây audio (tránh lấy dư)
            samples_to_take = int(self.input_sr * self.BUFFER_SECONDS)
            raw_audio = self.buffer_raw[:samples_to_take]
            
            # QUAN TRỌNG: Cắt bỏ 3 giây vừa lấy ra khỏi bộ đệm để hứng 3 giây tiếp theo
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
                    print("[Realtime] silence / no speech (Im lặng)")
                    continue

                print("\n" + "=" * 70)
                
                # 🔥 FIX LOGIC IN KẾT QUẢ ĐÚNG VỚI FUSION AI MỚI
                risk_final = str(result.get('risk', '')).lower()
                status_final = str(result.get('status', '')).lower()
                
                is_alert = False
                # Nếu Keyword chốt High/Medium/Cheating, HOẶC PhoBERT bật cờ Alert
                if risk_final in ['cheating', 'high', 'medium'] or status_final == 'alert':
                    is_alert = True
                    reason_text = str(result.get('fusion_reason', '')).lower()
                    if 'ai catch' in reason_text:
                        nguoi_bat = "🤖 AI PhoBERT (Bọc lót Keyword)"
                    else:
                        nguoi_bat = "🔑 Keyword (Bộ luật cứng)"
                        
                    print(f"🚨 [CẢNH BÁO GIAN LẬN] - Phát hiện bởi: {nguoi_bat}")
                else:
                    print(f"✅ [AN TOÀN] - Cả Keyword và AI đều đồng ý an toàn")
                    
                print(f"🗣️ Transcript : '{result.get('transcription', '')}'")
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
        print("Realtime audio worker started")
        print(f"Device        : {self.device_id}")
        print(f"Input SR      : {self.input_sr}")
        print(f"Target SR     : {self.TARGET_SR}")
        print(f"Channels      : {self.CHANNELS}")
        print(f"Chunk seconds : {self.CHUNK_SECONDS}")
        print(f"Process every : {self.PROCESS_INTERVAL}s")
        print("=" * 70)
        print("Listening... (Ctrl+C to stop)")
        print("Tip: giữ im lặng ~1 giây đầu để lấy noise profile.\n")

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
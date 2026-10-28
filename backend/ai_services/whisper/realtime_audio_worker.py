import time
import queue
import threading

import numpy as np
import sounddevice as sd

# Đảm bảo đường dẫn import đúng với cấu trúc thư mục của bạn
from backend.ai_services.whisper.audio_pipeline import AudioPipeline
from backend.ai_services.pipeline import audio_scan

class RealtimeAudioWorker:
    TARGET_SR = 16000
    CHANNELS = 1
    CHUNK_SECONDS = 0.5 

    def __init__(self):
        self.pipeline = AudioPipeline()
        self.audio_queue = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

        self.device_id = None
        self.input_sr = self.TARGET_SR

        # Thuật toán cắt câu theo RMS dùng chung với luồng offline (pipeline/audio_scan.py).
        # Bản Windows không ném đi 1 giây lấy noise profile như bản Ubuntu.
        self.chunker = audio_scan.DynamicSpeechChunker(self.input_sr, noise_profile_seconds=0.0)

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
        # Stereo Mix / loopback thu tiếng loa của máy, không phải mic thí sinh -> loại bỏ
        blocked = ["stereo mix", "loopback", "what u hear", "wave out"]
        keywords = ["microphone", "array", "lenovo", "realtek"]

        def usable(idx):
            dev = devices[idx]
            return dev["max_input_channels"] > 0 and not any(b in dev["name"].lower() for b in blocked)

        # 1) Mic mặc định mà người dùng đã chọn trong Windows Sound settings
        try:
            default_in = sd.default.device[0]
            if default_in is not None and default_in >= 0 and usable(default_in):
                return int(default_in)
        except Exception:
            pass

        # 2) Tìm theo từ khoá
        candidates = [i for i in range(len(devices)) if usable(i)]
        for idx in candidates:
            if any(k in devices[idx]["name"].lower() for k in keywords):
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
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

    # =========================
    # Worker loop (🔥 LOGIC CẮT CÂU MỚI ĐÃ ĐƯỢC TỐI ƯU)
    # =========================
    def _worker_loop(self):
        print("[Worker] Started. Đang chờ âm thanh...")

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None:  # sentinel từ stop()
                break

            # 1..6. Gom sample -> đo RMS đuôi câu -> cắt câu 16kHz (logic dùng chung với luồng offline)
            audio_16k = self.chunker.feed(chunk)
            if audio_16k is None:
                continue

            try:
                result = self.pipeline.process_audio(audio_16k, timestamp=time.time(), source="realtime-mic")

                if not result or result.get("status") == "idle": 
                    continue

                transcription = result.get('transcription', '').strip()
                status_final = str(result.get('status', '')).lower()

                is_alert = (status_final == 'alert')

                if not transcription and not is_alert:
                    continue

                print("\n" + "=" * 70)
                if is_alert:
                    reason_text = str(result.get('fusion_reason', '')).lower()
                    nguoi_bat = "🤖 AI PhoBERT (Ngữ cảnh)" if 'ai' in reason_text else "🔑 Keyword (Bộ luật cứng)"
                    print(f"🚨 [CẢNH BÁO GIAN LẬN] - Bắt bởi: {nguoi_bat}")
                else:
                    print(f"✅ [AN TOÀN]")

                print(f"🗣️ Transcript : '{transcription}'")

                if is_alert:
                    print(f"🧠 Lý do      : {result.get('fusion_reason', '')}")
                    keywords = result.get('matched_keywords', [])
                    if keywords:
                        print("Từ khóa bị bắt:")
                        for kw in keywords:
                            # Hỗ trợ format list string hoặc list dict tùy chuẩn của KeywordDetector
                            kw_str = kw.get('keyword', kw) if isinstance(kw, dict) else kw
                            print(f"   - '{kw_str}'")

                self.total_sent += 1
                if is_alert: self.total_alert += 1

                print(f"Thống kê Alert: {self.total_alert}/{self.total_sent}")
                print("=" * 70)

            except Exception as e:
                print(f"[Worker] Lỗi xử lý AI: {e}")

        print("[Worker] Stopped.")

    # =========================
    # Stop (gọi được từ luồng khác)
    # =========================
    def stop(self):
        self.stop_event.set()
        try:
            self.audio_queue.put_nowait(None)
        except queue.Full:
            pass

    # =========================
    # Start 
    # =========================
    def start(self):
        self.list_input_devices()
        self.resolve_device_and_samplerate()
        # Chunker phải theo đúng sample rate của thiết bị đã chọn
        self.chunker = audio_scan.DynamicSpeechChunker(self.input_sr, noise_profile_seconds=0.0)
        self.worker_thread.start()

        blocksize = int(self.input_sr * self.CHUNK_SECONDS)

        print("=" * 70)
        print("Realtime Audio Worker Started (DYNAMIC VAD CHUNKING OPTIMIZED)")
        print(f"Device        : {self.device_id}")
        print(f"Input SR      : {self.input_sr}")
        print(f"Target SR     : {self.TARGET_SR}")
        print("=" * 70)
        print("🎤 Đang lắng nghe... (Nhấn Ctrl+C để dừng)")

        try:
            with sd.InputStream(
                device=self.device_id,
                samplerate=self.input_sr,
                channels=self.CHANNELS,
                dtype="float32",
                blocksize=blocksize,
                callback=self._audio_callback,
            ):
                while not self.stop_event.is_set():
                    time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[Main] Đang tắt hệ thống...")
        finally:
            self.stop_event.set()
            self.worker_thread.join(timeout=2.0)
            print("[Main] Đã dừng hoàn toàn.")

if __name__ == "__main__":
    RealtimeAudioWorker().start()
import time
import queue
import threading
import os
import sys
from contextlib import contextmanager

import numpy as np


def _vad_threshold() -> float:
    """Ngưỡng RMS cắt câu; `PROCTOR_VAD_THRESHOLD` cho phép hạ khi mic laptop quá nhỏ."""
    raw = os.environ.get("PROCTOR_VAD_THRESHOLD", "").strip()
    try:
        value = float(raw) if raw else audio_scan.DynamicSpeechChunker.SILENCE_THRESHOLD
    except ValueError:
        value = audio_scan.DynamicSpeechChunker.SILENCE_THRESHOLD
    return max(0.001, value)


def _configure_alsa_plugins() -> None:
    """Trỏ ALSA tới plugin pulse nằm trong `~/.local/lib/alsa-lib` (nếu có).

    WSLg cho Windows dùng được micro qua PulseAudio socket `/mnt/wslg/PulseServer`, nhưng ảnh
    Ubuntu không cài sẵn `libasound2-plugins`, nên `pcm.!default { type pulse }` trong
    `~/.asoundrc` báo "cannot open shared library" và PyAudio chỉ thấy `default` giả.
    README §5.1 ghi cách giải nén gói đó vào `~/.local/lib/alsa-lib` **không cần sudo**; ở đây
    chỉ cần đặt `ALSA_PLUGIN_DIR` trước khi libasound được nạp (tức trước `import pyaudio`).
    Nếu máy đã cài `libasound2-plugins` (plugin ở `/usr/lib/*/alsa-lib`) thì biến này không cần.
    """
    if os.environ.get("ALSA_PLUGIN_DIR"):
        return  # người dùng đã tự đặt -> tôn trọng
    local_dir = os.path.expanduser("~/.local/lib/alsa-lib")
    if os.path.isfile(os.path.join(local_dir, "libasound_module_pcm_pulse.so")):
        os.environ["ALSA_PLUGIN_DIR"] = local_dir


_configure_alsa_plugins()

import pyaudio  # noqa: E402  (phải import SAU khi đặt ALSA_PLUGIN_DIR)

from backend.ai_services.whisper.audio_pipeline import AudioPipeline  # noqa: E402
from backend.ai_services.pipeline import audio_scan  # noqa: E402

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
        # Thread tạo trong `start()` (không phải ở đây) để worker có thể dừng rồi chạy lại:
        # một đối tượng Thread của Python chỉ start được MỘT lần.
        self.worker_thread = None
        self._stop_lock = threading.Lock()
        self._stopped = False

        self.audio_stream = None
        self.pyaudio_instance = None
        self.started = False
        self.last_error = None
        self._silent_warned = False

        self.input_sr = self.TARGET_SR
        self.device_index = None          # None = thiết bị mặc định; đổi bằng PROCTOR_MIC_DEVICE=<index>
        self.device_name = ""

        # Số đo để UI trả lời được "mic có nghe thấy gì không?" — RMS dưới ngưỡng VAD thì
        # không câu nào được cắt, nên không có gì hiện trên giao diện dù người dùng đang nói.
        self.last_rms = 0.0
        self.last_peak = 0.0
        self.max_rms = 0.0
        self.chunks_captured = 0
        self.segments_cut = 0

        # Thuật toán cắt câu theo RMS dùng chung với luồng offline (pipeline/audio_scan.py).
        # Hạ ngưỡng khi mic yếu: PROCTOR_VAD_THRESHOLD=0.012
        self.chunker = audio_scan.DynamicSpeechChunker(
            self.input_sr, silence_threshold=_vad_threshold()
        )

        self.total_sent = 0
        self.total_alert = 0

    # =========================
    # Audio helpers (Lõi PyAudio cho Ubuntu)
    # =========================
    def _audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            print(f"[PyAudio] Cảnh báo luồng âm thanh: {status}")

        chunk = np.frombuffer(in_data, dtype=np.float32)
        if chunk.size:
            self.last_rms = float(np.sqrt(np.mean(chunk ** 2)))
            self.last_peak = float(np.abs(chunk).max())
            if self.last_rms > self.max_rms:
                self.max_rms = self.last_rms
        self.chunks_captured += 1
        if self.chunks_captured >= 20 and self.max_rms == 0.0 and not self._silent_warned:
            self._silent_warned = True
            print("[Audio] ⚠️ 10 giây không có mẫu nào khác 0 -> thiết bị vào đang câm. "
                  "Chọn mic khác bằng PROCTOR_MIC_DEVICE=<index> (xem danh sách ở trên).")
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
            self.segments_cut += 1

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
        # `stop()` được gọi từ 2 nơi: nút/timer bên ngoài và `finally` của `start()`.
        # Bản cũ đọc `self.audio_stream` 3 lần -> luồng kia set None ở giữa -> văng
        # "'NoneType' object has no attribute 'close'". Giữ tham chiếu cục bộ + khoá.
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self.started = False

            print("[Audio] 🛑 Đang dừng RealtimeAudioWorker...")
            self.stop_event.set()
            stream, self.audio_stream = self.audio_stream, None
            if stream is not None:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                except Exception as e:
                    print(f"[Audio] Lỗi stop stream: {e}")

                try:
                    stream.close()
                except Exception as e:
                    print(f"[Audio] Lỗi close stream: {e}")

            instance, self.pyaudio_instance = self.pyaudio_instance, None
            if instance is not None:
                try:
                    instance.terminate()
                except Exception as e:
                    print(f"[Audio] Lỗi terminate PyAudio: {e}")

            # Đánh thức worker thread nếu đang chờ queue
            try:
                self.audio_queue.put_nowait(None)
            except Exception:
                pass

            if self.worker_thread is not None and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2.0)
            print("[Audio] ✅ RealtimeAudioWorker đã dừng.")

    # =========================
    # Start
    # =========================
    def start(self):
        with self._stop_lock:
            if self.started:
                return
            self.started = True
            self._stopped = False
        self.stop_event.clear()

        print("[Audio] 🔊 Khởi động PyAudio...")
        with suppress_alsa_logs():
            p = pyaudio.PyAudio()
        self.pyaudio_instance = p

        # PROCTOR_MIC_DEVICE=<index> để chọn mic khác khi mặc định là DMIC câm hoặc thiết bị lạ.
        raw_device = os.environ.get("PROCTOR_MIC_DEVICE", "").strip()
        self.device_index = int(raw_device) if raw_device.isdigit() else None
        try:
            info = (
                p.get_device_info_by_index(self.device_index)
                if self.device_index is not None
                else p.get_default_input_device_info()
            )
            self.input_sr = int(info["defaultSampleRate"])
            self.device_name = str(info.get("name", ""))
        except Exception as e:
            self.input_sr = self.TARGET_SR
            print(f"[Audio] Không lấy được thông tin mic ({e}) -> dùng {self.input_sr} Hz")

        # Chunker làm việc theo sample rate thật của card âm thanh
        self.chunker = audio_scan.DynamicSpeechChunker(
            self.input_sr, silence_threshold=_vad_threshold()
        )
        # Thread mới mỗi lần start -> dừng rồi bật lại được (Thread của Python chỉ start 1 lần).
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        blocksize = int(self.input_sr * self.CHUNK_SECONDS)

        print("=" * 70)
        print("Realtime audio worker started (PyAudio + Dynamic VAD)")
        print(f"Input device  : [{self.device_index if self.device_index is not None else 'default'}] {self.device_name}")
        print(f"Input SR      : {self.input_sr}")
        print(f"Target SR     : {self.TARGET_SR}")
        print(f"VAD threshold : {self.chunker.threshold} (RMS; đổi bằng PROCTOR_VAD_THRESHOLD)")
        print("=" * 70)
        print("Listening... (Ctrl+C to stop)")
        print("Tip: giữ im lặng ~1 giây đầu để lấy noise profile.\n")

        # Mở stream: lỗi ở bước này là "không có micro" thật (thiếu plugin pulse / quyền
        # micro Windows) -> in gợi ý cụ thể. Lỗi ở vòng chờ bên dưới chỉ là do stop() đóng
        # stream trước, không phải sự cố.
        try:
            with suppress_alsa_logs():
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=self.CHANNELS,
                    rate=self.input_sr,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=blocksize,
                    stream_callback=self._audio_callback
                )
            stream.start_stream()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            print(f"[Audio] ❌ Không mở được micro: {e}")
            try:
                print("[Audio] Các thiết bị vào khả dụng (đặt PROCTOR_MIC_DEVICE=<số> để chọn):")
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev.get("maxInputChannels", 0) > 0:
                        print(f"   [{i}] {dev['name']} @ {int(dev['defaultSampleRate'])} Hz")
            except Exception:
                pass
            print("[Audio] Gợi ý (WSL): cần libasound2-plugins để ALSA đi qua PulseAudio của WSLg.")
            print("        sudo apt install -y libasound2-plugins pulseaudio-utils")
            print("        rồi `~/.asoundrc` có pcm.!default { type pulse } (README §5.1).")
            print("        Trên Windows: Settings > Privacy > Microphone > cho phép desktop apps.")
            self.stop()
            return

        self.audio_stream = stream
        if self.stop_event.is_set():
            # stop() có thể đã chạy xong trước khi ta kịp gán -> tự đóng, tránh rò stream.
            try:
                stream.close()
            except Exception:
                pass
            self.audio_stream = None
            return
        # Chờ tới khi có người gọi stop(). Dùng stop_event (không dùng stream.is_active())
        # vì stop() có thể đã đóng stream -> is_active() raise "Stream not open".
        while not self.stop_event.is_set():
            time.sleep(0.1)
        self.stop()

if __name__ == "__main__":
    RealtimeAudioWorker().start()
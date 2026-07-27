import random
import time

class AudioWhisper:
    def __init__(self, model_path=None):
        print("[MOCK] Đã khởi tạo AudioWhisper (VAD + Whisper giả lập)")

    def process_audio(self, audio_chunk, timestamp):
        # Tỉ lệ 2% phát hiện tiếng nhắc bài
        if random.random() < 0.02:
            return {
                "module": "audio_whisper",
                "status": "alert",
                "timestamp": timestamp,
                "details": {
                    "transcription": "cau 5 dap an la A phai khong",
                    "keyword_detected": True
                }
            }
        return None # Im lặng hoặc tiếng ồn môi trường
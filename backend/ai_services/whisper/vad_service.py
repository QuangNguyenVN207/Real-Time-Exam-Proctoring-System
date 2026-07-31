import librosa
import numpy as np
import torch 

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps
)

class VADService:
    TARGET_SR = 16000

    def __init__(self):
        print("[VAD] Loading model...")
        self.model = load_silero_vad()
        print("[VAD] Model loaded!")

    def detect(self, audio_path):
        audio, sample_rate = librosa.load(audio_path, sr=16000)
        return self.detect_array(audio)

    def detect_array(self, audio):
        speech_timestamps = get_speech_timestamps(
            torch.from_numpy(audio),
            self.model,
            sampling_rate=self.TARGET_SR,
            # Các tham số chặn nhiễu mạnh tay hơn:
            threshold=0.65,              # Mặc định là 0.5. Tăng lên để phớt lờ tiếng ồn nhẹ.
            min_speech_duration_ms=300,  # Bắt buộc phải phát ra tiếng liên tục 0.3s mới tính là nói (loại bỏ tiếng gõ phím).
            min_silence_duration_ms=500, # Phải im lặng 0.5s mới ngắt câu (để không cắt nát chữ kéo dài).
            speech_pad_ms=150            # Lấy dư ra 150ms ở hai đầu để âm thanh mượt, không bị cụt cụt.
        )
        return speech_timestamps
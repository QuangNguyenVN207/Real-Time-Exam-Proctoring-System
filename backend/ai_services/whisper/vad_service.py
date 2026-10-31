import librosa
import numpy as np
import os
import torch 

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps
)

class VADService:
    TARGET_SR = 16000
    #: 0.65 chặn ồn tốt nhưng cũng bỏ tiếng nói nhỏ/nói xa. Giờ đã có cổng độ tự tin
    #: (audio_pipeline.ASR_MIN_SPEECH_PROB) chặn ảo giác nên có thể hạ nếu cần bắt lời nhỏ hơn:
    #: PROCTOR_VAD_SPEECH_THRESHOLD=0.5
    THRESHOLD = float(os.environ.get("PROCTOR_VAD_SPEECH_THRESHOLD", "0.65"))

    def __init__(self):
        print("[VAD] Loading model...")
        self.model = load_silero_vad()
        print("[VAD] Model loaded!")

    def detect(self, audio_path):
        audio, sample_rate = librosa.load(audio_path, sr=16000)
        return self.detect_array(audio)

    def detect_array(self, audio):
        # Silero giữ trạng thái LSTM giữ các lần gọi: không reset thì kết quả của câu này
        # phụ thuộc câu trước (đã gặp: cùng 1 file, chạy lẻ ra 0.65 s tiếng, chạy trong pipeline
        # ra 0.9 s -> cổng chặn độ dài hoạt động không ổn định).
        self.model.reset_states()
        speech_timestamps = get_speech_timestamps(
            torch.from_numpy(audio),
            self.model,
            sampling_rate=self.TARGET_SR,
            # Các tham số chặn nhiễu mạnh tay hơn:
            threshold=self.THRESHOLD,      # Mặc định là 0.5; 0.65 để phớt lờ tiếng ồn nhẹ.
            min_speech_duration_ms=300,  # Bắt buộc phải phát ra tiếng liên tục 0.3s mới tính là nói (loại bỏ tiếng gõ phím).
            min_silence_duration_ms=500, # Phải im lặng 0.5s mới ngắt câu (để không cắt nát chữ kéo dài).
            speech_pad_ms=150            # Lấy dư ra 150ms ở hai đầu để âm thanh mượt, không bị cụt cụt.
        )
        self.model.reset_states()
        return speech_timestamps

    def mean_probability(self, audio):
        """Độ tự tin trung bình của Silero trên từng cửa sổ 512 mẫu của đoạn đã cắt.

        `get_speech_timestamps` chỉ trả về mốc thời gian nên một tiếng ho/va chạm 0.7 s vẫn
        lọt qua. Đo 2026-10-29: câu nói thật (data/test_audio) cho 0.76-0.85, còn đoạn nhiễu
        lọt lưới trong data/video_test_demo.mp4 chỉ 0.08-0.13 -> dùng làm cổng chặn ảo giác.
        """
        if len(audio) < 512:
            return 0.0
        self.model.reset_states()
        try:
            probs = [
                float(self.model(torch.from_numpy(np.ascontiguousarray(audio[i:i + 512])), self.TARGET_SR))
                for i in range(0, len(audio) - 511, 512)
            ]
        finally:
            self.model.reset_states()
        return float(np.mean(probs)) if probs else 0.0
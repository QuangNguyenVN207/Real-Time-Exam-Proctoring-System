import numpy as np
import torch
import librosa

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from backend.ai_services.whisper.config import PHOWHISPER_MODEL


class PhoWhisperService:
    TARGET_SR = 16000

    def __init__(self):
        print("[PhoWhisper] Loading model...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[PhoWhisper] Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(PHOWHISPER_MODEL)

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            PHOWHISPER_MODEL,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.model.eval()

        try:
            self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="vi",
                task="transcribe",
            )
        except Exception:
            self.forced_decoder_ids = None

        print("[PhoWhisper] Model loaded!")

    def _prepare_audio(self, audio):
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=self.TARGET_SR)

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        return audio

    def transcribe(self, audio):
        audio = self._prepare_audio(audio)

        if audio.size == 0:
            return {
                "language": "vi",
                "text": "",
                "segments": [],
            }

        inputs = self.processor(
            audio,
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
        )

        input_features = inputs["input_features"].to(
            self.device,
            dtype=self.dtype,
        )

        generate_kwargs = dict(
            input_features=input_features,
            do_sample=False,         # Giữ nguyên False để tránh tính ngẫu nhiên (chỉ dùng greedy/beam)
            num_beams=3,             # Có thể cân nhắc giảm xuống 1 hoặc 2 nếu cần tốc độ realtime nhanh hơn, 3 là tốt cho độ chính xác
            temperature=0.0,
            repetition_penalty=1.1,  # Tăng nhẹ lên 1.1 để phạt gắt hơn các từ lặp lại
            no_repeat_ngram_size=3,  # CỰC KỲ QUAN TRỌNG: Ngăn chặn lỗi lặp lại 1 cụm từ vô tận (bệnh phổ biến của Whisper)
            early_stopping=True,     # Dừng sớm quá trình beam search khi đã tìm thấy sequence tốt nhất
            max_new_tokens=128,
            
            # (Tùy chọn nâng cao nếu HuggingFace version hỗ trợ) 
            # Giúp bỏ qua các đoạn mà mô hình có độ tự tin cực thấp (thường là tiếng ồn)
            # suppress_tokens=[...], 
        )

        if self.forced_decoder_ids is not None:
            generate_kwargs["forced_decoder_ids"] = self.forced_decoder_ids
        else:
            generate_kwargs["language"] = "vi"
            generate_kwargs["task"] = "transcribe"

        with torch.no_grad():
            predicted_ids = self.model.generate(**generate_kwargs)

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0].strip()

        return {
            "language": "vi",
            "text": text,
            "segments": [
                {
                    "start": 0,
                    "end": len(audio) / self.TARGET_SR,
                    "text": text,
                }
            ],
        }
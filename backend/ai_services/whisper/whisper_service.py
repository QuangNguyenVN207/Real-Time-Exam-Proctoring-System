import numpy as np
import torch
import librosa

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from whisper.config import PHOWHISPER_MODEL


class PhoWhisperService:
    TARGET_SR = 16000

    def __init__(self, model_path=None):
        print("[PhoWhisper] Loading model...")

        self.model_path = model_path or PHOWHISPER_MODEL

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[PhoWhisper] Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(self.model_path)

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
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

    def transcribe(self, audio):

        segments, info = self.model.transcribe(
            audio,
            language="vi",
            beam_size=1,
            temperature=0,
            vad_filter=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.7,
            compression_ratio_threshold=2.2,
            log_prob_threshold=-0.7,
            initial_prompt="Đây là hội thoại tiếng Việt trong kỳ thi trực tuyến."
        )

        results = []
        full_text = ""

        for segment in segments:
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

            full_text += segment.text + " "

        return {
            "language": info.language,
            "text": full_text.strip(),
            "segments": results
        }
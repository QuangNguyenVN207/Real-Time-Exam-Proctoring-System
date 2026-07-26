from faster_whisper import WhisperModel
import torch

from whisper.config import (
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
)


class WhisperService:

    def __init__(self):

        print("[Whisper] Loading model...")

        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"

        print(f"[Whisper] Device: {device}")
        print(f"[Whisper] Compute: {compute_type}")

        self.model = WhisperModel(
            WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
        )

        print("[Whisper] Model loaded!")

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
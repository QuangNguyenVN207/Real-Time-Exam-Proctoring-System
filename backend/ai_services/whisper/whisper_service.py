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
            beam_size=3,
            language=WHISPER_LANGUAGE,
            condition_on_previous_text=False,
            vad_filter=False,
            temperature=0.0
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
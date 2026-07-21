from faster_whisper import WhisperModel
from whisper.config import (
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)


class WhisperService:
    def __init__(self):
        print("[Whisper] Loading model...")

        self.model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )

        print("[Whisper] Model loaded!")

    def transcribe(self, audio):

        segments, info = self.model.transcribe(
            audio,
            beam_size=5,
            language=WHISPER_LANGUAGE,
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


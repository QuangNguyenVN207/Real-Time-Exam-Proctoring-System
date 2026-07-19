from faster_whisper import WhisperModel


class WhisperService:
    def __init__(
        self,
        model_size="tiny",
        device="cuda",
        compute_type="float16"
    ):
        print("[Whisper] Loading model...")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

        print("[Whisper] Model loaded!")

    def transcribe(self, audio_path: str):

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5
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
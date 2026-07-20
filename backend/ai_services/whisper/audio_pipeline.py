from whisper.whisper_service import WhisperService
from whisper.keyword_detector import KeywordDetector


class AudioPipeline:

    def __init__(self):

        self.whisper = WhisperService()

        self.detector = KeywordDetector()

    def process(self, audio_path: str):

        # Bước 1: Chuyển audio thành text
        transcript = self.whisper.transcribe(audio_path)

        # Bước 2: Phát hiện từ khóa
        keyword_result = self.detector.detect(
            transcript["text"]
        )

        # Bước 3: Ghép kết quả
        return {
            "language": transcript["language"],
            "text": transcript["text"],
            "segments": transcript["segments"],
            "alert": keyword_result["alert"],
            "score": keyword_result["score"],
            "matched": keyword_result["matched"]
        }

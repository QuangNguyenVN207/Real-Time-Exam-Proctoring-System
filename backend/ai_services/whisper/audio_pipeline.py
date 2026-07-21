from whisper.whisper_service import WhisperService
from whisper.keyword_detector import KeywordDetector
from whisper.vad_service import VADService
from whisper.audio_utils import (
    load_audio,
    extract_speech
)


class AudioPipeline:

    def __init__(self):
        
        self.vad = VADService()

        self.whisper = WhisperService()

        self.detector = KeywordDetector()

    def process(self, audio_path: str):

        audio = load_audio(audio_path)

        speech_segments = self.vad.detect(audio_path)

        speech_audio = extract_speech(
            audio,
            speech_segments
        )

        if len(speech_audio) == 0:

            return {
                "language": "",
                "text": "",
                "segments": [],
                "speech_segments": [],
                "alert": False,
                "score": 0,
                "matched": []
            }

        # 1. Speech -> Text
        transcript = self.whisper.transcribe(
            speech_audio
        )

        # 2. Detect keyword
        keyword_result = self.detector.detect(
            transcript["text"]
        )

        # 3. Merge result
        return {

            "language": transcript["language"],

            "text": transcript["text"],

            "segments": transcript["segments"],

            "speech_segments": speech_segments,

            "alert": keyword_result["alert"],

            "score": keyword_result["score"],

            "matched": keyword_result["matched"]

        }
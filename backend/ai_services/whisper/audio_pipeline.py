import time
from whisper.whisper_service import WhisperService
from whisper.keyword_detector import KeywordDetector
from whisper.vad_service import VADService
from whisper.audio_logger import AudioLogger
from whisper.audio_utils import (
    load_audio,
    extract_speech
)


class AudioPipeline:

    def __init__(self):
        
        self.vad = VADService()

        self.whisper = WhisperService()

        self.detector = KeywordDetector()

        self.logger = AudioLogger()

    def process(self, audio_path: str):

        audio = load_audio(audio_path)

        speech_segments = self.vad.detect(audio_path)

        speech_audio = extract_speech(
            audio,
            speech_segments
        )

        if len(speech_audio) == 0:

            return {
                "module": "audio_whisper",
                "status": "idle",
                "transcription": "",
                "keyword_detected": False,
                "matched": [],
                "timestamp": int(time.time())
            }

        # 1. Speech -> Text
        transcript = self.whisper.transcribe(
            speech_audio
        )

        # 2. Detect keyword
        keyword_result = self.detector.detect(
            transcript["text"]
        )

        if keyword_result["alert"]:

            self.logger.write(
                transcript["text"],
                keyword_result["matched"]
            )

        # 3. Merge result
        return {

            "module": "audio_whisper",

            "status": "alert" if keyword_result["alert"] else "normal",

            "language": transcript["language"],

            "transcription": transcript["text"],

            "speech_segments": speech_segments,

            "keyword_detected": keyword_result["alert"],

            "score": keyword_result["score"],

            "matched": keyword_result["matched"],

            "timestamp": int(time.time())

        }
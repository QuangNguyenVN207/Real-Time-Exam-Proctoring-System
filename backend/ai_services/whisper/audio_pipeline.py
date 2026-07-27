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
        """
        Xử lý từ file wav/mp3
        """

        audio = load_audio(audio_path)

        return self.process_audio(audio)

    def process_audio(self, audio):
        """
        Xử lý trực tiếp từ numpy array
        (microphone hoặc audio đã load)
        """

        # ---------------- VAD ---------------- #

        speech_segments = self.vad.detect_array(audio)

        speech_audio = extract_speech(
            audio,
            speech_segments
        )

        if len(speech_audio) == 0:

            return {
                "module": "audio_whisper",
                "status": "idle",
                "language": "",
                "transcription": "",
                "speech_segments": [],
                "keyword_detected": False,
                "score": 0,
                "matched": [],
                "timestamp": int(time.time())
            }

        # ---------------- Whisper ---------------- #

        transcript = self.whisper.transcribe(
            speech_audio
        )

        # ---------------- Keyword Detect ---------------- #

        keyword_result = self.detector.detect(
            transcript["text"]
        )

        # ---------------- Log ---------------- #

        if keyword_result["alert"]:

            self.logger.write(
                transcript["text"],
                keyword_result["matched"]
            )

        # ---------------- JSON Result ---------------- #

        return {

            "module": "audio_whisper",

            "status": "alert"
            if keyword_result["alert"]
            else "normal",

            "language": transcript["language"],

            "transcription": transcript["text"],

            "speech_segments": speech_segments,

            "keyword_detected": keyword_result["alert"],

            "score": keyword_result["score"],

            "matched": keyword_result["matched"],

            "timestamp": int(time.time())
        }
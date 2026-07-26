import time

from whisper.phowhisper_service import PhoWhisperService
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

        self.whisper = PhoWhisperService()

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

        # ==========================================================
        # Voice Activity Detection
        # ==========================================================

        speech_segments = self.vad.detect_array(audio)

        speech_audio = extract_speech(
            audio,
            speech_segments
        )

        if len(speech_audio) == 0:

            return {

                "module": "audio_phowhisper",

                "status": "idle",

                "language": "",

                "transcription": "",

                "speech_segments": [],

                "keyword_detected": False,

                "confidence": 0,

                "risk": "safe",

                "keyword_score": 0,

                "rule_bonus": 0,

                "context_bonus": 0,

                "penalty": 0,

                "matched": [],

                "matched_rules": [],

                "matched_context": [],

                "matched_negative": [],

                "timestamp": int(time.time())
            }

        # ==========================================================
        # PhoWhisper
        # ==========================================================

        transcript = self.whisper.transcribe(
            speech_audio
        )

        # ==========================================================
        # Keyword Detection
        # ==========================================================

        keyword_result = self.detector.detect(
            transcript["text"]
        )

        # ==========================================================
        # Logger
        # ==========================================================

        if keyword_result["alert"]:

            self.logger.write(

                text=transcript["text"],

                confidence=keyword_result["confidence"],

                risk=keyword_result["risk"],

                matched=keyword_result["matched"],

                matched_rules=keyword_result["matched_rules"],

                source="pipeline"

            )

        # ==========================================================
        # Final Result
        # ==========================================================

        return {

            "module": "audio_phowhisper",

            "status": keyword_result["risk"],

            "language": transcript["language"],

            "transcription": transcript["text"],

            "speech_segments": speech_segments,

            "keyword_detected": keyword_result["alert"],

            "confidence": keyword_result["confidence"],

            "risk": keyword_result["risk"],

            "keyword_score": keyword_result["keyword_score"],

            "rule_bonus": keyword_result["rule_bonus"],

            "context_bonus": keyword_result["context_bonus"],

            "penalty": keyword_result["penalty"],

            "matched": keyword_result["matched"],

            "matched_rules": keyword_result["matched_rules"],

            "matched_context": keyword_result["matched_context"],

            "matched_negative": keyword_result["matched_negative"],

            "timestamp": int(time.time())
        }
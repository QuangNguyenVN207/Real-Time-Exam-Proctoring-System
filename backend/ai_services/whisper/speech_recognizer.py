from whisper.config import SPEECH_MODEL
from whisper.whisper_service import WhisperService
from whisper.phowhisper_service import PhoWhisperService


class SpeechRecognizer:

    def __init__(self):

        if SPEECH_MODEL.lower() == "phowhisper":

            self.model = PhoWhisperService()
            self.name = "audio_phowhisper"

        else:

            self.model = WhisperService()
            self.name = "audio_whisper"

    def transcribe(self, audio):

        return self.model.transcribe(audio)
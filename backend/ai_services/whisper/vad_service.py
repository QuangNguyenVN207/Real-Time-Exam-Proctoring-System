import librosa
import numpy as np
import torch 

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps
)


class VADService:

    TARGET_SR = 16000

    def __init__(self):

        print("[VAD] Loading model...")

        self.model = load_silero_vad()

        print("[VAD] Model loaded!")

    def detect(self, audio_path):

        audio, sample_rate = librosa.load(
            audio_path,
            sr=16000
        )

        return self.detect_array(audio)

    def detect_array(self, audio):

        speech_timestamps = get_speech_timestamps(
            torch.from_numpy(audio),
            self.model
        )

        return speech_timestamps
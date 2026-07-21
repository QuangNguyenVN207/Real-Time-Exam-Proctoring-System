import librosa
import numpy as np

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

    def detect(self, audio_path: str):

        # Load audio và tự chuyển về mono + 16kHz
        audio, sample_rate = librosa.load(
            audio_path,
            sr=self.TARGET_SR,
            mono=True
        )

        audio = np.asarray(audio, dtype=np.float32)

        timestamps = get_speech_timestamps(
            audio,
            self.model,
            sampling_rate=self.TARGET_SR
        )

        return timestamps
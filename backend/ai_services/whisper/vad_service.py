from silero_vad import load_silero_vad, get_speech_timestamps
import soundfile as sf


class VADService:

    def __init__(self):

        print("[VAD] Loading model...")

        self.model = load_silero_vad()

        print("[VAD] Model loaded!")

    def detect(self, audio_path: str):

        audio, sample_rate = sf.read(audio_path)

        # Nếu stereo -> chuyển về mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        timestamps = get_speech_timestamps(
            audio,
            self.model,
            sampling_rate=sample_rate
        )

        return timestamps
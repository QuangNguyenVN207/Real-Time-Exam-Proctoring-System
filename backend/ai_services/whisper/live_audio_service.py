import time

from whisper.audio_pipeline import AudioPipeline
from whisper.microphone_service import MicrophoneService


class LiveAudioService:

    def __init__(self):

        self.pipeline = AudioPipeline()

        self.microphone = MicrophoneService()

    def start(self):

        print("=" * 60)
        print("Live Audio Service Started")
        print("=" * 60)

        while True:

            print("\nRecording ...")

            audio = self.microphone.record()

            result = self.pipeline.process_audio(audio)

            # Không có tiếng nói
            if result["status"] == "idle":
                print("No speech detected.")
                continue

            print("-" * 50)

            print("Language :", result["language"])

            print("Transcript :", result["transcription"])

            print("Alert :", result["keyword_detected"])

            print("Matched :", result["matched"])

            print("-" * 50)

            time.sleep(0.2)
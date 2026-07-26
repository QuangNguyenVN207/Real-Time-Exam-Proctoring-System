import time

from whisper.audio_pipeline import AudioPipeline


class AudioWhisper:

    def __init__(self, model_path=None):

        print("[AudioWhisper] Loading Audio Pipeline...")

        self.pipeline = AudioPipeline()

        print("[AudioWhisper] Ready!")

    def process_audio(self, audio_chunk, timestamp):

        result = self.pipeline.process_audio(audio_chunk)

        if result is None:
            return None

        return {
            "module": "audio_whisper",
            "status": result["status"],
            "timestamp": timestamp,
            "transcription": result["transcription"],
            "keyword_detected": result["keyword_detected"]
        }
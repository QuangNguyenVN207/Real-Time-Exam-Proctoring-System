from whisper.microphone_service import MicrophoneService
from whisper.audio_pipeline import AudioPipeline

mic = MicrophoneService(duration=3)
pipeline = AudioPipeline()

while True:

    audio = mic.record()

    result = pipeline.process_audio(audio)

    print("=" * 50)
    print(result)
from whisper.microphone_service import MicrophoneService

mic = MicrophoneService()

audio = mic.record(3)

print(audio.shape)

print(audio[:20])
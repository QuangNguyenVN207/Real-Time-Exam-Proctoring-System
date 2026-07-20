from pathlib import Path
from whisper.whisper_service import WhisperService

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_PATH = BASE_DIR / "samples" / "input1.wav"

service = WhisperService()

result = service.transcribe(str(AUDIO_PATH))

print("Language:", result["language"])
print()

print("Transcript:")
print(result["text"])

print("\nSegments:")
for s in result["segments"]:
    print(f"[{s['start']:.2f}s -> {s['end']:.2f}s] {s['text']}")
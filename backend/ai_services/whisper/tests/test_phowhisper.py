from pathlib import Path

from whisper.phowhisper_service import PhoWhisperService


BASE_DIR = Path(__file__).resolve().parent.parent


samples = BASE_DIR / "samples" / "cheating"
service = PhoWhisperService()

for audio in sorted(samples.glob("*.wav")):

    print("=" * 60)
    print(audio.name)

    result = service.transcribe(str(audio))

    print(result["text"])
from pathlib import Path

from whisper.vad_service import VADService

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"

vad = VADService()

wav_files = list(SAMPLES_DIR.rglob("*.wav"))

print(f"Tìm thấy {len(wav_files)} file.\n")

for audio_path in wav_files:

    print("=" * 60)
    print(audio_path.relative_to(SAMPLES_DIR))

    speech = vad.detect(str(audio_path))

    print(f"Speech segments: {len(speech)}")

    for i, seg in enumerate(speech, start=1):
        start = seg["start"] / 16000
        end = seg["end"] / 16000

        print(
            f"{i}. {start:.2f}s -> {end:.2f}s"
        )

    print()
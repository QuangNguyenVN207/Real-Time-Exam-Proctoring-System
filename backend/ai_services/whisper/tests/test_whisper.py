from pathlib import Path
from whisper.audio_pipeline import AudioPipeline
from whisper.whisper_service import WhisperService

BASE_DIR = Path(__file__).resolve().parent.parent

SAMPLES_DIR = BASE_DIR / "samples"

SAMPLES_DIR = BASE_DIR / "samples"

pipeline = AudioPipeline()

# Lấy tất cả file .wav
audio_files = sorted(SAMPLES_DIR.glob("*.wav"))

print(f"Tìm thấy {len(audio_files)} file.\n")

wav_files = list(SAMPLES_DIR.rglob("*.wav"))

print(f"Tìm thấy {len(wav_files)} file.\n")

for audio_file in wav_files:

    print("=" * 70)
    print(f"Đang xử lý: {audio_file.relative_to(SAMPLES_DIR)}")

    result = service.transcribe(str(audio_file))

    print("Language :", result["language"])
    print("Transcript:", result["text"])

    print("\nSegments:")
    for s in result["segments"]:
        print(
            f"{s['start']:.2f}s -> {s['end']:.2f}s : {s['text']}"
        )

    print()
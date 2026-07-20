from pathlib import Path

from whisper.audio_pipeline import AudioPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"

pipeline = AudioPipeline()

# Lấy tất cả file .wav
audio_files = sorted(SAMPLES_DIR.glob("*.wav"))

print(f"Tìm thấy {len(audio_files)} file.\n")

for audio_file in audio_files:

    print("=" * 70)
    print(f"Đang xử lý: {audio_file.name}")

    result = pipeline.process(str(audio_file))

    print("Language:", result["language"])
    print("Transcript:")
    print(result["text"])
    print()
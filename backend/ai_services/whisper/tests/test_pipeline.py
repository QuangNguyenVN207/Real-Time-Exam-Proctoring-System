from pathlib import Path

from whisper.audio_pipeline import AudioPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"

pipeline = AudioPipeline()

# Tìm tất cả file .wav trong samples và các thư mục con
audio_files = sorted(SAMPLES_DIR.rglob("*.wav"))

print(f"Tìm thấy {len(audio_files)} file.\n")

for audio_file in audio_files:

    print("=" * 70)
    print(f"Đang xử lý: {audio_file.relative_to(SAMPLES_DIR)}")

    result = pipeline.process(str(audio_file))

    print("Language :", result["language"])
    print("Transcript:", result["text"])
    print("\nSpeech Segments:")

    for seg in result["speech_segments"]:

        start = seg["start"] / 16000
        end = seg["end"] / 16000

        print(f"{start:.2f}s -> {end:.2f}s")
    print("Alert     :", result["alert"])
    print("Score     :", result["score"])

    if result["matched"]:
        print("Matched:")
        for item in result["matched"]:
            print(
                f"  - {item['keyword']} "
                f"(score={item['score']:.1f}, "
                f"severity={item['severity']}, "
                f"category={item['category']})"
            )
    else:
        print("Matched: None")

    print()
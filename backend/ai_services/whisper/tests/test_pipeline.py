from pathlib import Path
from pprint import pprint

from pprint import pprint

from whisper.audio_pipeline import AudioPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"

pipeline = AudioPipeline()

# ==========================
# Metrics
# ==========================

TP = 0
TN = 0
FP = 0
FN = 0

# Tìm tất cả file .wav trong samples và các thư mục con
audio_files = sorted(SAMPLES_DIR.rglob("*.wav"))

print(f"Tìm thấy {len(audio_files)} file.\n")

for audio_file in audio_files:

    print("=" * 70)
    print(f"Đang xử lý: {audio_file.relative_to(SAMPLES_DIR)}")

    result = pipeline.process(str(audio_file))

    pprint(result)
    print("Language :", result["language"])
    print("Transcript:", result["transcription"])

    print("\nSpeech Segments:")

    for seg in result["speech_segments"]:

        start = seg["start"] / 16000
        end = seg["end"] / 16000

        print(f"{start:.2f}s -> {end:.2f}s")

    print("Alert     :", result["keyword_detected"])
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

    # ==========================
    # Evaluation
    # ==========================

    prediction = result["keyword_detected"]

    folder = audio_file.parent.name.lower()

    if folder == "cheating":

        if prediction:
            TP += 1
        else:
            FN += 1

    elif folder == "normal":

        if prediction:
            FP += 1
        else:
            TN += 1

    print()

# ==========================
# Report
# ==========================

total = TP + TN + FP + FN

accuracy = (TP + TN) / total if total else 0

precision = TP / (TP + FP) if (TP + FP) else 0

recall = TP / (TP + FN) if (TP + FN) else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) else 0
)

print("\n")
print("=" * 70)
print("Evaluation Report")
print("=" * 70)

print(f"Total Files : {total}")
print()

print(f"TP : {TP}")
print(f"TN : {TN}")
print(f"FP : {FP}")
print(f"FN : {FN}")

print()

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-score : {f1 * 100:.2f}%")

print("=" * 70)

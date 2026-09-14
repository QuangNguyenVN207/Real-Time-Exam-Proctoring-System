import warnings
from transformers import logging

# Tắt các cảnh báo chung của Python (như TracerWarning)
warnings.filterwarnings("ignore")

# Chỉ hiển thị log lỗi (ERROR), ẩn các cảnh báo (WARNING) và thông tin (INFO) từ Hugging Face
logging.set_verbosity_error()
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from backend.ai_services.whisper.audio_pipeline import AudioPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"

pipeline = AudioPipeline()

y_true = []
y_pred = []

wav_files = list(SAMPLES_DIR.rglob("*.wav"))

print(f"Tìm thấy {len(wav_files)} file.\n")

for audio_file in wav_files:

    print("=" * 70)
    print(audio_file.relative_to(SAMPLES_DIR))

    result = pipeline.process(str(audio_file))

    truth = audio_file.parent.name == "cheating"
    # print("Actual pipeline result:", result)
    pred = (result["status"] == "alert")

    y_true.append(truth)
    y_pred.append(pred)

    print("Truth :", truth)
    print("Pred  :", pred)
    print("Text  :", result["transcription"])

print("\n===================== RESULT =====================\n")

print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
print(f"F1-score : {f1_score(y_true, y_pred):.4f1}")

print("\nConfusion Matrix")
print(confusion_matrix(y_true, y_pred))
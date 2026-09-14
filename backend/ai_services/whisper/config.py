import os
from pathlib import Path


SPEECH_MODEL = "phowhisper"

# PhoWhisper is used as a pretrained ASR model.  It is intentionally not
# fine-tuned by this project; Hugging Face caches the pinned model locally
# after the first download.
PHOWHISPER_MODEL = os.getenv("PHOWHISPER_MODEL", "vinai/PhoWhisper-small")
PHOWHISPER_REVISION = os.getenv(
    "PHOWHISPER_REVISION",
    "a86b604c346caf7148c37512eafe783a16420adb",
)
PHOWHISPER_DEVICE = os.getenv("PHOWHISPER_DEVICE", "auto")

# PhoBERT is the only fine-tuned component in the audio module.
WHISPER_DIR = Path(__file__).resolve().parent
PHOBERT_MODEL_DIR = Path(
    os.getenv("PHOBERT_MODEL_DIR", str(WHISPER_DIR / "phobert" / "weights"))
).resolve()
PHOBERT_CHEATING_THRESHOLD = float(os.getenv("PHOBERT_CHEATING_THRESHOLD", "0.60"))

WHISPER_MODEL = "small"
WHISPER_LANGUAGE = "vi"
WHISPER_DEVICE = "auto"
WHISPER_COMPUTE_TYPE = "int8"

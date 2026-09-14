from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from underthesea import word_tokenize

from backend.ai_services.whisper.config import PHOBERT_MODEL_DIR


class PhobertService:
    """Inference-only wrapper for the fine-tuned binary PhoBERT classifier."""

    LABELS = {0: "Normal", 1: "Cheating"}
    MAX_LEN = 128

    def __init__(self, model_path: str | Path | None = None, device: str = "auto"):
        self.model_path = Path(model_path or PHOBERT_MODEL_DIR).resolve()
        self._validate_artifact()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.device = torch.device(device)
        print(f"[PhoBERT] Loading fine-tuned model from {self.model_path}")
        print(f"[PhoBERT] Device: {self.device}")

        # A local-only load prevents an accidental fallback to the unfine-tuned
        # base classifier when the release artifact is missing.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        print("[PhoBERT] Model loaded successfully")

    def _validate_artifact(self) -> None:
        required = [
            "config.json",
            "model.safetensors",
            "tokenizer_config.json",
            "vocab.txt",
            "bpe.codes",
        ]
        missing = [name for name in required if not (self.model_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Incomplete fine-tuned PhoBERT artifact at "
                f"{self.model_path}. Missing: {', '.join(missing)}"
            )

    @staticmethod
    def preprocess(text: str) -> str:
        """Apply the same Vietnamese word segmentation used during training."""
        return word_tokenize(text.lower().strip(), format="text")

    def predict(self, text: str) -> dict:
        if not text:
            return {
                "text_processed": "",
                "label": "Normal",
                "confidence": 1.0,
                "all_probs": {"Normal": 1.0, "Cheating": 0.0},
            }

        processed_text = self.preprocess(text)
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.MAX_LEN,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        predicted_class_id = int(probs.argmax())
        all_probs = {
            self.LABELS[index]: float(probs[index])
            for index in range(len(self.LABELS))
        }
        return {
            "text_processed": processed_text,
            "label": self.LABELS[predicted_class_id],
            "confidence": float(probs[predicted_class_id]),
            "all_probs": all_probs,
        }


if __name__ == "__main__":
    service = PhobertService()
    for sample in (
        "thầy ơi cho em hỏi câu số 3",
        "ê nhắc tao câu năm m đáp án gì đấy",
        "hôm nay trời nóng quá",
    ):
        print(sample, service.predict(sample))

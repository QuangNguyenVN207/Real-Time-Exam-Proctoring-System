from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from backend.ai_services.whisper.config import (
    PHOWHISPER_DEVICE,
    PHOWHISPER_MODEL,
    PHOWHISPER_REVISION,
)


class PhoWhisperService:
    """Pretrained PhoWhisper ASR; no project-specific ASR fine-tuning is used."""

    TARGET_SR = 16000

    def __init__(
        self,
        model_name: str = PHOWHISPER_MODEL,
        revision: str = PHOWHISPER_REVISION,
        device: str = PHOWHISPER_DEVICE,
        local_files_only: bool = True,
    ):
        model_name = str(model_name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        model_path = Path(model_name).expanduser()
        is_local_model = model_path.is_dir()
        looks_like_local_path = model_path.is_absolute() or any(
            separator in model_name for separator in ("/", "\\")
        ) and not model_name.startswith("vinai/")
        if looks_like_local_path and not is_local_model:
            raise FileNotFoundError(
                "Packaged PhoWhisper model is missing: "
                f"{model_path.resolve()}. Extract the locked Whisper release ZIP "
                "before starting the audio module."
            )

        model_source = str(model_path.resolve()) if is_local_model else model_name
        self.model_name = model_source
        self.revision = revision
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        print(
            f"[PhoWhisper] Loading pretrained {model_source}@{revision} "
            f"on {self.device}"
        )
        load_kwargs = {"local_files_only": local_files_only or is_local_model}
        if not is_local_model:
            load_kwargs["revision"] = revision
        self.processor = AutoProcessor.from_pretrained(
            model_source,
            **load_kwargs,
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_source,
            dtype=self.dtype,
            **load_kwargs,
        ).to(self.device)
        self.model.eval()

        # Transformers 5.x prefers language/task over forced_decoder_ids.
        # Clear the legacy value shipped in generation_config to avoid applying
        # both mechanisms at the same time.
        self.model.generation_config.forced_decoder_ids = None
        print("[PhoWhisper] Model loaded")

    def _prepare_audio(self, audio):
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=self.TARGET_SR, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio

    def transcribe(self, audio) -> dict:
        audio = self._prepare_audio(audio)
        if audio.size == 0:
            return {"language": "vi", "text": "", "segments": []}

        inputs = self.processor(
            audio,
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
            return_attention_mask=True,
        )
        generate_kwargs = {
            "input_features": inputs["input_features"].to(
                self.device,
                dtype=self.dtype,
            ),
            "do_sample": False,
            "num_beams": 3,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "max_length": 128,
        }
        if "attention_mask" in inputs:
            generate_kwargs["attention_mask"] = inputs["attention_mask"].to(
                self.device
            )
        generate_kwargs["language"] = "vi"
        generate_kwargs["task"] = "transcribe"

        with torch.inference_mode():
            predicted_ids = self.model.generate(**generate_kwargs)

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0].strip()
        return {
            "language": "vi",
            "text": text,
            "segments": [
                {"start": 0, "end": len(audio) / self.TARGET_SR, "text": text}
            ],
        }

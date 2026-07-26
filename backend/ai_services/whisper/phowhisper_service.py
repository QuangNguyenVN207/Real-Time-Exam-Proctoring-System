import torch
import librosa

from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq
)

from whisper.config import (
    PHOWHISPER_MODEL,
)


class PhoWhisperService:

    TARGET_SR = 16000

    def __init__(self):

        print("[PhoWhisper] Loading model...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[PhoWhisper] Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            PHOWHISPER_MODEL
        )

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            PHOWHISPER_MODEL,
            torch_dtype=dtype
        ).to(self.device)

        self.model.eval()

        print("[PhoWhisper] Model loaded!")

    def transcribe(self, audio):

        if isinstance(audio, str):

            audio, _ = librosa.load(
                audio,
                sr=self.TARGET_SR
            )

        inputs = self.processor(
            audio,
            sampling_rate=self.TARGET_SR,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(
            self.device,
            dtype=self.model.dtype
        )

        predicted_ids = self.model.generate(
            input_features,
            language="vi",
            task="transcribe",
            do_sample=False,
            num_beams=5,
            temperature=0.0,
            repetition_penalty=1.05,
        )

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return {

            "language": "vi",

            "text": text.strip(),

            "segments": [
                {
                    "start": 0,
                    "end": len(audio) / self.TARGET_SR,
                    "text": text.strip()
                }
            ]
        }
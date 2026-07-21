import time
from pathlib import Path

from whisper.whisper_service import WhisperService
from whisper.vad_service import VADService
from whisper.audio_utils import load_audio, extract_speech

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"

whisper = WhisperService()
vad = VADService()

wav_files = list(SAMPLES_DIR.rglob("*.wav"))

print(f"Tìm thấy {len(wav_files)} file.\n")

for audio_file in wav_files:

    print("=" * 70)
    print(audio_file.relative_to(SAMPLES_DIR))

    # -------------------------
    # Không dùng VAD
    # -------------------------
    start = time.perf_counter()

    whisper.transcribe(str(audio_file))

    no_vad_time = time.perf_counter() - start

    # -------------------------
    # Có VAD
    # -------------------------
    audio = load_audio(audio_file)

    speech_segments = vad.detect(str(audio_file))

    speech_audio = extract_speech(
        audio,
        speech_segments
    )

    start = time.perf_counter()

    whisper.transcribe(speech_audio)

    vad_time = time.perf_counter() - start

    print(f"Không VAD : {no_vad_time:.3f}s")
    print(f"Có VAD    : {vad_time:.3f}s")

    improvement = (1 - vad_time / no_vad_time) * 100

    print(f"Tăng tốc  : {improvement:.1f}%")
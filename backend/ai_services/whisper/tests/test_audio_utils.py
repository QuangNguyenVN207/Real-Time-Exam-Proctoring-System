from pathlib import Path

from whisper.audio_utils import (
    load_audio,
    extract_speech
)

from whisper.vad_service import VADService

BASE_DIR = Path(__file__).resolve().parent.parent

audio_path = BASE_DIR / "samples" / "cheating" / "input1.wav"

audio = load_audio(audio_path)

vad = VADService()

segments = vad.detect(str(audio_path))

speech = extract_speech(
    audio,
    segments
)

SAMPLE_RATE = 16000

original_duration = len(audio) / SAMPLE_RATE
speech_duration = len(speech) / SAMPLE_RATE

print("=" * 60)
print(f"Audio gốc      : {original_duration:.2f} giây")
print(f"Audio sau VAD  : {speech_duration:.2f} giây")
print(f"Đã loại bỏ     : {original_duration - speech_duration:.2f} giây im lặng")
print(f"Tỷ lệ giữ lại  : {(speech_duration / original_duration) * 100:.1f}%")
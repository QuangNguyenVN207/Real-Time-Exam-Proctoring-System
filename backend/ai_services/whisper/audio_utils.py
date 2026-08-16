import numpy as np
import librosa
from scipy.signal import butter, lfilter

TARGET_SR = 16000

# Khởi tạo bộ lọc dải tần
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=300.0, highcut=3400.0, fs=16000, order=4):
    """Giữ lại dải tần giọng nói, triệt tiêu tiếng ù quạt và tiếng chói"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y.astype(np.float32)

def load_audio(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )
    # Gọt tạp âm ngay sau khi load file
    audio = apply_bandpass_filter(audio)
    return audio

def extract_speech(audio, timestamps):
    clips = []
    for seg in timestamps:
        start = seg["start"]
        end = seg["end"]
        clips.append(audio[start:end])

    if len(clips) == 0:
        return np.array([], dtype=np.float32)

    return np.concatenate(clips)
import numpy as np
import librosa


TARGET_SR = 16000


def load_audio(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    return audio


def extract_speech(audio, timestamps):

    clips = []

    for seg in timestamps:

        start = seg["start"]
        end = seg["end"]

        clips.append(
            audio[start:end]
        )

    if len(clips) == 0:

        return np.array([], dtype=np.float32)

    return np.concatenate(clips)
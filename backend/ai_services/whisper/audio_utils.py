import numpy as np
import librosa
from scipy.signal import butter, lfilter

TARGET_SR = 16000


def _load_audio_with_pyav(audio_path):
    """Decode usable frames while discarding corrupt packets in damaged MP3s."""
    import av

    chunks = []
    with av.open(
        str(audio_path),
        options={"err_detect": "ignore_err", "fflags": "+discardcorrupt"},
    ) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format="flt",
            layout="mono",
            rate=TARGET_SR,
        )
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                continue
            for frame in frames:
                for output_frame in resampler.resample(frame):
                    chunks.append(output_frame.to_ndarray().reshape(-1))
        for output_frame in resampler.resample(None):
            chunks.append(output_frame.to_ndarray().reshape(-1))

    if not chunks:
        raise RuntimeError(f"PyAV could not recover any audio frames from {audio_path}")
    return np.concatenate(chunks).astype(np.float32, copy=False)

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
    try:
        audio, _ = librosa.load(
            audio_path,
            sr=TARGET_SR,
            mono=True,
        )
    except Exception as primary_error:
        try:
            audio = _load_audio_with_pyav(audio_path)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Unable to decode {audio_path} with librosa or PyAV"
            ) from fallback_error
    # Filtering belongs to AudioPipeline.process_audio so file input and live
    # numpy input follow exactly the same path.  Applying it here too filtered
    # file-based tests twice and made them incomparable with realtime input.
    return np.asarray(audio, dtype=np.float32)

def extract_speech(audio, timestamps):
    clips = []
    for seg in timestamps:
        start = seg["start"]
        end = seg["end"]
        clips.append(audio[start:end])

    if len(clips) == 0:
        return np.array([], dtype=np.float32)

    return np.concatenate(clips)

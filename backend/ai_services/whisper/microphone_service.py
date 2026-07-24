import sounddevice as sd
import numpy as np
import queue

class MicrophoneService:

    def __init__(self,
                 sample_rate=16000,
                 duration=3):

        self.sample_rate = sample_rate
        self.duration = duration

    def record(self):

        print("Listening...")

        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        return audio.flatten()

    def stream(self):

        q = queue.Queue()

        def callback(indata, frames, time, status):

            if status:
                print(status)

            q.put(indata.copy())

        stream = sd.InputStream(

            samplerate=self.sample_rate,

            channels=1,

            dtype="float32",

            blocksize=8000,

            callback=callback

        )

        return stream, q
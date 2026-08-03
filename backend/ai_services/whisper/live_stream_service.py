import asyncio
import numpy as np

from whisper.microphone_service import MicrophoneService
from whisper.audio_pipeline import AudioPipeline
from whisper.websocket_client import WebSocketClient


class LiveStreamService:

    BUFFER_SECONDS = 4

    def __init__(self):

        self.microphone = MicrophoneService()

        self.pipeline = AudioPipeline()

        self.websocket = WebSocketClient()

        self.sample_rate = self.microphone.sample_rate

        self.buffer = np.array([], dtype=np.float32)

    async def start(self):

        stream, q = self.microphone.stream()

        print("=" * 60)
        print("Live Audio Service Started")
        print("=" * 60)
        print("Listening... (Ctrl+C to stop)\n")

        with stream:

            try:

                while True:

                    chunk = q.get().flatten()

                    self.buffer = np.concatenate(
                        (self.buffer, chunk)
                    )

                    if len(self.buffer) < self.sample_rate * self.BUFFER_SECONDS:
                        continue

                    result = self.pipeline.process_audio(
                        self.buffer
                    )

                    self.buffer = np.array([], dtype=np.float32)

                    if result["status"] == "idle":

                        print("[VAD] No speech")

                        continue

                    print("\n[VAD] Speech detected")

                    print("=" * 60)
                    print("Transcript :", result["transcription"])
                    print("Alert      :", result["keyword_detected"])
                    print("Score      :", result["score"])
                    print("Matched    :", result["matched"])

                    await self.websocket.send(result)

            except KeyboardInterrupt:

                print("\nStopped.")

                await self.websocket.close()


if __name__ == "__main__":

    asyncio.run(
        LiveStreamService().start()
    )
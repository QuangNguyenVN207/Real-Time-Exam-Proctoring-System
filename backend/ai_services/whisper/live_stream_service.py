import asyncio
import numpy as np

from whisper.microphone_service import MicrophoneService
from whisper.vad_service import VADService
from whisper.audio_pipeline import AudioPipeline
from whisper.websocket_client import WebSocketClient


class LiveStreamService:

    # khoảng im lặng để kết thúc 1 câu
    SILENCE_CHUNKS = 3

    # câu tối thiểu
    MIN_SECONDS = 1

    def __init__(self):

        self.microphone = MicrophoneService()

        self.vad = VADService()

        self.pipeline = AudioPipeline()

        self.websocket = WebSocketClient()

        self.sample_rate = self.microphone.sample_rate

        self.utterance = np.array([], dtype=np.float32)

        self.speaking = False

        self.silence_counter = 0


    def reset(self):

        self.utterance = np.array([], dtype=np.float32)

        self.speaking = False

        self.silence_counter = 0


    async def process_current_utterance(self):

        if len(self.utterance) < self.sample_rate * self.MIN_SECONDS:

            self.reset()

            return

        result = self.pipeline.process_audio(
            self.utterance
        )

        self.reset()

        if result["status"] == "idle":

            print("[VAD] No speech")

            return

        print("\n" + "=" * 70)

        print("[Whisper]")

        print(result["transcription"])

        print()

        print("Risk       :", result["risk"])

        print("Confidence :", result["confidence"])

        print("Keyword    :", result["keyword_score"])

        print("Rule Bonus :", result["rule_bonus"])

        print("Context    :", result["context_bonus"])

        print("Penalty    :", result["penalty"])

        print("Matched    :", result["matched"])

        print("Rules      :", result["matched_rules"])

        try:

            await self.websocket.send(result)

        except Exception as e:

            print("[WebSocket]", e)


    async def start(self):

        stream, q = self.microphone.stream()

        print("=" * 60)

        print("Live Audio Service Started")

        print("=" * 60)

        print("Listening... Ctrl+C to stop\n")

        try:

            await self.websocket.connect()

        except Exception as e:

            print("[WebSocket]", e)

        with stream:

            try:

                while True:

                    chunk = q.get().flatten()

                    if len(chunk) == 0:

                        continue

                    speech = self.vad.detect_array(chunk)

                    has_speech = len(speech) > 0

                    # -------------------------
                    # Có tiếng nói
                    # -------------------------

                    if has_speech:

                        self.utterance = np.concatenate(
                            (
                                self.utterance,
                                chunk
                            )
                        )

                        self.speaking = True

                        self.silence_counter = 0

                        continue

                    # -------------------------
                    # Chưa bắt đầu nói
                    # -------------------------

                    if not self.speaking:

                        continue

                    # -------------------------
                    # Đang nói nhưng gặp im lặng
                    # -------------------------

                    self.silence_counter += 1

                    if self.silence_counter < self.SILENCE_CHUNKS:

                        continue

                    await self.process_current_utterance()

            except KeyboardInterrupt:

                print("\nStopped.")

            finally:

                if self.speaking:

                    await self.process_current_utterance()

                await self.websocket.close()


if __name__ == "__main__":

    asyncio.run(
        LiveStreamService().start()
    )
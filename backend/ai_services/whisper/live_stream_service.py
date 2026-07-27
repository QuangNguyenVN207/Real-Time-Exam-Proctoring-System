import asyncio
import time

import numpy as np

from whisper.microphone_service import MicrophoneService
from whisper.vad_service import VADService
from whisper.audio_pipeline import AudioPipeline
from whisper.websocket_client import WebSocketClient


class LiveStreamService:
    # khoảng im lặng để kết thúc 1 câu
    SILENCE_CHUNKS = 3

    # câu tối thiểu (giây)
    MIN_SECONDS = 1.0

    # tránh giữ utterance quá lâu
    MAX_SECONDS = 8.0

    def __init__(self):

        self.microphone = MicrophoneService()
        self.vad = VADService()
        self.pipeline = AudioPipeline()
        self.websocket = WebSocketClient()

        self.sample_rate = self.microphone.sample_rate

        self.utterance = np.array([], dtype=np.float32)
        self.speaking = False
        self.silence_counter = 0

        self.total_sent = 0
        self.total_alert = 0

    def reset(self):

        self.utterance = np.array([], dtype=np.float32)
        self.speaking = False
        self.silence_counter = 0

    async def process_current_utterance(self):

        if len(self.utterance) < int(self.sample_rate * self.MIN_SECONDS):
            self.reset()
            return

        start_time = time.perf_counter()

        result = self.pipeline.process_audio(self.utterance)

        elapsed = time.perf_counter() - start_time

        self.reset()

        if result is None or result.get("status") == "idle":
            print("[VAD] No speech")
            return

        self.total_sent += 1
        if result.get("keyword_detected"):
            self.total_alert += 1

        print("\n" + "=" * 70)
        print("[PhoWhisper]")
        print(f"Module     : {result.get('module', '')}")
        print(f"Status     : {result.get('status', '')}")
        print(f"Language   : {result.get('language', '')}")
        print(f"Transcript : {result.get('transcription', '')}")
        print(f"Risk       : {result.get('risk', '')}")
        print(f"Confidence : {result.get('confidence', 0)}")
        print(f"Keyword    : {result.get('keyword_score', 0)}")
        print(f"Rule Bonus : {result.get('rule_bonus', 0)}")
        print(f"Context    : {result.get('context_bonus', 0)}")
        print(f"Penalty    : {result.get('penalty', 0)}")
        print(f"Matched    : {result.get('matched', [])}")
        print(f"Rules      : {result.get('matched_rules', [])}")
        print(f"Time       : {elapsed:.3f}s")
        print(f"Alerts     : {self.total_alert}/{self.total_sent}")

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
                    chunk = q.get().flatten().astype(np.float32)

                    if len(chunk) == 0:
                        continue

                    speech = self.vad.detect_array(chunk)
                    has_speech = len(speech) > 0

                    # -------------------------
                    # Có tiếng nói
                    # -------------------------
                    if has_speech:
                        self.utterance = np.concatenate(
                            (self.utterance, chunk)
                        )

                        self.speaking = True
                        self.silence_counter = 0

                        # Nếu một câu nói quá dài, chốt luôn
                        if len(self.utterance) >= int(self.sample_rate * self.MAX_SECONDS):
                            await self.process_current_utterance()

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
                if self.speaking and len(self.utterance) > 0:
                    await self.process_current_utterance()

                await self.websocket.close()


if __name__ == "__main__":
    asyncio.run(LiveStreamService().start())
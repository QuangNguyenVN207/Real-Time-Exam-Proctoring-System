import json
import websockets


class WebSocketClient:

    def __init__(self, uri="ws://localhost:8765"):

        self.uri = uri
        self.websocket = None

    async def connect(self):

        if self.websocket is None:

            self.websocket = await websockets.connect(self.uri)

            print("[WebSocket] Connected")

    async def send(self, data):

        await self.connect()

        print("[WebSocket] Sending...")

        await self.websocket.send(
            json.dumps(data, ensure_ascii=False)
        )

    async def close(self):

        if self.websocket:

            await self.websocket.close()

            self.websocket = None
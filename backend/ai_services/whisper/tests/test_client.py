import asyncio
import json
import websockets


async def main():

    async with websockets.connect(
        "ws://localhost:8765"
    ) as websocket:

        print("Connected.\n")

        while True:

            message = await websocket.recv()

            data = json.loads(message)

            print("=" * 50)
            print(data)


if __name__ == "__main__":

    asyncio.run(main())
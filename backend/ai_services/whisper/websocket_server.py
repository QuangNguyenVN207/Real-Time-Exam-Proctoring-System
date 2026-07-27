import asyncio
import websockets

clients = set()


async def handler(websocket):

    print("[Server] Client connected")

    clients.add(websocket)

    try:

        async for message in websocket:

            dead = []

            for client in clients:

                if client != websocket:

                    try:
                        await client.send(message)
                    except:
                        dead.append(client)

            for client in dead:
                clients.remove(client)

    except websockets.ConnectionClosed:
        pass

    finally:

        if websocket in clients:
            clients.remove(websocket)

        print("[Server] Client disconnected")


async def main():

    async with websockets.serve(handler, "localhost", 8765):

        print("WebSocket Server: ws://localhost:8765")

        await asyncio.Future()


if __name__ == "__main__":

    asyncio.run(main())
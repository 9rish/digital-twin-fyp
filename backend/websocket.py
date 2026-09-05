from fastapi import WebSocket
from typing import List

class ConnectionManager:
    def __init__(self):
        # Store all active websocket connections
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        """Sends a JSON message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # If a connection drops unexpectedly, ignore it.
                # Usually we'd remove it, but it's safer to handle in disconnect().
                pass

manager = ConnectionManager()

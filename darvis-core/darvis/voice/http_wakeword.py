import asyncio
import json
import os
from asyncio import Queue
from dataclasses import dataclass
from typing import Optional

from darvis.core.events import EventType


@dataclass
class HttpWakeWordConfig:
    host: str = "127.0.0.1"
    port: int = 8002
    reconnect_delay: float = 2.0
    ping_interval: float = 25.0

    @classmethod
    def from_env(cls) -> "HttpWakeWordConfig":
        host = os.environ.get("WAKEWORD_HOST", "127.0.0.1")
        port = int(os.environ.get("WAKEWORD_PORT", "8002"))
        return cls(host=host, port=port)

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class HttpWakeWordDetector:
    def __init__(
        self,
        event_queue: Queue[EventType],
        config: Optional[HttpWakeWordConfig] = None
    ):
        self._queue = event_queue
        self._config = config or HttpWakeWordConfig.from_env()
        self._running = False
        self._connected = False
        self._websocket = None
        self._task: Optional[asyncio.Task] = None
        self._http_client = None

    async def start(self) -> bool:
        import httpx

        self._http_client = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=5.0
        )

        try:
            response = await self._http_client.get("/health")
            if response.status_code != 200:
                print(f"[WAKEWORD] Service unhealthy: {response.status_code}")
                return False
            self._connected = True
            print(f"[WAKEWORD] Connected to service at {self._config.base_url}")
            return True
        except Exception as e:
            print(f"[WAKEWORD] Cannot connect to service: {e}")
            return False

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        if self._http_client:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._http_client.aclose())
            except RuntimeError:
                pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def run(self) -> None:
        import websockets

        self._running = True

        while self._running:
            try:
                async with websockets.connect(self._config.ws_url) as websocket:
                    self._websocket = websocket
                    self._connected = True
                    print("[WAKEWORD] WebSocket connected")

                    while self._running:
                        try:
                            message = await asyncio.wait_for(
                                websocket.recv(),
                                timeout=self._config.ping_interval
                            )
                            data = json.loads(message)
                            event_type = data.get("type")

                            if event_type == "wake_word":
                                word = data.get("word", "unknown")
                                score = data.get("score", 0.0)
                                print(f"[WAKEWORD] Wake detected: {word} (score: {score:.3f})")
                                await self._queue.put(EventType.WAKE_WORD)

                            elif event_type == "cancel_word":
                                word = data.get("word", "unknown")
                                print(f"[WAKEWORD] Cancel detected: {word}")
                                await self._queue.put(EventType.CANCEL)

                            elif event_type == "quit_word":
                                word = data.get("word", "unknown")
                                print(f"[WAKEWORD] Quit detected: {word}")
                                await self._queue.put(EventType.QUIT)

                            elif event_type == "ping":
                                await websocket.send(json.dumps({"type": "pong"}))

                        except asyncio.TimeoutError:
                            await websocket.send(json.dumps({"type": "ping"}))

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                if self._running:
                    print(f"[WAKEWORD] Connection lost: {e}")
                    print(f"[WAKEWORD] Reconnecting in {self._config.reconnect_delay}s...")
                    await asyncio.sleep(self._config.reconnect_delay)

        self._websocket = None
        self._connected = False

    def get_task(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run())
        return self._task

    async def pause(self) -> bool:
        if not self._http_client:
            return False

        try:
            response = await self._http_client.post("/pause")
            return response.status_code == 200
        except Exception as e:
            print(f"[WAKEWORD] Failed to pause: {e}")
            return False

    async def resume(self) -> bool:
        if not self._http_client:
            return False

        try:
            response = await self._http_client.post("/resume")
            return response.status_code == 200
        except Exception as e:
            print(f"[WAKEWORD] Failed to resume: {e}")
            return False

    async def activate_cancel(self) -> bool:
        if not self._http_client:
            return False

        try:
            response = await self._http_client.post("/cancel/activate")
            return response.status_code == 200
        except Exception as e:
            print(f"[WAKEWORD] Failed to activate cancel: {e}")
            return False

    async def deactivate_cancel(self) -> bool:
        if not self._http_client:
            return False

        try:
            response = await self._http_client.post("/cancel/deactivate")
            return response.status_code == 200
        except Exception as e:
            print(f"[WAKEWORD] Failed to deactivate cancel: {e}")
            return False

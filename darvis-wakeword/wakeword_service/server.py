import asyncio
import json
from contextlib import asynccontextmanager
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from wakeword_service.config import WakeWordConfig
from wakeword_service.detector import WakeWordDetector, DetectionType


detector: WakeWordDetector = None  # type: ignore
connected_clients: Set[WebSocket] = set()
detection_queue: asyncio.Queue = None  # type: ignore
main_loop: asyncio.AbstractEventLoop = None  # type: ignore


def on_detection(detection_type: DetectionType, word: str, score: float) -> None:
    event = {
        "type": detection_type.value,
        "word": word,
        "score": score
    }

    if main_loop and detection_queue:
        main_loop.call_soon_threadsafe(detection_queue.put_nowait, event)


async def broadcast_detections():
    while True:
        event = await detection_queue.get()

        disconnected = set()
        for client in connected_clients:
            try:
                await client.send_json(event)
            except Exception:
                disconnected.add(client)

        connected_clients.difference_update(disconnected)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, detection_queue, main_loop

    main_loop = asyncio.get_running_loop()
    detection_queue = asyncio.Queue()

    config = WakeWordConfig.from_env()
    detector = WakeWordDetector(config, on_detection=on_detection)

    print("[WAKEWORD] Starting wake word service...")
    detector.load()
    detector.start()

    broadcast_task = asyncio.create_task(broadcast_detections())

    yield

    print("[WAKEWORD] Shutting down...")
    broadcast_task.cancel()
    detector.stop()
    detector.unload()


app = FastAPI(title="DARVIS Wake Word Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "wakeword"}


@app.get("/status")
async def status():
    return {
        "paused": detector.is_paused,
        "model_loaded": detector.is_loaded,
        "wake_words": detector.wake_words,
        "cancel_words": detector.cancel_words,
        "quit_words": detector.quit_words,
        "connected_clients": len(connected_clients)
    }


@app.post("/pause")
async def pause():
    detector.pause()
    return {"status": "paused"}


@app.post("/resume")
async def resume():
    detector.resume()
    return {"status": "resumed"}


@app.post("/cancel/activate")
async def activate_cancel():
    detector.activate_cancel()
    return {"status": "cancel_activated"}


@app.post("/cancel/deactivate")
async def deactivate_cancel():
    detector.deactivate_cancel()
    return {"status": "cancel_deactivated"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"[WAKEWORD] Client connected ({len(connected_clients)} total)")

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WAKEWORD] WebSocket error: {e}")
    finally:
        connected_clients.discard(websocket)
        print(f"[WAKEWORD] Client disconnected ({len(connected_clients)} total)")

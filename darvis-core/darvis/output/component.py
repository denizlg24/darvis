import asyncio
import os
import threading
from asyncio import Queue
from dataclasses import dataclass
from typing import Optional

import httpx

from darvis.core.events import EventType
from darvis.core.fsm import TransitionResult
from darvis.core.handlers import DaemonContext, HandlerRegistry
from darvis.core.states import State
from darvis.core.task_registry import ResourceName, TaskName
from darvis.utils.audio_playback import play_wav_bytes, stop_playback


@dataclass
class TTSConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8003
    voice: str = "am_michael"
    speed: float = 1.0
    timeout: float = 30.0
    audio_queue_size: int = 3

    @classmethod
    def from_env(cls) -> "TTSConfig":
        return cls(
            enabled=os.environ.get("DARVIS_TTS_ENABLED", "true").lower() == "true",
            host=os.environ.get("TTS_HOST", "127.0.0.1"),
            port=int(os.environ.get("TTS_PORT", "8003")),
            voice=os.environ.get("TTS_VOICE", "am_michael"),
            speed=float(os.environ.get("TTS_SPEED", "1.0")),
            timeout=float(os.environ.get("TTS_TIMEOUT", "30.0")),
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class AudioChunk:
    sentence_num: int
    text: str
    audio_bytes: bytes


class TTSComponent:
    def __init__(
        self,
        event_queue: Queue[EventType],
        config: Optional[TTSConfig] = None
    ):
        self._queue = event_queue
        self._config = config or TTSConfig.from_env()
        self._client: Optional[httpx.AsyncClient] = None
        self._current_task: Optional[asyncio.Task] = None
        self._synthesis_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._available = False
        self._cancelled = threading.Event()
        self._audio_queue: Optional[asyncio.Queue[Optional[AudioChunk]]] = None

    async def start(self) -> None:
        if not self._config.enabled:
            print("[TTS] Disabled via config")
            return

        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=self._config.timeout
        )

        try:
            response = await self._client.get("/health")
            if response.status_code == 200:
                self._available = True
                print(f"[TTS] Service connected ({self._config.base_url})")
            else:
                print(f"[TTS] Service unhealthy: {response.status_code}")
        except httpx.ConnectError:
            print(f"[TTS] Service unavailable at {self._config.base_url}")
        except Exception as e:
            print(f"[TTS] Service check failed: {e}")

    def stop(self) -> None:
        self._cancel_all()
        stop_playback()
        self._client = None
        self._available = False

    def register_handlers(self, registry: HandlerRegistry) -> None:
        registry.on_enter(State.TTS, self._on_enter_tts)
        registry.on_enter(State.CANCEL_REQUESTED, self._on_cancel)

    def _cancel_all(self) -> None:
        self._cancelled.set()
        for task in [self._current_task, self._synthesis_task, self._playback_task]:
            if task and not task.done():
                task.cancel()
        self._current_task = None
        self._synthesis_task = None
        self._playback_task = None

    async def _on_enter_tts(
        self,
        result: TransitionResult,
        context: DaemonContext
    ) -> None:
        self._cancelled.clear()

        streaming_active = context.get_resource(ResourceName.STREAMING_ACTIVE)
        sentence_queue = context.get_resource(ResourceName.SENTENCE_QUEUE)

        if streaming_active and sentence_queue:
            self._current_task = asyncio.create_task(
                self._run_streaming_tts(sentence_queue, context)
            )
        else:
            chat_response = context.get_resource(ResourceName.CHAT_RESPONSE)

            if not chat_response or not chat_response.get("text"):
                print("[TTS] No chat response available")
                await self._queue.put(EventType.TTS_DONE)
                return

            text = chat_response["text"]
            print(f"[TTS] Speaking: {text}")

            if not self._available or not self._client:
                print(f"\n>>> {text}\n")
                await self._emit_done_or_continue(context)
                return

            self._current_task = asyncio.create_task(
                self._run_tts(text, context)
            )

        context.active_tasks[TaskName.TTS] = self._current_task

    async def _run_streaming_tts(
        self,
        sentence_queue: asyncio.Queue,
        context: DaemonContext
    ) -> None:
        self._audio_queue = asyncio.Queue(maxsize=self._config.audio_queue_size)

        try:
            print("[TTS] Starting streaming pipeline...")

            self._synthesis_task = asyncio.create_task(
                self._synthesis_worker(sentence_queue)
            )

            self._playback_task = asyncio.create_task(
                self._playback_worker()
            )

            await asyncio.gather(
                self._synthesis_task,
                self._playback_task,
                return_exceptions=True
            )

            print("[TTS] Streaming pipeline complete")
            await self._emit_done_or_continue(context)

        except asyncio.CancelledError:
            print("[TTS] Streaming pipeline cancelled")
            stop_playback()
            raise
        except Exception as e:
            print(f"[TTS] Pipeline error: {e}")
            await self._emit_done_or_continue(context)

    async def _synthesis_worker(self, sentence_queue: asyncio.Queue) -> None:
        sentence_num = 0

        try:
            while True:
                if self._cancelled.is_set():
                    break

                try:
                    sentence = await asyncio.wait_for(
                        sentence_queue.get(),
                        timeout=120.0
                    )
                except asyncio.TimeoutError:
                    print("[TTS] Synthesis worker: timeout waiting for sentence")
                    break

                if sentence is None:
                    print("[TTS] Synthesis worker: end of sentences")
                    break

                sentence_num += 1
                print(f"[TTS] Synthesizing sentence {sentence_num}: {sentence[:50]}...")

                if not self._available or not self._client:
                    print(f"\n>>> {sentence}\n")
                    continue

                audio_bytes = await self._synthesize_sentence(sentence)

                if audio_bytes and not self._cancelled.is_set():
                    chunk = AudioChunk(
                        sentence_num=sentence_num,
                        text=sentence,
                        audio_bytes=audio_bytes
                    )
                    await self._audio_queue.put(chunk)
                    print(f"[TTS] Queued audio {sentence_num} for playback")

        except asyncio.CancelledError:
            print("[TTS] Synthesis worker cancelled")
            raise
        finally:
            if self._audio_queue:
                await self._audio_queue.put(None)

    async def _playback_worker(self) -> None:
        try:
            while True:
                if self._cancelled.is_set():
                    break

                try:
                    chunk = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=60.0
                    )
                except asyncio.TimeoutError:
                    print("[TTS] Playback worker: timeout waiting for audio")
                    break

                if chunk is None:
                    print("[TTS] Playback worker: end of audio")
                    break

                print(f"[TTS] Playing sentence {chunk.sentence_num}")

                if not self._cancelled.is_set():
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        play_wav_bytes,
                        chunk.audio_bytes,
                        True
                    )

        except asyncio.CancelledError:
            print("[TTS] Playback worker cancelled")
            stop_playback()
            raise

    async def _synthesize_sentence(self, text: str) -> Optional[bytes]:
        if not self._client:
            return None

        try:
            response = await self._client.post(
                "/synthesize",
                json={
                    "text": text,
                    "voice": self._config.voice,
                    "speed": self._config.speed
                }
            )

            if response.status_code != 200:
                print(f"[TTS] Synthesis failed: {response.status_code}")
                return None

            return response.content

        except httpx.TimeoutException:
            print("[TTS] Synthesis timeout")
            return None
        except httpx.ConnectError:
            print("[TTS] Service connection lost")
            self._available = False
            return None
        except Exception as e:
            print(f"[TTS] Synthesis error: {e}")
            return None

    async def _run_tts(self, text: str, context: DaemonContext) -> None:
        try:
            response = await self._client.post(
                "/synthesize",
                json={
                    "text": text,
                    "voice": self._config.voice,
                    "speed": self._config.speed
                }
            )

            if response.status_code != 200:
                print(f"[TTS] Synthesis failed: {response.status_code}")
                await self._emit_done_or_continue(context)
                return

            audio_bytes = response.content

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                play_wav_bytes,
                audio_bytes,
                True
            )

            await self._emit_done_or_continue(context)

        except asyncio.CancelledError:
            print("[TTS] Cancelled")
            stop_playback()
            raise
        except httpx.TimeoutException:
            print("[TTS] Timeout waiting for synthesis")
            await self._emit_done_or_continue(context)
        except httpx.ConnectError:
            print("[TTS] Service connection lost")
            self._available = False
            await self._emit_done_or_continue(context)
        except Exception as e:
            print(f"[TTS] Error: {e}")
            await self._emit_done_or_continue(context)

    async def _emit_done_or_continue(self, context: DaemonContext) -> None:
        session_active = context.get_resource(ResourceName.SESSION_ACTIVE)
        exit_requested = context.get_resource(ResourceName.EXIT_REQUESTED)

        if session_active and not exit_requested:
            print("[TTS] Session active, continuing to listen...")
            await self._queue.put(EventType.WAKE_WORD)
        else:
            await self._queue.put(EventType.TTS_DONE)

    async def _on_cancel(
        self,
        result: TransitionResult,
        context: DaemonContext
    ) -> None:
        self._cancel_all()
        stop_playback()

        sentence_queue = context.get_resource(ResourceName.SENTENCE_QUEUE)
        if sentence_queue:
            while not sentence_queue.empty():
                try:
                    sentence_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        if self._audio_queue:
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

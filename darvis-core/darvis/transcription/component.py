import asyncio
from asyncio import Queue
from typing import Optional

from darvis.core.events import EventType
from darvis.core.fsm import TransitionResult
from darvis.core.handlers import DaemonContext, HandlerRegistry
from darvis.core.states import State
from darvis.core.task_registry import ResourceName, TaskName
from darvis.transcription.base import (
    Transcriber,
    TranscriptionResult,
    TranscriptionStatus,
)


class TranscriptionComponent:
    def __init__(
        self,
        event_queue: Queue[EventType],
        transcriber: Transcriber
    ):
        self._queue = event_queue
        self._transcriber = transcriber
        self._current_task: Optional[asyncio.Task] = None
        self._enabled = True

    async def start(self) -> None:
        if self._enabled:
            print(f"[TRANSCRIPTION] Loading {self._transcriber.name} model...")
            await self._transcriber.load()
            print(f"[TRANSCRIPTION] Model loaded")

    def stop(self) -> None:
        self._cancel_current()
        self._transcriber.unload()

    def register_handlers(self, registry: HandlerRegistry) -> None:
        registry.on_enter(State.TRANSCRIBING, self._on_enter_transcribing)
        registry.on_enter(State.CANCEL_REQUESTED, self._on_cancel)
        registry.on_enter(State.IDLE, self._on_enter_idle)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False
        self._cancel_current()

    def _cancel_current(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            self._current_task = None

    async def _on_enter_transcribing(
        self,
        result: TransitionResult,
        context: DaemonContext
    ) -> None:
        if not self._enabled or not self._transcriber.is_loaded:
            print("[TRANSCRIPTION] Disabled or not loaded, skipping")
            await self._queue.put(EventType.TRANSCRIPTION_READY)
            return

        audio_data = context.get_resource(ResourceName.AUDIO_BUFFER)
        if not audio_data:
            print("[TRANSCRIPTION] No audio data available")
            await self._queue.put(EventType.TRANSCRIPTION_READY)
            return

        self._current_task = asyncio.create_task(
            self._run_transcription(audio_data, context)
        )
        context.active_tasks[TaskName.TRANSCRIPTION] = self._current_task

    async def _run_transcription(
        self,
        audio_data: dict,
        context: DaemonContext
    ) -> None:
        try:
            audio = audio_data["data"]
            sample_rate = audio_data["sample_rate"]

            print(f"[TRANSCRIPTION] Processing {len(audio)/sample_rate:.2f}s of audio...")

            result = await self._transcriber.transcribe(audio, sample_rate)

            self._handle_result(result, context)

            if result.status != TranscriptionStatus.CANCELLED:
                await self._queue.put(EventType.TRANSCRIPTION_READY)

        except asyncio.CancelledError:
            print("[TRANSCRIPTION] Cancelled")
            raise

    def _handle_result(
        self,
        result: TranscriptionResult,
        context: DaemonContext
    ) -> None:
        match result.status:
            case TranscriptionStatus.SUCCESS:
                print(f"[TRANSCRIPTION] Result: {result.text}")
                context.set_resource(ResourceName.TRANSCRIPTION_RESULT, {
                    "text": result.text,
                    "confidence": result.confidence,
                    "duration": result.duration_seconds
                })
            case TranscriptionStatus.EMPTY:
                print("[TRANSCRIPTION] No speech detected")
            case TranscriptionStatus.CANCELLED:
                print("[TRANSCRIPTION] Cancelled")
            case TranscriptionStatus.ERROR:
                print(f"[TRANSCRIPTION] Error: {result.error}")

    async def _on_cancel(
        self,
        result: TransitionResult,
        context: DaemonContext
    ) -> None:
        self._cancel_current()

    async def _on_enter_idle(
        self,
        result: TransitionResult,
        context: DaemonContext
    ) -> None:
        context.clear_resource(ResourceName.TRANSCRIPTION_RESULT)

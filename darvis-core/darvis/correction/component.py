import asyncio
from asyncio import Queue
from typing import Optional

from darvis.core.events import EventType
from darvis.core.fsm import TransitionResult
from darvis.core.handlers import DaemonContext, HandlerRegistry
from darvis.core.states import State
from darvis.core.task_registry import ResourceName, TaskName
from darvis.correction.base import Corrector, CorrectionStatus


class CorrectionComponent:
    def __init__(
        self,
        event_queue: Queue[EventType],
        corrector: Corrector
    ):
        self._queue = event_queue
        self._corrector = corrector
        self._current_task: Optional[asyncio.Task] = None
        self._enabled = True

    async def start(self) -> None:
        if self._enabled:
            print(f"[CORRECTION] Loading {self._corrector.name} model...")
            await self._corrector.load()
            print(f"[CORRECTION] Model loaded")

    def stop(self) -> None:
        self._cancel_current()
        self._corrector.unload()

    def register_handlers(self, registry: HandlerRegistry) -> None:
        registry.on_enter(State.CORRECTING, self._on_enter_correcting)
        registry.on_enter(State.CANCEL_REQUESTED, self._on_cancel)
        registry.on_enter(State.IDLE, self._on_enter_idle)

    def get_corrector(self) -> Corrector:
        return self._corrector

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False
        self._cancel_current()

    def _cancel_current(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            self._current_task = None

    async def _on_enter_correcting(
        self,
        result: TransitionResult,
        context: DaemonContext
    ) -> None:
        if not self._enabled or not self._corrector.is_loaded:
            print("[CORRECTION] Disabled or not loaded, skipping")
            await self._queue.put(EventType.CORRECTION_READY)
            return

        transcription = context.get_resource(ResourceName.TRANSCRIPTION_RESULT)
        if not transcription or not transcription.get("text"):
            print("[CORRECTION] No transcription available")
            await self._queue.put(EventType.CORRECTION_READY)
            return

        self._current_task = asyncio.create_task(
            self._run_correction(transcription["text"], context)
        )
        context.active_tasks[TaskName.CORRECTION] = self._current_task

    async def _run_correction(
        self,
        text: str,
        context: DaemonContext
    ) -> None:
        try:
            print(f"[CORRECTION] Processing: {text}")

            result = await self._corrector.correct(text)

            self._handle_result(result, context)

            if result.status != CorrectionStatus.CANCELLED:
                await self._queue.put(EventType.CORRECTION_READY)

        except asyncio.CancelledError:
            print("[CORRECTION] Cancelled")
            raise

    def _handle_result(
        self,
        result,
        context: DaemonContext
    ) -> None:
        match result.status:
            case CorrectionStatus.SUCCESS:
                if result.text != result.original_text:
                    print(f"[CORRECTION] Corrected: {result.text}")
                else:
                    print(f"[CORRECTION] No changes needed")
                context.set_resource(ResourceName.CORRECTED_TEXT, {
                    "text": result.text,
                    "original": result.original_text
                })
            case CorrectionStatus.TIMEOUT:
                print(f"[CORRECTION] Timeout, using original")
                context.set_resource(ResourceName.CORRECTED_TEXT, {
                    "text": result.text,
                    "original": result.original_text
                })
            case CorrectionStatus.EMPTY:
                print("[CORRECTION] No text to correct")
            case CorrectionStatus.CANCELLED:
                print("[CORRECTION] Cancelled")
            case CorrectionStatus.ERROR:
                print(f"[CORRECTION] Error: {result.error}")
                if result.original_text:
                    context.set_resource(ResourceName.CORRECTED_TEXT, {
                        "text": result.original_text,
                        "original": result.original_text
                    })

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
        context.clear_resource(ResourceName.CORRECTED_TEXT)

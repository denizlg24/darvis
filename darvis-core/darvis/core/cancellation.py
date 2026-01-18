import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

from darvis.core.states import State
from darvis.core.task_registry import ResourceName, TaskName


class CancelResult(Enum):
    SUCCESS = auto()
    PARTIAL = auto()
    FAILED = auto()


@dataclass
class CancellationContext:
    from_state: State
    active_tasks: dict[TaskName, asyncio.Task]
    resources: dict[ResourceName, Any]


@dataclass
class CancellationOutcome:
    result: CancelResult
    from_state: State
    message: str
    cleaned_tasks: list[TaskName]
    cleaned_resources: list[ResourceName]


CancelHandler = Callable[[CancellationContext], Awaitable[CancellationOutcome]]


class CancellationProcessor:
    def __init__(self):
        self._handlers: dict[State, CancelHandler] = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        self.register_handler(State.LISTENING, self._cancel_listening)
        self.register_handler(State.VOICE_CAPTURE, self._cancel_voice_capture)
        self.register_handler(State.TRANSCRIBING, self._cancel_transcribing)
        self.register_handler(State.TTS, self._cancel_tts)

    def register_handler(self, state: State, handler: CancelHandler) -> None:
        self._handlers[state] = handler

    async def process(self, context: CancellationContext) -> CancellationOutcome:
        handler = self._handlers.get(context.from_state)

        if handler is None:
            return CancellationOutcome(
                result=CancelResult.SUCCESS,
                from_state=context.from_state,
                message=f"No cleanup needed for {context.from_state.name}",
                cleaned_tasks=[],
                cleaned_resources=[]
            )

        try:
            return await handler(context)
        except Exception as e:
            return CancellationOutcome(
                result=CancelResult.FAILED,
                from_state=context.from_state,
                message=f"Cancellation failed: {e}",
                cleaned_tasks=[],
                cleaned_resources=[]
            )

    async def _cancel_listening(self, ctx: CancellationContext) -> CancellationOutcome:
        cleaned_tasks = []

        if TaskName.LISTEN in ctx.active_tasks:
            ctx.active_tasks[TaskName.LISTEN].cancel()
            cleaned_tasks.append(TaskName.LISTEN)

        return CancellationOutcome(
            result=CancelResult.SUCCESS,
            from_state=ctx.from_state,
            message="Listening cancelled",
            cleaned_tasks=cleaned_tasks,
            cleaned_resources=[]
        )

    async def _cancel_voice_capture(self, ctx: CancellationContext) -> CancellationOutcome:
        cleaned_tasks = []
        cleaned_resources = []

        if TaskName.CAPTURE in ctx.active_tasks:
            ctx.active_tasks[TaskName.CAPTURE].cancel()
            cleaned_tasks.append(TaskName.CAPTURE)

        if ResourceName.AUDIO_BUFFER in ctx.resources:
            ctx.resources[ResourceName.AUDIO_BUFFER] = None
            cleaned_resources.append(ResourceName.AUDIO_BUFFER)

        return CancellationOutcome(
            result=CancelResult.SUCCESS,
            from_state=ctx.from_state,
            message="Voice capture cancelled, audio discarded",
            cleaned_tasks=cleaned_tasks,
            cleaned_resources=cleaned_resources
        )

    async def _cancel_transcribing(self, ctx: CancellationContext) -> CancellationOutcome:
        cleaned_tasks = []

        if TaskName.TRANSCRIPTION in ctx.active_tasks:
            ctx.active_tasks[TaskName.TRANSCRIPTION].cancel()
            cleaned_tasks.append(TaskName.TRANSCRIPTION)

        return CancellationOutcome(
            result=CancelResult.SUCCESS,
            from_state=ctx.from_state,
            message="Transcription cancelled",
            cleaned_tasks=cleaned_tasks,
            cleaned_resources=[]
        )

    async def _cancel_tts(self, ctx: CancellationContext) -> CancellationOutcome:
        cleaned_tasks = []

        if TaskName.TTS in ctx.active_tasks:
            ctx.active_tasks[TaskName.TTS].cancel()
            cleaned_tasks.append(TaskName.TTS)

        return CancellationOutcome(
            result=CancelResult.SUCCESS,
            from_state=ctx.from_state,
            message="TTS cancelled",
            cleaned_tasks=cleaned_tasks,
            cleaned_resources=[]
        )


_processor: Optional[CancellationProcessor] = None


def get_cancellation_processor() -> CancellationProcessor:
    global _processor
    if _processor is None:
        _processor = CancellationProcessor()
    return _processor

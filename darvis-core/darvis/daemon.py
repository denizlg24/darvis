import asyncio
import os
import signal
from asyncio import Queue
from pathlib import Path
from typing import Optional

from darvis.core.cancellation import (
    CancellationContext,
    CancelResult,
    get_cancellation_processor,
)
from darvis.core.events import EventType
from darvis.core.fsm import StateMachine, TransitionResult
from darvis.core.handlers import DaemonContext, HandlerRegistry
from darvis.core.states import State
from darvis.core.task_registry import ResourceName, TaskName
from darvis.chat.component import ChatComponent
from darvis.correction.base import Corrector, StubCorrector
from darvis.correction.component import CorrectionComponent
from darvis.correction.llm_corrector import LLMCorrector, LLMCorrectorConfig
from darvis.input.component import InputComponent
from darvis.output.component import TTSComponent
from darvis.transcription.base import StubTranscriber, Transcriber
from darvis.transcription.component import TranscriptionComponent
from darvis.transcription.http_transcriber import HttpTranscriber, HttpTranscriberConfig
from darvis.utils.keyboard_input import DefaultHotkeys
from darvis.voice.component import VoiceComponent, VoiceConfig


def _create_voice_config() -> VoiceConfig:
    return VoiceConfig(
        wake_words=["darvis", "jarvis"],
        hotkey=DefaultHotkeys.PUSH_TO_TALK
    )


def _create_transcriber() -> Transcriber:
    use_http = os.environ.get("DARVIS_USE_HTTP_TRANSCRIPTION", "true").lower() == "true"

    if use_http:
        return HttpTranscriber(HttpTranscriberConfig.from_env())

    print("[DARVIS] HTTP transcription disabled, using stub transcriber")
    return StubTranscriber()


def _get_correction_model_path() -> Optional[str]:
    model_path = os.environ.get("DARVIS_CORRECTION_MODEL")
    if model_path and Path(model_path).exists():
        return model_path

    default_path = Path("models/qwen2.5-1.5b-instruct-q4.gguf")
    if default_path.exists():
        return str(default_path)

    return None


def _create_corrector() -> Corrector:
    model_path = _get_correction_model_path()

    if model_path:
        return LLMCorrector(LLMCorrectorConfig(model_path=model_path))

    print("[DARVIS] No correction model found, using stub corrector")
    return StubCorrector()


class Daemon:
    def __init__(
        self,
        transcriber: Optional[Transcriber] = None,
        corrector: Optional[Corrector] = None,
        voice_config: Optional[VoiceConfig] = None
    ):
        self.queue: Queue[EventType] = asyncio.Queue()
        self.fsm = StateMachine(State.INIT)
        self.running = True

        self._registry = HandlerRegistry()
        self._cancellation = get_cancellation_processor()
        self._context = DaemonContext(
            active_tasks={},
            resources={}
        )

        self._voice = VoiceComponent(self.queue, voice_config or _create_voice_config())
        self._input = InputComponent(self.queue)
        self._transcription = TranscriptionComponent(
            self.queue,
            transcriber or _create_transcriber()
        )

        actual_corrector = corrector or _create_corrector()
        self._correction = CorrectionComponent(
            self.queue,
            actual_corrector
        )

        llm_corrector = actual_corrector if isinstance(actual_corrector, LLMCorrector) else None
        self._chat = ChatComponent(self.queue, llm_corrector=llm_corrector)

        self._tts = TTSComponent(self.queue)
        self._components = [self._voice, self._input, self._transcription, self._correction, self._chat, self._tts]

    async def start(self):
        print("[DARVIS] Initializing...")

        for component in self._components:
            component.register_handlers(self._registry)
            await component.start()

        wake_task = self._voice.get_wake_word_task()
        if wake_task:
            if isinstance(wake_task, asyncio.Task):
                self._context.active_tasks[TaskName.WAKE_WORD] = wake_task
            else:
                self._context.active_tasks[TaskName.WAKE_WORD] = asyncio.create_task(wake_task)

        await self.queue.put(EventType.LOADED)
        print("[DARVIS] Ready")

        await self._event_loop()

    async def _event_loop(self):
        while self.running:
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                result = self.fsm.dispatch(event)
                await self._handle_result(result)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _handle_result(self, result: TransitionResult):
        if result.transitioned:
            print(f"[FSM] {result.from_state.name} -> {result.to_state.name}")

            if result.preempted:
                print(f"[FSM] Action preempted: {result.reason}")

            if result.requires_rollback:
                await self._process_cancellation(result)

            await self._registry.dispatch_entry(result, self._context)
        else:
            print(f"[FSM] No transition for {result.event_type.name} in {result.from_state.name}")

        if result.terminal:
            self.stop()

    async def _process_cancellation(self, result: TransitionResult) -> None:
        context = CancellationContext(
            from_state=result.from_state,
            active_tasks=self._context.active_tasks,
            resources=self._context.resources
        )

        outcome = await self._cancellation.process(context)

        if outcome.result == CancelResult.SUCCESS:
            print(f"[CANCEL] {outcome.message}")
            cleaned = [t.name for t in outcome.cleaned_tasks] + [r.name for r in outcome.cleaned_resources]
            if cleaned:
                print(f"[CANCEL] Cleaned: {', '.join(cleaned)}")
        elif outcome.result == CancelResult.PARTIAL:
            print(f"[CANCEL] Partial: {outcome.message}")
        else:
            print(f"[CANCEL] Failed: {outcome.message}")

        cancel_result = self.fsm.dispatch(EventType.CANCEL_PROCESSED)
        await self._handle_result(cancel_result)

    def stop(self) -> None:
        print("\n[DARVIS] Shutting down...")
        self.running = False

        for component in self._components:
            component.stop()

        for task in self._context.active_tasks.values():
            task.cancel()


def _handle_signal(daemon: Daemon) -> None:
    daemon.stop()


async def _async_main() -> None:
    daemon = Daemon()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: _handle_signal(daemon))
        except NotImplementedError:
            pass

    try:
        await daemon.start()
    except KeyboardInterrupt:
        daemon.stop()


def run() -> None:
    print("[DARVIS] Starting daemon...")
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        print("\n[DARVIS] Interrupted.")

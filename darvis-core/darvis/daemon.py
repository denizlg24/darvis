import asyncio
import os
import signal
from asyncio import Queue
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
from darvis.core.task_registry import TaskName
from darvis.chat.component import ChatComponent
from darvis.input.component import InputComponent
from darvis.output.component import TTSComponent
from darvis.services.process_manager import ProcessManager
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


class Daemon:
    def __init__(
        self,
        transcriber: Optional[Transcriber] = None,
        voice_config: Optional[VoiceConfig] = None,
        manage_services: bool = True
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

        
        self._manage_services = manage_services
        self._process_manager: Optional[ProcessManager] = None
        if manage_services:
            self._process_manager = ProcessManager()

        self._voice = VoiceComponent(self.queue, voice_config or _create_voice_config())
        self._input = InputComponent(self.queue)
        self._transcription = TranscriptionComponent(
            self.queue,
            transcriber or _create_transcriber()
        )

        
        self._chat = ChatComponent(self.queue)

        self._tts = TTSComponent(self.queue)
        self._components = [self._voice, self._input, self._transcription, self._chat, self._tts]

    async def start(self):
        print("[DARVIS] Initializing...")

        
        if self._process_manager:
            services_ok = await self._process_manager.start_all()
            if not services_ok:
                print("[DARVIS] Warning: Some microservices failed to start")
                print("[DARVIS] Continuing anyway - some features may be unavailable")

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
        self._voice.play_sound("cancel")

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
        if not self.running:
            return  

        print("\n[DARVIS] Shutting down...")
        self.running = False

        
        for component in self._components:
            try:
                component.stop()
            except Exception as e:
                print(f"[DARVIS] Error stopping component: {e}")

        
        for name, task in self._context.active_tasks.items():
            if task and not task.done():
                task.cancel()

        
        if self._process_manager:
            self._process_manager.stop_all()


def _handle_signal(daemon: Daemon) -> None:
    daemon.stop()


async def _async_main() -> None:
    
    
    manage_services = os.environ.get("DARVIS_MANAGE_SERVICES", "true").lower() == "true"

    daemon = Daemon(manage_services=manage_services)

    loop = asyncio.get_running_loop()

    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: _handle_signal(daemon))
        except NotImplementedError:
            
            pass

    try:
        await daemon.start()
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        if daemon.running:
            daemon.stop()

        
        await asyncio.sleep(0.1)


def run() -> None:
    print("[DARVIS] Starting daemon...")
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        pass  
    except SystemExit:
        pass
    finally:
        print("[DARVIS] Goodbye.")

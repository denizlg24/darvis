import asyncio
from asyncio import Queue
from typing import Optional

from darvis.core.events import EventType
from darvis.core.handlers import HandlerRegistry
from darvis.utils.keyboard_input import DefaultHotkeys, get_keyboard_controller


class InputComponent:
    def __init__(self, event_queue: Queue[EventType]):
        self._queue = event_queue
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._setup_hotkeys()

        keyboard = get_keyboard_controller()
        keyboard.start()

    def stop(self) -> None:
        keyboard = get_keyboard_controller()
        keyboard.stop()

    def register_handlers(self, registry: HandlerRegistry) -> None:
        pass

    def _setup_hotkeys(self) -> None:
        keyboard = get_keyboard_controller()
        keyboard.register_hotkey(
            DefaultHotkeys.CANCEL,
            on_press=self._on_cancel,
            suppress=False
        )
        keyboard.register_hotkey(
            DefaultHotkeys.QUIT,
            on_press=self._on_quit,
            suppress=False
        )

    def _on_cancel(self) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, EventType.CANCEL
            )

    def _on_quit(self) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, EventType.QUIT
            )

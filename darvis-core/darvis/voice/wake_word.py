import asyncio
from asyncio import Queue
from typing import Optional

from darvis.core.events import EventType
from darvis.utils.keyboard_input import (
    DefaultHotkeys,
    KeyboardController,
    get_keyboard_controller,
)


class WakeWordListener:

    def __init__(
        self,
        wake_word: str,
        required_hits: int,
        event_queue: Queue,
        hotkey: str = DefaultHotkeys.PUSH_TO_TALK,
        keyboard_controller: Optional[KeyboardController] = None
    ):
        self.wake_word = wake_word
        self.required_hits = required_hits
        self.queue = event_queue
        self.hotkey = hotkey
        self.running = False
        self._paused = False
        self._wake_event: Optional[asyncio.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._keyboard = keyboard_controller or get_keyboard_controller()
        self._setup_hotkey()

    def _setup_hotkey(self) -> None:
        self._keyboard.register_hotkey(
            self.hotkey,
            on_press=self._on_hotkey_press,
            suppress=False
        )

    def _on_hotkey_press(self) -> None:
        if self._wake_event is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._wake_event.set)

    def _detect_wake_word(self) -> bool:
        if self._wake_event is None or self._paused:
            return False

        if self._wake_event.is_set():
            self._wake_event.clear()
            return True
        return False

    async def run(self) -> None:
        self.running = True
        self._loop = asyncio.get_running_loop()
        self._wake_event = asyncio.Event()
        self._keyboard.start()

        try:
            while self.running:
                await asyncio.sleep(0.05)

                if self._detect_wake_word():
                    await self.queue.put(EventType.WAKE_WORD)
        finally:
            self._keyboard.stop()

    def stop(self) -> None:
        self.running = False
        if self._wake_event:
            self._wake_event.set()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

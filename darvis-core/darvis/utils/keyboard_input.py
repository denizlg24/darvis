import asyncio
import keyboard
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import Lock
from typing import Optional


class KeyState(Enum):
    PRESSED = auto()
    RELEASED = auto()


@dataclass
class HotkeyBinding:
    key: str
    on_press: Optional[Callable[[], None]] = None
    on_release: Optional[Callable[[], None]] = None
    suppress: bool = False


class KeyboardController:
    _instance: Optional["KeyboardController"] = None
    _lock: Lock = Lock()

    def __new__(cls) -> "KeyboardController":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._bindings: dict[str, HotkeyBinding] = {}
        self._key_states: dict[str, KeyState] = {}
        self._async_events: dict[str, asyncio.Event] = {}
        self._running = False
        self._hooks: list[Callable] = []
        self._initialized = True

    def register_hotkey(
        self,
        key: str,
        on_press: Optional[Callable[[], None]] = None,
        on_release: Optional[Callable[[], None]] = None,
        suppress: bool = False
    ) -> None:
        binding = HotkeyBinding(
            key=key,
            on_press=on_press,
            on_release=on_release,
            suppress=suppress
        )
        self._bindings[key] = binding
        self._key_states[key] = KeyState.RELEASED

        if self._running:
            self._attach_binding(binding)

    def unregister_hotkey(self, key: str) -> None:
        if key in self._bindings:
            del self._bindings[key]
            del self._key_states[key]
            if key in self._async_events:
                del self._async_events[key]

    def get_async_event(self, key: str) -> asyncio.Event:
        if key not in self._async_events:
            self._async_events[key] = asyncio.Event()
        return self._async_events[key]

    def is_pressed(self, key: str) -> bool:
        return self._key_states.get(key) == KeyState.PRESSED

    def _attach_binding(self, binding: HotkeyBinding) -> None:
        def on_press_handler():
            self._key_states[binding.key] = KeyState.PRESSED

            if binding.key in self._async_events:
                event = self._async_events[binding.key]
                event.set()

            if binding.on_press:
                binding.on_press()

        def on_release_handler():
            self._key_states[binding.key] = KeyState.RELEASED

            if binding.key in self._async_events:
                event = self._async_events[binding.key]
                event.clear()

            if binding.on_release:
                binding.on_release()

        hook = keyboard.on_press_key(
            binding.key,
            lambda _: on_press_handler(),
            suppress=binding.suppress
        )
        self._hooks.append(hook)

        hook = keyboard.on_release_key(
            binding.key,
            lambda _: on_release_handler(),
            suppress=binding.suppress
        )
        self._hooks.append(hook)

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        for binding in self._bindings.values():
            self._attach_binding(binding)

    def stop(self) -> None:
        self._running = False
        keyboard.unhook_all()
        self._hooks.clear()

        for key in self._key_states:
            self._key_states[key] = KeyState.RELEASED

        for event in self._async_events.values():
            event.clear()

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None


class DefaultHotkeys:
    PUSH_TO_TALK = "f4"
    CANCEL = "escape"
    QUIT = "q"


def get_keyboard_controller() -> KeyboardController:
    return KeyboardController()

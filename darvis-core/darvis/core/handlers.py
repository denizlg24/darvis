from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from darvis.core.events import EventType
from darvis.core.fsm import TransitionResult
from darvis.core.states import State
from darvis.core.task_registry import ResourceName, TaskName


@dataclass
class DaemonContext:
    active_tasks: dict[TaskName, Any]
    resources: dict[ResourceName, Any]

    def set_resource(self, name: ResourceName, value: Any) -> None:
        self.resources[name] = value

    def get_resource(self, name: ResourceName) -> Optional[Any]:
        return self.resources.get(name)

    def clear_resource(self, name: ResourceName) -> None:
        if name in self.resources:
            del self.resources[name]


StateEntryHandler = Callable[[TransitionResult, DaemonContext], Awaitable[None]]
StateExitHandler = Callable[[TransitionResult, DaemonContext], Awaitable[None]]


@runtime_checkable
class Component(Protocol):
    async def start(self) -> None: ...
    def stop(self) -> None: ...
    def register_handlers(self, registry: "HandlerRegistry") -> None: ...


@dataclass
class HandlerRegistry:
    _entry_handlers: dict[State, list[StateEntryHandler]] = field(default_factory=dict)
    _exit_handlers: dict[State, list[StateExitHandler]] = field(default_factory=dict)
    _transition_handlers: dict[tuple[State, State], list[StateEntryHandler]] = field(
        default_factory=dict
    )

    def on_enter(self, state: State, handler: StateEntryHandler) -> None:
        if state not in self._entry_handlers:
            self._entry_handlers[state] = []
        self._entry_handlers[state].append(handler)

    def on_exit(self, state: State, handler: StateExitHandler) -> None:
        if state not in self._exit_handlers:
            self._exit_handlers[state] = []
        self._exit_handlers[state].append(handler)

    def on_transition(
        self, from_state: State, to_state: State, handler: StateEntryHandler
    ) -> None:
        key = (from_state, to_state)
        if key not in self._transition_handlers:
            self._transition_handlers[key] = []
        self._transition_handlers[key].append(handler)

    async def dispatch_entry(
        self, result: TransitionResult, context: DaemonContext
    ) -> None:
        transition_key = (result.from_state, result.to_state)
        if transition_key in self._transition_handlers:
            for handler in self._transition_handlers[transition_key]:
                await handler(result, context)

        if result.to_state in self._entry_handlers:
            for handler in self._entry_handlers[result.to_state]:
                await handler(result, context)

    async def dispatch_exit(
        self, result: TransitionResult, context: DaemonContext
    ) -> None:
        if result.from_state in self._exit_handlers:
            for handler in self._exit_handlers[result.from_state]:
                await handler(result, context)

from darvis.core.events import EventType
from darvis.core.states import State
from darvis.core.transitions import TRANSITIONS
from dataclasses import dataclass
from darvis.utils.reasons import TransitionReasons
from typing import Optional


@dataclass(frozen=True)
class TransitionResult:
    from_state: State
    to_state: State
    event_type: EventType

    transitioned: bool

    preempted: bool = False
    requires_dispatch: bool = False
    requires_rollback: bool = False
    terminal: bool = False

    reason: Optional[str] = None


class StateMachine:
    def __init__(self, initial_state: State):
        self._state = initial_state

    @property
    def state(self) -> State:
        return self._state

    def dispatch(self, event: EventType):
        current = self._state

        if event == EventType.CANCEL:
            self._state = State.CANCEL_REQUESTED
            return TransitionResult(
                from_state=current,
                to_state=State.CANCEL_REQUESTED,
                event_type=EventType.CANCEL,
                transitioned=True,
                preempted=True,
                requires_rollback=True,
                reason=TransitionReasons.USER_CANCEL
            )

        if event == EventType.QUIT and current != State.VOICE_CAPTURE:
            self._state = State.SHUTDOWN
            return TransitionResult(
                from_state=current,
                to_state=State.SHUTDOWN,
                event_type=EventType.QUIT,
                transitioned=True,
                terminal=True,
                reason=TransitionReasons.USER_QUIT
            )

        key = (current, event)

        if key not in TRANSITIONS:
            return TransitionResult(
                from_state=current,
                to_state=current,
                event_type=event,
                transitioned=False,
                reason=TransitionReasons.INVALID_EVENT
            )

        next_state = TRANSITIONS[key]
        self._state = next_state

        return TransitionResult(
            from_state=current,
            to_state=next_state,
            event_type=event,
            transitioned=True
        )

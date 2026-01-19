from enum import Enum, auto


class State(Enum):
    INIT = auto()
    IDLE = auto()

    LISTENING = auto()
    VOICE_CAPTURE = auto()
    TRANSCRIBING = auto()
    CHAT = auto()

    TTS = auto()

    CANCEL_REQUESTED = auto()
    SHUTDOWN = auto()

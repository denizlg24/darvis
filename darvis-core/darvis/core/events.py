from enum import Enum, auto


class EventType(Enum):
    LOADED = auto()

    WAKE_WORD = auto()
    LISTEN_START = auto()
    SILENCE_DETECTED = auto()
    TRANSCRIPTION_READY = auto()
    CHAT_READY = auto()
    TOOL_CALL_DETECTED = auto()
    TOOL_EXECUTION_DONE = auto()

    TTS_DONE = auto()

    CANCEL = auto()
    CANCEL_PROCESSED = auto()
    QUIT = auto()
    ERROR = auto()

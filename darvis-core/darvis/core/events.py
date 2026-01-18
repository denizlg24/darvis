from enum import Enum, auto


class EventType(Enum):
    LOADED = auto()

    WAKE_WORD = auto()
    LISTEN_START = auto()
    SILENCE_DETECTED = auto()
    TRANSCRIPTION_READY = auto()
    CORRECTION_READY = auto()
    CHAT_READY = auto()

    TTS_DONE = auto()

    CANCEL = auto()
    CANCEL_PROCESSED = auto()
    QUIT = auto()
    ERROR = auto()

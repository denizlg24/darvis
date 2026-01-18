from enum import Enum, auto


class TaskName(Enum):
    WAKE_WORD = auto()
    CANCEL_LISTENER = auto()
    QUIT_LISTENER = auto()
    LISTEN = auto()
    CAPTURE = auto()
    TRANSCRIPTION = auto()
    CORRECTION = auto()
    CHAT = auto()
    TTS = auto()
    LLM_INFERENCE = auto()
    SESSION_TIMEOUT = auto()


class ResourceName(Enum):
    AUDIO_BUFFER = auto()
    AUDIO_STREAM = auto()
    TRANSCRIPTION_RESULT = auto()
    CORRECTED_TEXT = auto()
    CHAT_RESPONSE = auto()
    CONVERSATION_HISTORY = auto()
    LLM_CONTEXT = auto()
    EXIT_REQUESTED = auto()
    SESSION_ACTIVE = auto()
    SENTENCE_QUEUE = auto()
    STREAMING_ACTIVE = auto()
    USER_MESSAGE = auto()

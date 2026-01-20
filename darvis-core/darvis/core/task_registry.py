from enum import Enum, auto


class TaskName(Enum):
    WAKE_WORD = auto()
    CANCEL_LISTENER = auto()
    QUIT_LISTENER = auto()
    LISTEN = auto()
    CAPTURE = auto()
    TRANSCRIPTION = auto()
    CHAT = auto()
    TTS = auto()
    LLM_INFERENCE = auto()
    SESSION_TIMEOUT = auto()
    TOOL_EXECUTION = auto()


class ResourceName(Enum):
    AUDIO_BUFFER = auto()
    AUDIO_STREAM = auto()
    TRANSCRIPTION_RESULT = auto()
    CHAT_INPUT = auto()
    CHAT_RESPONSE = auto()
    CONVERSATION_HISTORY = auto()
    LLM_CONTEXT = auto()
    EXIT_REQUESTED = auto()
    SESSION_ACTIVE = auto()
    SENTENCE_QUEUE = auto()
    STREAMING_ACTIVE = auto()
    USER_MESSAGE = auto()
    TOOLS_SCHEMA = auto()
    ACTIVE_TOOL = auto()
    TOOL_RESULT = auto()
    TOOL_FEEDBACK_TEXT = auto()

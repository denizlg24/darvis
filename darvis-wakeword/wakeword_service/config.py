import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WakeWordConfig:
    wake_words: list[str] = field(default_factory=lambda: ["darvis", "jarvis"])
    cancel_words: list[str] = field(default_factory=lambda: ["cancel", "stop", "abort", "never mind"])
    quit_words: list[str] = field(default_factory=lambda: ["quit", "exit", "shut down", "goodbye"])
    model_path: Optional[str] = None
    sample_rate: int = 16000
    chunk_size: int = 4000
    channels: int = 1
    cooldown_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> "WakeWordConfig":
        wake_words_str = os.environ.get("WAKEWORD_WORDS", "darvis,jarvis")
        wake_words = [w.strip().lower() for w in wake_words_str.split(",")]

        cancel_words_str = os.environ.get("WAKEWORD_CANCEL_WORDS", "cancel,stop,abort,never mind")
        cancel_words = [w.strip().lower() for w in cancel_words_str.split(",")]

        quit_words_str = os.environ.get("WAKEWORD_QUIT_WORDS", "quit,exit,shutdown,goodbye")
        quit_words = [w.strip().lower() for w in quit_words_str.split(",")]

        model_path = os.environ.get("WAKEWORD_MODEL_PATH")
        if not model_path:
            default_path = "models/vosk"
            if os.path.exists(default_path):
                model_path = default_path

        return cls(
            wake_words=wake_words,
            cancel_words=cancel_words,
            quit_words=quit_words,
            model_path=model_path,
            sample_rate=int(os.environ.get("WAKEWORD_SAMPLE_RATE", "16000")),
            chunk_size=int(os.environ.get("WAKEWORD_CHUNK_SIZE", "4000")),
            cooldown_seconds=float(os.environ.get("WAKEWORD_COOLDOWN", "2.0")),
        )


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8002

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            host=os.environ.get("WAKEWORD_HOST", "127.0.0.1"),
            port=int(os.environ.get("WAKEWORD_PORT", "8002")),
        )

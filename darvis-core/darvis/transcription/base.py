from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np


class TranscriptionStatus(Enum):
    SUCCESS = auto()
    EMPTY = auto()
    CANCELLED = auto()
    ERROR = auto()


@dataclass
class TranscriptionResult:
    status: TranscriptionStatus
    text: str = ""
    confidence: float = 0.0
    duration_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.status == TranscriptionStatus.SUCCESS and len(self.text.strip()) > 0


class Transcriber(ABC):
    @abstractmethod
    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> TranscriptionResult:
        pass

    @abstractmethod
    async def load(self) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class StubTranscriber(Transcriber):
    def __init__(self, response: str = "[stub transcription]"):
        self._response = response
        self._loaded = False

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> TranscriptionResult:
        if not self._loaded:
            return TranscriptionResult(
                status=TranscriptionStatus.ERROR,
                error="Transcriber not loaded"
            )

        duration = len(audio) / sample_rate if sample_rate > 0 else 0
        return TranscriptionResult(
            status=TranscriptionStatus.SUCCESS,
            text=self._response,
            confidence=1.0,
            duration_seconds=duration
        )

    async def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def name(self) -> str:
        return "stub"

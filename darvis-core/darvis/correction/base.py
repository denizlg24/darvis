from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class CorrectionStatus(Enum):
    SUCCESS = auto()
    EMPTY = auto()
    CANCELLED = auto()
    ERROR = auto()
    TIMEOUT = auto()


@dataclass
class CorrectionResult:
    status: CorrectionStatus
    text: str = ""
    original_text: str = ""
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.status == CorrectionStatus.SUCCESS and bool(self.text)


class Corrector(ABC):
    @abstractmethod
    async def correct(self, text: str) -> CorrectionResult:
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


class StubCorrector(Corrector):
    def __init__(self):
        self._loaded = False

    async def correct(self, text: str) -> CorrectionResult:
        return CorrectionResult(
            status=CorrectionStatus.SUCCESS,
            text=text,
            original_text=text
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

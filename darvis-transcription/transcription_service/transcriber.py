import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from transcription_service.config import WhisperConfig


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

    def to_dict(self) -> dict:
        return {
            "status": self.status.name,
            "text": self.text,
            "confidence": self.confidence,
            "duration": self.duration_seconds,
            "error": self.error
        }


class WhisperTranscriber:
    def __init__(self, config: Optional[WhisperConfig] = None):
        self._config = config or WhisperConfig.from_env()
        self._model = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._cancelled = threading.Event()
        self._model_info: dict = {}

    def load(self) -> None:
        from faster_whisper import WhisperModel

        model_path = self._config.get_model_path()

        print(f"[TRANSCRIPTION] Loading faster-whisper model...")
        print(f"[TRANSCRIPTION] Model: {model_path or self._config.model_size}")
        print(f"[TRANSCRIPTION] Device: {self._config.device}")
        print(f"[TRANSCRIPTION] Compute type: {self._config.compute_type}")

        with self._lock:
            self._model = WhisperModel(
                model_path or self._config.model_size,
                device=self._config.device,
                compute_type=self._config.compute_type,
                download_root=self._config.download_root
            )
            self._model_info = {
                "model": model_path or self._config.model_size,
                "device": self._config.device,
                "compute_type": self._config.compute_type
            }

        print(f"[TRANSCRIPTION] Model loaded successfully")

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._model_info = {}
        self._executor.shutdown(wait=False)

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    @property
    def model_info(self) -> dict:
        with self._lock:
            return self._model_info.copy()

    def cancel(self) -> None:
        self._cancelled.set()

    def reset_cancel(self) -> None:
        self._cancelled.clear()

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        if not self.is_loaded:
            return TranscriptionResult(
                status=TranscriptionStatus.ERROR,
                error="Model not loaded"
            )

        if len(audio) == 0:
            return TranscriptionResult(
                status=TranscriptionStatus.EMPTY,
                error="No audio data"
            )

        self.reset_cancel()

        return self._transcribe_sync(audio, sample_rate)

    def _transcribe_sync(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> TranscriptionResult:
        duration = len(audio) / sample_rate

        if self._cancelled.is_set():
            return TranscriptionResult(
                status=TranscriptionStatus.CANCELLED,
                duration_seconds=duration
            )

        with self._lock:
            if self._model is None:
                return TranscriptionResult(
                    status=TranscriptionStatus.ERROR,
                    error="Model unloaded during transcription"
                )

            try:
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

                if audio.max() > 1.0 or audio.min() < -1.0:
                    audio = audio / 32768.0

                segments, info = self._model.transcribe(
                    audio,
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500
                    )
                )

                text_parts = []
                total_confidence = 0.0
                segment_count = 0

                for segment in segments:
                    if self._cancelled.is_set():
                        return TranscriptionResult(
                            status=TranscriptionStatus.CANCELLED,
                            duration_seconds=duration
                        )
                    text_parts.append(segment.text.strip())
                    total_confidence += segment.avg_logprob
                    segment_count += 1

                text = " ".join(text_parts).strip()

                if not text:
                    return TranscriptionResult(
                        status=TranscriptionStatus.EMPTY,
                        duration_seconds=duration
                    )

                avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.0
                confidence = min(1.0, max(0.0, 1.0 + avg_confidence))

                return TranscriptionResult(
                    status=TranscriptionStatus.SUCCESS,
                    text=text,
                    confidence=confidence,
                    duration_seconds=duration
                )

            except Exception as e:
                print(f"[TRANSCRIPTION] Error: {e}")
                return TranscriptionResult(
                    status=TranscriptionStatus.ERROR,
                    duration_seconds=duration,
                    error=str(e)
                )

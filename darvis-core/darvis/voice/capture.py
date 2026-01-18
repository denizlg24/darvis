import asyncio
import threading
import time
from asyncio import Queue
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
import sounddevice as sd

from darvis.core.events import EventType


class SpeechState(Enum):
    WAITING_FOR_SPEECH = auto()
    SPEECH_DETECTED = auto()
    SILENCE_AFTER_SPEECH = auto()


@dataclass
class CaptureConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    block_size: int = 1024

    silence_threshold_db: float = -40.0
    speech_threshold_db: float = -30.0

    silence_duration_listening: float = 3.0
    silence_duration_after_speech: float = 2.5
    min_speech_duration: float = 0.3
    max_capture_duration: float = 30.0


@dataclass
class AudioBuffer:
    data: list[np.ndarray] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, chunk: np.ndarray) -> None:
        with self._lock:
            self.data.append(chunk.copy())

    def get_audio(self) -> np.ndarray:
        with self._lock:
            if not self.data:
                return np.array([], dtype=np.float32)
            return np.concatenate(self.data)

    def clear(self) -> None:
        with self._lock:
            self.data.clear()

    def duration(self, sample_rate: int) -> float:
        with self._lock:
            total_samples = sum(chunk.shape[0] for chunk in self.data)
            return total_samples / sample_rate


class VoiceCapture:
    def __init__(
        self,
        event_queue: Queue[EventType],
        config: Optional[CaptureConfig] = None
    ):
        self.queue = event_queue
        self.config = config or CaptureConfig()

        self._stream: Optional[sd.InputStream] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

        self._speech_state = SpeechState.WAITING_FOR_SPEECH
        self._state_lock = threading.Lock()

        self._last_speech_time: float = 0.0
        self._speech_start_time: float = 0.0
        self._capture_start_time: float = 0.0

        self.audio_buffer = AudioBuffer()

        self._level_history: deque[float] = deque(maxlen=10)

    def _db_from_rms(self, rms: float) -> float:
        if rms < 1e-10:
            return -100.0
        return 20 * np.log10(rms)

    def _calculate_level(self, audio: np.ndarray) -> float:
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        return self._db_from_rms(rms)

    def _get_smoothed_level(self, current_level: float) -> float:
        self._level_history.append(current_level)
        return sum(self._level_history) / len(self._level_history)

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags
    ) -> None:
        if status:
            print(f"[CAPTURE] Stream status: {status}")

        if not self._running:
            return

        current_time = time.monotonic()
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()

        level = self._calculate_level(audio)
        smoothed_level = self._get_smoothed_level(level)

        is_speech = smoothed_level > self.config.speech_threshold_db
        is_silence = smoothed_level < self.config.silence_threshold_db

        with self._state_lock:
            self._process_audio_state(
                audio, current_time, is_speech, is_silence
            )

    def _process_audio_state(
        self,
        audio: np.ndarray,
        current_time: float,
        is_speech: bool,
        is_silence: bool
    ) -> None:
        if self._speech_state == SpeechState.WAITING_FOR_SPEECH:
            self.audio_buffer.append(audio)

            if is_speech:
                self._speech_state = SpeechState.SPEECH_DETECTED
                self._speech_start_time = current_time
                self._last_speech_time = current_time
                self._emit_event(EventType.LISTEN_START)

            elif is_silence:
                elapsed = current_time - self._capture_start_time
                if elapsed > self.config.silence_duration_listening:
                    self._emit_event(EventType.SILENCE_DETECTED)

        elif self._speech_state == SpeechState.SPEECH_DETECTED:
            self.audio_buffer.append(audio)

            total_duration = current_time - self._capture_start_time
            if total_duration > self.config.max_capture_duration:
                self._speech_state = SpeechState.SILENCE_AFTER_SPEECH
                self._emit_event(EventType.SILENCE_DETECTED)
                return

            if is_speech:
                self._last_speech_time = current_time
            elif is_silence:
                silence_duration = current_time - self._last_speech_time
                if silence_duration > self.config.silence_duration_after_speech:
                    speech_duration = self._last_speech_time - self._speech_start_time
                    if speech_duration >= self.config.min_speech_duration:
                        self._speech_state = SpeechState.SILENCE_AFTER_SPEECH
                        self._emit_event(EventType.SILENCE_DETECTED)
                    else:
                        self._reset_state()

    def _emit_event(self, event: EventType) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self.queue.put_nowait, event
            )

    def _reset_state(self) -> None:
        self._speech_state = SpeechState.WAITING_FOR_SPEECH
        self._last_speech_time = 0.0
        self._speech_start_time = 0.0
        self._capture_start_time = time.monotonic()
        self.audio_buffer.clear()
        self._level_history.clear()

    async def start(self) -> None:
        if self._running:
            return

        self._loop = asyncio.get_running_loop()
        self._running = True
        self._reset_state()

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.block_size,
            callback=self._audio_callback
        )
        self._stream.start()

    def stop(self) -> None:
        self._running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def reset(self) -> None:
        with self._state_lock:
            self._reset_state()

    def get_captured_audio(self) -> tuple[np.ndarray, int]:
        return self.audio_buffer.get_audio(), self.config.sample_rate

    @property
    def has_speech(self) -> bool:
        with self._state_lock:
            return self._speech_state != SpeechState.WAITING_FOR_SPEECH

    @property
    def speech_duration(self) -> float:
        return self.audio_buffer.duration(self.config.sample_rate)

import json
import queue
import threading
import time
from enum import Enum
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel

from wakeword_service.config import WakeWordConfig


SetLogLevel(-1)


class DetectionType(Enum):
    WAKE = "wake_word"
    CANCEL = "cancel_word"
    QUIT = "quit_word"


class WakeWordDetector:
    def __init__(
        self,
        config: Optional[WakeWordConfig] = None,
        on_detection: Optional[Callable[[DetectionType, str, float], None]] = None
    ):
        self._config = config or WakeWordConfig()
        self._on_detection = on_detection
        self._model: Optional[Model] = None
        self._recognizer: Optional[KaldiRecognizer] = None
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._paused = False
        self._cancel_active = False
        self._lock = threading.Lock()
        self._loaded = False
        self._last_detection_time: dict[DetectionType, float] = {}
        self._audio_queue: queue.Queue = queue.Queue()
        self._process_thread: Optional[threading.Thread] = None

    def load(self) -> None:
        if not self._config.model_path:
            raise RuntimeError("No Vosk model path configured")

        print(f"[WAKEWORD] Loading Vosk model: {self._config.model_path}")

        self._model = Model(self._config.model_path)
        self._recognizer = KaldiRecognizer(self._model, self._config.sample_rate)
        self._recognizer.SetWords(True)

        self._loaded = True
        print(f"[WAKEWORD] Model loaded")
        print(f"[WAKEWORD]   Wake words: {self._config.wake_words}")
        print(f"[WAKEWORD]   Cancel words: {self._config.cancel_words}")
        print(f"[WAKEWORD]   Quit words: {self._config.quit_words}")

    def unload(self) -> None:
        self.stop()
        with self._lock:
            self._recognizer = None
            self._model = None
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def wake_words(self) -> list[str]:
        return self._config.wake_words

    @property
    def cancel_words(self) -> list[str]:
        return self._config.cancel_words

    @property
    def quit_words(self) -> list[str]:
        return self._config.quit_words

    def pause(self) -> None:
        with self._lock:
            self._paused = True
        print("[WAKEWORD] Wake detection paused")

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            if self._recognizer:
                self._recognizer.Reset()
        print("[WAKEWORD] Wake detection resumed")

    def activate_cancel(self) -> None:
        with self._lock:
            self._cancel_active = True
        print("[WAKEWORD] Cancel detection activated")

    def deactivate_cancel(self) -> None:
        with self._lock:
            self._cancel_active = False
        print("[WAKEWORD] Cancel detection deactivated")

    def start(self) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        if self._running:
            return

        self._running = True
        self._paused = False
        self._cancel_active = False

        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        self._process_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self._process_thread.start()

        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            dtype=np.int16,
            blocksize=self._config.chunk_size,
            callback=self._audio_callback
        )
        self._stream.start()
        print("[WAKEWORD] Audio stream started")

    def stop(self) -> None:
        self._running = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._process_thread and self._process_thread.is_alive():
            self._audio_queue.put(None)
            self._process_thread.join(timeout=2.0)
            self._process_thread = None

        print("[WAKEWORD] Audio stream stopped")

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            if status.input_overflow:
                pass
            else:
                print(f"[WAKEWORD] Audio status: {status}")

        if not self._running:
            return

        try:
            self._audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    def _process_audio_loop(self) -> None:
        while self._running:
            try:
                audio_data = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if audio_data is None:
                break

            with self._lock:
                if self._recognizer is None:
                    continue

                audio_bytes = audio_data.tobytes()

                if self._recognizer.AcceptWaveform(audio_bytes):
                    result = json.loads(self._recognizer.Result())
                    self._check_for_keywords(result.get("text", ""))
                else:
                    partial = json.loads(self._recognizer.PartialResult())
                    self._check_for_keywords(partial.get("partial", ""))

    def _check_for_keywords(self, text: str) -> None:
        if not text:
            return

        text_lower = text.lower()
        current_time = time.time()

        
        for word in self._config.quit_words:
            if word in text_lower:
                if self._can_trigger(DetectionType.QUIT, current_time):
                    print(f"[WAKEWORD] Quit detected: '{word}' in '{text}'")
                    self._last_detection_time[DetectionType.QUIT] = current_time
                    if self._recognizer:
                        self._recognizer.Reset()
                    if self._on_detection:
                        self._on_detection(DetectionType.QUIT, word, 1.0)
                    return

        
        if self._cancel_active:
            for word in self._config.cancel_words:
                if word in text_lower:
                    if self._can_trigger(DetectionType.CANCEL, current_time):
                        print(f"[WAKEWORD] Cancel detected: '{word}' in '{text}'")
                        self._last_detection_time[DetectionType.CANCEL] = current_time
                        if self._recognizer:
                            self._recognizer.Reset()
                        if self._on_detection:
                            self._on_detection(DetectionType.CANCEL, word, 1.0)
                        return

        
        if not self._paused:
            for word in self._config.wake_words:
                if word in text_lower:
                    if self._can_trigger(DetectionType.WAKE, current_time):
                        print(f"[WAKEWORD] Wake detected: '{word}' in '{text}'")
                        self._last_detection_time[DetectionType.WAKE] = current_time
                        if self._recognizer:
                            self._recognizer.Reset()
                        if self._on_detection:
                            self._on_detection(DetectionType.WAKE, word, 1.0)
                        return

    def _can_trigger(self, detection_type: DetectionType, current_time: float) -> bool:
        last_time = self._last_detection_time.get(detection_type, 0.0)
        return current_time - last_time >= self._config.cooldown_seconds

    def set_callback(self, callback: Callable[[DetectionType, str, float], None]) -> None:
        self._on_detection = callback

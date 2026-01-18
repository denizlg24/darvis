import asyncio
import wave
from asyncio import Queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from darvis.core.constants import SESSION_TIMEOUT_SECONDS
from darvis.core.events import EventType
from darvis.core.fsm import TransitionResult
from darvis.core.handlers import DaemonContext, HandlerRegistry
from darvis.core.states import State
from darvis.core.task_registry import ResourceName
from darvis.voice.capture import VoiceCapture


@dataclass
class VoiceConfig:
    wake_words: list[str] = None
    hotkey: Optional[str] = None
    play_feedback_sound: bool = True
    sounds_dir: str = "sounds"
    session_timeout: float = SESSION_TIMEOUT_SECONDS
    session_enabled: bool = True

    def __post_init__(self):
        if self.wake_words is None:
            self.wake_words = ["darvis", "jarvis"]


class VoiceComponent:
    def __init__(self, event_queue: Queue[EventType], config: VoiceConfig):
        self._queue = event_queue
        self._config = config
        self._wake_detector = None
        self._voice_capture: Optional[VoiceCapture] = None
        self._sounds: dict[str, tuple[np.ndarray, int]] = {}
        self._session_timeout_task: Optional[asyncio.Task] = None
        self._wakeword_connected = False

    async def start(self) -> None:
        self._voice_capture = VoiceCapture(event_queue=self._queue)
        self._load_sounds()

        from darvis.voice.http_wakeword import HttpWakeWordDetector

        wake_detector = HttpWakeWordDetector(event_queue=self._queue)
        self._wakeword_connected = await wake_detector.start()

        if self._wakeword_connected:
            self._wake_detector = wake_detector
            print("[VOICE] Wake word service connected")
        else:
            print("[VOICE] Wake word service unavailable, using hotkey fallback")
            if self._config.hotkey:
                from darvis.voice.wake_word import WakeWordListener
                self._wake_detector = WakeWordListener(
                    wake_word=self._config.wake_words[0],
                    required_hits=1,
                    event_queue=self._queue,
                    hotkey=self._config.hotkey
                )

    def _load_sounds(self) -> None:
        sounds_dir = Path(self._config.sounds_dir)
        if not sounds_dir.exists():
            return

        sound_files = {
            "listening": ["listening.wav", "ready.wav", "wake.wav"],
            "captured": ["captured.wav", "done.wav", "acknowledged.wav"],
            "shutdown": ["shutdown.wav", "goodbye.wav", "exit.wav"],
            "session_start": ["session_start.wav"],
            "session_continue": ["session_continue.wav", "continue.wav"],
            "session_end": ["session_end.wav", "end.wav"],
        }

        for sound_name, filenames in sound_files.items():
            for filename in filenames:
                path = sounds_dir / filename
                if path.exists():
                    audio, sample_rate = self._load_wav(path)
                    if audio is not None:
                        self._sounds[sound_name] = (audio, sample_rate)
                        break

    def _load_wav(self, path: Path) -> tuple[Optional[np.ndarray], int]:
        try:
            with wave.open(str(path), 'rb') as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()

                raw_data = wf.readframes(n_frames)

                if sample_width == 1:
                    audio = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                elif sample_width == 2:
                    audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    return None, 0

                if n_channels > 1:
                    audio = audio.reshape(-1, n_channels).mean(axis=1)

                return audio, sample_rate
        except Exception as e:
            print(f"[VOICE] Failed to load sound {path}: {e}")
            return None, 0

    def _play_sound(self, name: str) -> None:
        if not self._config.play_feedback_sound:
            return

        if name in self._sounds:
            audio, sample_rate = self._sounds[name]
            sd.play(audio, samplerate=sample_rate)
        else:
            if name == "listening" or name == "session_start":
                self._play_fallback_beep(440, 0.1)
            elif name == "captured":
                self._play_fallback_beep(523, 0.08)
            elif name == "shutdown" or name == "session_end":
                self._play_fallback_beep(262, 0.2)
            elif name == "session_continue":
                self._play_fallback_beep(660, 0.05)

    def _play_fallback_beep(self, frequency: float, duration: float) -> None:
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        envelope = np.ones_like(t)
        fade_samples = int(sample_rate * 0.01)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        beep = 0.3 * np.sin(2 * np.pi * frequency * t) * envelope
        sd.play(beep, samplerate=sample_rate)

    def stop(self) -> None:
        if self._wake_detector:
            self._wake_detector.stop()
        if self._voice_capture:
            self._voice_capture.stop()
        self._cancel_session_timeout()

    def register_handlers(self, registry: HandlerRegistry) -> None:
        registry.on_enter(State.LISTENING, self._on_enter_listening)
        registry.on_enter(State.VOICE_CAPTURE, self._on_enter_voice_capture)
        registry.on_enter(State.TRANSCRIBING, self._on_enter_transcribing)
        registry.on_enter(State.IDLE, self._on_enter_idle)
        registry.on_enter(State.CANCEL_REQUESTED, self._on_cancel)
        registry.on_enter(State.SHUTDOWN, self._on_shutdown)

    def get_wake_word_task(self):
        if self._wake_detector and hasattr(self._wake_detector, 'get_task'):
            return self._wake_detector.get_task()
        elif self._wake_detector:
            return self._wake_detector.run()
        return None

    def _cancel_session_timeout(self) -> None:
        if self._session_timeout_task and not self._session_timeout_task.done():
            self._session_timeout_task.cancel()
            self._session_timeout_task = None

    async def _start_session(self, context: DaemonContext) -> None:
        context.set_resource(ResourceName.SESSION_ACTIVE, True)

        if self._wake_detector and hasattr(self._wake_detector, 'pause'):
            await self._wake_detector.pause()

        self._play_sound("session_start")
        print("[SESSION] Started")

    async def _end_session(self, context: DaemonContext, reason: str = "timeout") -> None:
        context.set_resource(ResourceName.SESSION_ACTIVE, False)
        context.clear_resource(ResourceName.EXIT_REQUESTED)
        context.clear_resource(ResourceName.CONVERSATION_HISTORY)

        self._cancel_session_timeout()

        if self._wake_detector and hasattr(self._wake_detector, 'resume'):
            await self._wake_detector.resume()

        if self._wake_detector and hasattr(self._wake_detector, 'deactivate_cancel'):
            await self._wake_detector.deactivate_cancel()

        self._play_sound("session_end")
        print(f"[SESSION] Ended ({reason})")

    async def _on_enter_listening(
        self, result: TransitionResult, context: DaemonContext
    ) -> None:
        self._cancel_session_timeout()

        session_active = context.get_resource(ResourceName.SESSION_ACTIVE)
        from_tts = result.from_state == State.TTS

        if from_tts and session_active:
            self._play_sound("session_continue")
            print("[SESSION] Continuing...")
        elif not session_active:
            await self._start_session(context)

        if self._voice_capture:
            self._voice_capture.reset()
            await self._voice_capture.start()
            print("[VOICE] Listening for speech...")

    async def _on_enter_voice_capture(
        self, result: TransitionResult, context: DaemonContext
    ) -> None:
        print("[VOICE] Speech detected, capturing...")

    async def _on_enter_transcribing(
        self, result: TransitionResult, context: DaemonContext
    ) -> None:
        self._play_sound("captured")

        if self._wake_detector and hasattr(self._wake_detector, 'activate_cancel'):
            await self._wake_detector.activate_cancel()

        if self._voice_capture:
            audio_data, sample_rate = self._voice_capture.get_captured_audio()
            self._voice_capture.stop()

            context.set_resource(ResourceName.AUDIO_BUFFER, {
                "data": audio_data,
                "sample_rate": sample_rate
            })

            duration = len(audio_data) / sample_rate if sample_rate > 0 else 0
            print(f"[VOICE] Captured {duration:.2f}s of audio")

    async def _on_enter_idle(
        self, result: TransitionResult, context: DaemonContext
    ) -> None:
        if self._voice_capture:
            self._voice_capture.stop()

        if self._wake_detector and hasattr(self._wake_detector, 'deactivate_cancel'):
            await self._wake_detector.deactivate_cancel()

        session_active = context.get_resource(ResourceName.SESSION_ACTIVE)
        if session_active:
            await self._end_session(context, "returned to idle")

        if self._wake_detector and hasattr(self._wake_detector, 'resume'):
            await self._wake_detector.resume()

        context.clear_resource(ResourceName.AUDIO_BUFFER)

    async def _on_cancel(
        self, result: TransitionResult, context: DaemonContext
    ) -> None:
        if self._voice_capture:
            self._voice_capture.stop()

        if self._wake_detector and hasattr(self._wake_detector, 'deactivate_cancel'):
            await self._wake_detector.deactivate_cancel()

        session_active = context.get_resource(ResourceName.SESSION_ACTIVE)
        if session_active:
            await self._end_session(context, "cancelled")

    async def _on_shutdown(
        self, result: TransitionResult, context: DaemonContext
    ) -> None:
        self._play_sound("shutdown")
        sd.wait()

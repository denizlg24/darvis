import asyncio
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import torch

from darvis.core.events import EventType
from darvis.core.fsm import TransitionResult
from darvis.core.handlers import DaemonContext, HandlerRegistry
from darvis.core.states import State
from darvis.core.task_registry import ResourceName, TaskName

MODEL_URLS = {
    "v3_en": "https://models.silero.ai/models/tts/en/v3_en.pt",
    "v3_de": "https://models.silero.ai/models/tts/de/v3_de.pt",
    "v3_es": "https://models.silero.ai/models/tts/es/v3_es.pt",
    "v3_fr": "https://models.silero.ai/models/tts/fr/v3_fr.pt",
    "v4_ru": "https://models.silero.ai/models/tts/ru/v4_ru.pt",
    "v5_ru": "https://models.silero.ai/models/tts/ru/v5_ru.pt",
}


@dataclass
class TTSConfig:
    enabled: bool = True
    language: str = "en"
    model_id: str = "v3_en"
    speaker: str = "en_0"
    sample_rate: int = 48000
    model_path: Optional[str] = None
    auto_download: bool = True


class TTSComponent:
    def __init__(
        self,
        event_queue: Queue[EventType],
        config: Optional[TTSConfig] = None
    ):
        self._queue = event_queue
        self._config = config or TTSConfig()
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._current_task: Optional[asyncio.Task] = None
        self._loaded = False
        self._device = torch.device('cpu')

    async def start(self) -> None:
        if not self._config.enabled:
            return

        try:
            loop = asyncio.get_running_loop()
            print("[TTS] Loading Silero model...")
            self._model = await loop.run_in_executor(
                self._executor,
                self._load_model
            )
            self._loaded = self._model is not None
            if self._loaded:
                print("[TTS] Engine loaded (Silero)")
            else:
                print("[TTS] Engine failed to load")
        except Exception as e:
            print(f"[TTS] Failed to load engine: {e}")
            self._loaded = False

    def _load_model(self):
        torch.set_num_threads(4)

        if self._config.model_path:
            local_path = Path(self._config.model_path)
            if local_path.exists():
                return self._load_from_file(local_path)

        models_dir = Path("models/silero")
        model_path = models_dir / f"{self._config.model_id}.pt"

        if model_path.exists():
            return self._load_from_file(model_path)

        if self._config.auto_download and self._config.model_id in MODEL_URLS:
            models_dir.mkdir(parents=True, exist_ok=True)
            print(f"[TTS] Downloading {self._config.model_id}...")
            torch.hub.download_url_to_file(
                MODEL_URLS[self._config.model_id],
                str(model_path)
            )
            if model_path.exists():
                return self._load_from_file(model_path)

        print(f"[TTS] Loading via torch.hub...")
        result = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=self._config.language,
            speaker=self._config.model_id
        )
        model = result[0] if isinstance(result, tuple) else result
        model.to(self._device)
        return model

    def _load_from_file(self, path: Path):
        importer = torch.package.PackageImporter(str(path))
        model = importer.load_pickle("tts_models", "model")
        model.to(self._device)
        return model

    def stop(self) -> None:
        self._cancel_current()
        sd.stop()
        self._executor.shutdown(wait=False)
        self._model = None

    def register_handlers(self, registry: HandlerRegistry) -> None:
        registry.on_enter(State.TTS, self._on_enter_tts)
        registry.on_enter(State.CANCEL_REQUESTED, self._on_cancel)

    def _cancel_current(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            self._current_task = None

    async def _on_enter_tts(
        self,
        result: TransitionResult,
        context: DaemonContext
    ) -> None:
        chat_response = context.get_resource(ResourceName.CHAT_RESPONSE)

        if not chat_response or not chat_response.get("text"):
            print("[TTS] No chat response available")
            await self._queue.put(EventType.TTS_DONE)
            return

        text = chat_response["text"]
        print(f"[TTS] Speaking: {text}")

        if not self._loaded or not self._model:
            print(f"\n>>> {text}\n")
            await self._queue.put(EventType.TTS_DONE)
            return

        self._current_task = asyncio.create_task(
            self._run_tts(text, context)
        )
        context.active_tasks[TaskName.TTS] = self._current_task

    async def _run_tts(self, text: str, context: DaemonContext) -> None:
        try:
            loop = asyncio.get_running_loop()
            audio = await loop.run_in_executor(
                self._executor,
                self._synthesize,
                text
            )

            if audio is not None:
                await loop.run_in_executor(
                    self._executor,
                    self._play_audio,
                    audio
                )

            session_active = context.get_resource(ResourceName.SESSION_ACTIVE)
            exit_requested = context.get_resource(ResourceName.EXIT_REQUESTED)

            if session_active and not exit_requested:
                print("[TTS] Session active, continuing to listen...")
                await self._queue.put(EventType.WAKE_WORD)
            else:
                await self._queue.put(EventType.TTS_DONE)

        except asyncio.CancelledError:
            print("[TTS] Cancelled")
            sd.stop()
            raise
        except Exception as e:
            print(f"[TTS] Error: {e}")
            await self._queue.put(EventType.TTS_DONE)

    def _normalize_text(self, text: str) -> str:
        import re

        text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')

        text = re.sub(r'\s+', ' ', text)

        text = text.strip()

        if text and text[-1] not in '.!?':
            text = text + '.'

        return text

    def _synthesize(self, text: str) -> Optional[np.ndarray]:
        try:
            text = self._normalize_text(text)

            if not text:
                return None

            audio = self._model.apply_tts(
                text=text,
                speaker=self._config.speaker,
                sample_rate=self._config.sample_rate
            )
            audio_np = audio.numpy()

            fade_duration = int(self._config.sample_rate * 0.05)
            silence_duration = int(self._config.sample_rate * 0.15)

            if len(audio_np) > fade_duration:
                fade_curve = np.linspace(
                    1.0, 0.0, fade_duration).astype(audio_np.dtype)
                audio_np[-fade_duration:] *= fade_curve

            padding = np.zeros(silence_duration, dtype=audio_np.dtype)
            return np.concatenate([audio_np, padding])
        except Exception as e:
            print(f"[TTS] Synthesis error: {e}")
            return None

    def _play_audio(self, audio: np.ndarray) -> None:
        try:
            sd.play(audio, samplerate=self._config.sample_rate)
            sd.wait()
        except Exception as e:
            print(f"[TTS] Playback error: {e}")

    async def _on_cancel(
        self,
        result: TransitionResult,
        context: DaemonContext
    ) -> None:
        self._cancel_current()
        sd.stop()

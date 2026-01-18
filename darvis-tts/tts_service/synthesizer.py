import asyncio
import io
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from tts_service.config import TTSConfig

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


class KokoroSynthesizer:
    def __init__(self, config: TTSConfig):
        self._config = config
        self._model = None
        self._voices = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _download_file(self, url: str, dest: Path) -> None:
        print(f"[TTS] Downloading {dest.name}...")
        urllib.request.urlretrieve(url, str(dest))
        print(f"[TTS] Downloaded {dest.name}")

    def load(self) -> None:
        print("[TTS] Loading Kokoro model...")

        try:
            from kokoro_onnx import Kokoro

            model_dir = Path(self._config.model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / "kokoro-v1.0.onnx"
            voices_path = model_dir / "voices-v1.0.bin"

            if not model_path.exists():
                self._download_file(MODEL_URL, model_path)

            if not voices_path.exists():
                self._download_file(VOICES_URL, voices_path)

            print(f"[TTS] Loading from {self._config.model_dir}/")
            self._model = Kokoro(str(model_path), str(voices_path))

            self._loaded = True
            print(f"[TTS] Model loaded (default voice: {self._config.default_voice})")

        except Exception as e:
            print(f"[TTS] Failed to load model: {e}")
            raise

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        self._executor.shutdown(wait=False)

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None
    ) -> bytes:
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        voice = voice or self._config.default_voice
        speed = speed or self._config.speed

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._synthesize_sync,
            text,
            voice,
            speed
        )

    def _synthesize_sync(self, text: str, voice: str, speed: float) -> bytes:
        with self._lock:
            try:
                text = self._normalize_text(text)

                if not text:
                    return self._empty_wav()

                samples, sample_rate = self._model.create(
                    text,
                    voice=voice,
                    speed=speed,
                    lang="en-us"
                )

                return self._to_wav_bytes(samples, sample_rate)

            except Exception as e:
                print(f"[TTS] Synthesis error: {e}")
                raise

    def _normalize_text(self, text: str) -> str:
        text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())
        text = text.strip()

        if text and text[-1] not in '.!?':
            text = text + '.'

        return text

    def _to_wav_bytes(self, samples: np.ndarray, sample_rate: int) -> bytes:
        buffer = io.BytesIO()

        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        if samples.max() > 1.0 or samples.min() < -1.0:
            samples = samples / max(abs(samples.max()), abs(samples.min()))

        sf.write(buffer, samples, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()

    def _empty_wav(self) -> bytes:
        buffer = io.BytesIO()
        samples = np.zeros(100, dtype=np.float32)
        sf.write(buffer, samples, self._config.sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()

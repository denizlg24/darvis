import asyncio
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from darvis.transcription.base import (
    Transcriber,
    TranscriptionResult,
    TranscriptionStatus,
)


@dataclass
class HttpTranscriberConfig:
    host: str = "127.0.0.1"
    port: int = 8001
    timeout: float = 30.0
    retries: int = 3
    retry_delay: float = 0.5

    @classmethod
    def from_env(cls) -> "HttpTranscriberConfig":
        host = os.environ.get("TRANSCRIPTION_HOST", "127.0.0.1")
        port = int(os.environ.get("TRANSCRIPTION_PORT", "8001"))
        timeout = float(os.environ.get("TRANSCRIPTION_TIMEOUT", "30.0"))
        return cls(host=host, port=port, timeout=timeout)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class HttpTranscriber(Transcriber):
    def __init__(self, config: Optional[HttpTranscriberConfig] = None):
        self._config = config or HttpTranscriberConfig.from_env()
        self._client = None
        self._loaded = False
        self._model_info: dict = {}

    async def load(self) -> None:
        import httpx

        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=self._config.timeout
        )

        try:
            response = await self._client.get("/health")
            if response.status_code == 200:
                self._loaded = True
                model_response = await self._client.get("/model")
                if model_response.status_code == 200:
                    self._model_info = model_response.json()
                print(f"[TRANSCRIPTION] Connected to service at {self._config.base_url}")
            else:
                print(f"[TRANSCRIPTION] Service unhealthy: {response.status_code}")
                self._loaded = False
        except httpx.ConnectError:
            print(f"[TRANSCRIPTION] Cannot connect to service at {self._config.base_url}")
            self._loaded = False
        except Exception as e:
            print(f"[TRANSCRIPTION] Error connecting to service: {e}")
            self._loaded = False

    def unload(self) -> None:
        self._client = None
        self._loaded = False
        self._model_info = {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._client is not None

    @property
    def name(self) -> str:
        return "faster-whisper-http"

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> TranscriptionResult:
        import httpx

        if not self.is_loaded or self._client is None:
            return TranscriptionResult(
                status=TranscriptionStatus.ERROR,
                error="Transcription service not connected"
            )

        if len(audio) == 0:
            return TranscriptionResult(
                status=TranscriptionStatus.EMPTY,
                error="No audio data"
            )

        duration = len(audio) / sample_rate

        if audio.dtype == np.float32:
            audio_bytes = audio.tobytes()
            dtype = "float32"
        else:
            audio_float = audio.astype(np.float32)
            audio_bytes = audio_float.tobytes()
            dtype = "float32"

        last_error = None
        for attempt in range(self._config.retries):
            try:
                files = {"audio": ("audio.raw", audio_bytes, "application/octet-stream")}
                data = {"sample_rate": str(sample_rate), "dtype": dtype}

                response = await self._client.post(
                    "/transcribe",
                    files=files,
                    data=data
                )

                if response.status_code == 200:
                    result = response.json()
                    return self._parse_response(result, duration)
                elif response.status_code == 503:
                    return TranscriptionResult(
                        status=TranscriptionStatus.ERROR,
                        duration_seconds=duration,
                        error="Transcription service unavailable"
                    )
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"

            except httpx.TimeoutException:
                last_error = "Request timed out"
            except httpx.ConnectError:
                last_error = "Connection failed"
                self._loaded = False
                break
            except Exception as e:
                last_error = str(e)

            if attempt < self._config.retries - 1:
                await asyncio.sleep(self._config.retry_delay)

        return TranscriptionResult(
            status=TranscriptionStatus.ERROR,
            duration_seconds=duration,
            error=last_error or "Unknown error"
        )

    def _parse_response(
        self,
        response: dict,
        duration: float
    ) -> TranscriptionResult:
        status_str = response.get("status", "ERROR")

        status_map = {
            "SUCCESS": TranscriptionStatus.SUCCESS,
            "EMPTY": TranscriptionStatus.EMPTY,
            "CANCELLED": TranscriptionStatus.CANCELLED,
            "ERROR": TranscriptionStatus.ERROR
        }

        status = status_map.get(status_str, TranscriptionStatus.ERROR)

        return TranscriptionResult(
            status=status,
            text=response.get("text", ""),
            confidence=response.get("confidence", 0.0),
            duration_seconds=response.get("duration", duration),
            error=response.get("error")
        )

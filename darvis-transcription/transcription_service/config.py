import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class WhisperConfig:
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    model_path: Optional[str] = None
    download_root: str = "models/whisper"

    @classmethod
    def from_env(cls) -> "WhisperConfig":
        model_path = os.environ.get("DARVIS_WHISPER_MODEL")
        model_size = os.environ.get("DARVIS_WHISPER_SIZE", "small")
        device = os.environ.get("DARVIS_WHISPER_DEVICE", "cpu")
        compute_type = os.environ.get("DARVIS_WHISPER_COMPUTE", "int8")

        return cls(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            model_path=model_path
        )

    def get_model_path(self) -> Optional[str]:
        if self.model_path:
            path = Path(self.model_path)
            if path.exists():
                return str(path)

        return None


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8001

    @classmethod
    def from_env(cls) -> "ServerConfig":
        host = os.environ.get("TRANSCRIPTION_HOST", "127.0.0.1")
        port = int(os.environ.get("TRANSCRIPTION_PORT", "8001"))
        return cls(host=host, port=port)

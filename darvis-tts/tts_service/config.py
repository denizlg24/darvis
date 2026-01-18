import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TTSConfig:
    default_voice: str = "am_michael"
    model_dir: str = "models"
    sample_rate: int = 24000
    speed: float = 1.0

    @classmethod
    def from_env(cls) -> "TTSConfig":
        model_dir = os.environ.get("TTS_MODEL_DIR", "models")
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        return cls(
            default_voice=os.environ.get("TTS_DEFAULT_VOICE", "am_michael"),
            model_dir=model_dir,
            sample_rate=int(os.environ.get("TTS_SAMPLE_RATE", "24000")),
            speed=float(os.environ.get("TTS_SPEED", "1.0")),
        )


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8003

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            host=os.environ.get("TTS_HOST", "127.0.0.1"),
            port=int(os.environ.get("TTS_PORT", "8003")),
        )


AVAILABLE_VOICES = {
    "af_bella": "American Female - Bella",
    "af_sarah": "American Female - Sarah",
    "af_nicole": "American Female - Nicole",
    "af_sky": "American Female - Sky",
    "am_adam": "American Male - Adam",
    "am_michael": "American Male - Michael",
    "bf_emma": "British Female - Emma",
    "bf_isabella": "British Female - Isabella",
    "bm_george": "British Male - George",
    "bm_lewis": "British Male - Lewis",
}

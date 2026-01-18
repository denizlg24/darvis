import uvicorn

from tts_service.config import ServerConfig
from tts_service.server import app


def main():
    config = ServerConfig.from_env()

    print(f"[TTS] Starting server on {config.host}:{config.port}")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="warning"
    )


if __name__ == "__main__":
    main()

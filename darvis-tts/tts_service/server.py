from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from tts_service.config import TTSConfig, AVAILABLE_VOICES
from tts_service.synthesizer import KokoroSynthesizer


class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None


_synthesizer: Optional[KokoroSynthesizer] = None


def get_synthesizer() -> KokoroSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        raise RuntimeError("Synthesizer not initialized")
    return _synthesizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _synthesizer

    print("[TTS] Starting TTS service...")

    config = TTSConfig.from_env()
    _synthesizer = KokoroSynthesizer(config)

    try:
        _synthesizer.load()
        print("[TTS] Service ready")
        yield
    finally:
        print("[TTS] Shutting down TTS service...")
        if _synthesizer:
            _synthesizer.unload()
        print("[TTS] Service stopped")


app = FastAPI(
    title="DARVIS TTS Service",
    description="Kokoro TTS service for DARVIS",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    synthesizer = get_synthesizer()
    if not synthesizer.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.get("/voices")
async def list_voices():
    return {
        "voices": AVAILABLE_VOICES,
        "default": TTSConfig.from_env().default_voice
    }


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    synthesizer = get_synthesizer()

    if not synthesizer.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    if request.voice and request.voice not in AVAILABLE_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice: {request.voice}. Available: {list(AVAILABLE_VOICES.keys())}"
        )

    try:
        print(f"[TTS] Synthesizing: {request.text[:50]}{'...' if len(request.text) > 50 else ''}")

        audio_bytes = await synthesizer.synthesize(
            text=request.text,
            voice=request.voice,
            speed=request.speed
        )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=speech.wav"}
        )

    except Exception as e:
        print(f"[TTS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

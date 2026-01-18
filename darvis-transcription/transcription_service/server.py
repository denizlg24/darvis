from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from transcription_service.config import WhisperConfig
from transcription_service.transcriber import WhisperTranscriber, TranscriptionStatus


_transcriber: Optional[WhisperTranscriber] = None


def get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        raise RuntimeError("Transcriber not initialized")
    return _transcriber


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _transcriber

    print("[TRANSCRIPTION] Starting transcription service...")

    config = WhisperConfig.from_env()
    _transcriber = WhisperTranscriber(config)

    try:
        _transcriber.load()
        print("[TRANSCRIPTION] Service ready")
        yield
    finally:
        print("[TRANSCRIPTION] Shutting down transcription service...")
        if _transcriber:
            _transcriber.unload()
        print("[TRANSCRIPTION] Service stopped")


app = FastAPI(
    title="DARVIS Transcription Service",
    description="Faster-whisper transcription service for DARVIS",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    transcriber = get_transcriber()
    if not transcriber.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.get("/model")
async def model_info():
    transcriber = get_transcriber()
    if not transcriber.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return transcriber.model_info


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    sample_rate: int = Form(default=16000),
    dtype: str = Form(default="float32")
):
    transcriber = get_transcriber()

    if not transcriber.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        audio_bytes = await audio.read()

        if dtype == "float32":
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        elif dtype == "int16":
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported dtype: {dtype}")

        if len(audio_array) == 0:
            return JSONResponse(content={
                "status": "EMPTY",
                "text": "",
                "confidence": 0.0,
                "duration": 0.0,
                "error": "No audio data"
            })

        print(f"[TRANSCRIPTION] Processing {len(audio_array)/sample_rate:.2f}s of audio...")

        result = transcriber.transcribe(audio_array, sample_rate)

        return JSONResponse(content=result.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        print(f"[TRANSCRIPTION] Error processing request: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "ERROR",
                "text": "",
                "confidence": 0.0,
                "duration": 0.0,
                "error": str(e)
            }
        )

"""Persistent model server.

Holds all heavy weights (Praxy + XTTS TTS, faster-whisper STT) in memory
across app-backend restarts, so code changes to the main API never pay the
~10-minute model-load cost again. Runs on port 8002, alongside vLLM (8001).

Start (from D:\\voice-clone-agent\\backend):
    python -m uvicorn model_server:app --host 0.0.0.0 --port 8002

The main backend (port 8000) talks to this server over localhost.
"""
from __future__ import annotations

# Offline-first: never contact the HF Hub. All weights are cached locally.
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio
import io
import wave
from typing import Optional

import numpy as np
from fastapi import Body, FastAPI, Query
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel

from app.model_engines import RealSTTWorker, RealTTSWorker

app = FastAPI(title="Voice Clone Model Server")

_stt: Optional[RealSTTWorker] = None
_tts: Optional[RealTTSWorker] = None


def get_stt() -> RealSTTWorker:
    global _stt
    if _stt is None:
        _stt = RealSTTWorker()
    return _stt


def get_tts() -> RealTTSWorker:
    global _tts
    if _tts is None:
        _tts = RealTTSWorker()
    return _tts


class TTSRequest(BaseModel):
    text: str
    speaker_ref_path: Optional[str] = None
    language: Optional[str] = None


@app.get("/health")
async def health():
    tts = get_tts()
    return {
        "status": "ok",
        "praxy_loaded": tts._praxy_loaded,
        "xtts_loaded": tts._xtts is not None,
        "edge_enabled": tts._edge_voices is not None,
    }


@app.post("/v1/stt")
async def stt(
    pcm: bytes = Body(...),
    model: str = Query("medium"),
    language: str = Query("te"),
):
    arr = np.frombuffer(pcm, dtype=np.int16)
    res = await get_stt().transcribe_pcm(arr, language=language, model_size=model)
    return {
        "text": res.text,
        "language": res.language,
        "start": res.start,
        "end": res.end,
        "latency_ms": res.latency_ms,
        "model": model,
    }


class SttFileRequest(BaseModel):
    path: str
    model: str = "large-v3"
    language: Optional[str] = None


@app.post("/v1/stt_file")
async def stt_file(req: SttFileRequest):
    segments = []
    async for res in get_stt().transcribe_file(
        req.path, language=req.language, model_size=req.model
    ):
        segments.append({
            "t0": res.start,
            "t1": res.end,
            "text": res.text,
            "language": res.language,
            "latency_ms": res.latency_ms,
        })
    return {"segments": segments}


@app.post("/v1/tts")
async def tts(req: TTSRequest):
    res = await get_tts().synthesize(
        text=req.text,
        speaker_ref_wav=req.speaker_ref_path,
        language=req.language,
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(res.sample_rate)
        wf.writeframes(res.audio.tobytes())
    return Response(
        buf.getvalue(),
        media_type="audio/wav",
        headers={
            "X-TTS-Engine": res.engine,
            "X-TTS-Latency-ms": str(int(res.latency_ms)),
        },
    )


async def _do_prewarm() -> None:
    """Load every engine once, in the background, right after startup."""
    try:
        await get_tts().preload()
        logger.info("Model server: TTS preloaded (praxy/xtts)")
    except Exception as e:
        logger.warning(f"Model server: TTS preload failed: {e}")
    try:
        for size in ("medium", "small"):
            async with get_stt()._lock:
                await asyncio.to_thread(get_stt()._ensure_model, size)
        logger.info("Model server: STT preloaded (medium, small)")
    except Exception as e:
        logger.warning(f"Model server: STT preload failed: {e}")


@app.post("/v1/prewarm")
async def prewarm():
    asyncio.create_task(_do_prewarm())
    return {"status": "warming"}


@app.on_event("startup")
async def _startup_prewarm():
    asyncio.create_task(_do_prewarm())

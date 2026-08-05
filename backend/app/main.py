"""FastAPI application entry point.

Exposes:
- REST endpoints for speakers, calls, TTS, settings, health
- WebSocket endpoint for live call streaming
- Chunked upload endpoint for long audio (300+ minutes)
"""
from __future__ import annotations

# Offline-first: never contact the HF Hub at import/runtime. All weights are
# cached under D:\hf-models (HF_HOME). Applies to diffusers/transformers/
# huggingface_hub/faster-whisper (override with HF_HUB_OFFLINE=0 if needed).
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio
import base64
import json
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import (
    FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket,
    Depends, Header,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from .config import SETTINGS
from .models import (
    SpeakerCreate, SpeakerOut, SpeakerMeta,
    CallCreate, CallOut, SegmentOut,
    TTSRequest,
    ChunkedUploadInit, ChunkedUploadInitResponse, ChunkedUploadComplete,
    HealthOut, SettingsOut,
)
from .speaker_registry import SpeakerRegistry
from .call_logger import CallLogger, _safe_call_id
from .vad_worker import VADWorker
from .stt_worker import STTWorker
from .llm_worker import LLMWorker
from .tts_worker import TTSWorker
from .pipeline import Pipeline
from .websocket_handler import ConnectionManager, handle_call_websocket


# --- Singletons ------------------------------------------------------------

_speaker_reg = SpeakerRegistry()
_call_logger = CallLogger()

# Workers are created lazily on first use to keep startup fast and avoid
# loading all models into VRAM at boot.
_vad: Optional[VADWorker] = None
_stt: Optional[STTWorker] = None
_llm: Optional[LLMWorker] = None
_tts: Optional[TTSWorker] = None
_pipeline: Optional[Pipeline] = None
_ws_manager: Optional[ConnectionManager] = None


def get_vad() -> VADWorker:
    global _vad
    if _vad is None:
        _vad = VADWorker()
    return _vad


def get_stt() -> STTWorker:
    global _stt
    if _stt is None:
        _stt = STTWorker()
    return _stt


def get_llm() -> LLMWorker:
    global _llm
    if _llm is None:
        _llm = LLMWorker()
    return _llm


def get_tts() -> TTSWorker:
    global _tts
    if _tts is None:
        _tts = TTSWorker()
    return _tts


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(
            vad=get_vad(),
            stt=get_stt(),
            llm=get_llm(),
            tts=get_tts(),
            speaker_registry=_speaker_reg,
            call_logger=_call_logger,
        )
    return _pipeline


def get_ws_manager() -> ConnectionManager:
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = ConnectionManager(SETTINGS.max_concurrent_calls)
    return _ws_manager


# --- Lifespan --------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    SETTINGS.ensure_dirs()
    logger.info(f"Voice Clone Agent starting | mode={SETTINGS.pipeline_mode} device={SETTINGS.device} tts={SETTINGS.tts_engine}")
    logger.info(f"LLM enabled: {SETTINGS.llm_enabled}")
    # Heavy weights live in the persistent model server (port 8002); ask it
    # to warm up in the background. Nothing blocks startup here.
    preload_task = asyncio.create_task(_preload_engines())
    yield
    # Cleanup
    preload_task.cancel()
    try:
        if _llm is not None:
            await _llm.close()
    except Exception:
        pass


async def _preload_engines() -> None:
    """Ask the persistent model server to warm up its engines."""
    try:
        from .model_client import get_model_client
        ok = await get_model_client().prewarm()
        if ok:
            logger.info("Model server prewarm requested")
        else:
            logger.warning("Model server not reachable on :8002 — start it first (models will lazy-load)")
    except Exception as e:
        logger.warning(f"Model server prewarm error: {e}")


# --- App -------------------------------------------------------------------

app = FastAPI(
    title="Voice Clone Agent",
    version="0.1.0",
    description="Telugu/English zero-shot voice cloning agent (RTX 3060 6GB)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth ------------------------------------------------------------------

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if SETTINGS.auth_mode == "apikey":
        if not x_api_key or x_api_key != SETTINGS.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# --- Health ----------------------------------------------------------------

def _vram_stats() -> tuple[int, int, int]:
    """Return (total_mb, used_mb, free_mb)."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            used = torch.cuda.memory_allocated()
            free = total - used
            return int(total // (1024 * 1024)), int(used // (1024 * 1024)), int(free // (1024 * 1024))
    except Exception:
        pass
    return 0, 0, 0


@app.get("/health", response_model=HealthOut)
async def health():
    total, used, free = _vram_stats()
    ms = "down"
    try:
        from .model_client import get_model_client
        mh = await get_model_client().health()
        if mh:
            ms = "up" if (mh.get("praxy_loaded") or mh.get("xtts_loaded")) else "warming"
    except Exception:
        pass
    return HealthOut(
        status="ok",
        device=SETTINGS.device,
        pipeline_mode=SETTINGS.pipeline_mode,
        llm_enabled=SETTINGS.llm_enabled,
        tts_engine=SETTINGS.tts_engine,
        vram_total_mb=total,
        vram_used_mb=used,
        vram_free_mb=free,
        speakers_count=len(_speaker_reg.list_speakers()),
        calls_count=await _call_logger.count(),
        model_server_status=ms,
    )


@app.get("/settings", response_model=SettingsOut)
async def get_settings_endpoint(_: bool = Depends(verify_api_key)):
    total, used, free = _vram_stats()
    return SettingsOut(
        pipeline_mode=SETTINGS.pipeline_mode,
        max_concurrent_calls=SETTINGS.max_concurrent_calls,
        stt_model=SETTINGS.stt_model,
        stt_language=SETTINGS.stt_language,
        tts_engine=SETTINGS.tts_engine,
        xtts_language=SETTINGS.xtts_language,
        llm_enabled=SETTINGS.llm_enabled,
        llm_model=SETTINGS.llm_model,
        device=SETTINGS.device,
        vram_total_mb=total,
        vram_used_mb=used,
        vram_free_mb=free,
    )


# --- Speakers --------------------------------------------------------------

@app.get("/speakers", response_model=list[SpeakerOut])
async def list_speakers(_: bool = Depends(verify_api_key)):
    out = []
    for meta in _speaker_reg.list_speakers():
        out.append(SpeakerOut(
            speaker_id=meta["speaker_id"],
            display_name=meta["display_name"],
            language=meta["language"],
            ref_duration_s=meta.get("ref_duration_s", 0.0),
            prompt_text=meta.get("prompt_text") or None,
            created_at=meta["created_at"],
        ))
    return out


@app.post("/speakers", response_model=SpeakerOut)
async def create_speaker(
    speaker_id: str = Form(...),
    display_name: str = Form(...),
    language: str = Form("te"),
    ref_audio: UploadFile = File(...),
    _: bool = Depends(verify_api_key),
):
    if _speaker_reg.exists(speaker_id):
        raise HTTPException(status_code=409, detail=f"Speaker '{speaker_id}' already exists")
    suffix = Path(ref_audio.filename or "").suffix.lower().lstrip(".") or "wav"
    raw = await ref_audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty reference audio")
    meta = _speaker_reg.create(
        speaker_id=speaker_id,
        display_name=display_name,
        language=language,
        ref_audio_bytes=raw,
        ref_audio_format=suffix,
    )
    prompt_text = None
    try:
        ref = _speaker_reg.get_ref_wav(meta["speaker_id"])
        if ref is not None:
            stt = get_stt()
            parts = []
            async for res in stt.transcribe_file(str(ref), language=language or "auto"):
                parts.append(res.text)
            prompt_text = " ".join(parts).strip()
            _speaker_reg.update_meta(meta["speaker_id"], prompt_text=prompt_text)
    except Exception as e:
        logger.warning(f"Auto-transcription failed for speaker {meta['speaker_id']}: {e}")
    return SpeakerOut(
        speaker_id=meta["speaker_id"],
        display_name=meta["display_name"],
        language=meta["language"],
        ref_duration_s=meta["ref_duration_s"],
        prompt_text=prompt_text,
        created_at=meta["created_at"],
    )


@app.get("/speakers/{speaker_id}", response_model=SpeakerMeta)
async def get_speaker(speaker_id: str, _: bool = Depends(verify_api_key)):
    meta = _speaker_reg.get(speaker_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return SpeakerMeta(**meta)


@app.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str, _: bool = Depends(verify_api_key)):
    ok = _speaker_reg.delete(speaker_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"ok": True}


@app.get("/speakers/{speaker_id}/ref.wav")
async def download_speaker_ref(speaker_id: str, _: bool = Depends(verify_api_key)):
    ref = _speaker_reg.get_ref_wav(speaker_id)
    if ref is None or not ref.exists():
        raise HTTPException(status_code=404, detail="Reference audio not found")
    return FileResponse(str(ref), media_type="audio/wav", filename=f"{speaker_id}_ref.wav")


# --- TTS -------------------------------------------------------------------

@app.post("/tts/synthesize")
async def tts_synthesize(
    req: TTSRequest,
    _: bool = Depends(verify_api_key),
):
    """Synthesize a single utterance with voice cloning. Returns WAV bytes."""
    spk = _speaker_reg.get(req.speaker_id)
    if spk is None:
        raise HTTPException(status_code=404, detail="Speaker not found")
    ref = _speaker_reg.get_ref_wav(req.speaker_id)
    if ref is None or not ref.exists():
        raise HTTPException(status_code=400, detail="Speaker has no reference audio")
    tts = get_tts()
    res = await tts.synthesize(
        text=req.text,
        speaker_ref_wav=ref,
        language=req.language or spk.get("language", SETTINGS.xtts_language),
    )
    # Write to a temp WAV and return as file
    import wave
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=str(SETTINGS.calls_dir))
    tmp.close()
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(res.sample_rate)
            wf.writeframes(res.audio.tobytes())
    except Exception:
        Path(tmp.name).unlink(missing_ok=True)
        raise
    return FileResponse(
        tmp.name, media_type="audio/wav",
        filename=f"tts_{req.speaker_id}.wav",
        headers={"X-TTS-Latency-ms": str(int(res.latency_ms)),
                 "X-TTS-Engine": res.engine},
        background=BackgroundTask(_cleanup_tmp_file, tmp.name),
    )


def _cleanup_tmp_file(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


# --- Calls -----------------------------------------------------------------

@app.get("/calls", response_model=list[CallOut])
async def list_calls(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _: bool = Depends(verify_api_key),
):
    rows = await _call_logger.list_calls(limit=limit, offset=offset)
    return [CallOut(
        call_id=r["call_id"],
        speaker_id=r["speaker_id"],
        title=r.get("title"),
        started_at=r["started_at"],
        ended_at=r.get("ended_at"),
        duration_s=r.get("duration_s", 0.0),
        audio_path=r.get("audio_path"),
        transcript_path=r.get("transcript_path"),
        segment_count=r.get("segment_count", 0),
    ) for r in rows]


@app.get("/calls/{call_id}", response_model=CallOut)
async def get_call(call_id: str, _: bool = Depends(verify_api_key)):
    r = await _call_logger.get_call(call_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return CallOut(
        call_id=r["call_id"],
        speaker_id=r["speaker_id"],
        title=r.get("title"),
        started_at=r["started_at"],
        ended_at=r.get("ended_at"),
        duration_s=r.get("duration_s", 0.0),
        audio_path=r.get("audio_path"),
        transcript_path=r.get("transcript_path"),
        segment_count=r.get("segment_count", 0),
    )


@app.get("/calls/{call_id}/transcript", response_model=list[SegmentOut])
async def get_call_transcript(call_id: str, _: bool = Depends(verify_api_key)):
    try:
        segs = await _call_logger.get_transcript(call_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [SegmentOut(**s) for s in segs]


@app.get("/calls/{call_id}/audio")
async def get_call_audio(call_id: str, _: bool = Depends(verify_api_key)):
    r = await _call_logger.get_call(call_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Call not found")
    audio_path = r.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=404, detail="Audio not available for this call")
    return FileResponse(audio_path, media_type="audio/wav", filename=f"{call_id}.wav")


@app.delete("/calls/{call_id}")
async def delete_call(call_id: str, _: bool = Depends(verify_api_key)):
    try:
        await _call_logger.delete_call(call_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


# --- Chunked upload (300-min audio) ---------------------------------------

# In-memory tracking of partial uploads. For production scale, use Redis.
_uploads: dict[str, dict] = {}


@app.post("/upload/chunked/init", response_model=ChunkedUploadInitResponse)
async def chunked_upload_init(
    req: ChunkedUploadInit,
    _: bool = Depends(verify_api_key),
):
    upload_id = f"up_{uuid.uuid4().hex[:12]}"
    try:
        call_id = _safe_call_id(req.call_id or f"call_{uuid.uuid4().hex[:12]}")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid call_id")
    tmp_dir = SETTINGS.calls_dir / call_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _uploads[upload_id] = {
        "call_id": call_id,
        "filename": req.filename,
        "total_chunks": req.total_chunks,
        "received": 0,
        "chunks": [None] * req.total_chunks,
        "tmp_dir": tmp_dir,
    }
    # Pre-create call record (use the SAME call_id so segments don't orphan)
    await _call_logger.create_call(speaker_id="upload", title=req.filename, call_id=call_id)
    return ChunkedUploadInitResponse(upload_id=upload_id, call_id=call_id)


@app.post("/upload/chunked/{upload_id}/{chunk_index}")
async def chunked_upload_receive(
    upload_id: str,
    chunk_index: int,
    chunk: UploadFile = File(...),
    _: bool = Depends(verify_api_key),
):
    if upload_id not in _uploads:
        raise HTTPException(status_code=404, detail="Unknown upload_id")
    up = _uploads[upload_id]
    if chunk_index < 0 or chunk_index >= up["total_chunks"]:
        raise HTTPException(status_code=400, detail="Invalid chunk index")
    raw = await chunk.read()
    chunk_path = up["tmp_dir"] / f"chunk_{chunk_index:05d}.bin"
    chunk_path.write_bytes(raw)
    up["chunks"][chunk_index] = chunk_path
    up["received"] += 1
    return {"received": up["received"], "total": up["total_chunks"]}


@app.post("/upload/chunked/complete")
async def chunked_upload_complete(
    req: ChunkedUploadComplete,
    _: bool = Depends(verify_api_key),
):
    if req.upload_id not in _uploads:
        raise HTTPException(status_code=404, detail="Unknown upload_id")
    up = _uploads[req.upload_id]
    if up["received"] != up["total_chunks"]:
        raise HTTPException(
            status_code=400,
            detail=f"Missing chunks: got {up['received']} of {up['total_chunks']}",
        )
    # Concatenate chunks into final audio file
    final_path = up["tmp_dir"] / "audio.wav"
    with open(final_path, "wb") as fout:
        for cp in up["chunks"]:
            if cp is None:
                continue
            with open(cp, "rb") as fin:
                while True:
                    buf = fin.read(1024 * 1024)
                    if not buf:
                        break
                    fout.write(buf)
    # Clean up chunk files
    for cp in up["chunks"]:
        if cp is not None and cp.exists():
            cp.unlink()

    call_id = up["call_id"]

    # Kick off background transcription (STT-only)
    speaker_id = req.speaker_id or "upload"
    title = req.title or up.get("filename", "Uploaded audio")

    asyncio.create_task(_run_bg_transcription(call_id, str(final_path), speaker_id, title))

    # Free the in-memory upload tracking
    del _uploads[req.upload_id]
    return {"call_id": call_id, "audio_path": str(final_path), "transcription": "started"}


async def _run_bg_transcription(call_id: str, audio_path: str, speaker_id: str, title: str):
    try:
        pipeline = get_pipeline()
        await pipeline.transcribe_file_streaming(
            audio_path=audio_path,
            call_id=call_id,
            speaker_id=speaker_id,
            stt_model="large-v3",  # accuracy-first for uploads; slow CPU is fine here
        )
    except Exception as e:
        logger.exception(f"Background transcription failed for {call_id}: {e}")


# --- Analyze (long audio -> STT -> LLM -> TTS voice clone) -----------------

@app.post("/analyze")
async def analyze_audio(
    speaker_id: str = Form(...),
    audio: UploadFile = File(...),
    title: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    _: bool = Depends(verify_api_key),
):
    """Analyze an audio file end-to-end: STT -> LLM -> TTS (voice clone).

    Transcribes the whole file, generates ONE intelligent reply, and
    synthesizes a spoken bot response. Returns the call_id immediately; the
    analysis runs in the background (poll GET /calls/{call_id}/transcript).
    """
    spk = _speaker_reg.get(speaker_id)
    if spk is None:
        raise HTTPException(status_code=404, detail="Speaker not found")
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")

    call_id = f"call_{uuid.uuid4().hex[:12]}"
    tmp_dir = SETTINGS.calls_dir / call_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    in_path = tmp_dir / "input_audio.wav"
    in_path.write_bytes(raw)

    asyncio.create_task(
        _run_bg_analyze(call_id, str(in_path), speaker_id, title, language)
    )
    return {"call_id": call_id, "audio_path": str(in_path), "analysis": "started"}


async def _run_bg_analyze(
    call_id: str, audio_path: str, speaker_id: str,
    title: Optional[str], language: Optional[str],
):
    try:
        pipeline = get_pipeline()
        await pipeline.analyze_file(
            audio_path=audio_path,
            speaker_id=speaker_id,
            call_id=call_id,
            title=title or os.path.basename(audio_path),
            language=language,
        )
    except Exception as e:
        logger.exception(f"Background analysis failed for {call_id}: {e}")


# --- WebSocket -------------------------------------------------------------

@app.websocket("/ws/call")
async def ws_call(ws: WebSocket):
    await handle_call_websocket(ws, get_pipeline(), get_ws_manager())


# --- Root ------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "name": "Voice Clone Agent",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws/call",
    }

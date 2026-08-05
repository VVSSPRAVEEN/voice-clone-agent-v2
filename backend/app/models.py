"""Pydantic schemas for REST + WebSocket messages."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# --- Speakers ---------------------------------------------------------------

class SpeakerCreate(BaseModel):
    speaker_id: str = Field(..., description="Unique speaker identifier")
    display_name: str = Field(..., description="Human-readable name")
    language: str = Field("te", description="Default TTS language")


class SpeakerMeta(BaseModel):
    speaker_id: str
    display_name: str
    language: str
    ref_audio_path: str
    ref_duration_s: float
    prompt_text: Optional[str] = None
    created_at: datetime


class SpeakerOut(BaseModel):
    speaker_id: str
    display_name: str
    language: str
    ref_duration_s: float
    prompt_text: Optional[str] = None
    created_at: datetime


# --- Calls ------------------------------------------------------------------

class CallCreate(BaseModel):
    speaker_id: str
    title: Optional[str] = None


class CallOut(BaseModel):
    call_id: str
    speaker_id: str
    title: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    duration_s: float
    audio_path: Optional[str]
    transcript_path: Optional[str]
    segment_count: int


class SegmentOut(BaseModel):
    t0: float
    t1: float
    speaker: str
    text: str
    is_final: bool


# --- TTS --------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    speaker_id: str
    language: Optional[str] = None


# --- STT (chunked upload) ---------------------------------------------------

class ChunkedUploadInit(BaseModel):
    filename: str
    total_chunks: int
    call_id: Optional[str] = None


class ChunkedUploadInitResponse(BaseModel):
    upload_id: str
    call_id: str


class ChunkedUploadComplete(BaseModel):
    upload_id: str
    speaker_id: Optional[str] = None
    title: Optional[str] = None


# --- WebSocket messages -----------------------------------------------------

class WSClientHello(BaseModel):
    speaker_id: str
    call_id: Optional[str] = None
    title: Optional[str] = None
    sample_rate: int = 16000


class WSServerMessage(BaseModel):
    """Envelope for messages server -> client."""
    type: Literal[
        "hello",          # ack of ClientHello
        "transcript",     # partial or final STT
        "llm",            # LLM text response
        "audio",          # TTS audio chunk (base64 PCM)
        "audio_end",      # end of TTS for a turn
        "error",          # error message
        "vad",            # VAD event (speech_start / speech_end)
        "status",         # generic status
        "call_end",       # call ended, here is call_id
    ]
    data: dict | None = None
    call_id: Optional[str] = None


# --- Health -----------------------------------------------------------------

class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    device: str
    pipeline_mode: str
    llm_enabled: bool
    tts_engine: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    speakers_count: int
    calls_count: int


# --- Settings snapshot (read-only) -----------------------------------------

class SettingsOut(BaseModel):
    pipeline_mode: str
    max_concurrent_calls: int
    stt_model: str
    stt_language: str
    tts_engine: str
    xtts_language: str
    llm_enabled: bool
    llm_model: str
    device: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int

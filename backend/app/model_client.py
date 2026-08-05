"""HTTP client for the persistent model server (port 8002).

The app backend never loads TTS/STT weights itself anymore — it proxies
through this client, so restarting the API is instant. The model server
holds Praxy/XTTS/faster-whisper in memory and survives restarts.
"""
from __future__ import annotations

import io
import wave
from typing import Optional

import httpx
import numpy as np
from loguru import logger

from .stt_worker import STTResult
from .tts_worker import TTSResult

_MODEL_SERVER_URL = "http://127.0.0.1:8002"

_client: Optional["httpx.AsyncClient"] = None


def get_model_client() -> "ModelClient":
    global _client
    if _client is None:
        _client = ModelClient(_MODEL_SERVER_URL)
    return _client


class ModelClient:
    def __init__(self, base_url: str = _MODEL_SERVER_URL):
        self.base_url = base_url
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(3600.0, connect=3.0),  # large-v3 uploads take long
        )

    async def health(self) -> Optional[dict]:
        try:
            r = await self._http.get("/health", timeout=3.0)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    async def stt(
        self,
        pcm_int16: np.ndarray,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
    ) -> STTResult:
        r = await self._http.post(
            "/v1/stt",
            content=pcm_int16.tobytes(),
            params={
                "model": model_size or "medium",
                "language": language or "te",
            },
        )
        r.raise_for_status()
        d = r.json()
        return STTResult(
            text=d["text"],
            language=d.get("language", ""),
            start=d.get("start", 0.0),
            end=d.get("end", 0.0),
            latency_ms=d.get("latency_ms", 0.0),
        )

    async def stt_file(
        self,
        audio_path: str,
        model_size: Optional[str] = "large-v3",
        language: Optional[str] = None,
    ) -> list[dict]:
        r = await self._http.post("/v1/stt_file", json={
            "path": audio_path,
            "model": model_size,
            "language": language,
        })
        r.raise_for_status()
        return r.json().get("segments", [])

    async def tts(
        self,
        text: str,
        speaker_ref_wav: Optional[str],
        language: Optional[str] = None,
    ) -> TTSResult:
        r = await self._http.post("/v1/tts", json={
            "text": text,
            "speaker_ref_path": speaker_ref_wav,
            "language": language,
        })
        r.raise_for_status()
        with wave.open(io.BytesIO(r.content), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        return TTSResult(
            audio=np.frombuffer(raw, dtype=np.int16),
            sample_rate=sr,
            latency_ms=float(r.headers.get("X-TTS-Latency-ms", 0)),
            engine=r.headers.get("X-TTS-Engine", "remote"),
        )

    async def prewarm(self) -> bool:
        try:
            r = await self._http.post("/v1/prewarm", timeout=3.0)
            return r.status_code == 200
        except Exception as e:
            logger.warning(f"Model server prewarm failed: {e}")
            return False

"""STT worker proxy (app side).

Weights live in the persistent model server (port 8002); this class just
forwards calls over localhost via :mod:`app.model_client`. Keeps the same
public API (``STTResult``, ``transcribe_pcm``, ``transcribe_file``) so
pipelines and endpoints are unchanged.
"""
from __future__ import annotations

from typing import AsyncIterator, Optional

import numpy as np

from .model_engines import STTResult  # noqa: F401  (re-export for callers)


class STTWorker:
    """Remote proxy for faster-whisper (runs in the model server)."""

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
        beam_size: int | None = None,
    ):
        from .config import SETTINGS
        self.model_size = model_size or SETTINGS.stt_model
        self.language = language or SETTINGS.stt_language
        self.beam_size = beam_size if beam_size is not None else SETTINGS.stt_beam_size

    async def transcribe_pcm(
        self,
        pcm_int16: np.ndarray,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
    ) -> STTResult:
        from .model_client import get_model_client
        return await get_model_client().stt(
            pcm_int16,
            language=language,
            model_size=model_size or self.model_size,
        )

    async def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        chunk_seconds: float = 30.0,
        model_size: Optional[str] = None,
    ) -> AsyncIterator[STTResult]:
        from .model_client import get_model_client
        segments = await get_model_client().stt_file(
            audio_path,
            model_size=model_size,
            language=language,
        )
        for seg in segments:
            yield STTResult(
                text=seg["text"],
                language=seg.get("language", ""),
                start=seg.get("t0", 0.0),
                end=seg.get("t1", 0.0),
                latency_ms=seg.get("latency_ms", 0.0),
            )

    def unload(self) -> None:
        # Models live in the model server; nothing to unload here.
        pass

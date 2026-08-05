"""TTS worker proxy (app side).

Weights live in the persistent model server (port 8002); this class just
forwards calls over localhost via :mod:`app.model_client`. Keeps the same
public API (``TTSResult``, ``synthesize``, ``preload``) so pipelines and
endpoints are unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .model_engines import TTSResult  # noqa: F401  (re-export for callers)


class TTSWorker:
    """Remote proxy for the TTS engines (run in the model server)."""

    def __init__(self, engine: str | None = None):
        from .config import SETTINGS
        self.engine = (engine or SETTINGS.tts_engine).lower()
        self._praxy_loaded = False
        self._praxy_failed = False

    @property
    def praxy_ready(self) -> bool:
        # The model server owns praxy; assume ready unless told otherwise.
        return not self._praxy_failed

    async def preload(self) -> None:
        from .model_client import get_model_client
        await get_model_client().prewarm()

    async def synthesize(
        self,
        text: str,
        speaker_ref_wav: str | Path,
        language: Optional[str] = None,
    ) -> TTSResult:
        from .model_client import get_model_client
        return await get_model_client().tts(
            text,
            speaker_ref_wav=str(speaker_ref_wav) if speaker_ref_wav else None,
            language=language,
        )

    def unload(self) -> None:
        # Models live in the model server; nothing to unload here.
        pass

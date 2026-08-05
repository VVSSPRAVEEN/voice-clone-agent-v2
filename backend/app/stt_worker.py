"""faster-whisper STT worker.

Supports both streaming (one segment at a time) and batch (whole audio
file) transcription. Memory-aware: can be unloaded between calls to free
VRAM for the TTS stage.
"""
from __future__ import annotations

import asyncio
import gc
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np
from loguru import logger

from .config import SETTINGS
from .gpu_utils import should_use_cuda


@dataclass
class STTResult:
    text: str
    language: str
    start: float
    end: float
    latency_ms: float


class STTWorker:
    """faster-whisper wrapper."""

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
        beam_size: int | None = None,
    ):
        self.model_size = model_size or SETTINGS.stt_model
        if device is not None:
            self.device = device
        elif SETTINGS.stt_device_cuda and should_use_cuda(1400):
            self.device = "cuda"
        else:
            self.device = "cpu"
            if SETTINGS.stt_device_cuda:
                logger.warning("STT: insufficient free GPU memory; falling back to CPU")
        self.compute_type = compute_type or SETTINGS.stt_compute_type
        self.language = language or SETTINGS.stt_language
        self.beam_size = beam_size if beam_size is not None else SETTINGS.stt_beam_size
        self._model = None
        self._lock = asyncio.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        logger.info(
            f"Loading faster-whisper model={self.model_size} "
            f"device={self.device} compute_type={self.compute_type}"
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(SETTINGS.models_dir / "faster-whisper"),
        )
        logger.info("faster-whisper loaded")

    async def transcribe_pcm(
        self,
        pcm_int16: np.ndarray,
        language: Optional[str] = None,
    ) -> STTResult:
        """Transcribe a 16 kHz int16 PCM segment."""
        async with self._lock:
            await asyncio.to_thread(self._ensure_model)
            audio_f32 = pcm_int16.astype(np.float32) / 32768.0
            t0 = time.perf_counter()
            lang_arg = language or self.language
            if lang_arg == "auto":
                lang_arg = None

            def _transcribe() -> tuple[str, str, float, float, list]:
                # Run + fully consume the lazy generator off the event loop so
                # the decode never blocks the asyncio loop.
                segments, info = self._model.transcribe(
                    audio_f32,
                    language=lang_arg,
                    beam_size=self.beam_size,
                    vad_filter=False,  # we already VAD upstream
                )
                text_parts = []
                t_start = None
                t_end = None
                for seg in segments:
                    text_parts.append(seg.text.strip())
                    if t_start is None or seg.start < t_start:
                        t_start = seg.start
                    if t_end is None or seg.end > t_end:
                        t_end = seg.end
                return (" ".join(text_parts).strip(),
                        info.language, t_start or 0.0, t_end or 0.0,
                        text_parts)

            text, det_lang, start, end, _ = await asyncio.to_thread(_transcribe)
            elapsed = (time.perf_counter() - t0) * 1000
            return STTResult(
                text=text,
                language=det_lang if lang_arg is None else lang_arg,
                start=start,
                end=end,
                latency_ms=elapsed,
            )

    async def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        chunk_seconds: float = 30.0,
    ) -> AsyncIterator[STTResult]:
        """Stream-transcribe a long audio file in chunks.

        Yields STTResult objects as they complete. Never loads the full
        audio into memory at once.
        """
        import soundfile as sf
        async with self._lock:
            await asyncio.to_thread(self._ensure_model)
            # Use soundfile to get info first
            info = sf.info(audio_path)
            sr = info.samplerate
            total = info.frames
            chunk_samples = int(chunk_seconds * sr)
            lang_arg = language or self.language
            if lang_arg == "auto":
                lang_arg = None

            offset = 0
            while offset < total:
                # Read one chunk
                audio_f32, _ = sf.read(
                    audio_path,
                    start=offset,
                    frames=chunk_samples,
                    dtype="float32",
                    always_2d=False,
                )
                if audio_f32.ndim > 1:
                    audio_f32 = audio_f32.mean(axis=1)
                # Resample to 16 kHz if needed
                if sr != 16000:
                    import librosa
                    audio_f32 = librosa.resample(
                        audio_f32, orig_sr=sr, target_sr=16000
                    )
                t0 = time.perf_counter()

                def _transcribe_chunk() -> tuple[str, str, float, float]:
                    # Run + fully consume the lazy generator in a thread so the
                    # decode never blocks the asyncio loop.
                    segments, stt_info = self._model.transcribe(
                        audio_f32,
                        language=lang_arg,
                        beam_size=self.beam_size,
                        vad_filter=True,
                    )
                    text_parts = []
                    t_start = None
                    t_end = None
                    for seg in segments:
                        text_parts.append(seg.text.strip())
                        if t_start is None or seg.start < t_start:
                            t_start = seg.start
                        if t_end is None or seg.end > t_end:
                            t_end = seg.end
                    return (" ".join(text_parts).strip(),
                            stt_info.language, t_start or 0.0, t_end or 0.0)

                text, det_lang, t_start, t_end = await asyncio.to_thread(_transcribe_chunk)
                elapsed = (time.perf_counter() - t0) * 1000
                yield STTResult(
                    text=text,
                    language=det_lang if lang_arg is None else lang_arg,
                    start=(offset / sr) + t_start,
                    end=(offset / sr) + t_end,
                    latency_ms=elapsed,
                )
                offset += chunk_samples

    def unload(self) -> None:
        if self._model is not None:
            self._model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("STT model unloaded; VRAM cache cleared")

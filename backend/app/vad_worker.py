"""Silero VAD worker (ONNX, CPU).

Wraps the silero-vad library to detect speech segments in a 16 kHz PCM
stream. Used by the pipeline to chunk continuous audio into utterances
before sending them to STT.
"""
from __future__ import annotations

import asyncio
import gc
import json
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
from loguru import logger

from .config import SETTINGS


@dataclass
class VADSegment:
    """A detected speech segment."""
    t0: float          # start time in seconds (relative to stream start)
    t1: float          # end time in seconds
    samples: np.ndarray  # int16 PCM, 16 kHz mono
    is_final: bool = False  # True if this is the last segment of the stream


class VADWorker:
    """Silero VAD wrapper. CPU-only, negligible VRAM."""

    def __init__(
        self,
        threshold: float | None = None,
        min_speech_ms: int | None = None,
        max_speech_ms: int | None = None,
        silence_ms: int | None = None,
    ):
        self.threshold = threshold if threshold is not None else SETTINGS.vad_threshold
        self.min_speech_ms = min_speech_ms or SETTINGS.vad_min_speech_ms
        self.max_speech_ms = max_speech_ms or SETTINGS.vad_max_speech_ms
        self.silence_ms = silence_ms or SETTINGS.vad_silence_ms
        self.sample_rate = SETTINGS.vad_sample_rate
        self._model = None
        self._initialized = False

    def _ensure_model(self) -> None:
        if self._initialized:
            return
        try:
            # Preferred: silero-vad pip package
            from silero_vad import load_silero_vad, get_speech_timestamps
            self._model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
            self._use_onnx = False
            logger.info("Silero VAD loaded (torch)")
        except Exception as e_torch:
            logger.warning(f"silero-vad package unavailable ({e_torch}); falling back to ONNX runtime")
            self._load_onnx()
        self._initialized = True

    def _load_onnx(self) -> None:
        import onnxruntime as ort
        model_path = SETTINGS.models_dir / "silero-vad" / "silero_vad.onnx"
        if not model_path.exists():
            # Try the bundled cache path
            model_path = SETTINGS.models_dir / "hf" / "models" / "snakers4" / "silero-vad" / "silero_vad.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD ONNX model not found at {model_path}. "
                "Run `python scripts/download_models.py` first."
            )
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 1
        self._model = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._use_onnx = True
        # State for streaming
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        logger.info(f"Silero VAD loaded (ONNX) from {model_path}")

    def reset(self) -> None:
        if getattr(self, "_use_onnx", False):
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def _onnx_predict(self, pcm_int16: np.ndarray) -> float:
        """Run one 512-sample block through ONNX Silero."""
        import torch  # only for tensor conversion fallback
        x = pcm_int16.astype(np.float32) / 32768.0
        x = x.reshape(1, -1)
        sr = np.array(self.sample_rate, dtype=np.int64)
        ort_inputs = {
            "input": x,
            "h": self._h,
            "c": self._c,
            "sr": sr,
        }
        out, h_out, c_out = self._model.run(None, ort_inputs)
        self._h = h_out
        self._c = c_out
        return float(out[0, 0])

    def detect_segments(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Run VAD on a complete numpy array (16 kHz int16).

        Returns a list of (start_sample, end_sample) tuples.
        """
        self._ensure_model()
        if not self._use_onnx:
            timestamps = self._get_speech_timestamps(
                audio,
                self._model,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_ms,
                max_speech_duration_s=self.max_speech_ms / 1000.0,
                min_silence_duration_ms=self.silence_ms,
                return_seconds=False,
            )
            return [(int(t["start"]), int(t["end"])) for t in timestamps]
        else:
            # ONNX streaming inference
            return self._onnx_stream_detect(audio)

    def _onnx_stream_detect(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Stream 512-sample blocks through ONNX Silero, collect speech runs."""
        self.reset()
        block = 512
        segments: list[tuple[int, int]] = []
        in_speech = False
        cur_start = 0
        silent_blocks = 0
        speech_blocks = 0
        silence_needed = max(1, int(self.silence_ms * self.sample_rate / 1000 / block))
        min_speech_blocks = max(1, int(self.min_speech_ms * self.sample_rate / 1000 / block))
        max_speech_blocks = int(self.max_speech_ms * self.sample_rate / 1000 / block)

        for i in range(0, len(audio) - block + 1, block):
            chunk = audio[i:i + block]
            prob = self._onnx_predict(chunk)
            if prob >= self.threshold:
                if not in_speech:
                    in_speech = True
                    cur_start = i
                    speech_blocks = 0
                speech_blocks += 1
                silent_blocks = 0
                if speech_blocks >= max_speech_blocks:
                    segments.append((cur_start, i + block))
                    in_speech = False
            else:
                if in_speech:
                    silent_blocks += 1
                    if silent_blocks >= silence_needed:
                        if speech_blocks >= min_speech_blocks:
                            segments.append((cur_start, i))
                        in_speech = False
                        speech_blocks = 0
                        silent_blocks = 0

        if in_speech and speech_blocks >= min_speech_blocks:
            segments.append((cur_start, len(audio)))
        return segments

    def stream_segments(
        self,
        audio_iter: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[VADSegment]:
        """Consume an async iterator of PCM chunks; yield VADSegments.

        This is a simple accumulator: buffers incoming chunks, runs VAD on
        the accumulated buffer when it grows past a threshold, emits new
        segments. Suitable for live call use.
        """
        return self._stream_segments_impl(audio_iter)

    async def _stream_segments_impl(
        self,
        audio_iter: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[VADSegment]:
        await asyncio.to_thread(self._ensure_model)
        buffer = np.array([], dtype=np.int16)
        offset_s = 0.0
        last_emitted_end = 0
        async for chunk in audio_iter:
            buffer = np.concatenate([buffer, chunk])
            # Run VAD on what we have so far, emit any completed segments
            if len(buffer) < self.sample_rate:  # at least 1 s
                continue
            segments = self.detect_segments(buffer)
            for (s, e) in segments:
                if e <= last_emitted_end:
                    continue
                # Only emit segments that have a trailing silence (i.e. ended)
                if e < len(buffer) - int(self.silence_ms * self.sample_rate / 1000):
                    samples = buffer[s:e].copy()
                    yield VADSegment(
                        t0=offset_s + s / self.sample_rate,
                        t1=offset_s + e / self.sample_rate,
                        samples=samples,
                    )
                    last_emitted_end = e
            # Trim buffer up to last_emitted_end to bound memory
            if last_emitted_end > 0:
                offset_s += last_emitted_end / self.sample_rate
                buffer = buffer[last_emitted_end:].copy()
                last_emitted_end = 0

        # Final flush
        if len(buffer) > 0:
            segments = self.detect_segments(buffer)
            for (s, e) in segments:
                if e <= last_emitted_end:
                    continue
                samples = buffer[s:e].copy()
                yield VADSegment(
                    t0=offset_s + s / self.sample_rate,
                    t1=offset_s + e / self.sample_rate,
                    samples=samples,
                    is_final=True,
                )
        else:
            # Emit a final empty marker so downstream knows stream ended
            yield VADSegment(t0=offset_s, t1=offset_s, samples=np.array([], dtype=np.int16), is_final=True)

    def unload(self) -> None:
        self._model = None
        self._initialized = False
        gc.collect()
        logger.info("VAD model unloaded")

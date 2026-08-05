"""Pipeline orchestrator.

Two modes:

- ``parallel`` (default): VAD, STT, LLM, TTS run as concurrent asyncio
  tasks connected by bounded ``asyncio.Queue`` objects. Each segment flows
  through the pipeline as soon as it is ready, so a new utterance can be
  STT'd while the previous one is being TTS'd. Lowest perceived latency.

- ``sequential``: Each segment fully completes VAD→STT→LLM→TTS before the
  next one starts. More deterministic, slightly higher latency.

Both modes share the same worker instances (single model per stage) so
VRAM usage is bounded by max(STT, TTS) regardless of mode.
"""
from __future__ import annotations

import asyncio
import gc
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Optional

import numpy as np
from loguru import logger

from .config import SETTINGS
from .vad_worker import VADSegment, VADWorker
from .stt_worker import STTResult, STTWorker
from .llm_worker import LLMResult, LLMWorker
from .tts_worker import TTSResult, TTSWorker
from .speaker_registry import SpeakerRegistry
from .call_logger import CallLogger


@dataclass
class PipelineEvent:
    """An event emitted by the pipeline for the WebSocket layer."""
    kind: str   # "transcript" | "llm" | "audio" | "audio_end" | "vad" | "status" | "error"
    data: dict = field(default_factory=dict)


class Pipeline:
    def __init__(
        self,
        vad: VADWorker,
        stt: STTWorker,
        llm: LLMWorker,
        tts: TTSWorker,
        speaker_registry: SpeakerRegistry,
        call_logger: CallLogger,
    ):
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.speakers = speaker_registry
        self.calls = call_logger
        self.mode = SETTINGS.pipeline_mode

    async def run_streaming(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        speaker_id: str,
        call_id: Optional[str] = None,
        title: Optional[str] = None,
        on_event: Optional[Callable[[PipelineEvent], Awaitable[None]]] = None,
    ) -> str:
        """Run the full pipeline on a streaming audio input.

        ``audio_stream`` yields int16 PCM chunks at 16 kHz mono.
        ``on_event`` is an async callback for pipeline events (transcript,
        audio, etc.).

        Returns the ``call_id``.
        """
        # Look up speaker
        spk = self.speakers.get(speaker_id)
        if spk is None:
            raise ValueError(f"Unknown speaker_id: {speaker_id}")
        ref_wav = self.speakers.get_ref_wav(speaker_id)
        if ref_wav is None or not ref_wav.exists():
            raise ValueError(f"Speaker {speaker_id} has no reference audio")

        # Create call
        call_id = call_id or await self.calls.create_call(speaker_id, title)

        async def emit(ev: PipelineEvent):
            ev.data.setdefault("call_id", call_id)
            if on_event:
                try:
                    await on_event(ev)
                except Exception as e:
                    logger.warning(f"on_event callback failed: {e}")

        await emit(PipelineEvent(kind="status", data={"message": "pipeline_started"}))

        if self.mode == "parallel":
            await self._run_parallel(audio_stream, call_id, spk, ref_wav, emit)
        else:
            await self._run_sequential(audio_stream, call_id, spk, ref_wav, emit)

        await emit(PipelineEvent(kind="status", data={"message": "pipeline_finished"}))
        return call_id

    async def _run_parallel(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        call_id: str,
        speaker: dict,
        ref_wav,
        emit,
    ) -> None:
        """Parallel streaming mode: stages connected by asyncio queues."""
        # Bounded queues between stages
        vad_out: asyncio.Queue[VADSegment | None] = asyncio.Queue(maxsize=8)
        stt_out: asyncio.Queue[tuple[VADSegment, STTResult] | None] = asyncio.Queue(maxsize=8)
        llm_out: asyncio.Queue[tuple[VADSegment, STTResult, LLMResult] | None] = asyncio.Queue(maxsize=8)
        tts_out: asyncio.Queue[tuple[VADSegment, TTSResult] | None] = asyncio.Queue(maxsize=8)

        # --- VAD stage ---
        async def vad_stage():
            try:
                async for seg in self.vad.stream_segments(audio_stream):
                    if seg.samples.size == 0 and seg.is_final:
                        continue
                    await vad_out.put(seg)
            except Exception as e:
                logger.exception(f"VAD stage error: {e}")
                await emit(PipelineEvent(kind="error", data={"stage": "vad", "message": str(e)}))
            finally:
                await vad_out.put(None)

        # --- STT stage ---
        async def stt_stage():
            try:
                while True:
                    seg = await vad_out.get()
                    if seg is None:
                        break
                    if seg.samples.size == 0:
                        continue
                    # Send partial "listening" event
                    await emit(PipelineEvent(
                        kind="vad",
                        data={"event": "speech_end", "t0": seg.t0, "t1": seg.t1},
                    ))
                    res = await self.stt.transcribe_pcm(seg.samples)
                    if not res.text:
                        continue
                    # Persist + emit transcript
                    await self.calls.append_segment(
                        call_id, seg.t0, seg.t1,
                        speaker="user", text=res.text, is_final=True,
                    )
                    await emit(PipelineEvent(
                        kind="transcript",
                        data={
                            "t0": seg.t0, "t1": seg.t1,
                            "speaker": "user", "text": res.text,
                            "language": res.language,
                            "latency_ms": res.latency_ms,
                            "is_final": True,
                        },
                    ))
                    await stt_out.put((seg, res))
            except Exception as e:
                logger.exception(f"STT stage error: {e}")
                await emit(PipelineEvent(kind="error", data={"stage": "stt", "message": str(e)}))
            finally:
                await stt_out.put(None)

        # --- LLM stage ---
        async def llm_stage():
            try:
                while True:
                    item = await stt_out.get()
                    if item is None:
                        break
                    seg, stt_res = item
                    llm_res = await self.llm.generate(stt_res.text, stt_res.language)
                    await emit(PipelineEvent(
                        kind="llm",
                        data={
                            "text": llm_res.text,
                            "source": llm_res.source,
                            "latency_ms": llm_res.latency_ms,
                            "t0": seg.t0, "t1": seg.t1,
                        },
                    ))
                    await llm_out.put((seg, stt_res, llm_res))
            except Exception as e:
                logger.exception(f"LLM stage error: {e}")
                await emit(PipelineEvent(kind="error", data={"stage": "llm", "message": str(e)}))
            finally:
                await llm_out.put(None)

        # --- TTS stage ---
        async def tts_stage():
            try:
                while True:
                    item = await llm_out.get()
                    if item is None:
                        break
                    seg, stt_res, llm_res = item
                    tts_res = await self.tts.synthesize(
                        text=llm_res.text,
                        speaker_ref_wav=ref_wav,
                        language=stt_res.language,
                    )
                    # Persist bot transcript
                    bot_t0 = seg.t1
                    bot_t1 = bot_t0 + (len(tts_res.audio) / tts_res.sample_rate)
                    await self.calls.append_segment(
                        call_id, bot_t0, bot_t1,
                        speaker="bot", text=llm_res.text, is_final=True,
                    )
                    # Emit audio in 20 ms chunks for low-latency streaming
                    chunk_samples = int(tts_res.sample_rate * 0.02)
                    for i in range(0, len(tts_res.audio), chunk_samples):
                        chunk = tts_res.audio[i:i + chunk_samples]
                        await emit(PipelineEvent(
                            kind="audio",
                            data={
                                "pcm_int16": chunk.tolist(),
                                "sample_rate": tts_res.sample_rate,
                                "engine": tts_res.engine,
                            },
                        ))
                    await emit(PipelineEvent(kind="audio_end", data={
                        "t0": bot_t0, "t1": bot_t1,
                        "latency_ms": tts_res.latency_ms,
                    }))
            except Exception as e:
                logger.exception(f"TTS stage error: {e}")
                await emit(PipelineEvent(kind="error", data={"stage": "tts", "message": str(e)}))
            finally:
                await tts_out.put(None)

        # Run all stages concurrently
        await asyncio.gather(
            vad_stage(),
            stt_stage(),
            llm_stage(),
            tts_stage(),
            # Drain tts_out so the queue doesn't block
            _drain(tts_out),
        )
        # Finalize call
        await self.calls.finalize_call(call_id)

    async def _run_sequential(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        call_id: str,
        speaker: dict,
        ref_wav,
        emit,
    ) -> None:
        """Sequential mode: one segment at a time, full pipeline per segment."""
        try:
            async for seg in self.vad.stream_segments(audio_stream):
                if seg.samples.size == 0:
                    continue
                await emit(PipelineEvent(
                    kind="vad",
                    data={"event": "speech_end", "t0": seg.t0, "t1": seg.t1},
                ))
                # STT
                stt_res = await self.stt.transcribe_pcm(seg.samples)
                if not stt_res.text:
                    continue
                await self.calls.append_segment(
                    call_id, seg.t0, seg.t1,
                    speaker="user", text=stt_res.text, is_final=True,
                )
                await emit(PipelineEvent(
                    kind="transcript",
                    data={
                        "t0": seg.t0, "t1": seg.t1,
                        "speaker": "user", "text": stt_res.text,
                        "language": stt_res.language,
                        "latency_ms": stt_res.latency_ms,
                        "is_final": True,
                    },
                ))
                # LLM
                llm_res = await self.llm.generate(stt_res.text, stt_res.language)
                await emit(PipelineEvent(
                    kind="llm",
                    data={
                        "text": llm_res.text,
                        "source": llm_res.source,
                        "latency_ms": llm_res.latency_ms,
                    },
                ))
                # TTS
                tts_res = await self.tts.synthesize(
                    text=llm_res.text,
                    speaker_ref_wav=ref_wav,
                    language=stt_res.language,
                )
                bot_t0 = seg.t1
                bot_t1 = bot_t0 + (len(tts_res.audio) / tts_res.sample_rate)
                await self.calls.append_segment(
                    call_id, bot_t0, bot_t1,
                    speaker="bot", text=llm_res.text, is_final=True,
                )
                chunk_samples = int(tts_res.sample_rate * 0.02)
                for i in range(0, len(tts_res.audio), chunk_samples):
                    chunk = tts_res.audio[i:i + chunk_samples]
                    await emit(PipelineEvent(
                        kind="audio",
                        data={
                            "pcm_int16": chunk.tolist(),
                            "sample_rate": tts_res.sample_rate,
                            "engine": tts_res.engine,
                        },
                    ))
                await emit(PipelineEvent(kind="audio_end", data={
                    "t0": bot_t0, "t1": bot_t1, "latency_ms": tts_res.latency_ms,
                }))
        finally:
            await self.calls.finalize_call(call_id)

    async def transcribe_file_streaming(
        self,
        audio_path: str,
        call_id: str,
        speaker_id: str,
        language: Optional[str] = None,
        on_event: Optional[Callable[[PipelineEvent], Awaitable[None]]] = None,
    ) -> str:
        """Stream-transcribe a long audio file (300+ minutes supported).

        This is STT-only: no LLM/TTS. Used for chunked uploads.
        """
        async def emit(ev: PipelineEvent):
            ev.data.setdefault("call_id", call_id)
            if on_event:
                try:
                    await on_event(ev)
                except Exception as e:
                    logger.warning(f"on_event failed: {e}")

        await emit(PipelineEvent(kind="status", data={"message": "transcription_started"}))
        async for res in self.stt.transcribe_file(audio_path, language=language):
            await self.calls.append_segment(
                call_id, res.start, res.end,
                speaker=speaker_id, text=res.text, is_final=True,
            )
            await emit(PipelineEvent(
                kind="transcript",
                data={
                    "t0": res.start, "t1": res.end,
                    "speaker": speaker_id, "text": res.text,
                    "language": res.language,
                    "latency_ms": res.latency_ms,
                    "is_final": True,
                },
            ))
        await self.calls.finalize_call(call_id)
        await emit(PipelineEvent(kind="status", data={"message": "transcription_finished"}))
        return call_id

    async def analyze_file(
        self,
        audio_path: str,
        speaker_id: str,
        call_id: Optional[str] = None,
        title: Optional[str] = None,
        language: Optional[str] = None,
        on_event: Optional[Callable[[PipelineEvent], Awaitable[None]]] = None,
    ) -> str:
        """Analyze a (long) audio file end-to-end: STT -> LLM -> TTS.

        Transcribes the whole file in memory-safe chunks, accumulates the
        full transcript, composes ONE intelligent LLM response from it, then
        synthesizes a spoken reply using the speaker's reference voice
        (voice clone for XTTS; preset edge voice otherwise).

        Returns the ``call_id``.
        """
        spk = self.speakers.get(speaker_id)
        if spk is None:
            raise ValueError(f"Unknown speaker_id: {speaker_id}")
        ref_wav = self.speakers.get_ref_wav(speaker_id)
        if ref_wav is None or not ref_wav.exists():
            raise ValueError(f"No reference audio for speaker_id: {speaker_id}")

        call_id = call_id or f"call_{uuid.uuid4().hex[:12]}"
        await self.calls.create_call(speaker_id=speaker_id, title=title, call_id=call_id)

        async def emit(ev: PipelineEvent):
            ev.data.setdefault("call_id", call_id)
            if on_event:
                try:
                    await on_event(ev)
                except Exception as e:
                    logger.warning(f"on_event failed: {e}")

        await emit(PipelineEvent(kind="status", data={"message": "analysis_started"}))
        combined: list[str] = []
        try:
            # --- STT the whole file in chunks ---
            async for res in self.stt.transcribe_file(audio_path, language=language):
                if not res.text:
                    continue
                combined.append(res.text)
                await self.calls.append_segment(
                    call_id, res.start, res.end,
                    speaker="user", text=res.text, is_final=True,
                )
                await emit(PipelineEvent(
                    kind="transcript",
                    data={
                        "t0": res.start, "t1": res.end,
                        "speaker": "user", "text": res.text,
                        "language": res.language,
                        "latency_ms": res.latency_ms,
                        "is_final": True,
                    },
                ))
            if not combined:
                raise RuntimeError("No speech transcribed from the audio file")
            full_text = " ".join(combined)
            # Keep the LLM prompt bounded for very long files (Qwen2.5-3B: 32K ctx)
            if len(full_text) > 20_000:
                full_text = full_text[-20_000:]

            # --- LLM: one intelligent response to the full transcript ---
            llm_res = await self.llm.generate(full_text, language or SETTINGS.stt_language)
            await emit(PipelineEvent(
                kind="llm",
                data={
                    "text": llm_res.text,
                    "source": llm_res.source,
                    "latency_ms": llm_res.latency_ms,
                    "prompt_len": len(full_text),
                },
            ))

            # --- TTS: voice-clone reply ---
            tts_res = await self.tts.synthesize(
                text=llm_res.text,
                speaker_ref_wav=ref_wav,
                language=language or SETTINGS.stt_language,
            )
            t0 = next(
                (seg["t1"] for seg in await self.calls.get_transcript(call_id)
                 if seg.get("speaker") == "user"),
                0.0,
            )
            bot_t1 = t0 + (len(tts_res.audio) / tts_res.sample_rate)
            await self.calls.append_segment(
                call_id, t0, bot_t1,
                speaker="bot", text=llm_res.text, is_final=True,
            )
            chunk_samples = int(tts_res.sample_rate * 0.02)
            for i in range(0, len(tts_res.audio), chunk_samples):
                chunk = tts_res.audio[i:i + chunk_samples]
                await emit(PipelineEvent(
                    kind="audio",
                    data={
                        "pcm_int16": chunk.tolist(),
                        "sample_rate": tts_res.sample_rate,
                        "engine": tts_res.engine,
                    },
                ))
            await emit(PipelineEvent(kind="audio_end", data={
                "t0": t0, "t1": bot_t1, "latency_ms": tts_res.latency_ms,
            }))
            await emit(PipelineEvent(kind="status", data={"message": "analysis_finished"}))
            return call_id
        finally:
            await self.calls.finalize_call(call_id)


async def _drain(q: asyncio.Queue):
    """Consume and discard items until None sentinel arrives."""
    while True:
        item = await q.get()
        if item is None:
            break

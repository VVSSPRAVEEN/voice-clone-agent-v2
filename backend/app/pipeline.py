"""Pipeline orchestrator.

Single-mode pipeline for live calls (NO VAD): the client's mic audio is
buffered in full, then run through STT -> LLM -> TTS once per utterance.
Status events are emitted at each stage so the UI can show live progress.
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
from .vad_worker import VADWorker
from .stt_worker import STTResult, STTWorker
from .llm_worker import LLMResult, LLMWorker
from .tts_worker import TTSResult, TTSWorker
from .speaker_registry import SpeakerRegistry
from .call_logger import CallLogger


@dataclass
class PipelineEvent:
    """An event emitted by the pipeline for the WebSocket layer."""
    kind: str   # "transcript" | "llm" | "audio" | "audio_end" | "status" | "error"
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

        await self._run_sequential(audio_stream, call_id, spk, ref_wav, emit)

        await emit(PipelineEvent(kind="status", data={"message": "pipeline_finished"}))
        return call_id

    async def _run_sequential(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        call_id: str,
        speaker: dict,
        ref_wav,
        emit,
    ) -> None:
        """Sequential mode (no VAD): buffer the whole utterance, then run
        STT -> LLM -> TTS once on it. Status events show live progress."""
        parts: list = []
        sr = 16000
        try:
            buffer = np.array([], dtype=np.int16)
            last_status = 0.0
            async for chunk in audio_stream:
                buffer = np.concatenate([buffer, chunk])
                now = time.monotonic()
                if now - last_status >= 1.0:
                    last_status = now
                    await emit(PipelineEvent(kind="status", data={
                        "message": "listening",
                        "seconds": round(len(buffer) / 16000.0, 1),
                    }))

            if buffer.size == 0:
                await emit(PipelineEvent(kind="status", data={"message": "no_audio_received"}))
                return

            await emit(PipelineEvent(kind="status", data={
                "message": "audio_received",
                "seconds": round(len(buffer) / 16000.0, 1),
            }))

            # STT
            await emit(PipelineEvent(kind="status", data={"message": "stt_transcribing"}))
            stt_res = await self.stt.transcribe_pcm(buffer)
            if not stt_res.text:
                await emit(PipelineEvent(kind="status", data={"message": "no_speech_detected"}))
                return
            t0 = 0.0
            t1 = len(buffer) / 16000.0
            await self.calls.append_segment(
                call_id, t0, t1,
                speaker="user", text=stt_res.text, is_final=True,
            )
            await emit(PipelineEvent(
                kind="transcript",
                data={
                    "t0": t0, "t1": t1,
                    "speaker": "user", "text": stt_res.text,
                    "language": stt_res.language,
                    "latency_ms": stt_res.latency_ms,
                    "is_final": True,
                },
            ))

            # LLM
            await emit(PipelineEvent(kind="status", data={"message": "llm_thinking"}))
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
            await emit(PipelineEvent(kind="status", data={"message": "tts_synthesizing"}))
            tts_res = await self.tts.synthesize(
                text=llm_res.text,
                speaker_ref_wav=ref_wav,
                language=stt_res.language,
            )
            parts.append(tts_res.audio)
            sr = tts_res.sample_rate
            bot_t0 = t1
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
            audio_all = np.concatenate(parts) if parts else None
            await self.calls.finalize_call(call_id, audio_int16=audio_all, sample_rate=sr)

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
        detected_lang: Optional[str] = None
        tts_audio = None
        tts_sr = 16000
        try:
            # --- STT the whole file in chunks ---
            async for res in self.stt.transcribe_file(audio_path, language=language):
                if not res.text:
                    continue
                combined.append(res.text)
                if res.language:
                    detected_lang = res.language
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
            llm_res = await self.llm.generate(full_text, detected_lang or language or SETTINGS.stt_language)
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
                language=detected_lang or language or SETTINGS.stt_language,
            )
            tts_audio = tts_res.audio
            tts_sr = tts_res.sample_rate
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
            await self.calls.finalize_call(call_id, audio_int16=tts_audio, sample_rate=tts_sr)


async def _drain(q: asyncio.Queue):
    """Consume and discard items until None sentinel arrives."""
    while True:
        item = await q.get()
        if item is None:
            break

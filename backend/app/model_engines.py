"""Real model implementations for the persistent model server.

These classes are only instantiated inside ``model_server.py``. The app
backend (port 8000) talks to them over localhost via ``app.model_client``,
so restarting the API never re-loads weights.
"""
from __future__ import annotations

import asyncio
import gc
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
from loguru import logger

from .config import SETTINGS
from .gpu_utils import should_use_cuda


# --- STT -------------------------------------------------------------------

@dataclass
class STTResult:
    text: str
    language: str
    start: float
    end: float
    latency_ms: float


class RealSTTWorker:
    """faster-whisper wrapper (lives in the model server process)."""

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
        self._models: dict[str, object] = {}
        self._lock = asyncio.Lock()

    def _ensure_model(self, size: str | None = None) -> object:
        size = size or self.model_size
        if size in self._models:
            return self._models[size]
        from faster_whisper import WhisperModel
        logger.info(
            f"Loading faster-whisper model={size} "
            f"device={self.device} compute_type={self.compute_type}"
        )
        model = WhisperModel(
            size,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(SETTINGS.models_dir / "faster-whisper"),
        )
        self._models[size] = model
        logger.info(f"faster-whisper loaded: {size}")
        return model

    async def transcribe_pcm(
        self,
        pcm_int16: np.ndarray,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
    ) -> STTResult:
        """Transcribe a 16 kHz int16 PCM segment."""
        async with self._lock:
            model = await asyncio.to_thread(self._ensure_model, model_size)
            audio_f32 = pcm_int16.astype(np.float32) / 32768.0
            t0 = time.perf_counter()
            lang_arg = language or self.language
            if lang_arg == "auto":
                lang_arg = None

            def _transcribe() -> tuple[str, str, float, float, list]:
                # Run + fully consume the lazy generator off the event loop so
                # the decode never blocks the asyncio loop.
                segments, info = model.transcribe(
                    audio_f32,
                    language=lang_arg,
                    beam_size=self.beam_size,
                    vad_filter=False,  # no VAD: full buffer is transcribed once
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
        model_size: Optional[str] = None,
    ) -> AsyncIterator[STTResult]:
        """Stream-transcribe a long audio file in chunks.

        Yields STTResult objects as they complete. Never loads the full
        audio into memory at once.
        """
        import soundfile as sf
        async with self._lock:
            model = await asyncio.to_thread(self._ensure_model, model_size)
            info = sf.info(audio_path)
            sr = info.samplerate
            total = info.frames
            chunk_samples = int(chunk_seconds * sr)
            lang_arg = language or self.language
            if lang_arg == "auto":
                lang_arg = None

            offset = 0
            while offset < total:
                audio_f32, _ = sf.read(
                    audio_path,
                    start=offset,
                    frames=chunk_samples,
                    dtype="float32",
                    always_2d=False,
                )
                if audio_f32.ndim > 1:
                    audio_f32 = audio_f32.mean(axis=1)
                if sr != 16000:
                    import librosa
                    audio_f32 = librosa.resample(
                        audio_f32, orig_sr=sr, target_sr=16000
                    )
                t0 = time.perf_counter()

                def _transcribe_chunk() -> tuple[str, str, float, float]:
                    segments, stt_info = model.transcribe(
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
        if self._models:
            self._models.clear()
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("STT models unloaded; VRAM cache cleared")


# --- TTS -------------------------------------------------------------------

@dataclass
class TTSResult:
    audio: np.ndarray   # int16 PCM
    sample_rate: int
    latency_ms: float
    engine: str


async def _validate_edge_voice(voice: str) -> bool:
    import edge_tts
    voices = await edge_tts.list_voices()
    ok = any(v["ShortName"] == voice for v in voices)
    if not ok:
        logger.warning(
            f"Edge-TTS voice '{voice}' not found; falling back to te-IN-MohanNeural"
        )
    return ok


class RealTTSWorker:
    """TTS worker with zero-shot voice cloning (lives in the model server).

    Engines: ``xtts``, ``hybrid`` (praxy->xtts->sherpa, offline-first),
    ``sherpa``, ``ai4bharat``, ``edge`` (only if ``tts_allow_online``).
    """

    def __init__(self, engine: str | None = None):
        self.engine = (engine or SETTINGS.tts_engine).lower()
        self._xtts = None
        self._sherpa = None
        self._ai4bharat = None
        self._praxy = None
        self._praxy_failed = False
        self._praxy_loaded = False
        self._edge_voices = None
        self._lock = asyncio.Lock()

    def _ensure_model(self) -> None:
        if self.engine == "xtts":
            self._ensure_xtts()
        elif self.engine == "sherpa":
            self._ensure_sherpa()
        elif self.engine == "ai4bharat":
            self._ensure_ai4bharat()
        elif self.engine == "edge":
            if SETTINGS.tts_allow_online:
                self._ensure_edge()
            else:
                self._ensure_xtts()  # offline: substitute local engine
        elif self.engine == "hybrid":
            self._ensure_xtts()  # praxy/xtts are local; edge is only used live if allowed
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine}")

    def _ensure_edge(self) -> None:
        if not SETTINGS.tts_allow_online:
            raise RuntimeError(
                "Edge-TTS is disabled (tts_allow_online=False): it streams "
                "audio from Microsoft's servers. Use praxy/xtts/sherpa/ai4bharat."
            )
        if self._edge_voices is not None:
            return
        import threading
        import edge_tts
        voice = SETTINGS.edge_voice
        holder: dict = {}
        def _validate():
            holder["error"] = None
            try:
                ok = asyncio.run(_validate_edge_voice(voice))
                if not ok:
                    SETTINGS.edge_voice = "te-IN-MohanNeural"
                    holder["fixed"] = True
            except Exception as e:  # pragma: no cover - surfaced below
                holder["error"] = e
        t = threading.Thread(target=_validate, daemon=True)
        t.start()
        t.join()
        if holder.get("error"):
            raise holder["error"]
        import os
        os.environ["EDGE_VOICE"] = SETTINGS.edge_voice
        self._edge_voices = SETTINGS.edge_voice
        if holder.get("fixed"):
            logger.info("Edge-TTS configured voice invalid; reset to te-IN-MohanNeural")
        logger.info(f"Edge-TTS ready (voice={SETTINGS.edge_voice})")

    def _edge_voice_for(self, text: str, language: Optional[str]) -> str:
        if isinstance(self._edge_voices, str):
            SETTINGS.edge_voice = self._edge_voices
        return self._pick_edge_voice(text, language)

    @property
    def praxy_ready(self) -> bool:
        return not self._praxy_failed

    async def preload(self) -> None:
        """Warm up engines at startup so the first live call is fast."""
        if SETTINGS.tts_allow_online:
            try:
                self._ensure_edge()
            except Exception as e:
                logger.warning(f"Edge preload failed: {e}")
        else:
            logger.info("Edge-TTS skipped (offline mode)")
        try:
            if self._praxy is None:
                from .praxy_engine import PraxyEngine
                self._praxy = PraxyEngine()
            if not self._praxy_loaded:
                logger.info("Preloading Praxy (background, may take minutes on CPU)...")
                await asyncio.to_thread(self._praxy.ensure_loaded)
                self._praxy_loaded = True
                logger.info("Praxy preload complete")
        except Exception as e:
            logger.warning(f"Praxy preload failed: {e}")
        try:
            self._ensure_xtts()
            logger.info("XTTS preloaded")
        except Exception as e:
            logger.warning(f"XTTS preload failed: {e}")

    def _ensure_ai4bharat(self) -> None:
        if self._ai4bharat is not None:
            return
        import torch
        from TTS.api import TTS as CoquiTTS
        model_dir = SETTINGS.models_dir / "ai4bharat-te" / "te"
        if not (model_dir / "fastpitch" / "best_model.pth").exists():
            raise FileNotFoundError(
                f"AI4Bharat Telugu model not found under {model_dir}. "
                "Download te.zip from AI4Bharat/Indic-TTS releases and extract "
                "it into data/models/ai4bharat-te/te/."
            )
        device = "cuda" if SETTINGS.xtts_device == "cuda" and should_use_cuda(1500) else "cpu"
        if SETTINGS.xtts_device == "cuda" and device == "cpu":
            logger.warning("AI4Bharat TTS: insufficient free GPU memory; falling back to CPU")
        logger.info(f"Loading AI4Bharat Telugu FastPitch+HiFi-GAN on device={device}")
        self._ai4bharat = CoquiTTS(
            model_path=str(model_dir / "fastpitch" / "best_model.pth"),
            config_path=str(model_dir / "fastpitch" / "config.json"),
            vocoder_path=str(model_dir / "hifigan" / "best_model.pth"),
            vocoder_config_path=str(model_dir / "hifigan" / "config.json"),
        ).to(device)
        self._ai4bharat_device = device
        logger.info("AI4Bharat Telugu TTS loaded")

    def _ensure_xtts(self) -> None:
        if self._xtts is not None:
            return
        import torch
        import transformers.pytorch_utils as _tp
        if not hasattr(_tp, "isin_mps_friendly"):
            _tp.isin_mps_friendly = (
                lambda elements, test_elements: torch.isin(elements, test_elements)
            )
        from TTS.api import TTS as CoquiTTS
        device = "cuda" if SETTINGS.xtts_device == "cuda" and should_use_cuda(2500) else "cpu"
        if SETTINGS.xtts_device == "cuda" and device == "cpu":
            logger.warning("XTTS: insufficient free GPU memory; falling back to CPU")
        logger.info(f"Loading Coqui XTTS v2 on device={device}")
        self._xtts = CoquiTTS(SETTINGS.xtts_model).to(device)
        self._xtts_device = device
        logger.info("Coqui XTTS v2 loaded")

    def _ensure_sherpa(self) -> None:
        if self._sherpa is not None:
            return
        import sherpa_onnx
        telugu_model = SETTINGS.models_dir / "sherpa-onnx" / "telugu-vits"
        english_model = SETTINGS.models_dir / "sherpa-onnx" / "english-vits"
        if not telugu_model.exists() and not english_model.exists():
            raise FileNotFoundError(
                f"No sherpa-onnx VITS models found under {SETTINGS.models_dir / 'sherpa-onnx'}. "
                "Run `python scripts/download_models.py` or switch TTS_ENGINE=xtts."
            )
        if telugu_model.exists():
            model_dir = telugu_model
            lang = "te"
        else:
            model_dir = english_model
            lang = "en"
        model_file = next(model_dir.glob("*.onnx"))
        tokens_file = model_dir / "tokens.txt"
        lexicon = next(model_dir.glob("lexicon.txt"), None)
        self._sherpa = sherpa_onnx.OfflineTts(
            model="vits",
            vits_model=str(model_file),
            vits_lexicon=str(lexicon) if lexicon else "",
            vits_tokens=str(tokens_file),
            vits_data_dir="",
            num_threads=1,
            debug=False,
        )
        self._sherpa_lang = lang
        logger.info(f"sherpa-onnx VITS loaded (lang={lang}) from {model_dir}")

    async def synthesize(
        self,
        text: str,
        speaker_ref_wav: str | Path,
        language: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize speech with voice cloning."""
        async with self._lock:
            self._ensure_model()
            if self.engine == "xtts":
                return await self._synth_xtts(text, speaker_ref_wav, language)
            elif self.engine == "hybrid":
                return await self._synth_hybrid(text, speaker_ref_wav, language)
            elif self.engine == "ai4bharat":
                return await self._synth_ai4bharat(text)
            elif self.engine == "edge":
                if SETTINGS.tts_allow_online:
                    return await self._synth_edge(text, language)
                self._ensure_xtts()
                return await self._synth_xtts(text, speaker_ref_wav, language)
            else:
                return await self._synth_sherpa(text, language)

    async def _synth_xtts(
        self,
        text: str,
        speaker_ref_wav: str | Path,
        language: Optional[str],
    ) -> TTSResult:
        lang = (language or SETTINGS.xtts_language).lower()
        if lang.startswith("te"):
            lang_code = "te"
        elif lang.startswith("en"):
            lang_code = "en"
        else:
            lang_code = "en"  # safe default
        t0 = time.perf_counter()
        def _do():
            wav = self._xtts.tts(
                text=text,
                speaker_wav=str(speaker_ref_wav),
                language=lang_code,
                split_sentences=True,
            )
            arr = np.array(wav, dtype=np.float32)
            int16 = (arr * 32767.0).astype(np.int16)
            return int16
        audio = await asyncio.to_thread(_do)
        elapsed = (time.perf_counter() - t0) * 1000
        return TTSResult(
            audio=audio,
            sample_rate=24_000,
            latency_ms=elapsed,
            engine="xtts",
        )

    async def _synth_praxy(
        self,
        text: str,
        speaker_ref_wav: str | Path,
        language: str,
    ) -> TTSResult:
        if self._praxy is None:
            from .praxy_engine import PraxyEngine
            self._praxy = PraxyEngine()
        try:
            await asyncio.to_thread(self._praxy.ensure_loaded)
            self._praxy_loaded = True
        except Exception as exc:
            self._praxy_failed = True
            logger.warning(f"PraxyEngine load failed ({exc}); disabling praxy")
            raise
        t0 = time.perf_counter()
        audio, sr = await asyncio.to_thread(
            self._praxy.synth, text, str(speaker_ref_wav), language
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return TTSResult(
            audio=audio,
            sample_rate=sr,
            latency_ms=elapsed,
            engine="praxy",
        )

    async def _synth_hybrid(
        self,
        text: str,
        speaker_ref_wav: str | Path,
        language: Optional[str],
    ) -> TTSResult:
        lang = (language or SETTINGS.xtts_language).lower()
        if lang.startswith(("te", "ta")) and speaker_ref_wav:
            if self.praxy_ready:
                try:
                    return await self._synth_praxy(text, speaker_ref_wav, lang)
                except Exception as exc:
                    logger.warning(f"Praxy TTS failed ({exc}); falling back to local engines")
            try:
                self._ensure_xtts()
                return await self._synth_xtts(text, speaker_ref_wav, lang)
            except Exception as exc:
                logger.warning(f"XTTS fallback failed ({exc}); falling back to sherpa")
            if SETTINGS.tts_allow_online:
                return await self._synth_edge(text, language)
            self._ensure_sherpa()
            return await self._synth_sherpa(text, language)
        if lang.startswith(("en", "hi")):
            self._ensure_xtts()
            return await self._synth_xtts(text, speaker_ref_wav, lang)
        if SETTINGS.tts_allow_online:
            return await self._synth_edge(text, language)
        self._ensure_xtts()
        return await self._synth_xtts(text, speaker_ref_wav, lang)

    async def _synth_ai4bharat(self, text: str) -> TTSResult:
        t0 = time.perf_counter()
        def _do():
            wav = self._ai4bharat.tts(text=text)
            arr = np.array(wav, dtype=np.float32)
            return (arr * 32767.0).astype(np.int16)
        audio = await asyncio.to_thread(_do)
        elapsed = (time.perf_counter() - t0) * 1000
        return TTSResult(
            audio=audio,
            sample_rate=self._ai4bharat.synthesizer.output_sample_rate,
            latency_ms=elapsed,
            engine="ai4bharat",
        )

    @staticmethod
    def _pick_edge_voice(text: str, language: Optional[str]) -> str:
        scripts = [
            ("\u0c00", "\u0c7f", "te-IN-MohanNeural"),    # Telugu (configurable)
            ("\u0b80", "\u0bff", "ta-IN-PallaviNeural"),   # Tamil
            ("\u0a80", "\u0aff", "gu-IN-NiranjanNeural"),  # Gujarati
            ("\u0c80", "\u0cff", "kn-IN-GaganNeural"),     # Kannada
            ("\u0d00", "\u0d7f", "ml-IN-MidhunNeural"),    # Malayalam
            ("\u0980", "\u09ff", "bn-IN-TanishaaNeural"),  # Bengali + Assamese
            ("\u0900", "\u097f", "hi-IN-MadhurNeural"),    # Devanagari proxy
            ("\u0600", "\u06ff", "ur-IN-SalmanNeural"),    # Urdu
        ]
        for lo, hi_ch, voice in scripts:
            if any(lo <= ch <= hi_ch for ch in text):
                return voice
        lang = (language or SETTINGS.xtts_language).lower()
        if lang.startswith("en"):
            return "en-US-JennyNeural"
        return SETTINGS.edge_voice

    async def _synth_edge(self, text: str, language: Optional[str]) -> TTSResult:
        import edge_tts
        voice = self._edge_voice_for(text, language)
        t0 = time.perf_counter()
        buf = BytesIO()
        async for chunk in edge_tts.Communicate(text=text, voice=voice).stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        import pydub
        seg = pydub.AudioSegment.from_file(buf, format="mp3")
        seg = seg.set_channels(1).set_frame_rate(24000)
        samples = seg.get_array_of_samples()
        audio = np.array(samples, dtype=np.int16)
        elapsed = (time.perf_counter() - t0) * 1000
        return TTSResult(
            audio=audio,
            sample_rate=24_000,
            latency_ms=elapsed,
            engine="edge",
        )

    async def _synth_sherpa(
        self,
        text: str,
        language: Optional[str],
    ) -> TTSResult:
        t0 = time.perf_counter()
        audio = await asyncio.to_thread(
            self._sherpa.generate,
            text,
            sid=0,
            speed=1.0,
        )
        audio = np.array(audio, dtype=np.int16)
        elapsed = (time.perf_counter() - t0) * 1000
        return TTSResult(
            audio=audio,
            sample_rate=self._sherpa.sample_rate,
            latency_ms=elapsed,
            engine="sherpa",
        )

    def unload(self) -> None:
        if self._praxy is not None:
            self._praxy.unload()
        self._praxy = None
        self._praxy_failed = False
        self._praxy_loaded = False
        self._xtts = None
        self._sherpa = None
        self._ai4bharat = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("TTS model unloaded; VRAM cache cleared")

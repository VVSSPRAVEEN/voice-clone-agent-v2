"""TTS worker with zero-shot voice cloning.

Engines supported:

- ``xtts`` (default): Coqui XTTS v2 — multilingual zero-shot cloning from a
  3-10 s reference clip. English + Hindi (no Telugu in XTTS v2).
- ``edge``: Microsoft Edge-TTS (online). Full Telugu/English neural voices.
- ``ai4bharat``: AI4Bharat IndiTTS FastPitch+HiFi-GAN (Telugu; requires the
  original Coqui TTS, not the py3.12 coqui-tts fork — currently unusable).
- ``sherpa``: sherpa-onnx VITS — pre-bundled AI4Bharat models.

VRAM: ~4 GB for XTTS on GPU, ~1 GB for sherpa-onnx.
"""
from __future__ import annotations

import asyncio
import gc
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from .config import SETTINGS
from .gpu_utils import should_use_cuda

async def _validate_edge_voice(voice: str) -> bool:
    import edge_tts
    voices = await edge_tts.list_voices()
    ok = any(v["ShortName"] == voice for v in voices)
    if not ok:
        logger.warning(
            f"Edge-TTS voice '{voice}' not found; falling back to te-IN-MohanNeural"
        )
    return ok


@dataclass
class TTSResult:
    audio: np.ndarray   # int16 PCM at TTS_SAMPLE_RATE (24 kHz for XTTS)
    sample_rate: int
    latency_ms: float
    engine: str


class TTSWorker:
    def __init__(self, engine: str | None = None):
        self.engine = (engine or SETTINGS.tts_engine).lower()
        self._xtts = None
        self._sherpa = None
        self._ai4bharat = None
        self._praxy = None
        self._praxy_failed = False
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
            self._ensure_edge()
        elif self.engine == "hybrid":
            self._ensure_edge()
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine}")

    def _ensure_edge(self) -> None:
        # Edge-TTS is online (no local model to load); validate voices once.
        if self._edge_voices is not None:
            return
        import asyncio
        import threading
        import edge_tts
        voice = SETTINGS.edge_voice
        # Need a dedicated loop+thread: we may run inside an async event loop
        # (asyncio.run() would raise "cannot be called from a running loop").
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
        # Pick whichever exists; prefer Telugu
        if telugu_model.exists():
            model_dir = telugu_model
            lang = "te"
        else:
            model_dir = english_model
            lang = "en"
        # Find model + tokens files
        model_file = next(model_dir.glob("*.onnx"))
        tokens_file = model_dir / "tokens.txt"
        lexicon = next(model_dir.glob("lexicon.txt"), None)
        self._sherpa = sherpa_onnx.OfflineTts(
            model=f"vits",
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
        """Synthesize speech with voice cloning.

        For XTTS: ``speaker_ref_wav`` is a path to the reference 3-10 s clip.
        For sherpa: ``speaker_ref_wav`` is ignored (uses fixed preset voice).
        """
        async with self._lock:
            self._ensure_model()
            if self.engine == "xtts":
                return await self._synth_xtts(text, speaker_ref_wav, language)
            elif self.engine == "hybrid":
                return await self._synth_hybrid(text, speaker_ref_wav, language)
            elif self.engine == "ai4bharat":
                return await self._synth_ai4bharat(text)
            elif self.engine == "edge":
                return await self._synth_edge(text, language)
            else:
                return await self._synth_sherpa(text, language)

    async def _synth_xtts(
        self,
        text: str,
        speaker_ref_wav: str | Path,
        language: Optional[str],
    ) -> TTSResult:
        lang = (language or SETTINGS.xtts_language).lower()
        # XTTS uses ISO 639-1 codes; tel = "te", eng = "en"
        if lang.startswith("te"):
            lang_code = "te"
        elif lang.startswith("en"):
            lang_code = "en"
        else:
            lang_code = "en"  # safe default
        t0 = time.perf_counter()
        # XTTS sync API -> run in thread
        def _do():
            wav = self._xtts.tts(
                text=text,
                speaker_wav=str(speaker_ref_wav),
                language=lang_code,
                split_sentences=True,
            )
            arr = np.array(wav, dtype=np.float32)
            # XTTS native rate is 24 kHz
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
                    logger.warning(f"Praxy TTS failed ({exc}); falling back to Edge-TTS")
            return await self._synth_edge(text, language)
        if lang.startswith(("en", "hi")):
            self._ensure_xtts()
            return await self._synth_xtts(text, speaker_ref_wav, lang)
        return await self._synth_edge(text, language)

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
        # Full edge-TTS Indic + English coverage keyed by Unicode script block.
        # Script beats language tag: LLM/speaker tags are unreliable for
        # code-mixed replies (e.g. Telugu + "din").
        # Voice names verified live against edge-tts 7.2.8 (2026-08).
        scripts = [
            ("\u0c00", "\u0c7f", "te-IN-MohanNeural"),    # Telugu (configurable)
            ("\u0b80", "\u0bff", "ta-IN-PallaviNeural"),   # Tamil
            ("\u0a80", "\u0aff", "gu-IN-NiranjanNeural"),  # Gujarati
            ("\u0c80", "\u0cff", "kn-IN-GaganNeural"),     # Kannada
            ("\u0d00", "\u0d7f", "ml-IN-MidhunNeural"),    # Malayalam
            ("\u0980", "\u09ff", "bn-IN-TanishaaNeural"),  # Bengali + Assamese (shared script)
            ("\u0900", "\u097f", "hi-IN-MadhurNeural"),    # Devanagari: hi/mr/ne/sa (proxy hi)
            ("\u0600", "\u06ff", "ur-IN-SalmanNeural"),    # Urdu (Perso-Arabic)
            # Gurmukhi (pa) U+0A00-0A7F and Odia U+0B00-0B7F: no edge voice -> fallthrough
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

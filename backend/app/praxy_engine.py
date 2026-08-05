"""PraxyEngine: standalone Telugu/Tamil voice-cloning TTS using Chatterbox Multilingual + the Praxy R6 LoRA.

Recipe (verified):
  - Base model : ChatterboxMultilingualTTS (multilingual, language_id="hi" used as te/ta proxy)
  - Adapter    : Praxel/praxy-voice-r6 -> lora_state.pt (r=32, alpha=64, q/k/v/o projections)
  - Front-end  : BUPS-style romanisation via indic_transliteration.sanscript (Telugu/Tamil script -> ISO)
  - Output     : int16 mono PCM + sample rate (model.sr)

The model is loaded lazily on first synth() and guarded by a threading.Lock,
so importing this module never touches CUDA or the HF Hub.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Optional, Tuple

import numpy as np
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from huggingface_hub import hf_hub_download
from indic_transliteration.sanscript import ISO, TAMIL, TELUGU, transliterate
from loguru import logger
from peft import LoraConfig, get_peft_model

from .gpu_utils import should_use_cuda

__all__ = ["PraxyEngine"]

# Optional extra cache location for HF downloads.
_HF_CACHE_HINT = r"D:\hf-models\hub"

# Script ranges: Telugu U+0C00-U+0C7F, Tamil U+0B80-U+0BFF.
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]+")
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]+")

_LORA_REPO = "Praxel/praxy-voice-r6"
_LORA_FILE = "lora_state.pt"


def _romanise(text: str, script_re: "re.Pattern[str]", src_scheme: str) -> str:
    """Transliterate Indic-script runs to ISO roman; leave Latin/other runs unchanged.

    If the text contains no Indic-script characters at all, it is returned as-is.
    """
    if not script_re.search(text):
        return text
    parts: list[str] = []
    pos = 0
    for m in script_re.finditer(text):
        parts.append(text[pos : m.start()])  # Latin / punctuation / digits run - untouched
        parts.append(transliterate(m.group(), src_scheme, ISO))
        pos = m.end()
    parts.append(text[pos:])
    return "".join(parts)


class PraxyEngine:
    """Lazy, thread-safe wrapper around Chatterbox Multilingual + Praxy R6 LoRA."""

    def __init__(self, ckpt_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self._ckpt_path = ckpt_path
        self._device = device or ("cuda" if should_use_cuda(3600) else "cpu")
        self._model: Optional[ChatterboxMultilingualTTS] = None
        self._sr: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loading

    def _resolve_checkpoint(self) -> str:
        """Return the local path to the Praxy R6 LoRA state dict (download if needed)."""
        if self._ckpt_path:
            return self._ckpt_path
        cache_dir: Optional[str] = None
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dir = hf_home
        elif os.path.isdir(_HF_CACHE_HINT):
            cache_dir = _HF_CACHE_HINT
        logger.info(
            f"PraxyEngine: downloading {_LORA_REPO}/{_LORA_FILE}"
            + (f" (cache_dir={cache_dir})" if cache_dir else "")
        )
        return hf_hub_download(_LORA_REPO, _LORA_FILE, cache_dir=cache_dir)

    def ensure_loaded(self) -> None:
        """Lazily load the model. Thread-safe (serialised by a Lock)."""
        with self._lock:
            if self._model is not None:
                return

            # 1. Base model ----------------------------------------------------
            device = self._device
            try:
                logger.info(f"PraxyEngine: loading ChatterboxMultilingualTTS on {device} ...")
                model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            except Exception as exc:
                if device != "cpu":
                    logger.warning(f"PraxyEngine: CUDA load failed ({exc}); retrying on CPU")
                    device = "cpu"
                    self._device = "cpu"
                    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
                else:
                    raise

            # 2. Freeze the t3 backbone ----------------------------------------
            logger.info("PraxyEngine: freezing t3 backbone")
            for p in model.t3.parameters():
                p.requires_grad_(False)

            # 3. Wrap with LoRA (r=32, alpha=64, q/k/v/o) -----------------------
            logger.info("PraxyEngine: wrapping t3 with LoRA (r=32, alpha=64)")
            lora_cfg = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            model.t3 = get_peft_model(model.t3, lora_cfg)

            # 4. Load Praxy R6 LoRA weights ------------------------------------
            ckpt = self._resolve_checkpoint()
            logger.info(f"PraxyEngine: loading Praxy R6 LoRA weights from {ckpt}")
            sd = torch.load(ckpt, map_location=device)
            model.t3.load_state_dict(sd, strict=False, assign=True)
            model.t3.eval()

            self._model = model
            self._sr = int(model.sr)
            logger.info(f"PraxyEngine: model ready (device={device}, sample_rate={self._sr})")

    # ------------------------------------------------------------------ synth

    def synth(self, text: str, ref_wav: str, language: str = "te") -> Tuple[np.ndarray, int]:
        """Synthesise speech from Telugu or Tamil text, cloning the voice in ref_wav.

        Returns (int16 mono PCM np.ndarray, sample_rate). The text may be in
        Telugu or Tamil script; it is romanised (BUPS-style, script -> ISO) before
        synthesis. Latin-only text is passed through unchanged.
        """
        self.ensure_loaded()
        assert self._model is not None

        # Romanisation ----------------------------------------------------------
        lang = (language or "te").strip().lower()
        if lang.startswith("ta"):
            script_re, src_scheme = _TAMIL_RE, TAMIL
        else:
            script_re, src_scheme = _TELUGU_RE, TELUGU
        text_roman = _romanise(text.strip(), script_re, src_scheme)
        logger.debug(
            f"PraxyEngine: romanised ({lang}) '{text[:60]}' -> '{text_roman[:60]}'"
        )

        # Synthesis (language_id="hi" is the te/ta proxy in Chatterbox) ----------
        def _generate():
            model = self._model
            assert model is not None
            with torch.inference_mode():
                return model.generate(
                    text_roman,
                    language_id="hi",
                    audio_prompt_path=ref_wav,
                    exaggeration=0.7,
                    cfg_weight=0.5,
                    temperature=0.6,
                    repetition_penalty=2.0,
                    min_p=0.1,
                    top_p=1.0,
                )

        try:
            wav = _generate()
        except RuntimeError as exc:
            if self._device != "cuda":
                logger.exception("PraxyEngine: generation failed")
                raise
            logger.warning(f"PraxyEngine: CUDA generation failed ({exc}); retrying on CPU")
            self.unload()
            self._device = "cpu"
            self.ensure_loaded()
            try:
                wav = _generate()
            except Exception:
                logger.exception("PraxyEngine: generation failed after CPU retry")
                raise
        except Exception:
            logger.exception("PraxyEngine: generation failed")
            raise

        # PCM conversion ---------------------------------------------------------
        try:
            wav = wav.detach().cpu()
            if wav.ndim == 2:  # [1, T] -> [T]
                wav = wav.squeeze(0)
            if torch.is_floating_point(wav):
                wav = wav.clamp(-1.0, 1.0)  # Chatterbox returns float [-1, 1] typically
            else:
                wav = wav.float() / 32767.0
                wav = wav.clamp(-1.0, 1.0)
            pcm = (wav.numpy() * 32767).astype(np.int16)
        except Exception:
            logger.exception("PraxyEngine: PCM conversion failed")
            raise

        return pcm, self._sr

    # ------------------------------------------------------------------ teardown

    def unload(self) -> None:
        """Drop the model reference and free VRAM (best-effort)."""
        with self._lock:
            if self._model is not None:
                self._model = None
                self._sr = 0
                torch.cuda.empty_cache()
                logger.info("PraxyEngine: model unloaded; VRAM cache cleared")

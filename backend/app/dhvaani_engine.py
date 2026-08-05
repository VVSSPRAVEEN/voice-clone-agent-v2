"""DhVaani Telugu TTS engine (ARTPARK-IISc/DhVaani-0.5).

Research findings (verified 2026-08-05)
=======================================
* DhVaani-0.5 is NOT a sherpa-onnx model.  It is a custom PyTorch
  flow-matching zero-shot TTS built on k2-fsa/ZipVoice
  (model_type "dhvaani", architectures ["DhVaaniModel"] defined in
  modeling_dhvaani.py).  sherpa_onnx.OfflineTts / VITS cannot load it;
  the ".safetensors" + "tokens.txt" layout only looks sherpa-like.

* Files in the repo: config.json, tokens.txt, model.safetensors (~469 MB,
  122.8M F32 params), modeling_dhvaani.py, dhvaani.py, model.json,
  requirements.txt, samples/*.wav, and a vendored _backend/zipvoice/
  package (from github.com/k2-fsa/ZipVoice).

* Loading API (from the public ARTPARK-IISc/DhVaani-Demo Gradio space,
  app.py):

      model_path = snapshot_download("ARTPARK-IISc/DhVaani-0.5",
                                     token=os.getenv("HF_TOKEN"))
      sys.path.insert(0, model_path)
      sys.path.insert(0, os.path.join(model_path, "_backend"))
      from dhvaani import DhVaani
      model = DhVaani(model_path)
      model.synthesize(
          text=text,
          prompt_wav=prompt_wav,      # reference clip (3-10 s)
          prompt_text=prompt_text,    # exact transcript of the clip
          out_path=out_path,          # writes a 24 kHz WAV
          num_step=16, guidance_scale=1.0, speed=1.0, seed=666,
      )

  The DhVaani wrapper internally follows the ZipVoice torch pipeline
  (tokenizer from tokens.txt, VocosFbank features, ZipVoice flow-matching
  sampler, Vocos vocoder; sample rate = model.json["feature"]
  ["sampling_rate"] = 24000).

* The model is ZERO-SHOT VOICE CLONING: it requires a reference voice
  (prompt_wav + prompt_text).  Pure text->speech is impossible with this
  architecture, so call set_prompt(...) (or pass prompt_wav/prompt_text to
  __init__) before synth().

* Repo gating: the HF API reports gated="auto" and (as of this writing)
  anonymous downloads return "Access to model ... is restricted and you
  are not in the authorized list."  Loading therefore performs a
  snapshot_download with HF_TOKEN from the environment; if the token is
  not authorized, ensure_loaded() raises a clear error instead of
  synthesising with a half-downloaded model.

* Runtime deps (see demo requirements.txt): torch, torchaudio, safetensors,
  einops, vocos, pydub, soundfile, huggingface_hub.  vocos is required for
  the vocoder.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

MODEL_REPO_ID = "ARTPARK-IISc/DhVaani-0.5"
_DEFAULT_SAMPLE_RATE = 24000  # model.json feature.sampling_rate
_REQUIRED_MODULES = (
    "torch",
    "torchaudio",
    "safetensors",
    "einops",
    "vocos",
    "pydub",
    "soundfile",
    "huggingface_hub",
)

__all__ = ["DhVaaniEngine"]


class DhVaaniEngine:
    """Lazy, thread-safe wrapper around the DhVaani Telugu TTS model.

    Parameters
    ----------
    model_dir:
        Directory containing the downloaded snapshot (config.json,
        model.safetensors, dhvaani.py, model.json, tokens.txt,
        _backend/, ...).  If None, the HF cache under HF_HOME/hub is
        scanned for `models--ARTPARK-IISc--DhVaani-0.5/snapshots/*` and
        the first snapshot is used; if nothing is cached, the snapshot is
        downloaded via `huggingface_hub.snapshot_download`.
    device:
        Advisory: "cuda" or "cpu".  The DhVaani wrapper selects its own
        device (cuda -> mps -> cpu); this value is retained for callers
        that query the engine and for torch thread config on CPU.
    prompt_wav / prompt_text:
        Optional default reference voice (required for zero-shot
        synthesis).  Can also be set later via set_prompt().
    hf_token:
        Optional HuggingFace token for gated downloads.  Defaults to the
        HF_TOKEN environment variable.
    num_step / guidance_scale / speed / seed:
        Sampling hyper-parameters passed straight through to
        DhVaani.synthesize().
    """

    def __init__(
        self,
        model_dir: str | None = None,
        device: str | None = None,
        *,
        prompt_wav: str | None = None,
        prompt_text: str | None = None,
        hf_token: str | None = None,
        num_step: int = 16,
        guidance_scale: float = 1.0,
        speed: float = 1.0,
        seed: int = 666,
    ) -> None:
        self._model_dir = str(model_dir) if model_dir else None
        self._device = device
        self._prompt_wav = prompt_wav
        self._prompt_text = prompt_text
        self._hf_token = hf_token
        self._num_step = int(num_step)
        self._guidance_scale = float(guidance_scale)
        self._speed = float(speed)
        self._seed = int(seed)

        self._tts = None          # the dhvaani.DhVaani instance
        self._sr: int | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._tts is not None

    @property
    def model_dir(self) -> str | None:
        return self._model_dir

    @property
    def sample_rate(self) -> int | None:
        return self._sr

    def set_prompt(self, prompt_wav: str | os.PathLike, prompt_text: str) -> None:
        """Set the reference voice used by subsequent synth() calls.

        DhVaani is a zero-shot voice-cloning model: `prompt_wav` is a
        3-10 s clip of the speaker and `prompt_text` must be the exact
        transcript of what is spoken in that clip.
        """
        p = Path(prompt_wav)
        if not p.is_file():
            raise FileNotFoundError(f"Prompt wav not found: {p}")
        self._prompt_wav = str(p)
        self._prompt_text = prompt_text

    def ensure_loaded(self) -> None:
        """Load the model on first use (thread-safe, idempotent).

        Sets ``self._tts`` (the dhvaani.DhVaani instance) and ``self._sr``
        (model sample rate, 24000).
        """
        with self._lock:
            if self._tts is not None:
                return

            missing = [m for m in _REQUIRED_MODULES
                       if _find_spec(m) is None]
            if missing:
                raise RuntimeError(
                    "DhVaani engine missing required packages: "
                    + ", ".join(missing)
                    + ". Install them in the backend venv, e.g. "
                    '".venv\\\\Scripts\\\\pip.exe install ' + " ".join(missing) + '"'
                )

            model_dir = self._resolve_model_dir()

            # Mirror the official demo loading sequence exactly:
            # both the snapshot root (dhvaani.py) and its vendored
            # _backend package must be importable.
            root = str(model_dir)
            if root not in sys.path:
                sys.path.insert(0, root)
            backend = os.path.join(root, "_backend")
            if backend not in sys.path:
                sys.path.insert(0, backend)

            try:
                from dhvaani import DhVaani
            except Exception as exc:  # pragma: no cover - import failure
                raise RuntimeError(
                    f"Could not import dhvaani.DhVaani from {root}: {exc}"
                ) from exc

            tts = DhVaani(root)
            self._tts = tts
            self._sr = self._read_sample_rate(model_dir)

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesise `text` (Telugu script) and return ``(int16 mono PCM,
        sample_rate)``.

        Requires a reference voice: either set via set_prompt()/__init__
        (prompt_wav + prompt_text) or provided by a caller that calls
        synthesize() directly on :attr:`loaded` object.
        """
        self.ensure_loaded()

        prompt_wav = self._prompt_wav
        prompt_text = self._prompt_text
        if not prompt_wav:
            raise RuntimeError(
                "DhVaani is a zero-shot voice-cloning model: it needs a "
                "reference voice. Call set_prompt(prompt_wav, prompt_text) "
                "or construct DhVaaniEngine(..., prompt_wav=..., "
                "prompt_text=...) first."
            )
        if not Path(prompt_wav).is_file():
            raise FileNotFoundError(f"Prompt wav not found: {prompt_wav}")

        # The DhVaani wrapper writes a WAV file; generation is serialised
        # under the same lock that guards loading (the wrapper is not
        # guaranteed thread-safe).
        with self._lock:
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="dhvaani_")
            os.close(fd)
            try:
                self._tts.synthesize(
                    text=text,
                    prompt_wav=prompt_wav,
                    prompt_text=prompt_text,
                    out_path=tmp,
                    num_step=self._num_step,
                    guidance_scale=self._guidance_scale,
                    speed=self._speed,
                    seed=self._seed,
                )
                import soundfile as sf

                data, _sr = sf.read(tmp, dtype="float64", always_2d=True)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

        mono = data[:, 0]
        pcm = np.clip(mono, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        return pcm, int(self._sr or _sr or _DEFAULT_SAMPLE_RATE)

    def unload(self) -> None:
        """Release the model, free memory, and clear CUDA cache."""
        with self._lock:
            self._tts = None
            self._sr = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_model_dir(self) -> Path:
        if self._model_dir:
            p = Path(self._model_dir).expanduser()
            if not p.is_dir():
                raise FileNotFoundError(f"model_dir does not exist: {p}")
            return p

        cached = self._find_cached_snapshot()
        if cached is not None:
            return cached

        return self._download_snapshot()

    @staticmethod
    def _find_cached_snapshot() -> Path | None:
        """Locate the snapshot under the HF cache (HF_HOME/hub)."""
        candidates = []
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            candidates.append(Path(hf_home) / "hub")
        # Known deployment cache for this project.
        candidates.append(Path("D:/hf-models/hub"))
        candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

        for base in candidates:
            snap_dir = base / "models--ARTPARK-IISc--DhVaani-0.5" / "snapshots"
            try:
                snaps = sorted(snap_dir.glob("*"))
            except OSError:
                continue
            if snaps:
                return snaps[0]
        return None

    def _download_snapshot(self) -> Path:
        from huggingface_hub import snapshot_download

        token = self._hf_token or os.environ.get("HF_TOKEN")
        try:
            out = snapshot_download(repo_id=MODEL_REPO_ID, token=token)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download ARTPARK-IISc/DhVaani-0.5. The repo is "
                "gated (gated='auto') and requires an authorised HF token. "
                "Accept the gate at https://huggingface.co/ARTPARK-IISc/"
                "DhVaani-0.5 and make sure HF_TOKEN (backend/.env) belongs "
                "to an account in the authorised list, or pre-download the "
                f"snapshot into HF_HOME. Underlying error: {exc}"
            ) from exc
        return Path(out)

    def _read_sample_rate(self, model_dir: Path) -> int:
        model_json = model_dir / "model.json"
        try:
            with open(model_json, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            sr = int(cfg["feature"]["sampling_rate"])
            return sr or _DEFAULT_SAMPLE_RATE
        except Exception:
            return _DEFAULT_SAMPLE_RATE


def _find_spec(name: str):
    import importlib.util

    try:
        return importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
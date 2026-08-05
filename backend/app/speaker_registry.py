"""Speaker registry.

Manages speaker entries on disk: each speaker has a directory under
``data/speakers/{speaker_id}/`` containing:

- ``meta.json``          — display name, language, created_at, ref duration
- ``ref.wav``            — 3-10 s reference audio (16 kHz mono)

For XTTS, no separate embedding file is needed (the reference clip is used
at synthesis time). For a future embedding-based TTS engine, an
``embedding.pt`` file can be added here.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from .config import SETTINGS


_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _safe_id(raw: str) -> str:
    """Sanitize a user-supplied speaker id."""
    s = _SAFE_ID_RE.sub("_", raw.strip())
    s = s.strip("_")
    if not s:
        s = f"spk_{uuid.uuid4().hex[:8]}"
    return s


class SpeakerRegistry:
    def __init__(self, root: Path | None = None):
        self.root = root or SETTINGS.speakers_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, speaker_id: str) -> Path:
        return self.root / _safe_id(speaker_id)

    def exists(self, speaker_id: str) -> bool:
        return (self._dir(speaker_id) / "meta.json").exists()

    def list_speakers(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            meta_path = d / "meta.json"
            if meta_path.exists():
                out.append(json.loads(meta_path.read_text()))
        return out

    def get(self, speaker_id: str) -> dict | None:
        meta_path = self._dir(speaker_id) / "meta.json"
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text())

    def get_ref_wav(self, speaker_id: str) -> Path | None:
        d = self._dir(speaker_id)
        ref = d / "ref.wav"
        return ref if ref.exists() else None

    def create(
        self,
        speaker_id: str,
        display_name: str,
        language: str = "te",
        ref_audio_bytes: bytes | None = None,
        ref_audio_format: str = "wav",
    ) -> dict:
        sid = _safe_id(speaker_id)
        d = self._dir(sid)
        d.mkdir(parents=True, exist_ok=True)
        if ref_audio_bytes is not None:
            ref_path = d / "ref.wav"
            if ref_audio_format.lower() != "wav":
                # Convert via pydub
                from pydub import AudioSegment
                seg = AudioSegment.from_file(
                    _bytesio(ref_audio_bytes), format=ref_audio_format
                )
                seg = seg.set_channels(1).set_frame_rate(16000)
                seg.export(ref_path, format="wav")
            else:
                ref_path.write_bytes(ref_audio_bytes)
            duration = _wav_duration(ref_path)
        else:
            duration = 0.0
        meta = {
            "speaker_id": sid,
            "display_name": display_name,
            "language": language,
            "ref_audio_path": str(d / "ref.wav"),
            "ref_duration_s": duration,
            "prompt_text": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        logger.info(f"Speaker created: {sid} ({display_name}), ref={duration:.2f}s")
        return meta

    def update_meta(self, speaker_id: str, **fields) -> dict | None:
        meta = self.get(speaker_id)
        if meta is None:
            return None
        meta.update(fields)
        (self._dir(speaker_id) / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta

    def delete(self, speaker_id: str) -> bool:
        d = self._dir(speaker_id)
        if not d.exists():
            return False
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        logger.info(f"Speaker deleted: {speaker_id}")
        return True


def _bytesio(b: bytes):
    import io
    return io.BytesIO(b)


def _wav_duration(path: Path) -> float:
    import wave
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0

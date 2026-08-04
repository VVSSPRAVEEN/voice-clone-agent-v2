"""Audio helpers used by the Streamlit frontend."""
from __future__ import annotations

import io
import wave
from typing import Optional

import numpy as np


def pcm_int16_to_wav_bytes(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert int16 PCM numpy array to WAV file bytes (in-memory)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


def normalize_audio(int16_array: np.ndarray) -> np.ndarray:
    """Clip + dither slightly to avoid overflow on playback."""
    arr = np.clip(int16_array, -32768, 32767).astype(np.int16)
    return arr


def resample_linear(pcm: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Crude linear resampler for cases where librosa isn't available."""
    if orig_sr == target_sr:
        return pcm
    ratio = target_sr / orig_sr
    n_out = int(len(pcm) * ratio)
    indices = np.arange(n_out) / ratio
    idx_floor = np.clip(indices.astype(int), 0, len(pcm) - 1)
    idx_ceil = np.clip(idx_floor + 1, 0, len(pcm) - 1)
    frac = indices - idx_floor
    out = pcm[idx_floor] * (1 - frac) + pcm[idx_ceil] * frac
    return out.astype(np.int16)


def bytes_to_pcm_int16(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.int16).copy()


def split_audio_into_chunks(audio_path: str, chunk_seconds: float = 300.0,
                            sample_rate: int = 16000) -> list[bytes]:
    """Split a WAV file into N-second PCM chunks for chunked upload."""
    chunks = []
    with wave.open(audio_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        chunk_frames = int(chunk_seconds * sr)
        while True:
            frames = wf.readframes(chunk_frames)
            if not frames:
                break
            chunks.append(frames)
    return chunks


def get_wav_info(audio_path: str) -> dict:
    with wave.open(audio_path, "rb") as wf:
        return {
            "channels": wf.getnchannels(),
            "sampwidth": wf.getsampwidth(),
            "framerate": wf.getframerate(),
            "nframes": wf.getnframes(),
            "duration_s": wf.getnframes() / wf.getframerate(),
        }

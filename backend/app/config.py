"""Application configuration via Pydantic Settings.

All values come from environment variables (or .env file). See `.env.example`
for the full list and defaults.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    pipeline_mode: Literal["parallel", "sequential"] = "parallel"
    max_concurrent_calls: int = 1

    # --- Paths ---
    data_dir: Path = Path("/app/data")
    speakers_dir: Path = Path("/app/data/speakers")
    calls_dir: Path = Path("/app/data/calls")
    models_dir: Path = Path("/app/data/models")
    db_path: Path = Path("/app/data/calls.db")

    # --- GPU ---
    cuda_visible_devices: str = "0"
    pytorch_cuda_alloc_conf: str = "max_split_size_mb:128"
    force_cpu: bool = False

    # --- STT ---
    stt_model: str = "medium"
    stt_compute_type: str = "int8"
    stt_device_cuda: bool = True
    stt_language: str = "te"
    stt_beam_size: int = 1

    # --- VAD ---
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_max_speech_ms: int = 30_000
    vad_silence_ms: int = 500

    # --- TTS ---
    tts_engine: Literal["xtts", "sherpa", "ai4bharat", "edge"] = "edge"
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_device: Literal["cuda", "cpu"] = "cuda"
    xtts_language: str = "te"
    edge_voice: str = "te-IN-MohanNeural"  # Edge-TTS Telugu male; te-IN-ShrutiNeural = female

    # --- LLM (local vLLM) ---
    llm_enabled: bool = True
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct-AWQ"  # 4-bit AWQ fits 6GB VRAM
    llm_api_url: str = "http://127.0.0.1:8001"  # Local vLLM OpenAI-compatible server
    llm_system_prompt: str = (
        "You are a friendly bilingual voice agent. Reply in the same language "
        "the user spoke (Telugu or English). Keep replies under 2 sentences "
        "unless asked for detail."
    )
    llm_max_tokens: int = 128
    llm_temperature: float = 0.7

    # --- Audio I/O ---
    sample_rate: int = 16000
    tts_sample_rate: int = 24000
    channels: int = 1
    chunk_ms: int = 20

    # --- Storage ---
    storage_backend: Literal["local", "minio"] = "local"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "voice-clone-agent"
    minio_secure: bool = False

    # --- Auth ---
    auth_mode: Literal["none", "apikey"] = "none"
    api_key: str = "change-me"

    # --- Derived helpers ---
    @property
    def device(self) -> str:
        if self.force_cpu:
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    @property
    def stt_device(self) -> str:
        if self.force_cpu:
            return "cpu"
        return "cuda" if self.stt_device_cuda else "cpu"

    @property
    def vad_sample_rate(self) -> int:
        return 16000

    @field_validator("stt_language", mode="before")
    @classmethod
    def _normalize_lang(cls, v: str) -> str:
        if not v:
            return "auto"
        v = v.strip().lower()
        if v in ("auto", "auto-detect"):
            return "auto"
        return v

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir, self.speakers_dir, self.calls_dir,
            self.models_dir, self.models_dir / "hf",
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


SETTINGS = get_settings()

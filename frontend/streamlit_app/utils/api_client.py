"""API client for the Voice Clone Agent backend.

Wraps the REST and WebSocket endpoints in a thin async-friendly class.
For Streamlit, calls are synchronous from the UI's perspective but use
``httpx`` under the hood.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import httpx


def _backend_url() -> str:
    return os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")


def _api_key() -> Optional[str]:
    return os.environ.get("API_KEY")


class APIClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url or _backend_url()
        self.api_key = api_key or _api_key()
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=({"X-API-Key": self.api_key} if self.api_key else {}),
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    # --- Health / Settings ---
    def health(self) -> dict:
        return self._client.get("/health").json()

    def settings(self) -> dict:
        return self._client.get("/settings").json()

    # --- Speakers ---
    def list_speakers(self) -> list[dict]:
        return self._client.get("/speakers").json()

    def create_speaker(self, speaker_id: str, display_name: str,
                       language: str, ref_audio_bytes: bytes,
                       ref_format: str = "wav") -> dict:
        files = {"ref_audio": (f"ref.{ref_format}", ref_audio_bytes, f"audio/{ref_format}")}
        data = {
            "speaker_id": speaker_id,
            "display_name": display_name,
            "language": language,
        }
        r = self._client.post("/speakers", data=data, files=files)
        r.raise_for_status()
        return r.json()

    def delete_speaker(self, speaker_id: str) -> dict:
        r = self._client.delete(f"/speakers/{speaker_id}")
        r.raise_for_status()
        return r.json()

    def speaker_ref_url(self, speaker_id: str) -> str:
        return f"{self.base_url}/speakers/{speaker_id}/ref.wav"

    # --- TTS ---
    def synthesize_tts(self, text: str, speaker_id: str,
                       language: Optional[str] = None) -> tuple[bytes, dict]:
        r = self._client.post("/tts/synthesize", json={
            "text": text,
            "speaker_id": speaker_id,
            "language": language,
        })
        r.raise_for_status()
        headers = {
            "latency_ms": r.headers.get("X-TTS-Latency-ms", "?"),
            "engine": r.headers.get("X-TTS-Engine", "?"),
        }
        return r.content, headers

    # --- Calls ---
    def list_calls(self, limit: int = 100, offset: int = 0) -> list[dict]:
        r = self._client.get("/calls", params={"limit": limit, "offset": offset})
        r.raise_for_status()
        return r.json()

    def get_call(self, call_id: str) -> dict:
        r = self._client.get(f"/calls/{call_id}")
        r.raise_for_status()
        return r.json()

    def get_call_transcript(self, call_id: str) -> list[dict]:
        r = self._client.get(f"/calls/{call_id}/transcript")
        r.raise_for_status()
        return r.json()

    def get_call_audio_url(self, call_id: str) -> str:
        return f"{self.base_url}/calls/{call_id}/audio"

    def delete_call(self, call_id: str) -> dict:
        r = self._client.delete(f"/calls/{call_id}")
        r.raise_for_status()
        return r.json()

    # --- Chunked upload ---
    def init_chunked_upload(self, filename: str, total_chunks: int,
                            call_id: Optional[str] = None) -> dict:
        r = self._client.post("/upload/chunked/init", json={
            "filename": filename,
            "total_chunks": total_chunks,
            "call_id": call_id,
        })
        r.raise_for_status()
        return r.json()

    def upload_chunk(self, upload_id: str, chunk_index: int,
                     chunk_bytes: bytes) -> dict:
        files = {"chunk": (f"chunk_{chunk_index}.bin", chunk_bytes, "application/octet-stream")}
        r = self._client.post(
            f"/upload/chunked/{upload_id}/{chunk_index}",
            files=files,
        )
        r.raise_for_status()
        return r.json()

    def complete_chunked_upload(self, upload_id: str,
                                speaker_id: Optional[str] = None,
                                title: Optional[str] = None) -> dict:
        r = self._client.post("/upload/chunked/complete", json={
            "upload_id": upload_id,
            "speaker_id": speaker_id,
            "title": title,
        })
        r.raise_for_status()
        return r.json()

    # --- WebSocket ---
    def ws_call_url(self) -> str:
        url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        return f"{url}/ws/call"

    def close(self):
        self._client.close()

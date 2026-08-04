"""Call logger.

Persists call metadata + JSONL transcripts + audio files.

Layout under ``data/calls/``:

    data/calls/{call_id}/
        audio.wav         — mixed call audio (16 kHz mono)
        transcript.jsonl  — one JSON line per segment
        meta.json         — call metadata

Also writes a row into SQLite ``calls`` table for fast listing.
"""
from __future__ import annotations

import asyncio
import json
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from .config import SETTINGS


class CallLogger:
    def __init__(self, db_path: Path | None = None, calls_dir: Path | None = None):
        self.db_path = db_path or SETTINGS.db_path
        self.calls_dir = calls_dir or SETTINGS.calls_dir
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self._db = None
        self._lock = asyncio.Lock()

    async def _ensure_db(self):
        if self._db is not None:
            return
        import aiosqlite
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                call_id TEXT PRIMARY KEY,
                speaker_id TEXT NOT NULL,
                title TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                duration_s REAL DEFAULT 0,
                audio_path TEXT,
                transcript_path TEXT,
                segment_count INTEGER DEFAULT 0
            )
            """
        )
        await self._db.commit()

    async def create_call(
        self,
        speaker_id: str,
        title: str | None = None,
        call_id: str | None = None,
    ) -> str:
        call_id = call_id or f"call_{uuid.uuid4().hex[:12]}"
        call_dir = self.calls_dir / call_id
        call_dir.mkdir(parents=True, exist_ok=True)
        # Empty transcript file
        (call_dir / "transcript.jsonl").touch()
        # Empty audio file placeholder (we'll write a proper WAV when call ends)
        meta = {
            "call_id": call_id,
            "speaker_id": speaker_id,
            "title": title,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "duration_s": 0.0,
            "audio_path": str(call_dir / "audio.wav"),
            "transcript_path": str(call_dir / "transcript.jsonl"),
            "segment_count": 0,
        }
        (call_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        await self._ensure_db()
        async with self._lock:
            await self._db.execute(
                """INSERT INTO calls
                   (call_id, speaker_id, title, started_at, ended_at, duration_s, audio_path, transcript_path, segment_count)
                   VALUES (?, ?, ?, ?, NULL, 0, ?, ?, 0)""",
                (call_id, speaker_id, title, meta["started_at"],
                 str(call_dir / "audio.wav"), str(call_dir / "transcript.jsonl")),
            )
            await self._db.commit()
        logger.info(f"Call created: {call_id} (speaker={speaker_id})")
        return call_id

    async def append_segment(
        self,
        call_id: str,
        t0: float,
        t1: float,
        speaker: str,
        text: str,
        is_final: bool = True,
    ) -> None:
        call_dir = self.calls_dir / call_id
        if not call_dir.exists():
            logger.warning(f"append_segment: call dir not found: {call_dir}")
            return
        line = json.dumps({
            "t0": t0,
            "t1": t1,
            "speaker": speaker,
            "text": text,
            "is_final": is_final,
        }, ensure_ascii=False)
        async with self._lock:
            with open(call_dir / "transcript.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
            await self._ensure_db()
            await self._db.execute(
                "UPDATE calls SET segment_count = segment_count + 1 WHERE call_id = ?",
                (call_id,),
            )
            await self._db.commit()

    async def finalize_call(
        self,
        call_id: str,
        audio_int16: np.ndarray | None = None,
        sample_rate: int = 16000,
    ) -> None:
        call_dir = self.calls_dir / call_id
        if not call_dir.exists():
            return
        ended_at = datetime.now(timezone.utc).isoformat()
        # Read started_at from meta
        meta = json.loads((call_dir / "meta.json").read_text())
        started_at = datetime.fromisoformat(meta["started_at"])
        duration = (datetime.fromisoformat(ended_at) - started_at).total_seconds()

        if audio_int16 is not None and audio_int16.size > 0:
            wav_path = call_dir / "audio.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())

        meta["ended_at"] = ended_at
        meta["duration_s"] = duration
        (call_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        await self._ensure_db()
        async with self._lock:
            await self._db.execute(
                "UPDATE calls SET ended_at = ?, duration_s = ? WHERE call_id = ?",
                (ended_at, duration, call_id),
            )
            await self._db.commit()
        logger.info(f"Call finalized: {call_id} ({duration:.1f}s)")

    async def list_calls(self, limit: int = 100, offset: int = 0) -> list[dict]:
        await self._ensure_db()
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT call_id, speaker_id, title, started_at, ended_at, duration_s, audio_path, transcript_path, segment_count "
                "FROM calls ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = await cursor.fetchall()
        cols = ["call_id", "speaker_id", "title", "started_at", "ended_at",
                "duration_s", "audio_path", "transcript_path", "segment_count"]
        return [dict(zip(cols, r)) for r in rows]

    async def get_call(self, call_id: str) -> dict | None:
        await self._ensure_db()
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT call_id, speaker_id, title, started_at, ended_at, duration_s, audio_path, transcript_path, segment_count "
                "FROM calls WHERE call_id = ?",
                (call_id,),
            )
            row = await cursor.fetchone()
        if not row:
            return None
        cols = ["call_id", "speaker_id", "title", "started_at", "ended_at",
                "duration_s", "audio_path", "transcript_path", "segment_count"]
        return dict(zip(cols, row))

    async def get_transcript(self, call_id: str) -> list[dict]:
        path = self.calls_dir / call_id / "transcript.jsonl"
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
        return out

    async def delete_call(self, call_id: str) -> bool:
        call_dir = self.calls_dir / call_id
        import shutil
        if call_dir.exists():
            shutil.rmtree(call_dir, ignore_errors=True)
        await self._ensure_db()
        async with self._lock:
            await self._db.execute("DELETE FROM calls WHERE call_id = ?", (call_id,))
            await self._db.commit()
        return True

    async def count(self) -> int:
        await self._ensure_db()
        async with self._lock:
            cursor = await self._db.execute("SELECT COUNT(*) FROM calls")
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

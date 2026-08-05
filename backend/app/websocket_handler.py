"""WebSocket handler for live call streaming.

Protocol (JSON envelopes over text frames; binary audio over binary frames):

Client -> Server:
  - Text frame: {"type":"hello","speaker_id":"...","call_id":"...?","title":"...?","sample_rate":16000}
  - Binary frame: int16 PCM 16 kHz mono chunk (20-50 ms)
  - Text frame: {"type":"end"}            # graceful end

Server -> Client (text frames):
  - {"type":"hello","call_id":"..."}
  - {"type":"vad","event":"speech_end","t0":..,"t1":..}
  - {"type":"transcript","t0":..,"t1":..,"speaker":"user","text":"..","language":"te","is_final":true}
  - {"type":"llm","text":"..","source":"llm|fallback","latency_ms":..}
  - {"type":"audio","pcm_int16":[...],"sample_rate":24000,"engine":"xtts"}
  - {"type":"audio_end","t0":..,"t1":..,"latency_ms":..}
  - {"type":"status","message":"..."}
  - {"type":"error","stage":"...","message":"..."}
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import librosa
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from .config import SETTINGS
from .pipeline import Pipeline, PipelineEvent


class ConnectionManager:
    """Tracks active WebSocket connections; enforces MAX_CONCURRENT_CALLS."""
    def __init__(self, max_concurrent: int):
        self.max_concurrent = max(1, max_concurrent)
        self.active: set[WebSocket] = set()
        self._sem = asyncio.Semaphore(self.max_concurrent)

    async def accept(self, ws: WebSocket) -> bool:
        if len(self.active) >= self.max_concurrent:
            await ws.accept()
            await ws.send_text(json.dumps({
                "type": "error",
                "data": {"message": "Server busy: max concurrent calls reached. Please retry."},
            }))
            await ws.close(code=1013)
            return False
        await ws.accept()
        self.active.add(ws)
        return True

    def release(self, ws: WebSocket):
        self.active.discard(ws)

    async def acquire_slot(self) -> bool:
        await self._sem.acquire()
        return True

    def release_slot(self):
        self._sem.release()


async def handle_call_websocket(
    ws: WebSocket,
    pipeline: Pipeline,
    manager: ConnectionManager,
):
    accepted = await manager.accept(ws)
    if not accepted:
        return

    # Wait for hello
    try:
        hello_raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        manager.release(ws)
        try:
            await ws.close(code=1008)
        except Exception:
            pass
        return

    try:
        hello = json.loads(hello_raw)
    except Exception:
        await ws.send_text(json.dumps({"type": "error", "data": {"message": "Invalid hello JSON"}}))
        await ws.close()
        manager.release(ws)
        return

    if hello.get("type") != "hello" or "speaker_id" not in hello:
        await ws.send_text(json.dumps({"type": "error", "data": {"message": "Expected hello with speaker_id"}}))
        await ws.close()
        manager.release(ws)
        return

    speaker_id = hello["speaker_id"]
    title = hello.get("title")
    call_id = hello.get("call_id")
    client_sr = int(hello.get("sample_rate", 16000))
    stt_model = hello.get("stt_model")
    if stt_model not in (None, "medium", "small"):
        logger.warning(f"Ignoring unknown stt_model={stt_model!r} (client speaker={speaker_id})")
        stt_model = None

    # Acquire a processing slot (bounded concurrency)
    await manager.acquire_slot()
    try:
        # Create audio queue from incoming binary frames
        audio_q: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=64)
        client_gone = False

        async def audio_producer():
            nonlocal client_gone
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        client_gone = True
                        break
                    if "bytes" in msg and msg["bytes"] is not None:
                        raw = msg["bytes"]
                        if len(raw) % 2:
                            raw += b"\x00"  # tolerate odd-length frames
                        arr = np.frombuffer(raw, dtype=np.int16).copy()
                        if client_sr != 16000:
                            # Resample via librosa if needed
                            f32 = arr.astype(np.float32) / 32768.0
                            f32 = librosa.resample(f32, orig_sr=client_sr, target_sr=16000)
                            arr = (f32 * 32768.0).astype(np.int16)
                        await audio_q.put(arr)
                    elif "text" in msg and msg["text"] is not None:
                        try:
                            t = json.loads(msg["text"])
                            if t.get("type") == "end":
                                break
                        except Exception:
                            pass
            except WebSocketDisconnect:
                client_gone = True
            finally:
                # Bounded put so a full queue can't hang the producer (and
                # with it the only call slot) forever.
                for _ in range(10):
                    try:
                        audio_q.put_nowait(None)
                        break
                    except asyncio.QueueFull:
                        await asyncio.sleep(0.1)

        async def audio_stream():
            while True:
                chunk = await audio_q.get()
                if chunk is None:
                    return
                yield chunk

        async def on_event(ev: PipelineEvent):
            try:
                await ws.send_text(json.dumps({
                    "type": ev.kind,
                    "data": ev.data,
                    "call_id": ev.data.get("call_id"),
                }, ensure_ascii=False))
            except Exception as e:
                logger.warning(f"WS send failed: {e}")

        # Send hello ack
        await ws.send_text(json.dumps({"type": "hello", "data": {"message": "connected"}}))

        producer_task = asyncio.create_task(audio_producer())
        first_call_id = call_id
        try:
            while True:
                pipe_task = asyncio.create_task(pipeline.run_streaming(
                    audio_stream=audio_stream(),
                    speaker_id=speaker_id,
                    call_id=first_call_id,
                    title=title,
                    stt_model=stt_model,
                    on_event=on_event,
                ))
                first_call_id = None  # later turns get fresh call ids
                done, _ = await asyncio.wait(
                    {producer_task, pipe_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if producer_task in done:
                    # Client sent "end" or disconnected
                    if client_gone:
                        pipe_task.cancel()
                        try:
                            await pipe_task
                        except Exception:
                            pass
                        logger.info(
                            f"Client disconnected during call (speaker={speaker_id}); "
                            "pipeline cancelled"
                        )
                    else:
                        done_call = await pipe_task
                        await ws.send_text(json.dumps({
                            "type": "call_end",
                            "call_id": done_call,
                            "data": {"call_id": done_call},
                        }))
                    break
                # Pipeline finished on its own (silence flush) while the
                # producer keeps running -> streaming: report the turn and
                # continue listening for the next utterance on this socket.
                try:
                    done_call = pipe_task.result()
                except Exception as e:
                    logger.exception(f"Pipeline error (speaker={speaker_id}): {e}")
                    try:
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "data": {"stage": "pipeline", "message": str(e)},
                        }))
                    except Exception:
                        pass
                    break
                await ws.send_text(json.dumps({
                    "type": "call_end",
                    "call_id": done_call,
                    "data": {"call_id": done_call},
                }))
        finally:
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except Exception:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected (speaker={speaker_id})")
    except Exception as e:
        logger.exception(f"WebSocket handler error: {e}")
        try:
            await ws.send_text(json.dumps({
                "type": "error",
                "data": {"message": str(e)},
            }))
        except Exception:
            pass
    finally:
        manager.release(ws)
        manager.release_slot()

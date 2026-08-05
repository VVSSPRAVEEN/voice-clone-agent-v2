"""Streamlit WebRTC full-duplex live call component (streamlit-webrtc 0.47.9).

Mic frames (48 kHz stereo s16 from aiortc) are downmixed, resampled to
16 kHz mono and streamed to the backend over WebSocket. The backend
answers with JSON `audio` events carrying int16 PCM reply chunks at the
TTS rate (24 kHz); those are returned as av.AudioFrame objects so the
browser plays them live. When no reply audio is queued, silence frames
are returned to avoid echoing the caller's own mic (SENDRECV passthrough).
"""
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Optional

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer

from utils.api_client import APIClient


class CallAudioProcessor(AudioProcessorBase):
    def __init__(self, api: APIClient, speaker_id: str, title: str, stt_model: str = "medium"):
        self.api = api
        self.speaker_id = speaker_id
        self.title = title
        self.stt_model = stt_model
        self.ws = None
        self._recv_thread: Optional[threading.Thread] = None
        self._started = False
        self._lock = threading.Lock()
        self.out_queue: queue.Queue = queue.Queue(maxsize=4000)
        self.transcripts: list[dict] = []
        self.llm_responses: list[dict] = []
        self.status_log: list[tuple[str, str]] = []
        self.status: str = "connecting"
        self.call_id: Optional[str] = None

    # --- WebSocket plumbing (runs in the framework's worker thread) ---------

    @staticmethod
    def _ws_dead(ws) -> bool:
        try:
            return getattr(ws, "close_code", None) is not None
        except Exception:
            return False

    def _start_ws(self, sample_rate: int):
        if self._started and self.ws is not None and not self._ws_dead(self.ws):
            return
        import websockets.sync.client as ws_sync
        self.ws = ws_sync.connect(
            self.api.ws_call_url(),
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            open_timeout=5,
        )
        self.ws.send(json.dumps({
            "type": "hello",
            "speaker_id": self.speaker_id,
            "title": self.title,
            "sample_rate": sample_rate,
            "stt_model": self.stt_model,
        }))
        self._started = True
        self.status = "connected"
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def _send_audio(self, pcm16: np.ndarray):
        self._start_ws(16000)
        try:
            self.ws.send(pcm16.tobytes())
        except Exception:
            # Dead socket (e.g. after a backend restart): reconnect once.
            self._started = False
            self._start_ws(16000)
            try:
                self.ws.send(pcm16.tobytes())
            except Exception:
                pass

    def _recv_loop(self):
        while True:
            if self.ws is None or self._ws_dead(self.ws):
                break
            try:
                raw = self.ws.recv(timeout=0.1)
            except Exception:
                continue
            if raw is None:
                continue
            if isinstance(raw, bytes):
                continue  # backend sends all reply audio as JSON text
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            kind = msg.get("type")
            data = msg.get("data") or {}
            if kind == "transcript":
                with self._lock:
                    self.transcripts.append(data)
            elif kind == "llm":
                with self._lock:
                    self.llm_responses.append(data)
            elif kind == "audio":
                arr = data.get("pcm_int16")
                if arr:
                    try:
                        self.out_queue.put_nowait(np.array(arr, dtype=np.int16))
                    except queue.Full:
                        pass
            elif kind == "status":
                msg_txt = data.get("message", "status")
                with self._lock:
                    self.status_log.append((time.strftime("%H:%M:%S"), msg_txt))
            elif kind == "call_end":
                self.call_id = data.get("call_id") or msg.get("call_id")
            elif kind == "error":
                self.status = f"error: {data.get('message', 'unknown')}"
                with self._lock:
                    self.status_log.append((time.strftime("%H:%M:%S"), self.status))

    # --- streamlit-webrtc contract ------------------------------------------

    async def recv_queued(self, frames) -> list[av.AudioFrame]:
        """Async mode: send mic frames to the backend, return reply audio."""
        if frames:
            try:
                import librosa
            except Exception:
                librosa = None
            for frame in frames:
                arr = frame.to_ndarray()  # (channels, samples) s16 @48k
                if arr.ndim > 1:
                    mono = arr.mean(axis=0)
                else:
                    mono = arr
                f32 = mono.astype(np.float32) / 32768.0
                if librosa is not None:
                    f32 = librosa.resample(
                        f32,
                        orig_sr=frame.sample_rate or 48000,
                        target_sr=16000,
                    )
                pcm = (f32 * 32768.0).astype(np.int16)
                self._send_audio(pcm)

        out: list[np.ndarray] = []
        while True:
            try:
                out.append(self.out_queue.get_nowait())
            except queue.Empty:
                break

        if not out:
            # Silence so SENDRECV doesn't echo the caller's mic back.
            silence = np.zeros(960, dtype=np.int16)
            out = [silence]
        frames_out = []
        for pcm in out:
            f = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(pcm).reshape(1, -1),
                format="s16",
                layout="mono",
            )
            f.sample_rate = 24000
            frames_out.append(f)
        return frames_out

    def on_ended(self):
        try:
            if self.ws is not None and not self._ws_dead(self.ws):
                self.ws.send(json.dumps({"type": "end"}))
                self.ws.close()
        except Exception:
            pass
        self.status = "stopped"


def render_live_stream(api: APIClient, speaker_id: str, title: str, stt_model: str = "medium"):
    """Render the full-duplex streaming call. Returns the streamer context."""
    ctx = webrtc_streamer(
        key="live-call",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "audio": {"echoCancellation": True, "noiseSuppression": True},
            "video": False,
        },
        audio_processor_factory=lambda: CallAudioProcessor(api, speaker_id, title, stt_model),
        async_processing=True,
        desired_playing_state=st.session_state.get("live_playing", False),
    )
    if ctx.state.playing:
        st.session_state["live_playing"] = True
    else:
        st.session_state["live_playing"] = False
    return ctx

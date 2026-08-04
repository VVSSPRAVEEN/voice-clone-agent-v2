"""Streamlit WebRTC audio component wrapper."""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Optional

import numpy as np
import streamlit as st
from streamlit_webrtc import AudioProcessorBase, ClientSettings, WebRtcMode, webrtc_streamer

from .api_client import APIClient


class CallAudioProcessor(AudioProcessorBase):
    """WebRTC audio processor that streams mic audio to the backend over
    WebSocket and plays back TTS audio chunks received from the backend.
    """
    def __init__(self, api: APIClient, speaker_id: str, title: str):
        self.api = api
        self.speaker_id = speaker_id
        self.title = title
        self.ws = None
        self.ws_thread = None
        self.in_queue: queue.Queue = queue.Queue(maxsize=200)
        self.out_queue: queue.Queue = queue.Queue(maxsize=400)
        self.transcripts: list[dict] = []
        self.llm_responses: list[dict] = []
        self.status: str = "initializing"
        self.call_id: Optional[str] = None
        self._lock = threading.Lock()
        self._started = False

    def _start_ws(self, sample_rate: int):
        if self._started:
            return
        self._started = True
        import websockets.sync.client as ws_sync
        url = self.api.ws_call_url()
        self.ws = ws_sync.connect(url, max_size=None)
        # Send hello
        self.ws.send(json.dumps({
            "type": "hello",
            "speaker_id": self.speaker_id,
            "title": self.title,
            "sample_rate": sample_rate,
        }))
        # Start receiver thread
        self.ws_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.ws_thread.start()
        self.status = "connected"

    def _recv_loop(self):
        while True:
            try:
                raw = self.ws.recv(timeout=0.1)
                if isinstance(raw, bytes):
                    # PCM int16 audio chunk
                    pcm = np.frombuffer(raw, dtype=np.int16)
                    try:
                        self.out_queue.put_nowait(pcm)
                    except queue.Full:
                        # Drop if we can't keep up
                        pass
                else:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    self._handle_msg(msg)
            except Exception:
                # Timeout or error; keep looping
                pass

    def _handle_msg(self, msg: dict):
        kind = msg.get("type")
        data = msg.get("data") or {}
        if kind == "hello":
            self.status = "connected"
        elif kind == "transcript":
            with self._lock:
                self.transcripts.append(data)
        elif kind == "llm":
            with self._lock:
                self.llm_responses.append(data)
        elif kind == "audio":
            # Audio chunks come as base64 PCM via JSON; convert
            import base64
            pcm_b64 = data.get("pcm_b64") or data.get("pcm_int16_b64")
            if pcm_b64:
                pcm = np.frombuffer(base64.b64decode(pcm_b64), dtype=np.int16)
            else:
                # JSON list form
                arr = data.get("pcm_int16")
                if arr is None:
                    return
                pcm = np.array(arr, dtype=np.int16)
            try:
                self.out_queue.put_nowait(pcm)
            except queue.Full:
                pass
        elif kind == "status":
            self.status = data.get("message", "status")
        elif kind == "error":
            self.status = f"error: {data.get('message','')}"
        elif kind == "call_end":
            self.call_id = data.get("call_id") or msg.get("call_id")

    def recv_audio(self, frames: list) -> Optional["np.ndarray"]:
        """Called by streamlit-webrtc with incoming mic frames (we send to WS)."""
        import av
        # frames: list of av.AudioFrame at 48 kHz stereo by default
        if not frames:
            return None
        # Convert first frame to 16 kHz mono int16
        frame = frames[0]
        arr = frame.to_ndarray()  # shape: (channels, samples) float or int
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        # Resample to 16 kHz
        from_sr = frame.sample_rate or 48000
        target_sr = 16000
        if from_sr != target_sr:
            import librosa
            arr_f = arr.astype(np.float32)
            if arr_f.max() > 1.0 or arr_f.min() < -1.0:
                arr_f = arr_f / 32768.0
            arr_f = librosa.resample(arr_f, orig_sr=from_sr, target_sr=target_sr)
            pcm = (arr_f * 32768.0).astype(np.int16)
        else:
            pcm = arr.astype(np.int16)

        if not self._started:
            self._start_ws(target_sr)

        # Send raw bytes to backend
        try:
            self.ws.send(pcm.tobytes())
        except Exception:
            pass
        return None  # we don't echo back mic audio

    def recv_queued(self) -> Optional["np.ndarray"]:
        """Called by streamlit-webrtc to pull TTS audio for playback."""
        try:
            pcm = self.out_queue.get_nowait()
        except queue.Empty:
            return None
        # Convert to av.AudioFrame at 24 kHz mono
        import av
        from streamlit_webrtc import RTCConfiguration
        # streamlit-webrtc expects a generator of frames; here we return raw
        # numpy and let the framework wrap it
        return pcm

    def stop(self):
        try:
            if self.ws:
                self.ws.send(json.dumps({"type": "end"}))
                self.ws.close()
        except Exception:
            pass
        self.status = "stopped"


def render_webrtc_call(api: APIClient, speaker_id: str, title: str):
    """Render the WebRTC-based call UI. Returns the audio processor handle."""
    rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ctx = webrtc_streamer(
        key="voice-call",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        desired_playing_state=st.session_state.get("webrtc_playing", True),
        audio_processor_factory=lambda: CallAudioProcessor(api, speaker_id, title),
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )
    return ctx

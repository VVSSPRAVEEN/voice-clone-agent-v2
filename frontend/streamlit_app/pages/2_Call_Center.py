"""Page 2: Call Center — live WebRTC call with the agent."""
from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
import time
from typing import Optional

import numpy as np
import streamlit as st

from utils.api_client import APIClient
from components.transcript_viewer import render_transcript


st.set_page_config(page_title="Call Center", page_icon="📞", layout="wide")
st.title("📞 Call Center")
st.caption("Live WebRTC call with the voice clone agent. Speak in Telugu or English.")

api: APIClient = st.session_state.get("api") or APIClient()
st.session_state["api"] = api


# --- Refresh speakers ------------------------------------------------------
@st.cache_data(ttl=10)
def _list_speakers() -> list[dict]:
    try:
        return api.list_speakers()
    except Exception as e:
        st.error(f"Failed to load speakers: {e}")
        return []


speakers = _list_speakers()
if not speakers:
    st.warning("No speakers registered. Go to Speaker Lab first.")
    st.stop()


# --- Speaker picker --------------------------------------------------------
col1, col2 = st.columns([1, 1])
with col1:
    spk_options = {s["speaker_id"]: f"{s['display_name']} ({s['speaker_id']})" for s in speakers}
    selected_speaker = st.selectbox(
        "Speaker (cloned voice to use for replies)",
        options=list(spk_options.keys()),
        format_func=lambda k: spk_options[k],
    )
with col2:
    call_title = st.text_input("Call title (optional)", value="", placeholder="e.g. Test call 1")


# --- WebSocket-based call (no WebRTC required) -----------------------------
# We use a simpler recorder approach: streamlit audio recorder component.
st.divider()
st.markdown("### Live call (browser mic → WebSocket)")

st.markdown("""
Use the controls below to start a live call. The browser captures your mic
audio in chunks, streams it to the backend over WebSocket, and receives
back the transcript and the bot's spoken reply in the cloned voice.
""")

# Use streamlit's audio_input component (simple, native) for recording
# individual utterances. For continuous full-duplex, switch to WebRTC mode below.
col_a, col_b, col_c = st.columns([1, 1, 4])
with col_a:
    start_call = st.button("🎙️ Start call session", type="primary")
with col_b:
    end_call = st.button("⏹️ End call")

# Session state for live call
if "call_ws" not in st.session_state:
    st.session_state["call_ws"] = None
if "call_transcripts" not in st.session_state:
    st.session_state["call_transcripts"] = []
if "call_llm" not in st.session_state:
    st.session_state["call_llm"] = []
if "call_id" not in st.session_state:
    st.session_state["call_id"] = None
if "call_audio_chunks" not in st.session_state:
    st.session_state["call_audio_chunks"] = []


def _start_session():
    import websockets.sync.client as ws_sync
    url = api.ws_call_url()
    ws = ws_sync.connect(url, max_size=None)
    ws.send(json.dumps({
        "type": "hello",
        "speaker_id": selected_speaker,
        "title": call_title or None,
        "sample_rate": 16000,
    }))
    st.session_state["call_ws"] = ws
    st.session_state["call_transcripts"] = []
    st.session_state["call_llm"] = []
    st.session_state["call_audio_chunks"] = []
    # Start receiver thread
    t = threading.Thread(target=_recv_loop, daemon=True)
    st.session_state["call_recv_thread"] = t
    t.start()


def _recv_loop():
    ws = st.session_state.get("call_ws")
    if ws is None:
        return
    while ws is not None and ws.close_code is None:
        try:
            raw = ws.recv(timeout=0.2)
        except Exception:
            continue
        if raw is None:
            continue
        if isinstance(raw, bytes):
            # PCM int16 audio chunk
            pcm = np.frombuffer(raw, dtype=np.int16)
            st.session_state["call_audio_chunks"].append(pcm)
        else:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            kind = msg.get("type")
            data = msg.get("data") or {}
            if kind == "transcript":
                st.session_state["call_transcripts"].append(data)
            elif kind == "llm":
                st.session_state["call_llm"].append(data)
                # Also append the bot's text as a transcript segment
                st.session_state["call_transcripts"].append({
                    "t0": data.get("t0", 0),
                    "t1": data.get("t1", 0),
                    "speaker": "bot",
                    "text": data.get("text", ""),
                    "is_final": True,
                })
            elif kind == "call_end":
                st.session_state["call_id"] = data.get("call_id") or msg.get("call_id")
                break
            elif kind == "error":
                st.session_state["call_error"] = data.get("message", "unknown error")


def _end_session():
    ws = st.session_state.get("call_ws")
    if ws is not None:
        try:
            ws.send(json.dumps({"type": "end"}))
        except Exception:
            pass
        try:
            ws.close()
        except Exception:
            pass
    st.session_state["call_ws"] = None


if start_call:
    _start_session()
    st.success("Call session started. Record an utterance below.")

if end_call:
    _end_session()
    st.info("Call session ended.")


# --- Mic recorder ----------------------------------------------------------
st.divider()
st.markdown("### Record an utterance")
st.caption("Click record, speak (Telugu or English), then click stop. The audio is sent to the agent over WebSocket.")

# streamlit audio_input returns WAV bytes on completion
audio_value = st.audio_input("Click to record")
if audio_value is not None:
    wav_bytes = audio_value.getvalue()
    if wav_bytes and st.session_state.get("call_ws"):
        # Convert WAV bytes to 16 kHz mono int16 PCM
        import io, wave
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sr = wf.getframerate()
                n_ch = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())
            if sampwidth != 2:
                st.error("Only 16-bit PCM WAV supported by browser recorder.")
            else:
                arr = np.frombuffer(raw, dtype=np.int16)
                if n_ch > 1:
                    arr = arr.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
                if sr != 16000:
                    try:
                        import librosa
                        f32 = arr.astype(np.float32) / 32768.0
                        f32 = librosa.resample(f32, orig_sr=sr, target_sr=16000)
                        arr = (f32 * 32768.0).astype(np.int16)
                    except Exception:
                        st.warning("librosa not available; sending audio at original sample rate.")
                # Send as binary over WS
                ws = st.session_state["call_ws"]
                try:
                    ws.send(arr.tobytes())
                    st.success(f"Sent {len(arr)/16000:.1f}s of audio to the agent.")
                except Exception as e:
                    st.error(f"WS send failed: {e}")
        except Exception as e:
            st.error(f"WAV decode failed: {e}")
    elif not st.session_state.get("call_ws"):
        st.warning("Start a call session first.")


# --- Live transcript + bot audio -------------------------------------------
st.divider()
col_t, col_b = st.columns([2, 1])

with col_t:
    st.markdown("### Live transcript")
    render_transcript(st.session_state.get("call_transcripts", []))

with col_b:
    st.markdown("### Bot audio")
    chunks = st.session_state.get("call_audio_chunks", [])
    if chunks:
        # Concatenate all chunks at 24 kHz (XTTS native rate)
        all_pcm = np.concatenate(chunks)
        import io, wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(all_pcm.tobytes())
        st.audio(buf.getvalue(), format="audio/wav")
        st.caption(f"{len(all_pcm)/24000:.1f}s of bot audio")
    else:
        st.caption("No bot audio yet.")

    st.markdown("#### LLM responses")
    for r in st.session_state.get("call_llm", [])[-5:]:
        st.markdown(f"- {r.get('text','')} _({r.get('source','?')}, {r.get('latency_ms',0):.0f}ms)_")


# --- Refresh button (manual since WS is async) -----------------------------
if st.button("🔄 Refresh view"):
    st.rerun()

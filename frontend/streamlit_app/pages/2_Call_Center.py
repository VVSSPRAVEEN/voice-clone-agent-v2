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

call_mode = st.radio(
    "Call mode",
    ["📝 Record clip", "🔴 Live streaming (WebRTC)"],
    horizontal=True,
    help="Record clip: press record, speak, stop — reliable and simple. "
         "Live streaming: hands-free full-duplex call via WebRTC.",
)


# --- Live streaming mode (WebRTC full-duplex) ------------------------------
if call_mode == "🔴 Live streaming (WebRTC)":
    from components.webrtc_audio import render_live_stream

    ctx = render_live_stream(api, selected_speaker, call_title or None)
    if ctx.state.playing:
        proc = ctx.audio_processor
        st.divider()
        st.markdown("### Backend status")
        if proc is not None:
            st.caption(f"Status: `{proc.status}`")
            if proc.llm_responses:
                last = proc.llm_responses[-1]
                st.markdown(f"**Agent said:** {last.get('text','')} _({last.get('source','?')}, {last.get('latency_ms',0):.0f}ms)_")
            if proc.call_id:
                st.caption(f"Call: {proc.call_id}")
            with st.expander("Status log", expanded=False):
                for t, m in proc.status_log[-12:]:
                    st.code(f"{t}  {m}")
            st.markdown("### Transcript")
            for tr in proc.transcripts[-10:]:
                who = "🧑 user" if tr.get("speaker") == "user" else "🤖 bot"
                st.markdown(f"- **{who}:** {tr.get('text','')}")
        else:
            st.caption("Waiting for the WebRTC session to start… (grant mic access)")
        time.sleep(2)
        st.rerun()
    else:
        st.caption("Click ▶ Start to begin the live call. Speak, pause, and the agent replies in the cloned voice — multi-turn on one connection.")
    st.stop()


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
if "call_status_log" not in st.session_state:
    st.session_state["call_status_log"] = []


def _log_status(message: str, seconds: Optional[float] = None):
    entry = {"t": time.strftime("%H:%M:%S"), "message": message, "seconds": seconds}
    log = st.session_state.get("call_status_log", [])
    log.append(entry)
    st.session_state["call_status_log"] = log[-30:]


def _start_session():
    import websockets.sync.client as ws_sync
    url = api.ws_call_url()
    ws = ws_sync.connect(url, max_size=None, ping_interval=None, ping_timeout=None)
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
    st.session_state["call_status_log"] = []
    _log_status("connected to backend (ws)")
    st.session_state.pop("call_ws_error", None)
    # Start receiver thread
    t = threading.Thread(target=_recv_loop, daemon=True)
    st.session_state["call_recv_thread"] = t
    t.start()


def _ensure_live_ws():
    """Return the live websocket, reconnecting once if the old one died."""
    ws = st.session_state.get("call_ws")
    if ws is not None and _ws_dead(ws) is False:
        return ws
    _start_session()
    return st.session_state.get("call_ws")


def _ws_dead(ws) -> bool:
    """True if the websocket is closed. 'close_code' is not present on all
    versions of websockets.sync, so use getattr."""
    try:
        return getattr(ws, "close_code", None) is not None
    except Exception:
        return False


def _recv_loop():
    ws = st.session_state.get("call_ws")
    if ws is None:
        return
    failures = 0
    while True:
        if _ws_dead(ws):
            if st.session_state.get("call_ws") is ws:
                st.session_state["call_ws_error"] = "Connection closed by server."
                st.session_state["call_ws"] = None
            return
        try:
            raw = ws.recv(timeout=0.2)
            failures = 0
        except Exception as e:
            if _ws_dead(ws):
                if st.session_state.get("call_ws") is ws:
                    st.session_state["call_ws_error"] = "Connection closed by the agent."
                    st.session_state["call_ws"] = None
                return
            failures += 1
            if failures >= 5:
                # Socket is hard-dead (no close frame). Let the sender reconnect.
                if st.session_state.get("call_ws") is ws:
                    st.session_state["call_ws_error"] = "Connection to the agent was lost (socket died)."
                    st.session_state["call_ws"] = None
                return
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
                _log_status("call finished - reply done")
                # Session is over server-side; force a fresh connection
                # for the next utterance.
                if st.session_state.get("call_ws") is ws:
                    st.session_state["call_ws"] = None
                break
            elif kind == "status":
                _log_status(data.get("message", "status"), data.get("seconds"))
            elif kind == "error":
                st.session_state["call_error"] = data.get("message", "unknown error")
                _log_status(f"error: {data.get('message', 'unknown')}")


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
    st.session_state.pop("call_ws_error", None)
    st.success("Call session started. Record an utterance below.")

if end_call:
    _end_session()
    st.info("Call session ended.")

if st.session_state.get("call_ws_error"):
    st.error(f"{st.session_state['call_ws_error']} — press 'Start call session' to reconnect.")


# --- Mic recorder ----------------------------------------------------------
st.divider()
st.markdown("### Record an utterance")
st.caption("Click record, speak (Telugu or English), then click stop. The reply appears automatically — no need to click refresh.")

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
                # Send as binary over WS (auto-reconnect once if the socket died)
                ws = _ensure_live_ws()
                if ws is None:
                    st.error("Could not connect to the agent. Is the backend running on :8000?")
                else:
                    try:
                        ws.send(arr.tobytes())
                        ws.send(json.dumps({"type": "end"}))
                        st.success(f"Sent {len(arr)/16000:.1f}s of audio. Agent is replying — this can take 10-60s on CPU.")
                    except Exception as e:
                        st.error(f"WS send failed: {e} — press 'Start call session' to reconnect, then record again.")
        except Exception as e:
            st.error(f"WAV decode failed: {e}")
    elif not st.session_state.get("call_ws"):
        st.warning("Start a call session first.")


# --- Backend status (live progress from the pipeline) ----------------------
st.divider()
st.markdown("### Backend status")
st.caption("Live view of what the backend is doing with your audio — updates automatically.")

status_names = {
    "connected to backend (ws)": "Connected to backend",
    "pipeline_started": "Call session open",
    "listening": "Listening — audio streaming in",
    "audio_received": "Audio received, starting STT",
    "stt_transcribing": "Transcribing your speech…",
    "llm_thinking": "Agent is thinking…",
    "tts_synthesizing": "Synthesizing reply in the cloned voice…",
    "audio_end": "Reply audio sent",
    "pipeline_finished": "Call finished",
    "no_audio_received": "No audio received — record and send an utterance",
    "no_speech_detected": "No speech detected in the audio — try again",
}
status_log = st.session_state.get("call_status_log", [])
if status_log:
    last = status_log[-1]
    label = status_names.get(last["message"], last["message"])
    if last.get("seconds") is not None:
        label += f" — {last['seconds']}s of audio so far"
    if last["message"] == "listening":
        st.info(label)
    elif last["message"] in ("stt_transcribing", "llm_thinking", "tts_synthesizing"):
        st.warning(label)
    elif last["message"].startswith("error"):
        st.error(label)
    else:
        st.success(label)
    with st.expander(f"Status log ({len(status_log)} events)", expanded=False):
        for e in status_log[-12:]:
            sec = f" ({e['seconds']}s)" if e.get("seconds") is not None else ""
            st.code(f"{e['t']}  {e['message']}{sec}")
else:
    st.caption("No backend status yet — start a call session and record an utterance.")


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


# --- Auto-refresh while waiting for the bot's reply ------------------------
# The WS receiver thread fills session_state; Streamlit only re-renders on
# user interaction, so auto-rerun every few seconds until the reply audio
# arrives or the session ended. Once done, stop polling.
_waiting = (st.session_state.get("call_ws") is not None
            and not st.session_state.get("call_audio_chunks")
            and not st.session_state.get("call_id"))
if _waiting:
    time.sleep(4)
    st.rerun()

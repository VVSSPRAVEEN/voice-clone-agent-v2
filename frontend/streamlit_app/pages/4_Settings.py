"""Page 4: Settings — view config, GPU usage, and upload long audio."""
from __future__ import annotations

import math
import time

import streamlit as st

from utils.api_client import APIClient
from utils.audio_utils import get_wav_info


st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
st.title("⚙️ Settings")
st.caption("View backend configuration, VRAM usage, and upload long audio (300+ minutes).")

api: APIClient = st.session_state.get("api") or APIClient()
st.session_state["api"] = api


# --- Settings snapshot -----------------------------------------------------
st.markdown("### Backend configuration")
try:
    s = api.settings()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pipeline mode", s["pipeline_mode"])
        st.metric("Max concurrent calls", s["max_concurrent_calls"])
        st.metric("Device", s["device"])
    with col2:
        st.metric("STT model", s["stt_model"])
        st.metric("STT language", s["stt_language"])
        st.metric("TTS engine", s["tts_engine"])
        st.metric("TTS language", s["xtts_language"])
    with col3:
        st.metric("LLM enabled", str(s["llm_enabled"]))
        st.metric("LLM model", s.get("llm_model", "-") if s["llm_enabled"] else "—")
        vram_total = s.get("vram_total_mb", 0)
        vram_used = s.get("vram_used_mb", 0)
        if vram_total > 0:
            st.metric("VRAM total", f"{vram_total} MB")
            st.metric("VRAM used", f"{vram_used} MB ({vram_used/vram_total*100:.1f}%)")
            st.progress(min(vram_used / max(vram_total, 1), 1.0))
except Exception as e:
    st.error(f"Failed to load settings: {e}")


# --- Long audio upload (300-min) ------------------------------------------
st.divider()
st.markdown("### Upload long audio (300+ minutes supported)")
st.caption("""
Upload a long WAV/MP3 file. The backend splits it into 5-min chunks,
streams them through faster-whisper, and writes a JSONL transcript you can
browse in the History page. The file is **never** loaded fully into RAM.
""")

with st.form("long_upload_form", clear_on_submit=False):
    title = st.text_input("Title", value="", placeholder="e.g. 5-hour interview")
    speaker_id = st.text_input("Speaker ID label (for transcript)", value="upload", disabled=True)
    chunk_seconds = st.number_input(
        "Chunk size (seconds)", min_value=30, max_value=900, value=300, step=30,
    )
    up_file = st.file_uploader(
        "Audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        accept_multiple_files=False,
    )
    submitted = st.form_submit_button("Upload & transcribe", type="primary")

if submitted and up_file is not None:
    raw = up_file.read()
    total_bytes = len(raw)
    st.info(f"Uploading {total_bytes/1e6:.1f} MB file in chunks...")

    # Save to temp file first to compute duration
    import tempfile, os
    suffix = os.path.splitext(up_file.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        info = get_wav_info(tmp_path) if suffix.lower() == ".wav" else None
    except Exception:
        info = None

    if info:
        duration_s = info["duration_s"]
        st.caption(f"Detected WAV: {info['framerate']} Hz, {info['nchannels']}ch, {duration_s:.1f}s")
        # Compute chunks
        chunk_bytes_per_sec = total_bytes / duration_s if duration_s > 0 else total_bytes
        chunk_size_bytes = int(chunk_bytes_per_sec * chunk_seconds)
        total_chunks = math.ceil(total_bytes / chunk_size_bytes)
    else:
        # Unknown duration; fall back to 5 MB chunks
        chunk_size_bytes = 5 * 1024 * 1024
        total_chunks = math.ceil(total_bytes / chunk_size_bytes)

    st.caption(f"Splitting into {total_chunks} chunks of ~{chunk_size_bytes/1e6:.1f} MB each")

    # Initialize chunked upload
    try:
        init = api.init_chunked_upload(
            filename=up_file.name,
            total_chunks=total_chunks,
        )
        upload_id = init["upload_id"]
        call_id = init["call_id"]
        st.success(f"Upload session created (call_id={call_id})")
    except Exception as e:
        st.error(f"Init failed: {e}")
        st.stop()

    # Send chunks
    progress = st.progress(0.0)
    status = st.empty()
    for i in range(total_chunks):
        start = i * chunk_size_bytes
        end = min(start + chunk_size_bytes, total_bytes)
        chunk = raw[start:end]
        try:
            api.upload_chunk(upload_id, i, chunk)
        except Exception as e:
            st.error(f"Chunk {i} upload failed: {e}")
            st.stop()
        progress.progress((i + 1) / total_chunks)
        status.text(f"Uploaded chunk {i+1}/{total_chunks}")

    # Complete upload -> starts background transcription
    try:
        result = api.complete_chunked_upload(upload_id, speaker_id=speaker_id, title=title)
        st.success(f"Upload complete. Transcription started for call_id={result['call_id']}")
        st.info("Transcription runs in the background. Refresh the History page to see segments appear.")
    except Exception as e:
        st.error(f"Complete failed: {e}")

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass


# --- Manual refresh --------------------------------------------------------
st.divider()
if st.button("🔄 Refresh settings"):
    st.rerun()

"""Page 1: Speaker Lab — register speakers and test voice cloning."""
from __future__ import annotations

import io
import wave

import numpy as np
import streamlit as st

from utils.api_client import APIClient
from components.speaker_card import render_speaker_card


st.set_page_config(page_title="Speaker Lab", page_icon="🎤", layout="wide")
st.title("🎤 Speaker Lab")
st.caption("Register speakers with a 3-10 s reference clip, then test voice cloning.")

api: APIClient = st.session_state.get("api") or APIClient()
st.session_state["api"] = api


# --- Refresh speakers list -------------------------------------------------
@st.cache_data(ttl=5)
def _list_speakers() -> list[dict]:
    try:
        return api.list_speakers()
    except Exception as e:
        st.error(f"Failed to load speakers: {e}")
        return []


# --- Layout ----------------------------------------------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### Register new speaker")
    with st.form("new_speaker_form", clear_on_submit=False):
        sid = st.text_input("Speaker ID", value="", placeholder="e.g. ramesh_telugu")
        name = st.text_input("Display name", value="", placeholder="e.g. Ramesh (Telugu)")
        lang = st.selectbox("Default language", ["te", "en", "hi", "ta", "kn"], index=0)
        mic = st.audio_input("🎙️ Record from mic (3-10 s)")
        st.caption("— or —")
        ref = st.file_uploader(
            "Upload reference audio (3-10 s)",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            accept_multiple_files=False,
        )
        submitted = st.form_submit_button("Register", type="primary")
        if submitted:
            if not sid or not name or (ref is None and mic is None):
                st.error("All fields required — upload or record a reference clip.")
            else:
                if mic is not None:
                    raw = mic.getvalue()
                    fmt = "wav"
                else:
                    raw = ref.read()
                    fmt = (ref.name.split(".")[-1] if "." in ref.name else "wav")
                try:
                    meta = api.create_speaker(
                        speaker_id=sid.strip(),
                        display_name=name.strip(),
                        language=lang,
                        ref_audio_bytes=raw,
                        ref_format=fmt,
                    )
                    st.success(f"Created speaker: {meta['speaker_id']} ({meta['ref_duration_s']:.2f}s ref)")
                    _list_speakers.clear()
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.divider()
    st.markdown("### Existing speakers")
    speakers = _list_speakers()
    if not speakers:
        st.info("No speakers yet. Register one to get started.")
    for spk in speakers:
        def _select(s=spk):
            st.session_state["selected_speaker"] = s["speaker_id"]
        def _delete(s=spk):
            try:
                api.delete_speaker(s["speaker_id"])
                _list_speakers.clear()
                st.success(f"Deleted {s['speaker_id']}")
            except Exception as e:
                st.error(f"Delete failed: {e}")
        render_speaker_card(spk, on_select=_select, on_delete=_delete)

with col_right:
    st.markdown("### Test voice cloning")
    selected = st.session_state.get("selected_speaker")
    if not selected:
        if speakers:
            selected = speakers[0]["speaker_id"]
            st.session_state["selected_speaker"] = selected
        else:
            st.info("Register a speaker on the left to enable TTS testing.")
            st.stop()

    st.markdown(f"**Selected speaker:** `{selected}`")
    st.audio(api.speaker_ref_url(selected), format="audio/wav")

    st.markdown("#### Enter text to synthesize")
    default_text = "నమస్తే, ఎలా ఉన్నారు? నేను మీ వాయిస్ క్లోన్ చేసాను."
    text = st.text_area("Text", value=default_text, height=120)
    lang_override = st.selectbox("Language override", ["auto", "te", "en", "hi"], index=0)
    if st.button("🔊 Synthesize", type="primary"):
        if not text.strip():
            st.error("Enter some text first.")
        else:
            with st.spinner("Synthesizing..."):
                try:
                    wav_bytes, hdr = api.synthesize_tts(
                        text=text.strip(),
                        speaker_id=selected,
                        language=(None if lang_override == "auto" else lang_override),
                    )
                    st.audio(wav_bytes, format="audio/wav")
                    st.caption(f"Engine: {hdr.get('engine', '?')} · Latency: {hdr.get('latency_ms', '?')} ms")
                except Exception as e:
                    st.error(f"Synthesis failed: {e}")

    st.divider()
    st.markdown("### Tips")
    st.markdown("""
- **Reference audio quality matters.** Use a clean 3-10 s clip of natural
  speech (no music, no background noise). For best Telugu slang capture,
  use conversational Telugu rather than studio narration.
- **XTTS v2** supports zero-shot cloning across 16+ languages including
  Telugu and English. Mixed-language text is supported but specify the
  dominant language for best pronunciation.
- **Latency** of 500-1500 ms is normal on a 3060 for a 10-word utterance.
    """)

"""Transcript viewer component."""
from __future__ import annotations

import streamlit as st


def render_transcript(segments: list[dict]):
    """Render a list of transcript segments with timestamp + speaker labels."""
    if not segments:
        st.info("No transcript yet.")
        return
    st.markdown("### Transcript")
    for seg in segments:
        t0 = seg.get("t0", 0)
        t1 = seg.get("t1", 0)
        speaker = seg.get("speaker", "?")
        text = seg.get("text", "")
        is_final = seg.get("is_final", True)
        ts = f"[{t0:6.2f} → {t1:6.2f}]"
        speaker_label = "You" if speaker in ("user", "You") else "Bot"
        color = "#60a5fa" if speaker_label == "You" else "#a78bfa"
        css_class = "final" if is_final else "partial"
        st.markdown(
            f"<div class='seg {css_class}'>"
            f"<span style='color:#94a3b8;font-family:monospace'>{ts}</span> "
            f"<span style='color:{color};font-weight:600'>{speaker_label}:</span> "
            f"<span>{text}</span></div>",
            unsafe_allow_html=True,
        )


def render_call_segments(segments: list[dict]):
    """Render persisted call segments (from SQLite). Same shape as above."""
    render_transcript(segments)

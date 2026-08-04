"""Voice Clone Agent - Streamlit frontend entry point.

Run:  streamlit run streamlit_app/main.py
"""
from __future__ import annotations

import streamlit as st

from utils.api_client import APIClient

st.set_page_config(
    page_title="Voice Clone Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Custom CSS ------------------------------------------------------------
st.markdown("""
<style>
    .seg { padding: 6px 0; border-bottom: 1px solid #1e293b; line-height: 1.5; }
    .seg.partial { opacity: 0.6; }
    .seg.final { }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #e2e8f0; }
    .stMetric { background: #1e293b; padding: 12px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# --- Init API client (cached) ---------------------------------------------
@st.cache_resource
def get_api() -> APIClient:
    return APIClient()


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎙️ Voice Clone Agent")
    st.caption("Telugu / English · RTX 3060 6GB")
    st.divider()
    try:
        health = get_api().health()
        st.metric("Status", health.get("status", "?").upper())
        st.metric("Device", health.get("device", "?"))
        st.metric("Pipeline", health.get("pipeline_mode", "?"))
        st.metric("Speakers", health.get("speakers_count", 0))
        st.metric("Calls", health.get("calls_count", 0))
        vram_total = health.get("vram_total_mb", 0)
        vram_used = health.get("vram_used_mb", 0)
        if vram_total > 0:
            pct = (vram_used / vram_total) * 100
            st.metric("VRAM", f"{vram_used}/{vram_total} MB", f"{pct:.1f}%")
    except Exception as e:
        st.error(f"Backend unreachable: {e}")
        st.caption(f"Expected at: {get_api().base_url}")
    st.divider()
    st.markdown("**Pages**")
    st.markdown("- 🎤 [Speaker Lab](#speaker-lab)")
    st.markdown("- 📞 [Call Center](#call-center)")
    st.markdown("- 📜 [History](#history)")
    st.markdown("- ⚙️ [Settings](#settings)")


# --- Title -----------------------------------------------------------------
st.title("Voice Clone Agent")
st.caption("Bilingual (Telugu/English) zero-shot voice cloning agent for RTX 3060 6 GB.")
st.divider()

st.markdown("""
### Welcome

Use the navigation in the sidebar (or the pages at the top) to:

1. **🎤 Speaker Lab** — register a new speaker by uploading a 3-10 second
   reference clip, then test voice cloning with any text.
2. **📞 Call Center** — start a live WebRTC call. Your mic audio is streamed
   to the backend, transcribed, answered by the LLM (or rule-based
   responder if no LLM is configured), and the reply is spoken in the
   cloned voice.
3. **📜 History** — browse, replay, and read transcripts of past calls.
4. **⚙️ Settings** — view the current configuration, GPU usage, and
   pipeline mode.

> **Note on LLM:** The local LLM was intentionally removed. To enable
> LLM-powered replies, set `LLM_ENABLED=true` and point `LLM_API_BASE`
> at any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, OpenAI).
> When disabled, the pipeline uses a bilingual rule-based responder so
> the rest of the system still works end-to-end.
""")

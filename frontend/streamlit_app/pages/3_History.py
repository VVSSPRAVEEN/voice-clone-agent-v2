"""Page 3: Call History — list, replay, and read transcripts of past calls."""
from __future__ import annotations

import streamlit as st

from utils.api_client import APIClient
from components.transcript_viewer import render_call_segments


st.set_page_config(page_title="History", page_icon="📜", layout="wide")
st.title("📜 Call History")
st.caption("Browse, replay, and read transcripts of past calls.")

api: APIClient = st.session_state.get("api") or APIClient()
st.session_state["api"] = api


# --- Load calls ------------------------------------------------------------
@st.cache_data(ttl=10)
def _list_calls(limit: int = 100, offset: int = 0) -> list[dict]:
    try:
        return api.list_calls(limit=limit, offset=offset)
    except Exception as e:
        st.error(f"Failed to load calls: {e}")
        return []


calls = _list_calls()

if not calls:
    st.info("No calls recorded yet. Start one from the Call Center page.")
    st.stop()


# --- List ------------------------------------------------------------------
st.markdown(f"### {len(calls)} call(s)")

for call in calls:
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
        with col1:
            st.markdown(f"**{call.get('title') or call['call_id']}**")
            st.caption(f"ID: `{call['call_id']}` · Speaker: `{call['speaker_id']}`")
            st.caption(f"Started: {call.get('started_at','?')}")
        with col2:
            st.metric("Duration", f"{call.get('duration_s',0):.1f}s")
            st.metric("Segments", call.get("segment_count", 0))
        with col3:
            if st.button("Open", key=f"open_{call['call_id']}"):
                st.session_state["open_call_id"] = call["call_id"]
        with col4:
            if st.button("Delete", key=f"del_{call['call_id']}", type="secondary"):
                try:
                    api.delete_call(call["call_id"])
                    _list_calls.clear()
                    st.success("Deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")


# --- Detail view -----------------------------------------------------------
open_call_id = st.session_state.get("open_call_id")
if open_call_id:
    st.divider()
    st.markdown(f"### Call: `{open_call_id}`")
    try:
        call = api.get_call(open_call_id)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Metadata")
            st.json({
                "call_id": call["call_id"],
                "speaker_id": call["speaker_id"],
                "title": call.get("title"),
                "started_at": call["started_at"],
                "ended_at": call.get("ended_at"),
                "duration_s": call.get("duration_s", 0),
                "segment_count": call.get("segment_count", 0),
            })
            st.markdown("#### Audio")
            st.audio(api.get_call_audio_url(open_call_id), format="audio/wav")
        with col2:
            st.markdown("#### Transcript")
            segs = api.get_call_transcript(open_call_id)
            render_call_segments(segs)
            st.download_button(
                "Download transcript (JSONL)",
                data="\n".join(
                    __import__("json").dumps(s, ensure_ascii=False) for s in segs
                ),
                file_name=f"{open_call_id}_transcript.jsonl",
                mime="application/jsonl",
            )
    except Exception as e:
        st.error(f"Failed to load call: {e}")


# --- Refresh ---------------------------------------------------------------
if st.button("🔄 Refresh"):
    _list_calls.clear()
    st.rerun()

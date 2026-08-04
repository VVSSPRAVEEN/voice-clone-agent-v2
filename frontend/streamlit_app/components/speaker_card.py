"""Speaker card component."""
from __future__ import annotations

import streamlit as st


def render_speaker_card(speaker: dict, on_select=None, on_delete=None):
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {speaker.get('display_name', speaker['speaker_id'])}")
            st.caption(f"ID: `{speaker['speaker_id']}` · Lang: `{speaker.get('language', '?')}` · Ref: {speaker.get('ref_duration_s', 0):.1f}s")
        with col2:
            if on_select:
                if st.button("Select", key=f"sel_{speaker['speaker_id']}"):
                    on_select(speaker)
            if on_delete:
                if st.button("Delete", key=f"del_{speaker['speaker_id']}", type="secondary"):
                    on_delete(speaker)

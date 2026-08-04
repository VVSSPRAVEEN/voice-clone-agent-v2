"""Call control buttons."""
from __future__ import annotations

import streamlit as st


def render_call_controls(active: bool):
    """Returns ('start' | 'stop' | None)."""
    col1, col2, col3 = st.columns([1, 1, 4])
    action = None
    with col1:
        if st.button("Start call", disabled=active, type="primary"):
            action = "start"
    with col2:
        if st.button("End call", disabled=not active):
            action = "stop"
    with col3:
        if active:
            st.info("Call in progress...")
        else:
            st.caption("Ready to start a call.")
    return action

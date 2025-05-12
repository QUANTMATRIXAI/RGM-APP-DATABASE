# utils_progress.py
import streamlit as st
from db_utils import save_state, load_state   # your existing helpers

# Human‑friendly names for the wizard
STEPS = {
    "STEP_1": {"label": "Upload / Validate Data"},
    "STEP_2": {"label": "Base Price Estimator"},
    "STEP_3": {"label": "Promo Depth Estimator"},
    # add more here later
}

def mark_step_done(pid: int, step_code: str):
    """Flip a flag in the DB + session_state when a step is finished."""
    progress = load_state(pid, "progress", default={})
    progress[step_code] = True
    save_state(pid, "progress", progress)
    st.session_state["progress"] = progress

def auto_detect_progress(pid: int):
    """Read the saved dict into session_state on every run."""
    st.session_state["progress"] = load_state(pid, "progress", default={})

def show_completed_steps_sidebar():
    """Pretty sidebar list."""
    prog = st.session_state.get("progress", {})
    with st.sidebar:
        st.header("Progress")
        if not prog:
            st.write("No steps done yet.")
        else:
            for code, info in STEPS.items():
                if prog.get(code):
                    st.write(f"✅ {info['label']}")

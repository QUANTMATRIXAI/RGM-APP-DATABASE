import streamlit as st

def run(pid: int):
    if st.button("← Back to dashboard"):
        st.session_state.current_project = None        # clear router flags
        st.session_state.current_type    = None
        st.experimental_rerun()

    st.title("📊 mmm")
    st.write(f"Project id {pid} • build your Promo Analysis UI here …")

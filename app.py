"""Course AI — ChatGPT-style UI. Run: streamlit run app.py"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import rag_core

COURSE_NAME = os.getenv("COURSE_NAME", "My course")
JOBLIB_PATH = Path("embeddings.joblib")

st.set_page_config(
    page_title="Course AI",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
  #MainMenu, footer, header {visibility: hidden;}
  .block-container {
    padding-top: 1.5rem;
    padding-bottom: 6rem;
    max-width: 48rem;
  }
  .app-header {
    text-align: center;
    margin-bottom: 1.5rem;
  }
  .app-header h1 {
    font-size: 1.35rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: -0.02em;
  }
  .app-header p {
    color: #6b7280;
    font-size: 0.9rem;
    margin: 0.35rem 0 0;
  }
  .status-pill {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.5rem;
  }
  .status-ready { background: #dcfce7; color: #166534; }
  .status-wait { background: #f3f4f6; color: #4b5563; }
  .welcome-box {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    color: #374151;
  }
  .welcome-box h2 {
    font-size: 1.5rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
  }
  .welcome-box p { color: #6b7280; font-size: 0.95rem; }
  div[data-testid="stChatMessage"] {
    background: transparent !important;
  }
  .upload-row {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    background: #fafafa;
  }
</style>
""",
    unsafe_allow_html=True,
)

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.ready = False
    st.session_state.messages = []
    st.session_state.sources = {}
    st.session_state.indexed_videos = []


def set_ready(df) -> None:
    st.session_state.df = df
    st.session_state.ready = df is not None and not df.empty


# --- API key ---
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GROQ_API_KEY"):
    st.error(
        "Set **GEMINI_API_KEY** (recommended) or **GROQ_API_KEY** in `.env` or Streamlit Secrets. "
        "Get Gemini key: https://aistudio.google.com/apikey"
    )
    st.stop()

provider_label = "Gemini" if os.getenv("GEMINI_API_KEY") else "Groq"

# --- Header ---
ready = st.session_state.ready
status_class = "status-ready" if ready else "status-wait"
status_text = (
    f"● Ready · {len(st.session_state.df)} segments"
    if ready
    else "● Upload a video to start"
)
st.markdown(
    f"""
<div class="app-header">
  <h1>Course AI</h1>
  <p>Learn from your videos — powered by {provider_label}</p>
  <span class="status-pill {status_class}">{status_text}</span>
</div>
""",
    unsafe_allow_html=True,
)

# --- Sidebar (minimal) ---
with st.sidebar:
    st.caption("Settings")
    if JOBLIB_PATH.exists():
        if st.button("Load Sigma demo course"):
            with st.spinner("Re-indexing embeddings…"):
                set_ready(rag_core.load_joblib(JOBLIB_PATH))
                st.session_state.indexed_videos = ["Sigma Web Dev (demo)"]
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Demo course loaded. Ask me anything about the HTML/CSS lessons.",
                    }
                ]
            st.rerun()
    if st.button("New chat"):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.rerun()
    if st.button("Clear all data"):
        st.session_state.df = None
        st.session_state.ready = False
        st.session_state.messages = []
        st.session_state.sources = {}
        st.session_state.indexed_videos = []
        st.rerun()

# --- Welcome or messages ---
if not st.session_state.messages and not ready:
    st.markdown(
        """
<div class="welcome-box">
  <h2>What do you want to learn?</h2>
  <p>Attach a video or audio file below, then index it. Chat unlocks when processing finishes.</p>
</div>
""",
        unsafe_allow_html=True,
    )

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "✦"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and str(i) in st.session_state.sources:
            with st.expander("Sources"):
                st.markdown(st.session_state.sources[str(i)])

# --- Upload row (above chat input) ---
st.markdown('<div class="upload-row">', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Attach video or audio",
    type=["mp4", "webm", "mov", "mkv", "mp3", "wav", "m4a", "ogg"],
    label_visibility="visible",
)
col_a, col_b = st.columns([3, 1])
with col_a:
    lang = st.text_input("Language code (optional)", placeholder="en, hi …", label_visibility="collapsed")
with col_b:
    index_btn = st.button("Index", type="primary", use_container_width=True, disabled=uploaded is None)
st.markdown("</div>", unsafe_allow_html=True)

if index_btn and uploaded:
    st.session_state.messages.append(
        {"role": "user", "content": f"📎 Indexing: **{uploaded.name}**"}
    )
    with st.chat_message("assistant", avatar="✦"):
        status = st.status("Processing…", expanded=True)
        log = status.empty()

        def on_step(msg: str) -> None:
            log.write(msg)

        try:
            new_df = rag_core.process_uploaded_media(
                uploaded.getvalue(),
                uploaded.name,
                on_step=on_step,
                language=lang.strip() or None,
            )
            merged = rag_core.merge_dataframes(st.session_state.df, new_df)
            set_ready(merged)
            st.session_state.indexed_videos.append(uploaded.name)
            n = len(new_df)
            status.update(label="Done", state="complete", expanded=False)
            reply = (
                f"**{uploaded.name}** is ready ({n} segments indexed). "
                "Ask me anything about this course."
            )
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except subprocess.CalledProcessError:
            status.update(label="Failed", state="error")
            err = "ffmpeg is required for video files. On deploy, `packages.txt` includes it."
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        except Exception as e:
            status.update(label="Failed", state="error")
            err = str(e)
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})
    st.rerun()

# --- Chat input ---
prompt = st.chat_input(
    "Message Course AI…",
    disabled=not ready,
)

if prompt and ready:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✦"):
        try:
            answer, ctx = rag_core.answer_question(
                st.session_state.df,
                prompt,
                course_name=COURSE_NAME,
            )
            st.markdown(answer)
            src_lines = []
            for _, row in ctx.iterrows():
                ts = rag_core.format_timestamp(row["start"])
                src_lines.append(f"- **{row['title']}** `{ts}` — {row['text'][:180]}")
            src_md = "\n".join(src_lines)
            msg_idx = str(len(st.session_state.messages))
            st.session_state.sources[msg_idx] = src_md
        except Exception as e:
            answer = f"Something went wrong: {e}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

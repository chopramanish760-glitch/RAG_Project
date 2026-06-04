"""mY_Tutor — video learning assistant. Run: streamlit run app.py"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import rag_core

APP_NAME = "mY_Tutor"
COURSE_NAME = os.getenv("COURSE_NAME", "My course")
JOBLIB_PATH = Path("embeddings.joblib")
FAVICON = Path(__file__).parent / "favicon.png"

st.set_page_config(
    page_title=APP_NAME,
    page_icon=str(FAVICON) if FAVICON.exists() else "📘",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
  #MainMenu, footer, header {visibility: hidden;}
  .block-container {
    padding-top: 1rem;
    padding-bottom: 5.5rem;
    max-width: 44rem;
  }
  .brand-wrap {
    text-align: center;
    padding: 0.5rem 0 1.25rem;
  }
  .brand-logo {
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
  }
  .brand-title {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #312e81, #4f46e5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .brand-tag {
    color: #64748b;
    font-size: 0.88rem;
    margin-top: 0.25rem;
  }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.65rem;
  }
  .status-ready {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
  }
  .status-wait {
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
  }
  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
  }
  .dot-live { background: #10b981; animation: pulse 1.5s infinite; }
  .dot-idle { background: #94a3b8; }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
  }
  .welcome-card {
    background: linear-gradient(180deg, #f8fafc 0%, #fff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.75rem 1.25rem;
    text-align: center;
    margin-bottom: 1rem;
  }
  .welcome-card h2 {
    font-size: 1.15rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0 0 0.4rem;
  }
  .welcome-card p { color: #64748b; font-size: 0.9rem; margin: 0; }
  .upload-card {
    border: 1px dashed #c7d2fe;
    border-radius: 16px;
    padding: 1rem 1.1rem 0.75rem;
    background: #fafaff;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .upload-card:hover {
    border-color: #818cf8;
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.08);
  }
  .file-chip {
    display: inline-block;
    background: #eef2ff;
    color: #3730a3;
    padding: 0.2rem 0.55rem;
    border-radius: 8px;
    font-size: 0.75rem;
    margin: 0.15rem 0.2rem 0 0;
  }
  div[data-testid="stChatMessage"] {
    background: transparent !important;
  }
  .stButton > button[kind="primary"] {
    border-radius: 10px;
    font-weight: 600;
  }
  .job-card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.5rem;
    background: #fff;
  }
  .job-card.active {
    border-color: #818cf8;
    background: #f5f3ff;
    box-shadow: 0 2px 12px rgba(79, 70, 229, 0.12);
  }
  .job-card.done { border-color: #6ee7b7; background: #ecfdf5; }
  .job-card.error { border-color: #fca5a5; background: #fef2f2; }
  .job-card.wait { opacity: 0.55; }
  .job-title { font-weight: 600; font-size: 0.88rem; color: #1e293b; }
  .job-step { font-size: 0.8rem; color: #4f46e5; margin-top: 0.15rem; }
  .job-eta { font-size: 0.75rem; color: #64748b; }
</style>
""",
    unsafe_allow_html=True,
)

MAX_VIDEOS = 5
STAGE_LABELS = {
    "prepare": "Preparing upload",
    "extract": "Extracting audio",
    "transcribe": "Transcribing speech",
    "index": "Indexing for search",
    "done": "Complete",
    "error": "Failed",
    "queued": "Waiting in queue",
}
STAGE_PROGRESS = {
    "queued": 0.05,
    "prepare": 0.2,
    "extract": 0.4,
    "transcribe": 0.7,
    "index": 0.9,
    "done": 1.0,
    "error": 1.0,
}

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.ready = False
    st.session_state.messages = []
    st.session_state.sources = {}
    st.session_state.indexed_videos = []
    st.session_state.suggested_questions = []


def set_ready(df) -> None:
    st.session_state.df = df
    st.session_state.ready = df is not None and not df.empty


def refresh_suggestions() -> None:
    if st.session_state.ready and st.session_state.df is not None:
        with st.spinner("Preparing suggested questions…"):
            st.session_state.suggested_questions = rag_core.build_suggested_questions(
                st.session_state.df,
                course_name=COURSE_NAME,
                count=5,
            )


def render_suggested_questions() -> None:
    questions = st.session_state.get("suggested_questions") or []
    if not questions or not st.session_state.ready:
        return
    st.markdown("#### Suggested questions")
    st.caption("Based on your transcript — tap to ask")
    cols = st.columns(min(len(questions), 2))
    for i, q in enumerate(questions):
        with cols[i % 2]:
            if st.button(q, key=f"suggest_{i}_{hash(q) % 10**6}", use_container_width=True):
                ask_tutor(q)
                st.rerun()


def ask_tutor(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        answer, ctx = rag_core.answer_question(
            st.session_state.df,
            question,
            course_name=COURSE_NAME,
        )
        src_lines = []
        for _, row in ctx.iterrows():
            ts = rag_core.format_timestamp(row["start"])
            snippet = str(row["text"])[:160]
            src_lines.append(f"- **{row['title']}** `{ts}` — {snippet}")
        st.session_state.sources[str(len(st.session_state.messages))] = "\n".join(src_lines)
    except Exception as e:
        answer = f"Something went wrong: {e}"
    st.session_state.messages.append({"role": "assistant", "content": answer})


# --- API key ---
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GROQ_API_KEY"):
    st.error("Add **GEMINI_API_KEY** to `.env` or Streamlit Secrets.")
    st.stop()

ready = st.session_state.ready
dot_class = "dot-live" if ready else "dot-idle"
status_class = "status-ready" if ready else "status-wait"
if ready:
    status_text = f"Online · {len(st.session_state.df)} segments ready"
else:
    status_text = "Waiting for your video"

# --- Header ---
st.markdown(
    f"""
<div class="brand-wrap">
  <div class="brand-logo">mY</div>
  <h1 class="brand-title">{APP_NAME}</h1>
  <p class="brand-tag">Up to {MAX_VIDEOS} videos · Ask anything · Get timestamps</p>
  <span class="status-pill {status_class}">
    <span class="dot {dot_class}"></span> {status_text}
  </span>
</div>
""",
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    if JOBLIB_PATH.exists():
        if st.button("Load saved index", use_container_width=True):
            with st.spinner("Loading…"):
                set_ready(rag_core.load_joblib(JOBLIB_PATH))
                st.session_state.indexed_videos = ["Saved course index"]
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Saved index loaded. Ask me anything about the course.",
                    }
                ]
                refresh_suggestions()
            st.rerun()
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = {}
        # keep suggested_questions — still based on indexed transcript
        st.rerun()
    if st.session_state.ready and st.button("Refresh suggestions", use_container_width=True):
        refresh_suggestions()
        st.rerun()
    if st.button("Reset everything", use_container_width=True):
        st.session_state.df = None
        st.session_state.ready = False
        st.session_state.messages = []
        st.session_state.sources = {}
        st.session_state.indexed_videos = []
        st.session_state.suggested_questions = []
        st.rerun()

# Indexed file chips
if st.session_state.indexed_videos:
    chips = "".join(
        f'<span class="file-chip">{name}</span>'
        for name in st.session_state.indexed_videos
    )
    st.markdown(f"**Indexed:** {chips}", unsafe_allow_html=True)

# Welcome
if not st.session_state.messages and not ready:
    st.markdown(
        f"""
<div class="welcome-card">
  <h2>Start in 2 steps</h2>
  <p>1. Attach up to <strong>{MAX_VIDEOS}</strong> videos &nbsp;→&nbsp; 2. Tap <strong>Index all</strong> &nbsp;→&nbsp; chat here</p>
</div>
""",
        unsafe_allow_html=True,
    )

# Chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and str(i) in st.session_state.sources:
            with st.expander("View sources"):
                st.markdown(st.session_state.sources[str(i)])

render_suggested_questions()

def _format_eta(seconds: int) -> str:
    if seconds < 60:
        return f"~{max(1, seconds)} sec"
    return f"~{seconds // 60} min {seconds % 60} sec"


def _render_job_card(
    slot,
    *,
    video_num: int,
    name: str,
    stage: str,
    detail: str = "",
    progress: float = 0.0,
    eta_sec: int | None = None,
    size_mb: float = 0.0,
) -> None:
    css = "active" if stage not in ("done", "error", "queued") else stage
    if stage == "queued":
        css = "wait"
    elif stage == "done":
        css = "done"
    elif stage == "error":
        css = "error"
    label = STAGE_LABELS.get(stage, stage)
    eta_line = ""
    if eta_sec is not None and stage not in ("done", "error"):
        eta_line = f'<div class="job-eta">About {_format_eta(eta_sec)} remaining</div>'
    detail_line = f'<div class="job-step">{detail or label}</div>' if stage != "queued" else ""
    slot.markdown(
        f"""
<div class="job-card {css}">
  <div class="job-title">Video {video_num}: {name}</div>
  <div class="job-eta">{size_mb:.1f} MB</div>
  {detail_line}
  {eta_line}
</div>
""",
        unsafe_allow_html=True,
    )
    slot.progress(min(1.0, max(0.0, progress)), text=f"{label}…")


# Upload card
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    f"Drop up to {MAX_VIDEOS} videos or audio files",
    type=["mp4", "webm", "mov", "mkv", "mp3", "wav", "m4a", "ogg"],
    accept_multiple_files=True,
    help=f"Select 1–{MAX_VIDEOS} files · MP4, MP3, WAV · max ~100 MB each",
)
files = list(uploaded_files or [])
if len(files) > MAX_VIDEOS:
    st.warning(f"Only **{MAX_VIDEOS}** videos allowed. Processing the first {MAX_VIDEOS}.")
    files = files[:MAX_VIDEOS]

if files:
    total_mb = sum(len(f.getvalue()) for f in files) / (1024 * 1024)
    total_eta = sum(rag_core.estimate_processing_seconds(f.getvalue(), f.name) for f in files)
    st.caption(
        f"**{len(files)}** file(s) selected · **{total_mb:.1f} MB** total · "
        f"estimated **{_format_eta(total_eta)}** to process all"
    )

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    lang = st.text_input("Language (optional)", placeholder="en, hi, …")
with c2:
    st.caption("Leave blank to auto-detect")
with c3:
    index_btn = st.button(
        "Index all",
        type="primary",
        use_container_width=True,
        disabled=not files,
    )
st.markdown("</div>", unsafe_allow_html=True)

if index_btn and files:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": f"Indexing **{len(files)}** video(s)…",
        }
    )

    st.markdown("### Processing queue")
    job_slots = [st.empty() for _ in files]
    overall = st.progress(0, text="Starting queue…")
    overall_eta_label = st.empty()

    lang_code = lang.strip() or None
    results_ok = []
    results_fail = []
    batch_start = time.time()
    batch_eta = sum(
        rag_core.estimate_processing_seconds(f.getvalue(), f.name) for f in files
    )

    for idx, uploaded in enumerate(files):
        video_num = idx + 1
        file_bytes = uploaded.getvalue()
        size_mb = len(file_bytes) / (1024 * 1024)
        eta_this = rag_core.estimate_processing_seconds(file_bytes, uploaded.name)
        file_start = time.time()

        def make_on_step(vnum, slot, start_t, eta, fname):
            def on_step(stage: str, msg: str) -> None:
                elapsed = time.time() - start_t
                remaining = max(0, int(eta - elapsed))
                prog = STAGE_PROGRESS.get(stage, 0.5)
                _render_job_card(
                    slot,
                    video_num=vnum,
                    name=fname,
                    stage=stage,
                    detail=msg,
                    progress=prog,
                    eta_sec=remaining,
                    size_mb=size_mb,
                )
                batch_elapsed = time.time() - batch_start
                batch_remaining = max(0, int(batch_eta - batch_elapsed))
                done_frac = (vnum - 1 + prog) / len(files)
                overall.progress(done_frac, text=f"Video {vnum} of {len(files)} · {msg}")
                overall_eta_label.caption(
                    f"**Overall:** {_format_eta(batch_remaining)} remaining "
                    f"(~{_format_eta(batch_eta)} total estimated)"
                )

            return on_step

        # Show queued state for upcoming videos
        for j, slot in enumerate(job_slots):
            if j < idx:
                _render_job_card(
                    job_slots[j],
                    video_num=j + 1,
                    name=files[j].name,
                    stage="done",
                    detail="Finished",
                    progress=1.0,
                    size_mb=len(files[j].getvalue()) / (1024 * 1024),
                )
            elif j == idx:
                _render_job_card(
                    slot,
                    video_num=video_num,
                    name=uploaded.name,
                    stage="prepare",
                    detail="Upload received · starting…",
                    progress=0.15,
                    eta_sec=eta_this,
                    size_mb=size_mb,
                )
            else:
                wait_eta = rag_core.estimate_processing_seconds(
                    files[j].getvalue(), files[j].name
                )
                _render_job_card(
                    slot,
                    video_num=j + 1,
                    name=files[j].name,
                    stage="queued",
                    detail="Waiting…",
                    progress=0.05,
                    eta_sec=wait_eta,
                    size_mb=len(files[j].getvalue()) / (1024 * 1024),
                )

        on_step = make_on_step(video_num, job_slots[idx], file_start, eta_this, uploaded.name)

        try:
            new_df = rag_core.process_uploaded_media(
                file_bytes,
                uploaded.name,
                on_step=on_step,
                language=lang_code,
            )
            merged = rag_core.merge_dataframes(st.session_state.df, new_df)
            set_ready(merged)
            st.session_state.indexed_videos.append(uploaded.name)
            st.session_state.df = merged
            results_ok.append((uploaded.name, len(new_df)))
            _render_job_card(
                job_slots[idx],
                video_num=video_num,
                name=uploaded.name,
                stage="done",
                detail=f"Done · {len(new_df)} segments indexed",
                progress=1.0,
                size_mb=size_mb,
            )
        except subprocess.CalledProcessError:
            results_fail.append((uploaded.name, "ffmpeg required for video"))
            _render_job_card(
                job_slots[idx],
                video_num=video_num,
                name=uploaded.name,
                stage="error",
                detail="ffmpeg not found",
                progress=1.0,
                size_mb=size_mb,
            )
        except Exception as e:
            results_fail.append((uploaded.name, str(e)))
            _render_job_card(
                job_slots[idx],
                video_num=video_num,
                name=uploaded.name,
                stage="error",
                detail=str(e)[:80],
                progress=1.0,
                size_mb=size_mb,
            )

    overall.progress(1.0, text="Queue complete")
    elapsed_total = int(time.time() - batch_start)
    overall_eta_label.caption(f"Finished in **{_format_eta(elapsed_total)}**")

    if results_ok:
        names = ", ".join(f"**{n}**" for n, _ in results_ok)
        segs = sum(s for _, s in results_ok)
        with st.spinner("Generating suggested questions from transcript…"):
            st.session_state.suggested_questions = rag_core.build_suggested_questions(
                st.session_state.df,
                course_name=COURSE_NAME,
                count=5,
            )
        bullets = "\n".join(f"- {q}" for q in st.session_state.suggested_questions)
        reply = (
            f"Analysis complete! Indexed {names} ({segs} segments).\n\n"
            f"**Try asking:**\n{bullets}"
        )
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.balloons()
    if results_fail:
        fail_txt = "; ".join(f"{n}: {err}" for n, err in results_fail)
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Some files failed: {fail_txt}"}
        )
    st.rerun()

# Chat input
prompt = st.chat_input(
    f"Message {APP_NAME}…",
    disabled=not ready,
)

if prompt and ready:
    ask_tutor(prompt)
    st.rerun()

"""mY_Tutor — upload videos, auto-transcribe, chat. Run: streamlit run app.py"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

# Quiet Hugging Face cache warning on Windows (first FastEmbed download)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import rag_core

APP_NAME = "mY_Tutor"
COURSE_NAME = os.getenv("COURSE_NAME", "My course")
FAVICON = Path(__file__).parent / "favicon.png"
MAX_VIDEOS = 5

STEP_LABEL = {
    "prepare": "Reading file",
    "extract": "Getting audio",
    "transcribe": "Writing transcript",
    "save": "Building search index",
}
STEP_PCT = {"prepare": 0.15, "extract": 0.35, "transcribe": 0.75, "save": 0.92}

st.set_page_config(
    page_title=APP_NAME,
    page_icon=str(FAVICON) if FAVICON.exists() else "📘",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.cache_resource(rag_core._get_fastembed)()

st.markdown(
    """
<style>
  #MainMenu, footer {visibility: hidden;}
  .block-container { padding-top: 0.75rem; padding-bottom: 5rem; max-width: 42rem; }
  .hero { text-align: center; margin-bottom: 1rem; }
  .hero h1 {
    font-size: 2rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #312e81, #6366f1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero p { color: #64748b; font-size: 0.95rem; margin: 0.35rem 0 0; }
  .timer-box {
    background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
    border: 1px solid #c7d2fe; border-radius: 16px;
    padding: 1rem 1.25rem; margin: 0.75rem 0; text-align: center;
  }
  .timer-big { font-size: 2rem; font-weight: 700; color: #4338ca; }
  .ready-banner {
    background: #ecfdf5; border: 1px solid #6ee7b7; border-radius: 12px;
    padding: 0.75rem 1rem; text-align: center; color: #047857;
    font-weight: 600; margin-bottom: 1rem;
  }
</style>
""",
    unsafe_allow_html=True,
)

_DEFAULTS = {
    "df": None,
    "ready": False,
    "messages": [],
    "sources": {},
    "ready_videos": [],
    "suggested_questions": [],
    "done_sigs": set(),
    "pending_queue": [],
    "last_error": None,
    "last_success": None,
    "is_processing": False,
    "answering": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v if not isinstance(v, set) else set()


def set_ready(df) -> None:
    st.session_state.df = df
    st.session_state.ready = df is not None and not df.empty


def file_sig(name: str, data: bytes) -> tuple:
    digest = hashlib.sha256(data).hexdigest()[:16]
    return (name, len(data), digest)


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.sources = {}
    st.session_state.suggested_questions = []


def reset_lesson_state() -> None:
    """Drop indexed videos and chat — use when starting a new single upload."""
    reset_chat()
    st.session_state.df = None
    st.session_state.ready = False
    st.session_state.ready_videos = []
    st.session_state.done_sigs = set()
    st.session_state.last_success = None
    st.session_state.last_error = None


def fmt_time(seconds: int) -> str:
    s = max(0, int(seconds))
    m, sec = divmod(s, 60)
    return f"{m:02d}:{sec:02d}" if m else f"0:{sec:02d}"


def ask_tutor(question: str) -> None:
    if st.session_state.answering:
        return
    st.session_state.answering = True
    st.session_state.messages.append({"role": "user", "content": question})
    answer = ""
    try:
        with st.spinner("Searching transcript and writing answer (15–30s)…"):
            answer, ctx = rag_core.answer_question(
                st.session_state.df, question, course_name=COURSE_NAME
            )
        lines = []
        for _, row in ctx.iterrows():
            ts = rag_core.format_timestamp_range(row["start"], row["end"])
            lines.append(f"- **{row['title']}** `{ts}` — {str(row['text'])[:140]}")
        st.session_state.sources[str(len(st.session_state.messages))] = "\n".join(lines)
    except Exception as e:
        answer = f"Sorry, something went wrong: {e}"
    finally:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": rag_core.fix_timestamps_in_text(answer),
            }
        )
        st.session_state.answering = False


def queue_uploads(uploaded_files) -> None:
    """Save file bytes in session so they survive reruns."""
    files = list(uploaded_files[:MAX_VIDEOS])
    new_items: list[dict] = []
    for f in files:
        data = f.getvalue()
        sig = file_sig(f.name, data)
        if sig in st.session_state.done_sigs:
            continue
        already = {item["sig"] for item in st.session_state.pending_queue}
        if sig in already:
            continue
        new_items.append({"name": f.name, "bytes": data, "sig": sig})

    if not new_items:
        return

    # One new file to process = new lesson (even if an old chip is still in the uploader)
    if len(new_items) == 1:
        reset_lesson_state()
    else:
        reset_chat()

    st.session_state.pending_queue.extend(new_items)


def process_queue() -> None:
    """Process all pending files — no st.rerun(), status stays visible."""
    queue = st.session_state.pending_queue
    if not queue:
        return

    st.session_state.is_processing = True
    st.session_state.last_error = None
    batch_eta = max(
        60,
        sum(rag_core.estimate_processing_seconds(item["bytes"], item["name"]) for item in queue),
    )
    batch_start = time.time()

    st.markdown("#### Processing your upload")
    st.caption("Please keep this tab open. Transcription can take a few minutes.")
    progress = st.progress(0.0, text="Starting…")
    timer_box = st.empty()
    status_line = st.empty()

    def refresh_ui(video_i: int, total: int, stage: str, filename: str, msg: str) -> None:
        elapsed = time.time() - batch_start
        remaining = max(0, int(batch_eta - elapsed))
        pct = STEP_PCT.get(stage, 0.5)
        overall = min(0.98, ((video_i - 1) + pct) / total)
        label = STEP_LABEL.get(stage, stage)
        progress.progress(overall, text=f"Video {video_i}/{total} · {label}")
        status_line.markdown(f"**{filename}** — {msg}")
        timer_box.markdown(
            f'<div class="timer-box">'
            f'<div style="color:#64748b;font-size:14px;">Video {video_i} of {total} · {label}</div>'
            f'<div class="timer-big">{fmt_time(remaining)}</div>'
            f'<div style="color:#94a3b8;font-size:12px;">estimated time left · elapsed {fmt_time(int(elapsed))}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    ok_names = []
    still_pending = []
    replace_mode = len(queue) == 1
    for idx, item in enumerate(queue):
        num = idx + 1
        name = item["name"]
        data = item["bytes"]
        sig = item["sig"]
        refresh_ui(num, len(queue), "prepare", name, "Starting…")

        def on_step(stage: str, msg: str, _n=num, _name=name):
            refresh_ui(_n, len(queue), stage, _name, msg)

        try:
            new_df = rag_core.process_uploaded_media(
                data, name, on_step=on_step, language=None
            )
            if replace_mode:
                st.session_state.df = new_df.reset_index(drop=True)
                st.session_state.ready_videos = [name]
            else:
                st.session_state.df = rag_core.merge_dataframes(
                    st.session_state.df, new_df
                )
                if name not in st.session_state.ready_videos:
                    st.session_state.ready_videos.append(name)
            set_ready(st.session_state.df)
            st.session_state.done_sigs.add(sig)
            ok_names.append(name)
            st.success(f"**{name}** is ready")
        except Exception as e:
            st.session_state.last_error = f"**{name}:** {str(e)}"
            st.error(st.session_state.last_error)
            still_pending.append(item)

    st.session_state.pending_queue = still_pending
    elapsed = max(1, int(time.time() - batch_start))
    progress.progress(1.0, text="Finished")
    timer_box.markdown(
        f'<div class="timer-box" style="background:#ecfdf5;border-color:#6ee7b7;">'
        f'<div class="timer-big" style="color:#047857;">Done · {fmt_time(elapsed)}</div></div>',
        unsafe_allow_html=True,
    )

    if ok_names:
        st.session_state.last_success = ", ".join(ok_names)
        with st.spinner("Preparing question ideas…"):
            st.session_state.suggested_questions = rag_core.build_suggested_questions(
                st.session_state.df, course_name=COURSE_NAME, count=5
            )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Your video is ready: **{st.session_state.last_success}**. Ask below.",
            }
        )
        st.balloons()
    elif st.session_state.last_error:
        st.warning("Processing failed. Click **Try again** in the sidebar (left) or upload again.")

    st.session_state.is_processing = False


# --- API ---
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GROQ_API_KEY"):
    st.error("Add **GEMINI_API_KEY** or **GROQ_API_KEY** to `.env` or Streamlit Secrets.")
    st.stop()

def clear_all_session() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]


with st.sidebar:
    st.caption("mY_Tutor")
    st.markdown("Use **»** (top-left) if this panel is hidden.")
    st.markdown(
        "**Quota error (429)?**  \n"
        "Wait 1 min → **Try again**  \n"
        "Or add **GROQ_API_KEY** in `.env`  \n"
        "[Free Groq key](https://console.groq.com/keys)"
    )
    st.caption("First answer can take ~15–30s (search + AI).")
    if st.session_state.pending_queue and st.button("Try again", use_container_width=True):
        st.session_state.last_error = None
        st.rerun()
    if st.button("Clear all & start over", use_container_width=True):
        clear_all_session()
        st.rerun()

col_opts, _ = st.columns([1, 3])
with col_opts:
    if st.button("Clear all & start over", key="clear_main"):
        clear_all_session()
        st.rerun()

st.markdown(
    f'<div class="hero"><h1>{APP_NAME}</h1>'
    f"<p>Upload your lesson — transcribed with {rag_core.provider_label()}</p></div>",
    unsafe_allow_html=True,
)

if st.session_state.ready:
    titles = ", ".join(st.session_state.ready_videos) or "your video"
    st.markdown(
        f'<div class="ready-banner">Ready · {titles}</div>',
        unsafe_allow_html=True,
    )

if st.session_state.last_error and not st.session_state.is_processing:
    st.error(st.session_state.last_error)

if st.session_state.last_success and st.session_state.ready:
    st.caption(f"Last processed: {st.session_state.last_success}")

uploaded = st.file_uploader(
    "Upload video or audio",
    type=["mp4", "webm", "mov", "mkv", "mp3", "wav", "m4a", "ogg"],
    accept_multiple_files=True,
    label_visibility="collapsed",
    key="video_uploader",
)
st.caption(
    "New lesson? Remove the old file with **×**, then upload again, "
    "or click **Clear all & start over** above."
)

if uploaded:
    if len(uploaded) > MAX_VIDEOS:
        st.warning(f"Only the first {MAX_VIDEOS} files are used.")
    before = len(st.session_state.pending_queue)
    queue_uploads(list(uploaded))
    if (
        len(st.session_state.pending_queue) == before
        and not st.session_state.is_processing
        and st.session_state.ready
    ):
        st.info(
            "This file is already indexed. Click **×** on it and upload your new video, "
            "or **Clear all & start over** above."
        )

if st.session_state.pending_queue and not st.session_state.is_processing:
    process_queue()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and str(i) in st.session_state.sources:
            with st.expander("Sources"):
                st.markdown(st.session_state.sources[str(i)])

if st.session_state.ready and st.session_state.suggested_questions:
    st.markdown("##### Tap a question")
    cols = st.columns(2)
    busy = st.session_state.answering or st.session_state.is_processing
    for i, q in enumerate(st.session_state.suggested_questions):
        if cols[i % 2].button(
            q, key=f"q{i}", use_container_width=True, disabled=busy
        ):
            ask_tutor(q)
            st.rerun()

if (
    not st.session_state.ready
    and not st.session_state.pending_queue
    and not st.session_state.is_processing
    and not uploaded
):
    st.info("Drop a video above — processing starts automatically.")

prompt = st.chat_input(
    "Ask about your video…",
    disabled=(
        not st.session_state.ready
        or st.session_state.is_processing
        or st.session_state.answering
    ),
)
if prompt and st.session_state.ready:
    ask_tutor(prompt)
    st.rerun()

"""mY_Tutor — upload videos, auto-transcribe, chat. Run: streamlit run app.py"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from datetime import timedelta
from pathlib import Path

# Quiet Hugging Face cache warning on Windows (first FastEmbed download)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import rag_core

APP_NAME = "mY_Tutor"
FAVICON = Path(__file__).parent / "favicon.png"
MAX_VIDEOS = 5
MAX_UPLOAD_MB = rag_core.MAX_UPLOAD_MB

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
)

st.cache_resource(rag_core._get_fastembed)()

st.markdown(
    """
<style>
  footer {visibility: hidden;}
  /* No sidebar / left panel */
  section[data-testid="stSidebar"],
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"] {
    display: none !important;
  }
  .block-container { padding-top: 3rem; padding-bottom: 5rem; max-width: 42rem; }
  /* Reset — top-left only */
  .reset-anchor + div {
    position: fixed !important;
    top: 0.65rem !important;
    left: 1rem !important;
    z-index: 999 !important;
    width: auto !important;
    margin: 0 !important;
    padding: 0 !important;
  }
  .reset-anchor + div button {
    min-width: 4.5rem !important;
    height: 2.1rem !important;
    padding: 0 0.85rem !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
  }
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
    "videos": [],
    "active_video_num": 1,
    "messages": [],
    "sources": {},
    "suggested_questions": [],
    "done_sigs": set(),
    "pending_queue": [],
    "last_error": None,
    "is_processing": False,
    "processing_video_num": None,
    "answering": False,
    "_last_active_num": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v if not isinstance(v, set) else set()


def short_name(filename: str, max_len: int = 28) -> str:
    name = Path(filename).name
    return name if len(name) <= max_len else name[: max_len - 3] + "..."


def next_video_num() -> int:
    nums = [v["num"] for v in st.session_state.videos]
    nums += [p.get("video_num", 0) for p in st.session_state.pending_queue]
    return max(nums, default=0) + 1


def get_video(num: int) -> dict | None:
    for v in st.session_state.videos:
        if v["num"] == num:
            return v
    return None


def active_video() -> dict | None:
    v = get_video(st.session_state.active_video_num)
    if v:
        return v
    return st.session_state.videos[0] if st.session_state.videos else None


def active_df():
    v = active_video()
    if v and v.get("ready") and v.get("df") is not None:
        return v["df"]
    return None


def any_video_ready() -> bool:
    return any(v.get("ready") for v in st.session_state.videos)


def pick_pending_item() -> dict | None:
    queue = st.session_state.pending_queue
    if not queue:
        return None
    active = st.session_state.active_video_num
    for item in queue:
        if item.get("video_num") == active:
            return item
    return queue[0]


def sync_active_context() -> None:
    """Refresh chat when the user switches videos."""
    num = st.session_state.active_video_num
    if st.session_state.get("_last_active_num") == num:
        return
    st.session_state._last_active_num = num
    reset_chat()
    v = active_video()
    if not v:
        return
    if v.get("ready"):
        st.session_state.suggested_questions = list(v.get("suggested_questions") or [])
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Now asking about **Video {v['num']}** — {short_name(v['name'])}.",
            }
        )
    elif any(p.get("video_num") == num for p in st.session_state.pending_queue):
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"**Video {v['num']}** is queued. It will process when you continue.",
            }
        )


def file_sig(name: str, data: bytes) -> tuple:
    digest = hashlib.sha256(data).hexdigest()[:16]
    return (name, len(data), digest)


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.sources = {}
    st.session_state.suggested_questions = []


# Background transcription (so a ready video stays usable while another processes)
_BG_LOCK = threading.Lock()
_BG: dict = {"status": None}


def reset_lesson_state() -> None:
    """Drop all videos and chat."""
    reset_chat()
    st.session_state.videos = []
    st.session_state.active_video_num = 1
    st.session_state.done_sigs = set()
    st.session_state.pending_queue = []
    st.session_state.last_error = None
    st.session_state._last_active_num = None
    st.session_state.processing_video_num = None
    st.session_state.is_processing = False
    with _BG_LOCK:
        _BG["status"] = None


def can_chat() -> bool:
    if active_df() is None or st.session_state.answering:
        return False
    if not st.session_state.is_processing:
        return True
    proc = st.session_state.processing_video_num
    return proc is None or proc != st.session_state.active_video_num


def _process_item_core(item: dict) -> dict:
    num = item["video_num"]
    name = item["name"]
    data = item["bytes"]
    sig = item["sig"]
    label = f"Video {num}"
    new_df = rag_core.process_uploaded_media(
        data, name, language=None, video_label=label
    )
    questions = rag_core.build_suggested_questions(new_df, count=5)
    return {
        "num": num,
        "name": name,
        "sig": sig,
        "df": new_df.reset_index(drop=True),
        "questions": questions,
    }


def _apply_process_result(result: dict) -> None:
    num = result["num"]
    for v in st.session_state.videos:
        if v["num"] == num:
            v["df"] = result["df"]
            v["ready"] = True
            v["error"] = None
            v["suggested_questions"] = result["questions"]
            break
    st.session_state.done_sigs.add(result["sig"])
    st.session_state.pending_queue = [
        p for p in st.session_state.pending_queue if p["sig"] != result["sig"]
    ]
    if st.session_state.active_video_num == num:
        st.session_state.suggested_questions = result["questions"]


def _bg_worker(item: dict) -> None:
    try:
        result = _process_item_core(item)
        with _BG_LOCK:
            _BG["status"] = "done"
            _BG["result"] = result
    except Exception as e:
        with _BG_LOCK:
            _BG["status"] = "error"
            _BG["error"] = str(e)
            _BG["failed_item"] = item


def start_background_process(item: dict) -> None:
    with _BG_LOCK:
        if _BG.get("status") == "running":
            return
        _BG["status"] = "running"
        _BG["result"] = None
        _BG["error"] = None
    st.session_state.is_processing = True
    st.session_state.processing_video_num = item["video_num"]
    threading.Thread(target=_bg_worker, args=(item,), daemon=True).start()


def poll_background_job() -> None:
    with _BG_LOCK:
        status = _BG.get("status")
        if status not in ("done", "error"):
            return
        if status == "done":
            result = _BG["result"]
            _BG["status"] = None
        else:
            err = _BG["error"]
            item = _BG.get("failed_item", {})
            _BG["status"] = None

    st.session_state.is_processing = False
    st.session_state.processing_video_num = None

    if status == "done":
        _apply_process_result(result)
        num = result["num"]
        if st.session_state.active_video_num == num:
            reset_chat()
            st.session_state._last_active_num = num
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"**Video {num}** is ready. Ask below.",
                }
            )
        st.rerun()
    else:
        num = item.get("video_num", "?")
        name = item.get("name", "")
        st.session_state.last_error = f"**Video {num}** ({short_name(name)}): {err}"
        for v in st.session_state.videos:
            if v["num"] == num:
                v["error"] = err
                break
        st.session_state.pending_queue = [
            p for p in st.session_state.pending_queue if p.get("sig") != item.get("sig")
        ]
        st.rerun()


def kickoff_pending_work() -> None:
    """Start the next queued video — in background if another video is already ready."""
    poll_background_job()
    if st.session_state.is_processing or not st.session_state.pending_queue:
        return
    item = pick_pending_item()
    if not item:
        return
    active = active_video()
    if active and active.get("ready") and item["video_num"] != active["num"]:
        start_background_process(item)
    else:
        process_queue(item)


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
        switch = None
        if sum(1 for v in st.session_state.videos if v.get("ready")) > 1:
            switch = rag_core.suggest_video_switch(
                st.session_state.videos,
                st.session_state.active_video_num,
                question,
            )
        if switch:
            answer = (
                f"That looks like it belongs to **Video {switch['num']}** — "
                f"{short_name(switch['name'])}. "
                f"Please switch to that video above, then ask again."
            )
        else:
            with st.spinner("Thinking…"):
                df = active_df()
                if df is None:
                    raise ValueError("Select a processed video first.")
                answer, ctx = rag_core.answer_question(df, question)
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

    replacing = len(new_items) == 1 and not st.session_state.videos and not st.session_state.pending_queue
    if replacing:
        reset_lesson_state()
    else:
        reset_chat()

    start_num = next_video_num()
    for i, item in enumerate(new_items):
        num = start_num + i
        item["video_num"] = num
        st.session_state.videos.append(
            {
                "num": num,
                "name": item["name"],
                "sig": item["sig"],
                "ready": False,
                "df": None,
                "error": None,
                "suggested_questions": [],
            }
        )
        if len(new_items) == 1 and (replacing or not st.session_state.active_video_num):
            st.session_state.active_video_num = num

    if len(new_items) > 1:
        st.session_state.active_video_num = new_items[0]["video_num"]
    elif len(new_items) == 1 and not replacing:
        st.session_state.active_video_num = new_items[0]["video_num"]

    st.session_state.pending_queue.extend(new_items)


def process_queue(item: dict) -> None:
    """Process one video in the foreground (blocks until done)."""
    st.session_state.is_processing = True
    st.session_state.processing_video_num = item["video_num"]
    st.session_state.last_error = None
    num = item["video_num"]
    name = item["name"]
    sig = item["sig"]
    st.session_state.active_video_num = num

    batch_eta = max(60, rag_core.estimate_processing_seconds(item["bytes"], name))
    batch_start = time.time()

    st.markdown(f"#### Processing **Video {num}**")
    st.caption(f"{short_name(name)} · you can use other **Ready** videos once processing finishes.")
    progress = st.progress(0.0, text="Starting…")
    timer_box = st.empty()
    status_line = st.empty()

    def refresh_ui(stage: str, msg: str) -> None:
        elapsed = time.time() - batch_start
        remaining = max(0, int(batch_eta - elapsed))
        pct = STEP_PCT.get(stage, 0.5)
        label_step = STEP_LABEL.get(stage, stage)
        progress.progress(min(0.98, pct), text=f"Video {num} · {label_step}")
        status_line.markdown(f"**Video {num}** — {msg}")
        timer_box.markdown(
            f'<div class="timer-box">'
            f'<div style="color:#64748b;font-size:14px;">Video {num} · {label_step}</div>'
            f'<div class="timer-big">{fmt_time(remaining)}</div>'
            f'<div style="color:#94a3b8;font-size:12px;">estimated time left · elapsed {fmt_time(int(elapsed))}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    refresh_ui("prepare", "Starting…")

    def on_step(stage: str, msg: str) -> None:
        refresh_ui(stage, msg)

    try:
        result = _process_item_core(item)
        _apply_process_result(result)
        questions = result["questions"]
        elapsed = max(1, int(time.time() - batch_start))
        progress.progress(1.0, text="Finished")
        timer_box.markdown(
            f'<div class="timer-box" style="background:#ecfdf5;border-color:#6ee7b7;">'
            f'<div class="timer-big" style="color:#047857;">Done · {fmt_time(elapsed)}</div></div>',
            unsafe_allow_html=True,
        )
        st.success(f"**Video {num}** is ready — {short_name(name)}")
        reset_chat()
        st.session_state.suggested_questions = questions
        st.session_state._last_active_num = num
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"**Video {num}** is ready. Ask below.",
            }
        )
        if st.session_state.pending_queue:
            st.caption(
                f"{len(st.session_state.pending_queue)} more queued — keep chatting or switch videos."
            )
        st.balloons()
    except Exception as e:
        err = str(e)
        st.session_state.last_error = f"**Video {num}** ({short_name(name)}): {err}"
        for v in st.session_state.videos:
            if v["num"] == num:
                v["error"] = err
                break
        st.error(st.session_state.last_error)
        st.session_state.pending_queue = [
            p for p in st.session_state.pending_queue if p["sig"] != sig
        ]
    finally:
        st.session_state.is_processing = False
        st.session_state.processing_video_num = None


# --- API ---
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GROQ_API_KEY"):
    st.error("Add **GEMINI_API_KEY** or **GROQ_API_KEY** to `.env` or Streamlit Secrets.")
    st.stop()

def clear_all_session() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]


st.markdown('<div class="reset-anchor"></div>', unsafe_allow_html=True)
if st.button("Reset", key="reset", type="primary"):
    clear_all_session()
    st.rerun()

st.markdown(
    f'<div class="hero"><h1>{APP_NAME}</h1>'
    f"<p>Upload your lesson — we transcribe it automatically</p></div>",
    unsafe_allow_html=True,
)

if st.session_state.last_error and not st.session_state.is_processing:
    st.error(st.session_state.last_error)

uploaded = st.file_uploader(
    f"Upload video or audio (max {MAX_UPLOAD_MB} MB)",
    type=["mp4", "webm", "mov", "mkv", "mp3", "wav", "m4a", "ogg"],
    accept_multiple_files=True,
    label_visibility="collapsed",
    key="video_uploader",
)
if uploaded:
    if len(uploaded) > MAX_VIDEOS:
        st.warning(f"Only the first {MAX_VIDEOS} files are used.")
    valid_uploads = []
    for f in uploaded[:MAX_VIDEOS]:
        size_mb = len(f.getvalue()) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            st.error(
                f"**{f.name}** is {size_mb:.1f} MB. Max size is **{MAX_UPLOAD_MB} MB**."
            )
        else:
            valid_uploads.append(f)
    if valid_uploads:
        before = len(st.session_state.pending_queue)
        queue_uploads(valid_uploads)
        if (
            len(st.session_state.pending_queue) == before
            and not st.session_state.is_processing
            and any_video_ready()
        ):
            st.info("This file is already indexed. Click **×** on it or **Reset**, then upload again.")

if st.session_state.videos:
    def _video_option(v: dict) -> str:
        if v.get("ready"):
            tag = "Ready"
        elif st.session_state.processing_video_num == v["num"]:
            tag = "Processing"
        elif any(p.get("video_num") == v["num"] for p in st.session_state.pending_queue):
            tag = "Queued"
        elif v.get("error"):
            tag = "Failed"
        else:
            tag = "—"
        return f"Video {v['num']} · {short_name(v['name'])} ({tag})"

    options = [v["num"] for v in sorted(st.session_state.videos, key=lambda x: x["num"])]
    st.caption("Switch video — chat with any **Ready** video while others process")
    picked = st.radio(
        "Switch video",
        options=options,
        index=options.index(st.session_state.active_video_num)
        if st.session_state.active_video_num in options
        else 0,
        format_func=lambda n: _video_option(get_video(n) or {"num": n, "name": "?"}),
        horizontal=True,
        label_visibility="collapsed",
        key="video_switcher",
    )
    if picked != st.session_state.active_video_num:
        st.session_state.active_video_num = picked
        st.rerun()
    sync_active_context()

active = active_video()
if active and active.get("ready"):
    st.markdown(
        f'<div class="ready-banner">Ready · Video {active["num"]} · {short_name(active["name"])}</div>',
        unsafe_allow_html=True,
    )
elif any_video_ready():
    st.info("Selected video is not ready yet. Pick a **Ready** video or wait for processing.")

proc_num = st.session_state.processing_video_num
if st.session_state.is_processing and proc_num:
    pv = get_video(proc_num)
    pname = short_name(pv["name"]) if pv else ""
    st.info(
        f"Processing **Video {proc_num}** ({pname}) in the background. "
        f"You can keep chatting with other **Ready** videos."
    )

@st.fragment(run_every=timedelta(seconds=2))
def _poll_background_job() -> None:
    if st.session_state.is_processing and st.session_state.processing_video_num:
        poll_background_job()


_poll_background_job()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and str(i) in st.session_state.sources:
            with st.expander("Sources"):
                st.markdown(st.session_state.sources[str(i)])

if active_df() is not None and st.session_state.suggested_questions:
    st.markdown("##### Tap a question")
    cols = st.columns(2)
    busy = not can_chat()
    for i, q in enumerate(st.session_state.suggested_questions):
        if cols[i % 2].button(
            q, key=f"q{i}", use_container_width=True, disabled=busy
        ):
            ask_tutor(q)
            st.rerun()

if (
    active_df() is None
    and not st.session_state.pending_queue
    and not st.session_state.is_processing
    and not uploaded
):
    st.info("Drop a video above — processing starts automatically.")

prompt = st.chat_input(
    "Ask about your video…",
    disabled=not can_chat(),
)
if prompt and can_chat():
    ask_tutor(prompt)
    st.rerun()

kickoff_pending_work()

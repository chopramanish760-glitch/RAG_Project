"""RAG pipeline: Gemini (primary) or Groq (fallback) + FastEmbed."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash-lite")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))

_embedder = None
_groq_client = None
_gemini_client = None


def use_gemini() -> bool:
    return bool(GEMINI_API_KEY.strip())


def use_groq() -> bool:
    return bool(GROQ_API_KEY.strip())


def _ai_provider() -> str:
    return os.getenv("AI_PROVIDER", "auto").strip().lower()


def groq_primary() -> bool:
    """When true, use Groq for transcribe/chat; Gemini only if Groq fails."""
    if _ai_provider() == "groq":
        return use_groq()
    if _ai_provider() == "gemini":
        return False
    return use_groq() and os.getenv("PREFER_GROQ_TRANSCRIBE", "true").lower() in (
        "1",
        "true",
        "yes",
    )


def prefer_groq_transcribe() -> bool:
    """Use Groq Whisper for uploads (avoids Gemini free-tier video limits)."""
    if not use_groq():
        return False
    if groq_primary():
        return True
    val = os.getenv("PREFER_GROQ_TRANSCRIBE", "true").lower()
    return val in ("1", "true", "yes")


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota exceeded" in msg


def _retry_delay_seconds(exc: Exception, default: int = 45) -> int:
    match = re.search(r"retry in ([\d.]+)s", str(exc), re.I)
    if match:
        return min(90, int(float(match.group(1))) + 5)
    return default


def friendly_groq_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "429" in msg or "rate" in msg or "quota" in msg:
        return "Groq rate limit reached. Wait about a minute, then click **Try again**."
    return f"Groq error: {exc}"


def friendly_api_error(exc: Exception) -> str:
    if _is_rate_limit_error(exc):
        wait = _retry_delay_seconds(exc)
        extra = ""
        if use_groq():
            extra = " The app will try Groq automatically if configured."
        else:
            extra = (
                " Add a free **GROQ_API_KEY** in `.env` (https://console.groq.com/keys) "
                "so transcription can continue when Gemini is busy."
            )
        return (
            f"Gemini free-tier limit reached. Wait about **{wait} seconds**, then click "
            f"**Try again**. Check usage: https://ai.google.dev/gemini-api/docs/rate-limits.{extra}"
        )
    return str(exc)


def require_api_key() -> None:
    if use_gemini() or use_groq():
        return
    raise RuntimeError(
        "Set GEMINI_API_KEY or GROQ_API_KEY in .env or Streamlit Secrets. "
        "Gemini: https://aistudio.google.com/apikey"
    )


def _gemini():
    global _gemini_client
    require_api_key()
    if not use_gemini():
        raise RuntimeError("GEMINI_API_KEY is not set.")
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _groq():
    global _groq_client
    require_api_key()
    if not use_groq():
        raise RuntimeError("GROQ_API_KEY is not set.")
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _get_fastembed():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def format_timestamp(seconds: float) -> str:
    """Wall-clock time for video position (M:SS or H:MM:SS)."""
    s = max(0, int(round(float(seconds))))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def format_timestamp_range(start: float, end: float) -> str:
    a = format_timestamp(start)
    b = format_timestamp(end)
    return a if a == b else f"{a}–{b}"


_BAD_CLOCK_TS = re.compile(r"\b(\d{1,3}):(\d{2,})\b")


def fix_timestamps_in_text(text: str) -> str:
    """Fix LLM mistakes like 27:83 (meant 27.83 seconds → 0:27)."""

    def _fix(match: re.Match[str]) -> str:
        left, right = int(match.group(1)), int(match.group(2))
        if right < 60:
            return match.group(0)
        if left >= 180:
            return match.group(0)
        # Treat as decimal seconds (27:83 → 27.83s)
        secs = left + right / (10 ** len(match.group(2)))
        return format_timestamp(secs)

    return _BAD_CLOCK_TS.sub(_fix, text)


def _context_for_prompt(context_df: pd.DataFrame) -> str:
    rows = []
    for _, row in context_df.iterrows():
        rows.append(
            {
                "video": str(row.get("title", "Lesson")),
                "at": format_timestamp(float(row.get("start", 0))),
                "until": format_timestamp(float(row.get("end", 0))),
                "text": str(row.get("text", ""))[:400],
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


GEMINI_EMBED_BATCH_MAX = 100


def _gemini_embeddings(text_list: list[str]) -> list[list[float]]:
    """Gemini allows at most 100 texts per embed request."""
    if len(text_list) > GEMINI_EMBED_BATCH_MAX:
        model = _get_fastembed()
        return [vec.tolist() for vec in model.embed(text_list)]

    client = _gemini()
    all_vectors: list[list[float]] = []
    for start in range(0, len(text_list), GEMINI_EMBED_BATCH_MAX):
        batch = text_list[start : start + GEMINI_EMBED_BATCH_MAX]
        result = client.models.embed_content(model=GEMINI_EMBED_MODEL, contents=batch)
        all_vectors.extend([list(e.values) for e in result.embeddings])
    return all_vectors


def _use_gemini_embeddings() -> bool:
    return os.getenv("USE_GEMINI_EMBEDDINGS", "").lower() in ("1", "true", "yes") and use_gemini()


def _embed_fallback_needed(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        _is_rate_limit_error(exc)
        or "invalid_argument" in msg
        or "not_found" in msg
        or "404" in msg
    )


def create_embeddings(text_list: list[str]) -> list[list[float]]:
    """Local FastEmbed by default — avoids Gemini embed 404 / batch limits."""
    if not text_list:
        return []
    if _use_gemini_embeddings():
        try:
            return _gemini_embeddings(text_list)
        except Exception as e:
            if not _embed_fallback_needed(e):
                raise
    model = _get_fastembed()
    return [vec.tolist() for vec in model.embed(text_list)]


VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".mpeg", ".mpg", ".m4v"}


def find_ffmpeg() -> str:
    custom = os.getenv("FFMPEG_PATH", "").strip()
    if custom and Path(custom).exists():
        return custom
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in (
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"),
    ):
        if Path(candidate).exists():
            return candidate
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).exists():
            return bundled
    except ImportError:
        pass
    raise FileNotFoundError(
        "ffmpeg not found. Run: pip install imageio-ffmpeg  OR  winget install Gyan.FFmpeg"
    )


def video_to_mp3(video_path: str, mp3_path: str) -> None:
    ffmpeg = find_ffmpeg()
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "libmp3lame",
                "-q:a",
                "4",
                str(mp3_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e))[:200]
        raise RuntimeError(f"Could not extract audio from video. {err}") from e


def _safe_title(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\s\-]", "", stem, flags=re.UNICODE)
    return stem.strip() or "Uploaded video"


def _check_file_size(file_bytes: bytes) -> None:
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(
            f"File is {size_mb:.1f} MB (limit {MAX_UPLOAD_MB} MB). Use a shorter or compressed file."
        )


def _parse_json_segments(text: str) -> list[dict]:
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise ValueError("Gemini returned an invalid transcript format. Try again.") from None
        data = json.loads(match.group(0))
    if isinstance(data, dict) and "segments" in data:
        data = data["segments"]
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of transcript segments.")
    return data


def transcribe_with_gemini(
    media_path: str,
    title: str,
    *,
    language: str | None = None,
    on_pulse: Callable[[], None] | None = None,
) -> list[dict]:
    client = _gemini()
    path = Path(media_path)
    upload_path = str(path)
    # Safe copy for APIs that dislike spaces/parentheses in paths
    if re.search(r"[^\w.\-]", path.name):

        def _safe_copy() -> str:
            safe = re.sub(r"[^\w.\-]", "_", path.name) or "media.bin"
            dest = path.parent / safe
            dest.write_bytes(path.read_bytes())
            return str(dest)

        upload_path = _safe_copy()

    try:
        uploaded = client.files.upload(file=upload_path)
    except Exception as e:
        raise RuntimeError(f"Could not upload file to Gemini: {e}") from e

    for _ in range(120):
        meta = client.files.get(name=uploaded.name)
        state = getattr(meta, "state", None)
        state_name = getattr(state, "name", None) or str(state)
        if state_name == "ACTIVE":
            uploaded = meta
            break
        if state_name == "FAILED":
            raise RuntimeError("Gemini failed to process the uploaded audio file.")
        if on_pulse:
            on_pulse()
        time.sleep(1)

    ext = Path(media_path).suffix.lower()
    kind = "video" if ext in VIDEO_EXTENSIONS else "audio"
    lang_hint = f"Language: {language}." if language else "Detect the spoken language."
    prompt = f"""Transcribe this {kind} for a course titled "{title}".
{lang_hint}
Return ONLY a JSON array (no markdown), each item:
{{"start": <seconds float>, "end": <seconds float>, "text": "<spoken text>"}}
Split into natural phrase segments (roughly 5–30 seconds each)."""

    if on_pulse:
        on_pulse()
    response = None
    last_err = None
    for attempt in range(2):
        try:
            if on_pulse:
                on_pulse()
            response = client.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=[uploaded, prompt],
            )
            break
        except Exception as e:
            last_err = e
            if _is_rate_limit_error(e) and attempt == 0:
                time.sleep(_retry_delay_seconds(e))
                continue
            raise RuntimeError(friendly_api_error(e)) from e
    if response is None:
        raise RuntimeError(friendly_api_error(last_err or RuntimeError("No response")))
    if on_pulse:
        on_pulse()
    raw = (response.text or "").strip()
    if not raw:
        raise ValueError("Gemini returned an empty transcript. Try a shorter clip.")
    segments = _parse_json_segments(raw)

    chunks = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        chunks.append(
            {
                "number": "upload",
                "title": title,
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": text,
            }
        )
    return chunks


def transcribe_with_groq(
    audio_path: str,
    title: str,
    *,
    language: str | None = None,
) -> list[dict]:
    client = _groq()
    with open(audio_path, "rb") as f:
        kwargs = {
            "file": (Path(audio_path).name, f.read()),
            "model": GROQ_WHISPER_MODEL,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            kwargs["language"] = language
        result = client.audio.transcriptions.create(**kwargs)

    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    segments = data.get("segments") or []

    if segments:
        return [
            {
                "number": "upload",
                "title": title,
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": (seg.get("text") or "").strip(),
            }
            for seg in segments
            if (seg.get("text") or "").strip()
        ]

    full_text = (data.get("text") or "").strip()
    if not full_text:
        return []
    return [
        {
            "number": "upload",
            "title": title,
            "start": 0.0,
            "end": 0.0,
            "text": full_text,
        }
    ]


def transcribe_audio(
    audio_path: str,
    title: str,
    *,
    language: str | None = None,
    on_pulse: Callable[[], None] | None = None,
) -> list[dict]:
    if prefer_groq_transcribe():
        try:
            return transcribe_with_groq(audio_path, title, language=language)
        except Exception as e:
            if use_gemini() and not groq_primary():
                pass
            else:
                raise RuntimeError(friendly_groq_error(e)) from e
    if use_gemini() and not groq_primary():
        try:
            return transcribe_with_gemini(
                audio_path, title, language=language, on_pulse=on_pulse
            )
        except Exception as e:
            if _is_rate_limit_error(e) and use_groq():
                return transcribe_with_groq(audio_path, title, language=language)
            raise RuntimeError(friendly_api_error(e)) from e
    return transcribe_with_groq(audio_path, title, language=language)


def chunks_to_dataframe(chunks: list[dict], start_chunk_id: int = 0) -> pd.DataFrame:
    texts = [c["text"] for c in chunks]
    embeddings = create_embeddings(texts)
    rows = []
    for i, chunk in enumerate(chunks):
        row = dict(chunk)
        row["chunk_id"] = start_chunk_id + i
        row["embedding"] = embeddings[i]
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def process_uploaded_media(
    file_bytes: bytes,
    filename: str,
    *,
    on_step: Callable[[str, str], None] | None = None,
    language: str | None = None,
) -> pd.DataFrame:
    """Process media. on_step(stage, message) with stages: prepare|extract|transcribe|index."""

    def step(stage: str, msg: str) -> None:
        if on_step:
            on_step(stage, msg)

    require_api_key()
    _check_file_size(file_bytes)
    title = _safe_title(filename)
    ext = Path(filename).suffix.lower()
    is_audio = ext in {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}

    use_groq_path = prefer_groq_transcribe() or not use_gemini()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        media_path = tmp_path / f"input{ext or '.mp4'}"
        media_path.write_bytes(file_bytes)

        if is_audio:
            step("prepare", "Reading your file…")
            transcribe_path = str(media_path)
        elif use_groq_path:
            step("extract", "Getting audio from video…")
            transcribe_path = str(tmp_path / "audio.mp3")
            video_to_mp3(str(media_path), transcribe_path)
        else:
            step("prepare", "Uploading video to Gemini…")
            transcribe_path = str(media_path)

        _check_file_size(Path(transcribe_path).read_bytes())
        step("transcribe", "Writing transcript…")

        def pulse() -> None:
            step("transcribe", "Writing transcript…")

        try:
            chunks = transcribe_audio(
                transcribe_path,
                title=title,
                language=language,
                on_pulse=pulse,
            )
        except Exception as e:
            if _is_rate_limit_error(e) and use_groq():
                step("transcribe", "Gemini limit hit — switching to Groq…")
                if transcribe_path == str(media_path) and not is_audio:
                    step("extract", "Extracting audio for Groq…")
                    transcribe_path = str(tmp_path / "audio.mp3")
                    video_to_mp3(str(media_path), transcribe_path)
                chunks = transcribe_with_groq(
                    transcribe_path, title, language=language
                )
            else:
                raise RuntimeError(friendly_api_error(e)) from e

        if not chunks:
            raise ValueError("No speech detected in this file.")

        step("save", "Building search index…")
        return chunks_to_dataframe(chunks)


def estimate_processing_seconds(file_bytes: bytes, filename: str) -> int:
    """Rough ETA from file size (not exact)."""
    ext = Path(filename).suffix.lower()
    is_video = ext not in {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
    mb = len(file_bytes) / (1024 * 1024)
    base = 25 if is_video else 15
    per_mb = 10 if use_gemini() else 6
    return int(base + mb * per_mb)


def merge_dataframes(existing: pd.DataFrame | None, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new_df.reset_index(drop=True)
    next_id = (
        int(existing["chunk_id"].max()) + 1
        if "chunk_id" in existing.columns
        else len(existing)
    )
    new_df = new_df.copy()
    new_df["chunk_id"] = range(next_id, next_id + len(new_df))
    return pd.concat([existing, new_df], ignore_index=True)


def load_joblib(path: str | Path) -> pd.DataFrame:
    import joblib

    df = joblib.load(path)
    if "text" in df.columns:
        df = df.copy()
        df["embedding"] = create_embeddings(df["text"].tolist())
    return df


def build_prompt(query: str, context_df: pd.DataFrame, course_name: str) -> str:
    max_end = float(context_df["end"].max()) if "end" in context_df.columns else 0.0
    length_hint = format_timestamp(max_end)
    records = _context_for_prompt(context_df)
    return f"""You are a friendly course teaching assistant for "{course_name}".
Use the transcript excerpts below. When you mention a moment in the video, cite the exact "at" time from the excerpt (format like 0:27 or 1:05). Never write invalid clock times (seconds must be 00–59). Do not turn decimal seconds into colons (27.83 seconds is 0:27, not 27:83).
Video length is about {length_hint}. Do not cite times beyond that.
Be concise and helpful.

Transcript excerpts (times are already formatted):
{records}

Student question: {query}

If the question is unrelated to the course material, politely decline."""


def _answer_gemini(prompt: str) -> str:
    client = _gemini()
    last_err = None
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=prompt,
            )
            return (response.text or "").strip()
        except Exception as e:
            last_err = e
            if _is_rate_limit_error(e) and attempt == 0:
                time.sleep(_retry_delay_seconds(e))
                continue
            if _is_rate_limit_error(e) and use_groq():
                return _answer_groq(prompt)
            raise
    raise RuntimeError(friendly_api_error(last_err or RuntimeError("No response")))


def _answer_groq(prompt: str) -> str:
    client = _groq()
    response = client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You help students navigate course videos using transcript search results.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _prefer_groq_chat() -> bool:
    if not use_groq():
        return False
    if groq_primary():
        return True
    return os.getenv("PREFER_GROQ_CHAT", "true").lower() in ("1", "true", "yes")


def provider_label() -> str:
    if groq_primary() and use_groq():
        return "Groq"
    if use_gemini():
        return "Gemini"
    if use_groq():
        return "Groq"
    return "not configured"


def generate_answer(prompt: str) -> str:
    text = ""
    if _prefer_groq_chat() and use_groq():
        try:
            text = _answer_groq(prompt)
        except Exception:
            text = ""
    if not text and use_gemini():
        try:
            text = _answer_gemini(prompt)
        except Exception as e:
            if use_groq():
                text = _answer_groq(prompt)
            else:
                raise RuntimeError(friendly_api_error(e)) from e
    if not text:
        text = _answer_groq(prompt)
    return fix_timestamps_in_text(text)


def _fallback_suggested_questions(df: pd.DataFrame, count: int) -> list[str]:
    titles = [str(t) for t in df["title"].dropna().unique().tolist()]
    questions = [
        "Summarize all videos in this course",
        "What are the main topics covered across all lessons?",
    ]
    for title in titles[: max(0, count - 2)]:
        questions.append(f"What is explained in \"{title}\"?")
    if len(titles) > 1:
        questions.append("Which video should I start with and why?")
    if len(titles) >= 2:
        questions.append("Where are the most important concepts taught with timestamps?")
    return questions[:count]


def build_suggested_questions(
    df: pd.DataFrame,
    *,
    course_name: str = "My course",
    count: int = 5,
) -> list[str]:
    """Generate short question suggestions from indexed transcript."""
    if df is None or df.empty:
        return _fallback_suggested_questions(
            pd.DataFrame({"title": ["course"], "text": [""]}), count
        )

    titles = df["title"].dropna().unique().tolist()[:6]
    sample_rows = df.head(25)
    lines = []
    for _, row in sample_rows.iterrows():
        lines.append(f"[{row.get('title', 'Lesson')}] {str(row.get('text', ''))[:120]}")
    excerpt = "\n".join(lines)[:3500]
    title_list = ", ".join(str(t) for t in titles)

    prompt = f"""Course: "{course_name}"
Videos: {title_list}

Transcript samples:
{excerpt}

Write exactly {count} short questions a student would ask about THIS content.
Each question under 12 words. Reference actual topics from the transcript.
Return ONLY a JSON array of strings, no markdown.

Example: ["What is HTML?", "Where is CSS introduced?"]"""

    try:
        raw = generate_answer(prompt)
        text = raw.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if match:
                text = match.group(1).strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            cleaned = [str(q).strip() for q in parsed if str(q).strip()]
            if cleaned:
                return cleaned[:count]
    except (json.JSONDecodeError, TypeError, RuntimeError):
        pass

    return _fallback_suggested_questions(df, count)


def answer_question(
    df: pd.DataFrame,
    query: str,
    *,
    course_name: str = "My course",
    top_k: int = TOP_K,
) -> tuple[str, pd.DataFrame]:
    if df is None or df.empty:
        raise ValueError("Upload and index a video first.")

    q_emb = create_embeddings([query])[0]
    matrix = np.vstack(df["embedding"].values)
    similarities = cosine_similarity(matrix, [q_emb]).flatten()
    top_idx = similarities.argsort()[::-1][:top_k]
    context_df = df.loc[top_idx]
    prompt = build_prompt(query, context_df, course_name)
    return generate_answer(prompt), context_df

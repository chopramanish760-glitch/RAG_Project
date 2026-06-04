"""RAG pipeline: Gemini (primary) or Groq (fallback) + FastEmbed."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

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
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _gemini_embeddings(text_list: list[str]) -> list[list[float]]:
    client = _gemini()
    result = client.models.embed_content(model=GEMINI_EMBED_MODEL, contents=text_list)
    return [list(e.values) for e in result.embeddings]


def create_embeddings(text_list: list[str]) -> list[list[float]]:
    if not text_list:
        return []
    if use_gemini():
        return _gemini_embeddings(text_list)
    model = _get_fastembed()
    return [vec.tolist() for vec in model.embed(text_list)]


def video_to_mp3(video_path: str, mp3_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "4",
            mp3_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


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
    data = json.loads(text)
    if isinstance(data, dict) and "segments" in data:
        data = data["segments"]
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of segments.")
    return data


def transcribe_with_gemini(
    audio_path: str,
    title: str,
    *,
    language: str | None = None,
) -> list[dict]:
    client = _gemini()
    uploaded = client.files.upload(file=audio_path)

    for _ in range(120):
        meta = client.files.get(name=uploaded.name)
        state = getattr(meta, "state", None)
        state_name = getattr(state, "name", None) or str(state)
        if state_name == "ACTIVE":
            uploaded = meta
            break
        if state_name == "FAILED":
            raise RuntimeError("Gemini failed to process the uploaded audio file.")
        time.sleep(1)

    lang_hint = f"Language: {language}." if language else "Detect the spoken language."
    prompt = f"""Transcribe this audio for a course titled "{title}".
{lang_hint}
Return ONLY a JSON array (no markdown), each item:
{{"start": <seconds float>, "end": <seconds float>, "text": "<spoken text>"}}
Split into natural phrase segments (roughly 5–30 seconds each)."""

    response = client.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=[uploaded, prompt],
    )
    raw = (response.text or "").strip()
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
) -> list[dict]:
    if use_gemini():
        return transcribe_with_gemini(audio_path, title, language=language)
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

    provider = "Gemini" if use_gemini() else "Groq"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        media_path = tmp_path / f"input{ext or '.mp4'}"
        media_path.write_bytes(file_bytes)

        if is_audio:
            step("prepare", "Preparing audio…")
            audio_path = str(media_path)
        else:
            step("extract", "Extracting audio from video…")
            audio_path = str(tmp_path / "audio.mp3")
            video_to_mp3(str(media_path), audio_path)

        _check_file_size(Path(audio_path).read_bytes())
        step("transcribe", f"Transcribing with {provider}…")
        chunks = transcribe_audio(audio_path, title=title, language=language)
        if not chunks:
            raise ValueError("No speech detected in this file.")

        step("index", f"Building search index ({len(chunks)} segments)…")
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
    records = context_df[["title", "number", "start", "end", "text"]].to_json(orient="records")
    return f"""You are a friendly course teaching assistant for "{course_name}".
Use the transcript excerpts below. Cite video title and timestamp (MM:SS) when relevant.
Be concise and helpful.

Transcript excerpts:
{records}

Student question: {query}

If the question is unrelated to the course material, politely decline."""


def _answer_gemini(prompt: str) -> str:
    client = _gemini()
    response = client.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=prompt,
    )
    return (response.text or "").strip()


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


def generate_answer(prompt: str) -> str:
    if use_gemini():
        return _answer_gemini(prompt)
    return _answer_groq(prompt)


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

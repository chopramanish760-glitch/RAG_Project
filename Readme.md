# mY_Tutor

AI tutor for your videos — upload up to 5 lessons, get transcripts, then ask questions with timestamps.

## Quick start (local)

1. Get a **Gemini API key**: https://aistudio.google.com/apikey  
2. Copy `.env.example` → `.env` and set `GEMINI_API_KEY`.  
3. Install [ffmpeg](https://ffmpeg.org/download.html) (for video files).  
4. Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

5. Upload videos (wait until each is ready) → use suggested questions or chat.

**First run:** FastEmbed downloads a small model (~67 MB) once; HF Hub warnings on Windows are harmless.

## Deploy online

**Host:** [Streamlit Community Cloud](https://share.streamlit.io)

Full steps: **[DEPLOY.md](DEPLOY.md)**

## Stack

| Feature | Service |
|---------|---------|
| Transcription | Gemini |
| Chat & suggestions | Gemini |
| Embeddings | Local FastEmbed (BGE-small); optional Gemini |
| Fallback | Groq (optional) |

## Repo

https://github.com/chopramanish760-glitch/RAG_Project

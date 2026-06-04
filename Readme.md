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

5. Upload videos → **Index all** → use suggested questions or chat.

## Deploy online

**Host:** [Streamlit Community Cloud](https://share.streamlit.io)

Full steps: **[DEPLOY.md](DEPLOY.md)**

## Stack

| Feature | Service |
|---------|---------|
| Transcription | Gemini |
| Chat & suggestions | Gemini |
| Embeddings | Gemini `text-embedding-004` |
| Fallback | Groq (optional) |

## Repo

https://github.com/chopramanish760-glitch/RAG_Project

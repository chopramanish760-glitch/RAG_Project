# Course AI — RAG Teaching Assistant

ChatGPT-style UI: upload a video → fast Groq transcription → ask questions.

## Quick start (local)

1. Get a **Gemini API key**: https://aistudio.google.com/apikey  
2. Copy `.env.example` → `.env` and set `GEMINI_API_KEY` (or use the `.env` already created locally).  
3. Install [ffmpeg](https://ffmpeg.org/download.html) (for video files).  
4. Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

5. Attach a video → **Index** → chat when status shows **Ready**.

## Deploy online

**Host:** [Streamlit Community Cloud](https://share.streamlit.io) (free)

Full steps: see **[DEPLOY.md](DEPLOY.md)**

## Stack (fast APIs)

| Step | Service |
|------|---------|
| Transcription | Gemini (`gemini-2.0-flash` + audio upload) |
| Chat answers | Gemini (`gemini-2.0-flash`) |
| Search embeddings | Gemini (`text-embedding-004`) |
| Fallback | Groq Whisper + Llama if only `GROQ_API_KEY` is set |

## Batch pipeline (original scripts)

### Step 1 - Collect your videos
Move all your video files to the `videos` folder

### Step 2 - Convert to mp3
Run `video_to_mp3.py`

### Step 3 - Convert mp3 to json
Run `mp3_to_json.py`

### Step 4 - Convert json files to vectors
Run `preprocess_json.py` → creates `embeddings.joblib`

### Step 5 - CLI Q&A
Run `process_incoming.py`


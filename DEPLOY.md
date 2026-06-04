# Deploy Course AI (Streamlit Cloud)

**Recommended host:** [Streamlit Community Cloud](https://streamlit.io/cloud) — free, made for this stack, supports secrets + `packages.txt` (ffmpeg).

Alternatives: [Hugging Face Spaces](https://huggingface.co/spaces) (Docker), [Railway](https://railway.app/) (paid after trial).

---

## What you need before deploy

1. **GitHub account** — https://github.com
2. **Gemini API key** — https://aistudio.google.com/apikey (or Groq as fallback)  
3. This project pushed to a **public** GitHub repo (Streamlit free tier needs public repos)

---

## Step 1 — Push code to GitHub

Open terminal in the project folder:

```bash
git init
git add app.py rag_core.py requirements.txt packages.txt .streamlit DEPLOY.md Readme.md .env.example .gitignore
git commit -m "Course AI app ready for deploy"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Do **not** commit `.env` or API keys.

Optional: add `embeddings.joblib` only if you want the demo course button (file can be large).

---

## Step 2 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **New app**.
3. Choose:
   - **Repository:** `YOUR_USERNAME/YOUR_REPO`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Advanced settings** → **Secrets** and paste:

```toml
GEMINI_API_KEY = "your_gemini_key_here"
COURSE_NAME = "Sigma Web Development"
```

5. Click **Deploy**.

First build takes **5–10 minutes** (downloads FastEmbed model). Later deploys are faster.

---

## Step 3 — Test your live app

1. Open the URL Streamlit gives you (e.g. `https://your-app.streamlit.app`).
2. Upload a short **MP3** or small **MP4** (under ~24 MB for Groq).
3. Click **Index** — wait for “ready”.
4. Ask a question in the chat box.

---

## Local run (same as cloud)

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and set `GEMINI_API_KEY`.

Install **ffmpeg** for video: https://ffmpeg.org/download.html

```bash
streamlit run app.py
```

---

## Files Streamlit uses

| File | Purpose |
|------|---------|
| `app.py` | Main app (entry point) |
| `requirements.txt` | Python packages |
| `packages.txt` | Installs **ffmpeg** on Linux (for video → audio) |
| `.streamlit/config.toml` | Theme / server settings |
| Secrets | `GEMINI_API_KEY` |

---

## Limits & tips

| Topic | Detail |
|-------|--------|
| **Groq file size** | ~25 MB max per audio file |
| **Speed** | Groq Whisper + Llama is much faster than local Whisper/Ollama |
| **Cost** | Groq free tier has rate limits; fine for demos |
| **Private repo** | Streamlit paid plan, or use Hugging Face / Railway |
| **Custom domain** | Streamlit Teams / settings in Cloud dashboard |

---

## Hugging Face Spaces (alternative)

1. Create a Space → SDK **Streamlit**.
2. Upload the same files.
3. Add `GEMINI_API_KEY` under **Settings → Repository secrets**.
4. HF installs `packages.txt` via Dockerfile — for ffmpeg, use a `Dockerfile` with `RUN apt-get install -y ffmpeg` if needed.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| API key error | Add `GEMINI_API_KEY` in Streamlit Secrets and reboot app |
| ffmpeg error | Ensure `packages.txt` contains `ffmpeg` and redeploy |
| File too large | Shorter video or export audio-only MP3 |
| App sleeps / cold start | Normal on free tier; first user waits longer |
| Demo course button missing | Add `embeddings.joblib` to the repo (or only use upload) |

---

## Your live link

After deploy, share:

`https://<app-name>-<user>.streamlit.app`

Update `Readme.md` with that URL for your portfolio.

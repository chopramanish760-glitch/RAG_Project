# mY_Tutor

An AI-powered learning assistant that transforms educational videos into interactive lessons. Upload videos, generate transcripts, ask questions, and receive context-aware answers with timestamps using Retrieval-Augmented Generation (RAG).

## Features

* Upload and analyze video lessons
* Automatic transcription using Gemini
* AI-powered question answering
* Timestamp-based responses
* Suggested learning questions
* Multi-video support
* Semantic search with embeddings
* Retrieval-Augmented Generation (RAG)
* Fast and interactive Streamlit interface
* Local vector search with FastEmbed

## Tech Stack

### AI & LLMs

* Google Gemini
* Groq (optional fallback)
* RAG Pipeline

### Backend

* Python
* FastEmbed
* Vector Search

### Frontend

* Streamlit

### Media Processing

* FFmpeg

## How It Works

1. Upload a video lesson.
2. Generate transcript automatically.
3. Convert transcript into embeddings.
4. Store chunks for semantic retrieval.
5. Ask questions about the lesson.
6. Receive AI-generated answers with relevant timestamps.

## Installation

### Clone Repository

```bash
git clone https://github.com/chopramanish760-glitch/RAG_Project.git
cd RAG_Project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
```

### Run Application

```bash
streamlit run app.py
```

## Project Structure

```text
app.py
rag_core.py
requirements.txt
.env.example
DEPLOY.md
.streamlit/
```

## Use Cases

* Course revision
* Lecture Q&A
* Video learning assistant
* Educational content search
* AI-powered study companion

## Deployment

The application can be deployed using:

* Streamlit Community Cloud
* Render
* Docker
* Local Machine

## Author

Manish Chopra



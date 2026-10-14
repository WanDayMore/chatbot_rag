# 🛡️ Portfolio AI RAG Chatbot Service

A secure, lightweight, and cost-effective Retrieval-Augmented Generation (RAG) backend API for your portfolio website, powered by **Google Gemini** and **ChromaDB**.

---

## ✨ Features & Architecture

- **Zero Paid Subscriptions Required**:
  - Uses your Gemini API key (no third-party cloud DB or vector subscriptions required).
  - **Embedded ChromaDB**: Stores vector embeddings locally on disk for free.
  - **Cloudflare Turnstile**: Free, invisible CAPTCHA alternative to block bots, curl, and scrapers without user annoyance.
- **Semantic Query Caching**:
  - Automatically caches user questions and Gemini responses.
  - If a visitor asks an identical or semantically similar question (cosine similarity $\ge 0.92$, e.g., *"What is your tech stack?"* vs *"Which technologies do you know?"*), the cached response is returned in **<25ms** at **zero Gemini token cost**.
- **Multi-Tier Security & Abuse Prevention**:
  1. **Strict CORS Whitelist**: Only requests originating from your allowed portfolio domains are processed by browsers.
  2. **Cloudflare Turnstile Bot Defense**: Verifies cryptographic tokens on incoming requests.
  3. **Strict Rate Limiting**: Enforces 3 req/min, 10 req/hr, and 20 req/day per IP.
  4. **Daily Global Hard Cap Kill-Switch**: Caps overall requests at 500/day (configurable) to guarantee you never get a surprise bill.
- **Dual API Delivery**:
  - `POST /api/chat`: Standard JSON response with reply, sources, and latency.
  - `POST /api/chat/stream`: Server-Sent Events (SSE) streaming for real-time typewriter UI effect.

---

## 📁 Project Structure

```
chatbot_rag/
├── app/
│   ├── config.py              # Environment settings (Pydantic Settings)
│   ├── main.py                # FastAPI entrypoint & CORS middleware
│   ├── cache/
│   │   └── semantic_cache.py  # ChromaDB-backed vector similarity cache
│   ├── llm/
│   │   ├── gemini.py          # Google GenAI client (streaming & non-streaming)
│   │   └── prompts.py         # System prompt & portfolio persona guardrails
│   ├── models/
│   │   └── schemas.py         # Request & response Pydantic models
│   ├── rag/
│   │   ├── retriever.py       # ChromaDB vector retrieval with text-embedding-004
│   │   └── ingest.py          # Document loader (Markdown, TXT, JSON, PDF) & chunker
│   ├── routes/
│   │   └── chat.py            # /api/chat and /api/chat/stream endpoints
│   └── security/
│       ├── rate_limiter.py    # Sliding window rate limiter & daily hard cap
│       └── turnstile.py       # Cloudflare Turnstile token validator
├── data/
│   └── knowledge/             # Your portfolio documents (bio, resume, projects)
├── scripts/
│   └── ingest_cli.py          # CLI to index documents into ChromaDB
├── Dockerfile                 # Docker container for deployment
├── requirements.txt           # Dependencies
└── .env.example               # Environment variables template
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API Key ([Get one free from Google AI Studio](https://aistudio.google.com/))

### 2. Setup Environment

```bash
# Clone or open the repository
cd chatbot_rag

# Create virtual environment & activate
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure `.env`

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` with your values:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-pro
EMBEDDING_MODEL=text-embedding-004

ALLOWED_ORIGINS=http://localhost:3000,https://yourportfolio.com

# During local testing, keep TURNSTILE_ENABLED=false
TURNSTILE_ENABLED=false
TURNSTILE_SECRET_KEY=

# Rate Limits
RATE_LIMIT_PER_MINUTE=3
RATE_LIMIT_PER_HOUR=10
RATE_LIMIT_PER_DAY=20
DAILY_GLOBAL_HARD_CAP=500
SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.92
```

---

## 📚 Knowledge Base Ingestion

Add your resume, project summaries, bio, or FAQs into `data/knowledge/` in `.md`, `.txt`, `.json`, or `.pdf` format.

Run the ingestion CLI to chunk, embed, and store them into ChromaDB:

```bash
python scripts/ingest_cli.py --dir ./data/knowledge
```

---

## 🏃 Running the Server

Start the FastAPI development server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Health Check**: `http://localhost:8000/health`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`

---

## 🔒 Free Cloudflare Turnstile Setup (For Production)

To enable zero-friction, free bot defense on your live website:

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/) → **Turnstile**.
2. Click **Add Widget**:
   - Widget Mode: **Managed** (invisible challenge).
   - Domains: Add your portfolio domain (e.g., `yourportfolio.com`).
3. Copy:
   - **Site Key** (used in your frontend HTML/React).
   - **Secret Key** (put in your backend `.env` as `TURNSTILE_SECRET_KEY`).
4. Set `TURNSTILE_ENABLED=true` in your production `.env`.

---

## 🌐 Frontend Integration Examples

### Option A: Standard JSON Endpoint (`POST /api/chat`)

```javascript
async function askChatbot(userMessage, turnstileToken = null) {
  const response = await fetch("https://api.yourportfolio.com/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: userMessage,
      turnstile_token: turnstileToken
    })
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Error querying chatbot");
  }

  const data = await response.json();
  console.log("Answer:", data.reply);
  console.log("Served from cache?", data.cached);
  console.log("Sources:", data.sources);
  return data.reply;
}
```

### Option B: Streaming SSE Endpoint (`POST /api/chat/stream`)

```javascript
async function streamChatbot(userMessage, onChunk, turnstileToken = null) {
  const response = await fetch("https://api.yourportfolio.com/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: userMessage,
      turnstile_token: turnstileToken
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    const lines = text.split("\n");
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const payload = JSON.parse(line.replace("data: ", ""));
          if (payload.chunk) {
            onChunk(payload.chunk); // Append token to UI
          }
        } catch (e) {
          // ignore parse errors on heartbeat or end events
        }
      }
    }
  }
}
```

---

## 🚢 Free Deployment Guide

### Deploy to Render / Fly.io / Google Cloud Run:
This repository contains a production-ready `Dockerfile`.

1. **Google Cloud Run (Recommended)**:
   - Offers **2,000,000 free requests per month**.
   - Build and deploy:
     ```bash
     gcloud run deploy portfolio-chatbot --source . --port 8000 --allow-unauthenticated
     ```
   - Set environment variables in the Cloud Run console.

2. **Render**:
   - New Web Service -> Connect GitHub repo -> Environment: **Docker** -> Free tier.

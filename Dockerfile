# ================================================================
# 🧠 Brajen Semantic Engine v18.0 — Dockerfile
# ================================================================

FROM python:3.10-slim

# --- System setup ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl wget build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Copy dependencies first (layer caching) ---
COPY requirements.txt .

# --- Install dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Download Polish spaCy model (for NLP + semantic drift) ---
RUN python -m spacy download pl_core_news_lg || true

# --- Copy project files ---
COPY . .

# --- Environment variables ---
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV FIREBASE_CREDS_JSON=""
ENV DEBUG_MODE=false
ENV GEMINI_API_KEY=""

# --- Run Flask app (through Gunicorn for production) ---
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 master_api:app

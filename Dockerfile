# ================================================================
# 🧠 Brajen Semantic Engine v19.5 — Dockerfile (Final Review Ready)
# ================================================================

FROM python:3.10-slim

# --- System setup ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl wget build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Copy dependency list first (for build caching) ---
COPY requirements.txt .

# --- Install Python dependencies ---
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ================================================================
# 🧩 Install Polish spaCy model with fallback
# ================================================================
# 1️⃣ Try official model URL with version (stable)
# 2️⃣ If that fails — use spaCy CLI (auto-detect latest)
# 3️⃣ Prevent build failure on network hiccups
RUN python -m spacy validate || true && \
    (python -m spacy download pl_core_news_lg || \
     pip install https://github.com/explosion/spacy-models/releases/download/pl_core_news_lg-3.7.0/pl_core_news_lg-3.7.0.tar.gz || true)

# ================================================================
# 🧱 Copy application files
# ================================================================
COPY . .

# --- Environment configuration ---
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV FIREBASE_CREDS_JSON=""
ENV DEBUG_MODE=false
ENV GEMINI_API_KEY=""
ENV LANG=pl_PL.UTF-8
ENV LC_ALL=pl_PL.UTF-8

# --- Create non-root user for safety ---
RUN adduser --disabled-password --gecos '' brajenuser && chown -R brajenuser /app
USER brajenuser

# --- Healthcheck (optional for Render / Cloud Run) ---
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# ================================================================
# 🚀 Launch application
# ================================================================
# Gunicorn for production-ready serving
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 master_api:app

# ================================================================
# 🧠 Brajen Semantic Engine v19.6-LIGHT — Dockerfile (2 GB RAM Safe)
# ================================================================

FROM python:3.11-slim

# --- System setup ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl wget build-essential locales \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Polish locale ---
RUN sed -i '/pl_PL.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG=pl_PL.UTF-8
ENV LC_ALL=pl_PL.UTF-8

# --- Working directory ---
WORKDIR /app

# --- Dependencies ---
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ================================================================
# 🧩 Install smaller Polish SpaCy model (medium instead of large)
# ================================================================
RUN echo "⚙️ Installing lightweight Polish SpaCy model (md)..." && \
    (python -m spacy download pl_core_news_md || \
     pip install https://github.com/explosion/spacy-models/releases/download/pl_core_news_md-3.7.0/pl_core_news_md-3.7.0.tar.gz) && \
    python -m spacy validate && \
    python -m spacy info pl_core_news_md

# --- Optional test (NLP sanity check) ---
RUN python - <<'PYCODE'
import spacy
nlp = spacy.load("pl_core_news_md")
print("✅ SpaCy Polish model (MD) loaded successfully.")
PYCODE

# --- Copy project files ---
COPY . .

# --- Environment ---
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV FIREBASE_CREDS_JSON=""
ENV DEBUG_MODE=false
ENV GEMINI_API_KEY=""

# --- Non-root user ---
RUN adduser --disabled-password --gecos '' brajenuser && chown -R brajenuser /app
USER brajenuser

# --- Healthcheck ---
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

  # Force remove any cached spaCy large models
RUN pip uninstall -y pl-core-news-lg || true

# ================================================================
# 🚀 Run app (1 worker, low RAM mode)
# ================================================================
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 master_api:app

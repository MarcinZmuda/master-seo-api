# ================================================================
# 🧠 Brajen Semantic Engine v19.6 — Dockerfile (Render/Cloud Run Ready)
# ================================================================

FROM python:3.11-slim

# ================================================================
# 🔧 System setup
# ================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl wget build-essential locales \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Set locale (for Polish NLP) ---
RUN sed -i '/pl_PL.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG=pl_PL.UTF-8
ENV LC_ALL=pl_PL.UTF-8

# ================================================================
# 📁 Working directory
# ================================================================
WORKDIR /app

# ================================================================
# 📦 Dependency installation
# ================================================================
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ================================================================
# 🧩 Install Polish SpaCy model (3.7.0) — with fallback and verification
# ================================================================
RUN echo "⚙️ Installing Polish SpaCy model..." && \
    (python -m spacy download pl_core_news_lg || \
     pip install https://github.com/explosion/spacy-models/releases/download/pl_core_news_lg-3.7.0/pl_core_news_lg-3.7.0.tar.gz) && \
    python -m spacy validate && \
    python -m spacy info pl_core_news_lg

# ================================================================
# 🧠 Optional test (confirms NLP model installed correctly)
# ================================================================
RUN python - <<'PYCODE'
import spacy
try:
    nlp = spacy.load("pl_core_news_lg")
    print("✅ SpaCy Polish model loaded successfully.")
    doc = nlp("To jest test poprawnego działania modelu języka polskiego.")
    print("Example tokenization:", [t.text for t in doc[:5]])
except Exception as e:
    print("❌ SpaCy model test failed:", e)
PYCODE

# ================================================================
# 🧱 Copy application files
# ================================================================
COPY . .

# ================================================================
# ⚙️ Environment configuration
# ================================================================
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV FIREBASE_CREDS_JSON=""
ENV DEBUG_MODE=false
ENV GEMINI_API_KEY=""

# ================================================================
# 👤 Create non-root user (security best practice)
# ================================================================
RUN adduser --disabled-password --gecos '' brajenuser && chown -R brajenuser /app
USER brajenuser

# ================================================================
# 🩺 Healthcheck (for Render / Cloud Run)
# ================================================================
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# ================================================================
# 🚀 Launch Gunicorn (1 worker = less RAM, still multithreaded)
# ================================================================
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 master_api:app

# ================================================================
# 🧠 Brajen Semantic Engine v22.1 — Dockerfile
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

# --- Validate SpaCy Model ---
RUN python -m spacy validate

# --- Copy project files ---
COPY . .

# --- Environment Defaults ---
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV FINAL_REVIEW_MODEL="gemini-2.5-flash"

# --- Non-root user ---
RUN adduser --disabled-password --gecos '' brajenuser && chown -R brajenuser /app
USER brajenuser

# --- Healthcheck ---
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# ================================================================
# 🚀 Run app (1 worker, threaded for concurrency)
# ================================================================
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 master_api:app

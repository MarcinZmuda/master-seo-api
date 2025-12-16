#!/bin/bash
# ===============================================
# 🚀 run.sh — Render/Container bootstrap (v20)
# ===============================================
set -euo pipefail

echo "==============================================="
echo "🚀 SEO Master API starting..."
echo "🐍 Python: $(python --version)"
echo "📦 Environment: ${ENV:-production}"
echo "🌍 Port: ${PORT:-8080}"
echo "==============================================="

# --- Activate virtual environment if present ---
if [ -d "venv" ]; then
  source venv/bin/activate
  echo "✅ Virtualenv activated"
fi

# --- Ensure dependencies are installed ---
if [ -f "requirements.txt" ]; then
  echo "📦 Installing dependencies..."
  pip install --no-cache-dir -r requirements.txt
fi

# --- Ensure only lightweight spaCy model is present ---
python - <<'EOF'
import spacy
import sys
import subprocess
try:
    spacy.load("pl_core_news_md")
    print("✅ SpaCy model pl_core_news_md is available")
except Exception:
    print("⚙️ Installing SpaCy model: pl_core_news_md")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "https://github.com/explosion/spacy-models/releases/download/pl_core_news_md-3.7.0/pl_core_news_md-3.7.0-py3-none-any.whl"])
EOF

# --- Check Firestore credentials (required) ---
if [ -z "$FIREBASE_CREDS_JSON" ]; then
  echo "❌ FIREBASE_CREDS_JSON not set — Firebase is required"
  exit 1
else
  echo "✅ FIREBASE_CREDS_JSON environment variable detected"
fi

# --- Run basic healthcheck ---
echo "🔍 Running healthcheck..."
python - <<'EOF'
from master_api import app
try:
    print("✅ Master API import OK")
except Exception as e:
    print("❌ Master API import failed:", e)
    raise
EOF

# --- Start app ---
if command -v gunicorn &> /dev/null
then
  echo "🚀 Launching via Gunicorn (light mode)..."
  exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 2 master_api:app
else
  echo "⚙️ Gunicorn not found, starting Flask dev server..."
  python master_api.py
fi

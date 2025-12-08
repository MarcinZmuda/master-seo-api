#!/bin/bash
# ================================================================
# 🚀 Brajen Semantic Engine v19.6-LIGHT — Run Script (Safe for 2 GB)
# ================================================================

echo "🔥 Starting Brajen Semantic Engine (Master SEO API v19.6-LIGHT)"
echo "📅 $(date)"
echo "🐍 Python version: $(python3 --version)"
echo "📦 Environment: ${ENV:-production}"
echo "🌍 Port: ${PORT:-8080}"

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
python -m spacy validate | grep -q "pl_core_news_md" || {
  echo "⚙️ Installing lightweight SpaCy model: pl_core_news_md"
  python -m spacy download pl_core_news_md
}

# --- Force uninstall heavy model if exists ---
pip uninstall -y pl-core-news-lg || true

# --- Check Firestore credentials ---
if [ -z "$FIREBASE_CREDS_JSON" ]; then
  echo "⚠️ FIREBASE_CREDS_JSON not set (running in no-Firebase mode)"
else
  echo "✅ FIREBASE_CREDS_JSON environment variable detected"
fi

# --- Run basic healthcheck ---
echo "🔍 Running healthcheck..."
python - <<'EOF'
from master_api import app
try:
    print("✅ Master SEO API initialized successfully.")
except Exception as e:
    print("❌ Healthcheck failed:", e)
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

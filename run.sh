#!/bin/bash
# ================================================================
# 🚀 Brajen Semantic Engine v18.0 — Run Script
# ================================================================

echo "🔥 Starting Brajen Semantic Engine (Master SEO API v18.0)"
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

# --- Check for spaCy model ---
python -m spacy validate | grep -q "pl_core_news_lg" || {
  echo "⚙️ Installing missing spaCy model: pl_core_news_lg"
  python -m spacy download pl_core_news_lg
}

# --- Check Firestore credentials ---
if [ -z "$FIREBASE_CREDS_JSON" ]; then
  echo "❌ ERROR: Missing FIREBASE_CREDS_JSON environment variable!"
  exit 1
fi

# --- Run healthcheck first ---
echo "🔍 Running healthcheck..."
python - <<'EOF'
from firebase_admin import firestore
from master_api import app
try:
    db = firestore.client()
    print("✅ Firestore connected successfully.")
except Exception as e:
    print("❌ Firestore connection failed:", e)
EOF

# --- Start app with gunicorn or fallback ---
if command -v gunicorn &> /dev/null
then
  echo "🚀 Launching via Gunicorn..."
  exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 2 --threads 4 master_api:app
else
  echo "⚙️ Gunicorn not found, starting Flask dev server..."
  python master_api.py
fi

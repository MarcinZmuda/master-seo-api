#!/bin/bash
set -e

# Render ustawia PORT automatycznie.
# Jeśli nie – ustawiamy fallback na 10000.
PORT=${PORT:-10000}

echo "➡️ Starting Gunicorn on port ${PORT}..."

exec gunicorn master_api:app \
    --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --timeout 300 \
    --preload \
    --log-level info

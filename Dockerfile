# ===========================================================
# Dockerfile v12.25.6.6 - NO JAVA REQUIRED
# LanguageTool now uses Remote API
# RAM savings: ~500MB per worker
# ===========================================================

# Bazowy obraz z Pythonem
FROM python:3.11-slim

# Instalacja narzędzi build (BEZ JAVA!)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Katalog roboczy
WORKDIR /app

# Zależności Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy Polish model (if not already in requirements)
RUN python -m spacy download pl_core_news_lg || echo "spaCy model already installed"

# Kod aplikacji
COPY . .

# Uprawnienia do skryptu startowego
RUN chmod +x ./run.sh

# Health check (opcjonalne, ale zalecane)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Informacyjnie – aplikacja będzie nasłuchiwać na porcie z ENV (np. 10000)
EXPOSE 10000

# Start aplikacji
CMD ["./run.sh"]

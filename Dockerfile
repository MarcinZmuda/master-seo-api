# ===========================================================
# Dockerfile v12.25.6.7 - FIX SPACY DOWNLOAD ERROR
# LanguageTool uses Remote API (No Java)
# SpaCy model installed directly via pip URL (Fixes 404)
# ===========================================================

# Bazowy obraz z Pythonem
FROM python:3.11-slim

# Ustawienia zmiennych środowiskowych dla Pythona (lepsze logowanie i brak plików .pyc)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instalacja narzędzi build (BEZ JAVA!)
# build-essential jest potrzebny do kompilacji niektórych bibliotek Python (np. cffi, rapidfuzz)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Katalog roboczy
WORKDIR /app

# Zależności Pythona
COPY requirements.txt .
# Najpierw upgrade pip, potem instalacja requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# 🔧 FIX 404 ERROR: Instalacja modelu SpaCy bezpośrednio z pliku .whl
# Zamiast: python -m spacy download pl_core_news_lg
# Używamy bezpośredniego linku do wersji kompatybilnej ze spaCy 3.7.x
# ==============================================================================
RUN pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/pl_core_news_lg-3.7.0/pl_core_news_lg-3.7.0-py3-none-any.whl

# Kod aplikacji
COPY . .

# Uprawnienia do skryptu startowego
RUN chmod +x ./run.sh

# Health check (sprawdza czy aplikacja żyje co 30s)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Informacyjnie – aplikacja będzie nasłuchiwać na porcie z ENV (np. 10000)
EXPOSE 10000

# Start aplikacji
CMD ["./run.sh"]

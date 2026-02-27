#!/bin/bash

# Master SEO API v23.8 - Run Script

set -e

# Kolory
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   MASTER SEO API v23.8${NC}"
echo -e "${GREEN}========================================${NC}"

# Sprawdź czy Python jest zainstalowany
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 nie jest zainstalowany!${NC}"
    exit 1
fi

# Sprawdź zmienne środowiskowe
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo -e "${YELLOW}⚠️  GOOGLE_APPLICATION_CREDENTIALS nie ustawione${NC}"
    echo "   Ustaw: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/firebase-key.json"
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  GEMINI_API_KEY nie ustawione (Final Review nieaktywny)${NC}"
fi

# Opcje uruchomienia
MODE=${1:-dev}
PORT=${PORT:-8080}

case $MODE in
    dev)
        echo -e "${GREEN}🚀 Uruchamiam w trybie DEV (Flask debug)${NC}"
        echo ""
        export FLASK_ENV=development
        export FLASK_DEBUG=1
        python3 master_api.py
        ;;
    
    prod)
        echo -e "${GREEN}🚀 Uruchamiam w trybie PROD (Gunicorn)${NC}"
        echo ""
        gunicorn \
            --bind 0.0.0.0:$PORT \
            --workers 2 \
            --threads 2 \
            --timeout 180 \
            --keep-alive 5 \
            --worker-tmp-dir /dev/shm \
            --max-requests 200 \
            --max-requests-jitter 30 \
            --access-logfile - \
            --error-logfile - \
            master_api:app
        ;;
    
    docker)
        echo -e "${GREEN}🐳 Buduję i uruchamiam Docker${NC}"
        echo ""
        docker build -t master-seo-api:v23.8 .
        docker run -d \
            --name seo-api \
            -p $PORT:8080 \
            -e GOOGLE_APPLICATION_CREDENTIALS=/app/firebase-key.json \
            -e GEMINI_API_KEY=$GEMINI_API_KEY \
            -v $(pwd)/firebase-key.json:/app/firebase-key.json:ro \
            master-seo-api:v23.8
        echo -e "${GREEN}✅ Container uruchomiony na porcie $PORT${NC}"
        ;;
    
    docker-build)
        echo -e "${GREEN}🐳 Buduję obraz Docker${NC}"
        docker build -t master-seo-api:v23.8 .
        echo -e "${GREEN}✅ Obraz zbudowany: master-seo-api:v23.8${NC}"
        ;;
    
    test)
        echo -e "${GREEN}🧪 Uruchamiam testy${NC}"
        echo ""
        python3 -m pytest tests/ -v
        ;;
    
    install)
        echo -e "${GREEN}📦 Instaluję zależności${NC}"
        echo ""
        pip install -r requirements.txt
        python -m spacy download pl_core_news_md
        echo -e "${GREEN}✅ Zależności zainstalowane${NC}"
        ;;
    
    *)
        echo "Użycie: ./run.sh [dev|prod|docker|docker-build|test|install]"
        echo ""
        echo "  dev          - Flask development server (domyślne)"
        echo "  prod         - Gunicorn production server"
        echo "  docker       - Build i run w Docker"
        echo "  docker-build - Tylko build Docker image"
        echo "  test         - Uruchom testy"
        echo "  install      - Zainstaluj zależności"
        exit 1
        ;;
esac

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Kopiuj requirements i instaluj zależności
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pobierz model spaCy dla polskiego
RUN python -m spacy download pl_core_news_md

# Kopiuj kod aplikacji
COPY *.py .
COPY *.json .

# 🆕 v44.2: Kopiuj folder project_helpers/
COPY project_helpers/ ./project_helpers/

# 🆕 v37.0: Kopiuj folder medical_module/
COPY medical_module/ ./medical_module/

# 🆕 v44.2: Kopiuj folder tests/ (opcjonalne - dla CI/CD)
# COPY tests/ ./tests/

# Port
EXPOSE 8080

# Zmienne środowiskowe
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Uruchom
CMD ["python", "master_api.py"]

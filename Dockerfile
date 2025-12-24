FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Morfeusz2 - oficjalne repozytorium SGJP
RUN wget -qO- http://download.sgjp.pl/apt/sgjp.gpg.key | apt-key add - && \
    echo "deb http://download.sgjp.pl/apt/ubuntu jammy main" > /etc/apt/sources.list.d/sgjp.list && \
    apt-get update && \
    apt-get install -y morfeusz2 libmorfeusz2-dev python3-morfeusz2 && \
    rm -rf /var/lib/apt/lists/*

# Kopiuj requirements i instaluj zależności
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Morfeusz2 Python binding
RUN pip install --no-cache-dir morfeusz2 || echo "Morfeusz2 pip install failed, using system package"

# Pobierz model spaCy dla polskiego
RUN python -m spacy download pl_core_news_md

# Kopiuj kod aplikacji
COPY *.py .
COPY *.json .

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

# ===========================================================
# Dockerfile v12.25.6.6 FIXED - NO JAVA REQUIRED
# ===========================================================

FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download pl_core_news_lg

COPY . .

RUN chmod +x ./run.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

EXPOSE 10000

CMD ["./run.sh"]

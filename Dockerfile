FROM python:3.11-slim

# system dependencies required for spaCy
RUN apt-get update && apt-get install -y build-essential gcc

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download pl_core_news_sm

COPY . .

EXPOSE 10000

# Proper startup command: run master_api:app via gunicorn
CMD ["gunicorn", "master_api:app", "--bind", "0.0.0.0:10000"]

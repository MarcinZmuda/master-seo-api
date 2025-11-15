FROM python:3.11-slim

# Dependencies for spaCy
RUN apt-get update && apt-get install -y build-essential gcc

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model
RUN python -m spacy download pl_core_news_sm

# Copy application code
COPY . .

# Expose Render port
EXPOSE 10000

# Proper startup command
CMD ["gunicorn", "master_api:app", "--bind", "0.0.0.0:10000"]

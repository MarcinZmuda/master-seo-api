FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential gcc

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x ./run.sh

EXPOSE 10000

CMD ["gunicorn", "master_api:app", "--bind", "0.0.0.0:10000"]

# Krok 1: Wybierz oficjalny obraz Python jako bazę
FROM python:3.9-slim

# Krok 2: Zainstaluj narzędzia systemowe (jeśli będą potrzebne w przyszłości)
RUN apt-get update && apt-get install -y build-essential

# Krok 3: Ustaw katalog roboczy wewnątrz kontenera
WORKDIR /app

# Krok 4: Skopiuj plik z zależnościami i zainstaluj je
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Krok 5: Skopiuj resztę kodu aplikacji
COPY . .

# Krok 6: Poinformuj Docker, że aplikacja będzie działać na porcie 10000
EXPOSE 10000

# Krok 7: Zdefiniuj komendę, która uruchomi naszą NOWĄ aplikację master_api
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "master_api:app"]

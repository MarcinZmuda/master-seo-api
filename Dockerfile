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

# Krok 6: Skopiuj i przygotuj skrypt startowy
COPY run.sh .
RUN chmod +x ./run.sh

# Krok 7: Poinformuj Docker, że aplikacja będzie działać na porcie 10000
EXPOSE 10000

# Krok 8: Zdefiniuj komendę, która uruchomi nasz skrypt startowy
CMD ["./run.sh"]

# Bazowy obraz z Pythonem
FROM python:3.11-slim

# Instalacja Java (wymagana przez language_tool_python) + narzędzi build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        default-jre \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Katalog roboczy
WORKDIR /app

# Zależności Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kod aplikacji
COPY . .

# Uprawnienia do skryptu startowego
RUN chmod +x ./run.sh

# Informacyjnie – aplikacja będzie nasłuchiwać na porcie z ENV (np. 10000)
EXPOSE 10000

# Start aplikacji
CMD ["./run.sh"]

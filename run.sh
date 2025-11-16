#!/bin/sh
# Ten skrypt zapewnia, że Gunicorn jest uruchamiany w poprawnym kontekście

# Przejdź do katalogu aplikacji (chociaż Dockerfile już to robi, to dodatkowe zabezpieczenie)
cd /app

# Uruchom serwer Gunicorn, wskazując mu bezpośrednio plik i aplikację
exec gunicorn --bind 0.0.0.0:10000 master_api:app

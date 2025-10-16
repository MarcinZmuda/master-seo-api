import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe z pliku .env (przydatne do lokalnego testowania)
load_dotenv()

app = Flask(__name__)

# --- Konfiguracja adresów URL i kluczy API ---
# NAJLEPSZA PRAKTYKA: Przechowuj klucze w zmiennych środowiskowych na Renderze!
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract" # Upewnij się, że to poprawny URL
NGRAM_API_URL = "https://gpt-ngram-api-igyw.vercel.app/api/ngram_entity_analysis" # Przykładowy URL
HEADINGS_API_URL = "https://gpt-ngram-api-igyw.vercel.app/api/analyze_headings" # Przykładowy URL
SYNTHESIZE_API_URL = "https://gpt-ngram-api-igyw.vercel.app/api/synthesize_topics" # Przykładowy URL


# --- Funkcje pomocnicze do wywoływania innych API ---

def call_serpapi(topic):
    """Wywołuje SerpApi i zwraca wyniki."""
    params = {
        "api_key": SERPAPI_KEY,
        "q": topic,
        "gl": "pl",
        "hl": "pl",
        "engine": "google"
    }
    try:
        response = requests.get(SERPAPI_URL, params=params, timeout=20)
        response.raise_for_status()  # Rzuci błędem dla statusów 4xx/5xx
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Błąd podczas wywołania SerpApi: {e}")
        return None

def call_langextract(url):
    """Wywołuje API do ekstrakcji treści."""
    try:
        response = requests.post(LANGEXTRACT_API_URL, json={"url": url}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Błąd podczas ekstrakcji treści z {url}: {e}")
        return None
        
# TODO: Dodaj tutaj podobne funkcje dla ngram_api, headings_api i synthesize_api

# --- Główny endpoint naszego Master API ---

@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    data = request.get_json()
    topic = data.get("topic")

    if not topic:
        return jsonify({"error": "Brak parametru 'topic'"}), 400
    if not SERPAPI_KEY:
        return jsonify({"error": "Brak klucza SERPAPI_KEY w zmiennych środowiskowych"}), 500

    # 1. Pobierz dane z Google
    serp_data = call_serpapi(topic)
    if not serp_data:
        return jsonify({"error": "Nie udało się pobrać danych z SerpApi"}), 502

    organic_results = serp_data.get("organic_results", [])
    top_5_urls = [res.get("link") for res in organic_results[:5]]

    # 2. Ekstrakcja treści z każdego URL-a
    successful_sources = []
    source_processing_log = []
    combined_text = ""
    combined_h2s = []

    for url in top_5_urls:
        if len(successful_sources) >= 4:
            break # Mamy już 4 źródła
        
        content_data = call_langextract(url)
        if content_data and content_data.get("content"):
            successful_sources.append(content_data)
            source_processing_log.append({"url": url, "status": "Success"})
            combined_text += content_data.get("content", "") + "\n\n"
            combined_h2s.extend(content_data.get("h2", []))
        else:
            source_processing_log.append({"url": url, "status": "Failure"})

    # 3. TODO: Uruchom analizy (ngram, headings, synthesize)
    # W tym miejscu wywołałbyś kolejne API, przekazując im `combined_text` i `combined_h2s`
    # Na potrzeby tego przykładu, zwrócimy dane, które już mamy.
    
    # 4. Zbuduj finalną odpowiedź dla GPT
    final_response = {
        "identified_urls": top_5_urls,
        "processing_report": source_processing_log,
        "serp_features": {
            "ai_overview": serp_data.get("ai_overview"),
            "people_also_ask": serp_data.get("related_questions"),
            "featured_snippets": serp_data.get("answer_box") 
        },
        "analysis_results": {
            # TODO: Tutaj wstaw wyniki z API analitycznych
            "ngrams": "wynik z ngram_api",
            "headings_analysis": "wynik z headings_api",
            "synthesis": "wynik z synthesize_api"
        },
        "successful_sources_count": len(successful_sources)
    }

    return jsonify(final_response)


if __name__ == "__main__":
    app.run(port=os.getenv("PORT", 3000), debug=True)

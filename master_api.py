import os
import re
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe z pliku .env (przydatne do lokalnego testowania)
load_dotenv()

app = Flask(__name__)

# --- Konfiguracja adresów URL i kluczy API ---
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"
LANGEXTRACT_API_URL = "https://langextract-api.onrender.com/extract"

# --- POPRAWIONY ADRES URL ---
# Wskazuje na api/index.py (który jest mapowany na /api/)
NGRAM_API_URL = "https://gpt-ngram-api-igyw.vercel.app/api/" 
# ------------------------------

HEADINGS_API_URL = "https://gpt-ngram-api-igyw.vercel.app/api/analyze_headings"
SYNTHESIZE_API_URL = "https://gpt-ngram-api-igyw.vercel.app/api/synthesize_topics"

# --- Funkcje pomocnicze do wywoływania zewnętrznych API ---

def call_api_with_json(url, json_payload, service_name):
    """Generyczna funkcja do wywoływania API z payloadem JSON."""
    try:
        response = requests.post(url, json=json_payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Błąd podczas wywołania {service_name}: {e}")
        return {"error": f"Nie udało się połączyć z {service_name}", "details": str(e)}

def call_serpapi(topic):
    """Wywołuje SerpApi i zwraca wyniki."""
    params = {
        "api_key": SERPAPI_KEY, "q": topic, "gl": "pl", "hl": "pl", "engine": "google"
    }
    try:
        response = requests.get(SERPAPI_URL, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Błąd podczas wywołania SerpApi: {e}")
        return None

def call_langextract(url):
    """Wywołuje API do ekstrakcji treści."""
    return call_api_with_json(LANGEXTRACT_API_URL, {"url": url}, "LangExtract API")

# --- Nowe funkcje pomocnicze dla analiz S1 ---

# --- POPRAWIONA FUNKCJA (wysyła 'topic' jako 'main_keyword') ---
def call_ngram_api(text, topic):
    """Wywołuje API do analizy n-gramów i encji."""
    payload = {"text": text, "main_keyword": topic}
    return call_api_with_json(NGRAM_API_URL, payload, "N-gram API")
# -----------------------------------------------------------

def call_headings_api(headings_list):
    """Wywołuje API do analizy nagłówków."""
    return call_api_with_json(HEADINGS_API_URL, {"headings": headings_list}, "Headings API")

def call_synthesize_api(text):
    """Wywołuje API do syntezy tematów."""
    return call_api_with_json(SYNTHESIZE_API_URL, {"text": text}, "Synthesize API")


# --- GŁÓWNE ENDPOINTY APLIKACJI ---

@app.route("/api/s1_analysis", methods=["POST"])
def perform_s1_analysis():
    data = request.get_json()
    topic = data.get("topic")

    if not topic:
        return jsonify({"error": "Brak parametru 'topic'"}), 400
    if not SERPAPI_KEY:
        return jsonify({"error": "Brak klucza SERPAPI_KEY"}), 500

    # 1. Pobierz dane z Google
    serp_data = call_serpapi(topic)
    if not serp_data:
        return jsonify({"error": "Nie udało się pobrać danych z SerpApi"}), 502

    organic_results = serp_data.get("organic_results", [])
    top_5_urls = [res.get("link") for res in organic_results[:5]]

    # 2. Ekstrakcja treści z każdego URL-a
    successful_sources, source_processing_log = [], []
    combined_text, combined_h2s = "", []
    for url in top_5_urls:
        if len(successful_sources) >= 4: break
        
        content_data = call_langextract(url)
        if content_data and not content_data.get("error") and content_data.get("content"):
            successful_sources.append(content_data)
            source_processing_log.append({"url": url, "status": "Success"})
            combined_text += content_data.get("content", "") + "\n\n"
            combined_h2s.extend(content_data.get("h2", []))
        else:
            source_processing_log.append({"url": url, "status": "Failure", "reason": content_data.get("error", "Brak treści")})

    # 3. Uruchom analizy na połączonej treści
    
    # --- POPRAWIONE WYWOŁANIE (przekazuje 'topic') ---
    ngrams_result = call_ngram_api(combined_text, topic)
    # -----------------------------------------------

    headings_result = call_headings_api(combined_h2s)
    synthesis_result = call_synthesize_api(combined_text)

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
            "ngrams": ngrams_result,
            "headings_analysis": headings_result,
            "synthesis": synthesis_result
        },
        "successful_sources_count": len(successful_sources)
    }

    return jsonify(final_response)


# --- NOWY ENDPOINT DLA WERYFIKACJI S3 ---

def parse_keyword_string(keyword_data):
    """Parsuje string 'fraza (min-max)' na słownik."""
    if isinstance(keyword_data, dict): # Jeśli GPT przekaże już obiekt JSON
        return keyword_data

    keyword_dict = {}
    # Używamy regex do znalezienia frazy i opcjonalnego zakresu
    # np. "moja fraza (2-4)" lub "inna fraza"
    pattern = re.compile(r"^\s*(.+?)\s*(?:\((\d+)\s*-\s*(\d+)\))?\s*$")
    
    for line in keyword_data.splitlines():
        if not line.strip():
            continue
        match = pattern.match(line)
        if match:
            phrase, min_val, max_val = match.groups()
            phrase = phrase.strip()
            # Domyślne wartości, jeśli zakres nie został podany
            min_allowed = int(min_val) if min_val else 1
            max_allowed = int(max_val) if max_val else 5 # Domyślny max
            keyword_dict[phrase.lower()] = {
                "min_allowed": min_allowed,
                "max_allowed": max_allowed,
                "allowed_range": f"{min_allowed}-{max_allowed}"
            }
    return keyword_dict

@app.route("/api/s3_verify_keywords", methods=["POST"])
def verify_s3_keywords():
    data = request.get_json()
    text = data.get("text")
    keywords_with_ranges = data.get("keywords_with_ranges")

    if not isinstance(text, str) or not keywords_with_ranges:
        return jsonify({"error": "Brak 'text' lub 'keywords_with_ranges'"}), 400

    try:
        keywords_to_check = parse_keyword_string(keywords_with_ranges)
    except Exception as e:
        return jsonify({"error": f"Błąd parsowania keywords_with_ranges: {e}"}), 400

    text_lower = text.lower()
    keyword_report = {}

    for phrase, ranges in keywords_to_check.items():
        count = text_lower.count(phrase)
        status = "OK"
        if count < ranges["min_allowed"]:
            status = "UNDER"
        elif count > ranges["max_allowed"]:
            status = "OVER"
        
        keyword_report[phrase] = {
            "used": count,
            "min_allowed": ranges["min_allowed"],
            "max_allowed": ranges["max_allowed"],
            "allowed_range": ranges["allowed_range"],
            "status": status
        }
        
    return jsonify({"keyword_report": keyword_report})


if __name__ == "__main__":
    app.run(port=os.getenv("PORT", 3000), debug=True)
